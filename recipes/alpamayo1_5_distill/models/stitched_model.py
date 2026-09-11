# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Student VLM + the TEACHER's action expert, bolted together — the eval the KV term is for.

**Why this exists.**  ``L_KV`` trains the student's K/V cache to *be* the teacher's, and the
teacher's action expert is the thing that reads that cache.  Scoring the student's own
trajectory-token head therefore measures something the KV objective was never trying to
improve.  The endpoint that actually tests the hypothesis is: run the STUDENT's VLM,
hand its cache to the TEACHER's frozen expert, and score what the expert drives.

**Why this is even possible.**  The expert is not a cross-attention module with its own
K/V projections — it is a decoder that runs the noisy action tokens with
``past_key_values=prompt_cache`` (``alpamayo_r1.py:282-289``), i.e. it *self-attends over
the VLM's cache*.  So it consumes whatever K/V the VLM produced, and only the geometry has
to line up.  Verified against the real configs before this was written:

    expert config built on TEACHER vlm  vs  built on STUDENT vlm
      hidden_size          2048 == 2048       (from expert_cfg, NOT from the vlm)
      intermediate_size    8256 == 8256       (from expert_cfg)
      num_attention_heads    16 == 16         (from expert_cfg)
      head_dim              128 == 128        (from expert_cfg)
      num_hidden_layers      36 == 36         <- inherited from the vlm, and they match
      num_key_value_heads     8 == 8          <- inherited, and they match
      rope_theta        5000000 == 5000000    <- inherited, and they match
      rope_scaling  mrope_section [24,20,20]  <- inherited, identical
    only differing field: tie_word_embeddings, which is moot -- alpamayo_r1.py:98 does
    `del self.expert.embed_tokens`, so the expert has no embedding table at all.

``AlpamayoR1.__init__`` derives the expert config from ``self.vlm.config.text_config`` and
then overrides it with ``config.expert_cfg``.  The two fields where the towers actually
differ (hidden_size 4096 vs 2560, intermediate_size 12288 vs 9728) are *both* in
``expert_cfg``, so the expert comes out byte-identical either way and the teacher's 397
expert tensors load unchanged.  That is not luck — matching kv-heads x head_dim is why
Qwen3-VL-4B was picked as the student.

**What is NOT guaranteed** is that the student's cache is *semantically* what the expert
expects.  That is the hypothesis under test, not an assumption of the harness.
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import nullcontext
from typing import Any

import torch
import einops
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.models.base_model import IGNORE_INDEX
from alpamayo_r1.models.token_utils import to_special_token
from safetensors.torch import load_file

from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1
from alpamayo1_5_sft.models.sft_base_model import (
    ReasoningVLAOutput, TrainableReasoningVLA, load_alpamayo1_vlm,
)
from alpamayo1_5_distill.models.expert_conditioning import build_expert_conditioning
from alpamayo1_5_distill.models.layer_mix import LAYER_MIX_SHARPEN, LayerMixer

logger = logging.getLogger(__name__)

#: Everything in the teacher checkpoint that is NOT the VLM: the expert decoder, the
#: action-space projections. ``diffusion`` holds no parameters (it is a sampler).
TEACHER_PREFIXES = ("expert.", "action_in_proj.", "action_out_proj.")



class _SkippedExpertLayer(torch.nn.Module):
    """Identity stand-in for an ablated expert decoder layer.

    ⚠️ It REPLACES the layer in-place rather than shortening the ModuleList, so every
    surviving layer keeps its original index. That matters: expert layer ``l`` reads VLM
    cache layer ``l``, so re-indexing would silently re-pair every layer above the cut with
    the wrong cache and measure something else entirely.

    The skipped layer never appends its action K/V to cache slot ``l``; nothing reads that
    slot once the layer is gone, and the other slots are untouched.
    """

    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

    def forward(self, hidden_states, *args, **kwargs):
        # ⚠️ Return the BARE tensor, matching what this transformers version's Qwen3 decoder
        # layer returns. Wrapping it in a tuple crashes the caller with
        # "'tuple' object has no attribute 'dtype'" -- the enclosing model uses the result
        # directly, it does not unpack it.
        return hidden_states


def _apply_expert_pruning(model) -> None:
    """Ablate expert layers named by ``PRUNE_EXPERT_LAYERS`` (comma-separated indices).

    A measurement tool, not a deployment path -- the compute is unchanged, the layer is
    simply bypassed. It exists to test causally whether layers that CKA calls near-identity
    can be removed without hurting the trajectory.
    """
    spec = os.environ.get("PRUNE_EXPERT_LAYERS", "").strip()
    if not spec:
        return
    idx = sorted({int(x) for x in spec.split(",") if x.strip()})
    layers = model.expert.layers
    bad = [i for i in idx if not 0 <= i < len(layers)]
    if bad:
        raise ValueError(f"PRUNE_EXPERT_LAYERS out of range for {len(layers)} layers: {bad}")
    for i in idx:
        layers[i] = _SkippedExpertLayer(i)
    logger.warning(
        "[stitch] PRUNED expert layers %s -- %d of %d bypassed (identity)",
        idx, len(idx), len(layers),
    )
    print(f"[stitch] PRUNED expert layers {idx} ({len(idx)}/{len(layers)})", flush=True)


def _load_teacher_non_vlm(checkpoint_path: str, model: torch.nn.Module) -> None:
    """Load the teacher's expert + action projections, leaving ``vlm.*`` untouched.

    The complement of ``sft_base_model.load_alpamayo1_vlm``, which takes only ``vlm.*``.
    Between them the stitched model is: student VLM, teacher everything-else.
    """
    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"no model.safetensors.index.json under {checkpoint_path}")
    with open(index_path) as fh:
        weight_map = json.load(fh)["weight_map"]

    wanted = [k for k in weight_map if k.startswith(TEACHER_PREFIXES)]
    if not wanted:
        raise ValueError(
            f"{checkpoint_path} has no {TEACHER_PREFIXES} tensors -- is this a full Alpamayo "
            "checkpoint, or a vlm-only one?"
        )

    state: dict[str, torch.Tensor] = {}
    for shard in sorted({weight_map[k] for k in wanted}):
        shard_sd = load_file(os.path.join(checkpoint_path, shard))
        state.update({k: v for k, v in shard_sd.items() if k.startswith(TEACHER_PREFIXES)})
    # Captured BEFORE the remap below renames keys: the "was every wanted tensor actually read
    # from a shard?" check has to run on checkpoint names, not remapped ones.
    loaded_orig = set(state)

    # ⚠️ DEPTH REMAP, mirroring expert_holder._load. The expert's layer count is derived from
    # the VLM's text config, so stitching a 28-layer student (Cosmos-Reason2-2B) builds a
    # 28-layer expert while this checkpoint holds 36 -- and the strict checks below would fire
    # on unexpected=['expert.layers.28...']. PRUNE_EXPERT_LAYERS names the teacher layers to
    # DROP; the survivors in order are pi, and teacher expert layer pi(j) becomes slot j, the
    # SAME map training used. Getting this wrong does not crash, it silently evaluates a
    # differently-wired expert, so the count is asserted rather than trusted.
    # NOTE: this is the alternative to _apply_expert_pruning, not a companion to it. That one
    # keeps 36 slots and bypasses 8 (the ablation); this one builds 28 real ones.
    n_have = len(model.expert.layers)
    n_ckpt = 1 + max((int(m.group(1)) for m in
                      (re.match(r"expert\.layers\.(\d+)\.", k) for k in state) if m),
                     default=-1)
    if n_ckpt > n_have:
        drop = {int(x) for x in os.environ.get("PRUNE_EXPERT_LAYERS", "").split(",") if x.strip()}
        surv = [i for i in range(n_ckpt) if i not in drop]
        if len(surv) != n_have:
            raise RuntimeError(
                f"expert depth {n_have} but checkpoint has {n_ckpt}; PRUNE_EXPERT_LAYERS must "
                f"drop exactly {n_ckpt - n_have} layers (currently drops {len(drop)})")
        remap = {pi: j for j, pi in enumerate(surv)}
        out: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            m = re.match(r"(expert\.layers\.)(\d+)(\..*)", k)
            if not m:
                out[k] = v
                continue
            if int(m.group(2)) in remap:
                out[f"{m.group(1)}{remap[int(m.group(2))]}{m.group(3)}"] = v
        # `wanted`/`unloaded` below are keyed by the ORIGINAL names, so drop the 8 removed
        # layers from the expectation too -- otherwise a correct load reports them unloaded.
        for k in list(wanted):
            m = re.match(r"expert\.layers\.(\d+)\.", k)
            if m and int(m.group(1)) not in remap:
                wanted.remove(k)
        state = {k: v for k, v in out.items()}
        logger.warning("[stitch] expert DEPTH REMAP %d -> %d layers; dropped %s",
                       n_ckpt, n_have, sorted(drop))
        print(f"[stitch] expert REMAP {n_ckpt}->{n_have}, dropped {sorted(drop)}", flush=True)

    missing, unexpected = model.load_state_dict(state, strict=False)
    # `missing` is dominated by every vlm.* key, which is correct and expected here -- the
    # student's VLM weights are already in place and must NOT be overwritten. What must be
    # empty is `unexpected`, and no teacher-side key may be left unloaded.
    # ⚠️ compare on the PRE-remap names: the remap renames keys, so checking `state`
    # directly would report every renamed tensor as unloaded.
    unloaded = sorted(set(wanted) - loaded_orig)
    # A missing teacher-side key is only a real fault if it is a PARAMETER. Deterministic
    # buffers -- e.g. action_in_proj.sinus.*.freqs, the sinusoidal frequency tables -- are
    # recomputed at __init__ and are absent from the checkpoint by design, so demanding them
    # would reject a perfectly correct load. Anything that carries learned values must be
    # present, and that is what this distinguishes.
    params = dict(model.named_parameters())
    missing_params = [k for k in missing if k.startswith(TEACHER_PREFIXES) and k in params]
    missing_buffers = [k for k in missing if k.startswith(TEACHER_PREFIXES) and k not in params]
    if unexpected or missing_params or unloaded:
        raise RuntimeError(
            "teacher expert did not load cleanly: "
            f"unexpected={unexpected[:5]} missing_params={missing_params[:5]} unloaded={unloaded[:5]}"
        )
    if missing_buffers:
        logger.info(f"[stitch] {len(missing_buffers)} non-checkpointed buffers left at init: {missing_buffers}")
    logger.info(
        f"[stitch] loaded {len(state)} teacher tensors "
        f"({sum(1 for k in state if k.startswith('expert.'))} expert) from {checkpoint_path}; "
        f"vlm.* left as the student's"
    )


def _layer_mix_keys(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """``layer_mixer.*`` tensors in a checkpoint, across the sharded and single-file layouts."""
    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    single_path = os.path.join(checkpoint_path, "model.safetensors")
    state: dict[str, torch.Tensor] = {}
    if os.path.exists(index_path):
        with open(index_path) as fh:
            weight_map = json.load(fh)["weight_map"]
        wanted = [k for k in weight_map if k.startswith("layer_mixer.")]
        for shard in sorted({weight_map[k] for k in wanted}):
            shard_sd = load_file(os.path.join(checkpoint_path, shard))
            state.update({k: v for k, v in shard_sd.items() if k.startswith("layer_mixer.")})
    elif os.path.exists(single_path):
        shard_sd = load_file(single_path)
        state = {k: v for k, v in shard_sd.items() if k.startswith("layer_mixer.")}
    else:
        raise FileNotFoundError(f"no model.safetensors[.index.json] under {checkpoint_path}")
    return state


def _refuse_orphaned_layer_mix(checkpoint_path: str) -> None:
    """Raise if a checkpoint carries mixing matrices that this stitch would ignore."""
    if not checkpoint_path:
        return
    try:
        found = _layer_mix_keys(checkpoint_path)
    except FileNotFoundError:
        return                      # not a checkpoint dir; from_pretrained_vlm will say so
    if found:
        raise RuntimeError(
            f"{checkpoint_path} carries {len(found)} `layer_mixer.*` tensors but this config "
            "has layer_mix=False. Its VLM was trained to feed a FULL-depth expert through "
            "those matrices; evaluated without them the expert is built shallow and the "
            "matrices are silently dropped, which scores a model that was never trained. "
            "Use configs/sft_eval_stitched_2b_layermix_lcdrive.yaml."
        )


def _load_layer_mix(checkpoint_path: str, model: torch.nn.Module) -> None:
    """Load the trained ``layer_mixer.*`` from the student checkpoint. Raises if absent.

    ⚠️ THIS FUNCTION EXISTS BECAUSE OF A SILENT-REVERT TRAP. ``load_alpamayo1_vlm`` filters
    the checkpoint down to keys starting with ``vlm.`` (``sft_base_model.py:96,117``), so
    every non-VLM parameter a training arm learned is DROPPED without a word. For the mixing
    matrices that would mean evaluating the untrained tent init while the log says the
    checkpoint loaded fine -- a number that is finite, plausible, and about a model nobody
    trained. Hence the explicit load, and the hard failure when nothing is found.

    (``kv_projector`` has the same latent bug today. It has never bitten only because the
    shipped configs use ``kv_align: direct``, which allocates no parameters at all.)
    """
    directory = checkpoint_path
    state = _layer_mix_keys(directory)
    if not state:
        raise RuntimeError(
            f"layer_mix is enabled but {directory} contains no `layer_mixer.*` tensors. "
            "Either this checkpoint came from an arm trained WITHOUT layer_mix (in which case "
            "its 28-layer cache cannot drive this 36-layer expert at all), or the mixing "
            "matrices were dropped on save. Refusing to evaluate the untrained tent init "
            "while reporting it as a trained result."
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in unexpected if k.startswith("layer_mixer.")]
    if bad:
        raise RuntimeError(f"layer_mixer tensors do not match this mixer: {bad[:5]}")
    print(f"[stitch] loaded {len(state)} layer_mixer tensors from {directory}", flush=True)
    print(model.layer_mixer.describe(), flush=True)


class StitchedAlpamayoR1(AlpamayoR1, TrainableReasoningVLA):
    """``AlpamayoR1`` whose VLM is the distilled student and whose expert is the teacher's.

    Inherits ``AlpamayoR1`` for the expert/diffusion machinery and
    ``TrainableReasoningVLA`` for ``from_pretrained_vlm`` (which is what knows how to build
    a Qwen3-VL tower with the +4000 trajectory vocabulary).
    """

    @staticmethod
    def _read_raw_config(checkpoint_path: str) -> dict[str, Any]:
        """The whole ``config.json``, resolving both layouts ``_read_alpamayo_config`` accepts.

        Flat local dir, or an HF hub cache dir whose real config sits under
        ``snapshots/<sha>/``.
        """
        path = os.path.join(checkpoint_path, "config.json")
        if not os.path.exists(path):
            snaps = os.path.join(checkpoint_path, "snapshots")
            if os.path.isdir(snaps):
                candidates = sorted(os.listdir(snaps))
                if candidates:
                    path = os.path.join(snaps, candidates[-1], "config.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"config.json not found under {checkpoint_path}")
        with open(path) as fh:
            return json.load(fh)

    @classmethod
    def from_stitch(
        cls,
        vlm_name_or_path: str,
        alpamayo_config_path: str,
        checkpoint_path: str,
        teacher_checkpoint_path: str,
        layer_mix: bool = False,
        layer_mix_expert_layers: int = 36,
        layer_mix_blocks: int = 4,
        layer_mix_gain: bool = False,
        layer_mix_sharpen: float = LAYER_MIX_SHARPEN,
        layer_mix_pin_head: int = 0,
        layer_mix_pin_tail: int = 0,
        layer_mix_init: str = "checkpoint",
        **kwargs: Any,
    ) -> "StitchedAlpamayoR1":
        """Build student-VLM + teacher-expert.

        ⚠️ Parameters are EXPLICIT rather than ``**kwargs``-forwarded. ``from_pretrained_vlm``
        passes unrecognised kwargs straight into the config, where a typo becomes a silently
        ignored field instead of an error -- the bug class that cost this project a run.

        Args:
            vlm_name_or_path: the student's base tower (Qwen3-VL-4B snapshot).
            alpamayo_config_path: the 10B A1-format dir. Supplies the trajectory vocabulary
                AND ``expert_cfg`` / ``diffusion_cfg`` / ``action_*_cfg`` -- i.e. this is
                what makes the skeleton an AlpamayoR1 rather than a bare VLA.
            checkpoint_path: a trained arm's ``checkpoint-*`` dir (``vlm.*`` only). Named
                `checkpoint_path` and not something clearer on purpose: evaluate_hf.py:70
                assigns `cfg.model.checkpoint_path = cfg.evaluate.eval_ckpt` for every
                TrainableReasoningVLA subclass, so this name is what lets --eval_ckpt
                reach the student without patching the shared eval entry point.
            teacher_checkpoint_path: the 10B checkpoint to take the expert from.
            layer_mix: keep the expert at its FULL depth and synthesise its cache slots from
                the student's shallower stack, instead of pruning it to match. Must be set
                exactly as the training arm set it -- a student trained with the mix produces
                a cache that only means anything through that same mix, and one trained
                without it has no `layer_mixer.*` to load. See ``models/layer_mix.py``.
            layer_mix_init: where the mixing matrices come from. ``"checkpoint"`` (the
                default, and the only correct setting for scoring a trained arm) loads them
                from ``checkpoint_path`` and RAISES if they are absent. ``"tent"`` builds them
                fresh at the depth-matched init and loads nothing -- for probes that supply
                their own P (``scripts/layer_mix_oracle.py``) and for the zero-shot sanity
                eval of an arm trained WITHOUT the mix. ⚠️ ``"tent"`` silently ignores any
                trained matrices in the checkpoint, so it must never be the default.
        """
        # `from_pretrained_vlm` builds its config via the hard-coded target string
        # f"alpamayo_r1.models.base_model.{cls.config_class.__name__}"
        # (sft_base_model.py:342). That holds for ReasoningVLAConfig, which lives there --
        # but AlpamayoR1Config lives in `alpamayo_r1.config`, so the lookup raises
        # "Error locating target 'alpamayo_r1.models.base_model.AlpamayoR1Config'".
        # Publishing the alias is a two-line shim; the alternative is duplicating ~40 lines
        # of config assembly here, which would then silently drift from the sft recipe.
        import alpamayo_r1.models.base_model as _base_model

        if not hasattr(_base_model, cls.config_class.__name__):
            setattr(_base_model, cls.config_class.__name__, cls.config_class)

        # `_read_alpamayo_config` lifts ONLY the trajectory-tokenizer and image-resolution
        # fields (sft_base_model.py:~360) -- it knows nothing about AlpamayoR1. Every block
        # that defines the expert is therefore dropped, and the model builds with
        # action_space=None, failing later with a bare
        #   AttributeError: 'NoneType' object has no attribute 'get_action_space_dims'
        # which names neither the missing key nor the reason. Read them off the teacher's
        # config.json and pass them through; `from_pretrained_vlm` merges explicit kwargs
        # over whatever it read (sft_base_model.py:330).
        expert_keys = (
            "expert_cfg",
            "diffusion_cfg",
            "action_space_cfg",
            "action_in_proj_cfg",
            "action_out_proj_cfg",
            "expert_non_causal_attention",
            "keep_same_dtype",
        )
        teacher_cfg = cls._read_raw_config(alpamayo_config_path)
        expert_kwargs = {k: teacher_cfg[k] for k in expert_keys if k in teacher_cfg}
        if layer_mix:
            if os.environ.get("PRUNE_EXPERT_LAYERS", "").strip():
                raise ValueError(
                    "layer_mix=True with PRUNE_EXPERT_LAYERS set. These are alternatives: "
                    "mixing exists so the expert need NOT be pruned. Unset the env var."
                )
            # ⚠️ `num_hidden_layers` is deliberately ABSENT from the teacher's `expert_cfg`,
            # which is exactly why expert depth follows the VLM (AlpamayoR1.__init__ copies
            # text_config then overlays expert_cfg). Adding it here is the whole mechanism
            # that keeps the expert 36 deep on a 28-layer student.
            expert_kwargs["expert_cfg"] = {
                **(expert_kwargs.get("expert_cfg") or {}),
                "num_hidden_layers": int(layer_mix_expert_layers),
            }
        absent = [k for k in ("expert_cfg", "diffusion_cfg", "action_space_cfg") if k not in expert_kwargs]
        if absent:
            raise ValueError(
                f"{alpamayo_config_path}/config.json has no {absent} -- this is not an "
                "AlpamayoR1 checkpoint, so there is no action expert to stitch."
            )

        model = cls.from_pretrained_vlm(
            vlm_name_or_path=vlm_name_or_path,
            alpamayo_config_path=alpamayo_config_path,
            checkpoint_path=checkpoint_path,
            **expert_kwargs,
            **kwargs,
        )
        if not layer_mix:
            # ⚠️ THE SYMMETRIC TRAP. A checkpoint from a layer_mix arm evaluated with
            # layer_mix=False builds a 28-layer expert, drops the trained `layer_mixer.*`
            # silently (load_alpamayo1_vlm keeps only `vlm.`), and produces a perfectly
            # finite min_ade for a model that was never trained. Cheap to detect: the
            # tensors are right there in the checkpoint.
            _refuse_orphaned_layer_mix(checkpoint_path)
        if layer_mix:
            n_student = len(model.vlm.model.language_model.layers)
            ref = next(model.vlm.parameters())
            model.layer_mixer = LayerMixer(
                n_student=n_student,
                n_expert=int(layer_mix_expert_layers),
                n_blocks=int(layer_mix_blocks),
                gain=bool(layer_mix_gain),
                sharpen=float(layer_mix_sharpen),
                pin_head=int(layer_mix_pin_head),
                pin_tail=int(layer_mix_pin_tail),
            ).to(device=ref.device, dtype=ref.dtype)
            # BEFORE the teacher load, so a mixer-shape mismatch surfaces on its own terms
            # rather than inside the teacher loader's strict-key accounting.
            if layer_mix_init == "checkpoint":
                _load_layer_mix(checkpoint_path, model)
            elif layer_mix_init == "tent":
                print("[stitch] ⚠️ layer_mix_init=tent: mixing matrices left at the "
                      "depth-matched init; anything trained in the checkpoint is IGNORED. "
                      "Valid only for a probe that supplies its own P.", flush=True)
            else:
                raise ValueError(
                    f"layer_mix_init must be checkpoint|tent, got {layer_mix_init!r}")
        _load_teacher_non_vlm(teacher_checkpoint_path, model)
        if layer_mix and len(model.expert.layers) != int(layer_mix_expert_layers):
            raise RuntimeError(
                f"expert built {len(model.expert.layers)} deep, expected "
                f"{layer_mix_expert_layers}; the expert_cfg depth override did not take"
            )
        return model

    @classmethod
    def from_teacher(
        cls,
        checkpoint_path: str,
        vlm_name_or_path: str,
        teacher_checkpoint_path: str | None = None,
        sparse_pruned_expert: bool = False,
        # Accepted and ignored so this target is a DROP-IN for `from_stitch` in the same
        # config: the teacher reads its trajectory settings from `checkpoint_path` itself,
        # but the shared eval config supplies `alpamayo_config_path` and hydra cannot delete
        # a key that evaluate_hf.py:70 re-adds after composition.
        alpamayo_config_path: str | None = None,
        **kwargs: Any,
    ) -> "StitchedAlpamayoR1":
        """The unmodified teacher, through the IDENTICAL harness — the ceiling.

        Without this the student's expert-head numbers have no scale: the surviving 10B
        baselines in this tree (ade 1.2111 / min_ade 0.6413) come from the trajectory-TOKEN
        head, which is a different head entirely and not comparable to anything the expert
        produces. The ceiling has to be measured here, through the same rollout, sampler and
        metric code, or the comparison smuggles in a harness difference.

        Uses ``from_alpamayo_checkpoint`` rather than ``from_pretrained_vlm`` because the
        teacher's tower (Cosmos-Reason2-8B) is present locally as CONFIG ONLY — its weights
        live in the Alpamayo checkpoint under ``vlm.*``. ``from_pretrained_vlm`` would try to
        pull real weights from that snapshot and die with "no file named model.safetensors".
        """
        import alpamayo_r1.models.base_model as _base_model

        if not hasattr(_base_model, cls.config_class.__name__):
            setattr(_base_model, cls.config_class.__name__, cls.config_class)

        teacher_cfg = cls._read_raw_config(checkpoint_path)
        expert_kwargs = {
            k: teacher_cfg[k]
            for k in (
                "expert_cfg",
                "diffusion_cfg",
                "action_space_cfg",
                "action_in_proj_cfg",
                "action_out_proj_cfg",
                "expert_non_causal_attention",
                "keep_same_dtype",
            )
            if k in teacher_cfg
        }
        # NOT `from_alpamayo_checkpoint`: it assembles `config_kwargs` as a fixed literal
        # and never merges **kwargs (sft_base_model.py:~388-403), unlike from_pretrained_vlm
        # which does `config_kwargs.update(kwargs)`. So the expert blocks are silently
        # dropped there and the model builds with action_space=None. Assembled here instead.
        from hydra.utils import instantiate

        cfg_kwargs = {
            "vlm_name_or_path": vlm_name_or_path,
            "vlm_backend": teacher_cfg.get("vlm_backend", "qwenvl3"),
            "traj_tokenizer_cfg": teacher_cfg.get("traj_tokenizer_cfg"),
            "hist_traj_tokenizer_cfg": teacher_cfg.get("hist_traj_tokenizer_cfg"),
            "traj_vocab_size": teacher_cfg.get("traj_vocab_size"),
            "tokens_per_history_traj": teacher_cfg.get("tokens_per_history_traj"),
            "tokens_per_future_traj": teacher_cfg.get("tokens_per_future_traj"),
            "model_dtype": teacher_cfg.get("model_dtype", "bfloat16"),
            "attn_implementation": teacher_cfg.get("attn_implementation", "flash_attention_2"),
            "min_pixels": teacher_cfg.get("min_pixels"),
            "max_pixels": teacher_cfg.get("max_pixels"),
            "add_special_tokens": teacher_cfg.get("add_special_tokens", True),
            **expert_kwargs,
            **kwargs,  # lets ++model.attn_implementation=sdpa reach the build
        }
        config = instantiate(
            {
                "_target_": f"alpamayo_r1.config.{cls.config_class.__name__}",
                "_recursive_": False,
                "_convert_": "all",
                **cfg_kwargs,
            }
        )
        pretrained_modules = {}
        if config.traj_tokenizer_cfg is not None:
            pretrained_modules["traj_tokenizer"] = instantiate(config.traj_tokenizer_cfg)
        model = cls(config, pretrained_modules=pretrained_modules or None)
        model = load_alpamayo1_vlm(checkpoint_path, model)
        expert_source = teacher_checkpoint_path or checkpoint_path
        if sparse_pruned_expert:
            if teacher_checkpoint_path is None:
                raise ValueError(
                    "sparse_pruned_expert needs teacher_checkpoint_path=<sparse checkpoint>; "
                    "checkpoint_path remains the dense teacher source for the frozen VLM"
                )
            if not os.environ.get("PRUNE_EXPERT_LAYERS", "").strip():
                raise ValueError(
                    "sparse_pruned_expert requires PRUNE_EXPERT_LAYERS to name the absent slots"
                )
            # Replace the absent parameterised blocks BEFORE strict loading. The sparse CD
            # checkpoint deliberately has no tensors for these slots; loading first would
            # report them as missing parameters, while pruning afterwards is too late.
            _apply_expert_pruning(model)
            skipped = [i for i, layer in enumerate(model.expert.layers)
                       if isinstance(layer, _SkippedExpertLayer)]
            _load_teacher_non_vlm(expert_source, model)
            print(f"[stitch] sparse-pruned expert loaded: "
                  f"{len(model.expert.layers) - len(skipped)}/{len(model.expert.layers)} "
                  f"active, skipped={skipped}", flush=True)
        else:
            _load_teacher_non_vlm(expert_source, model)
            _apply_expert_pruning(model)
        return model

    # ------------------------------------------------------------------ prefill-only
    @torch.no_grad()
    def _prefill_prompt_cache(self, input_ids, tokenized_data):
        """One prefill of the prompt. Returns (cache, prefill_len, rope_deltas).

        ⚠️ ``logits_to_keep=1``: nothing here reads the LM head, and without it this pays a
        [T, 155697] vocab matmul per clip and throws it away -- the same waste the training
        forward already avoids.
        """
        out = self.vlm(
            input_ids=input_ids,
            use_cache=True,
            logits_to_keep=1,
            **tokenized_data,
        )
        cache = out.past_key_values
        return cache, cache.get_seq_length(), self.vlm.model.rope_deltas

    @staticmethod
    def _expand_cache(cache, n: int) -> None:
        """Repeat every cached K/V ``n`` times along BATCH, in place.

        The 6 trajectory samples of a clip share one prompt, so they share one prefill; only
        the diffusion noise differs. ``generate(num_return_sequences=n)`` instead expands the
        batch BEFORE prefill and pays for n identical prefills (plus n ViT passes).
        ⚠️ repeat_interleave, not repeat: the diffusion batch is laid out (b ns nj), so clip
        i's samples must be CONTIGUOUS or every trajectory is attributed to the wrong clip.
        """
        if n == 1:
            return
        layers = getattr(cache, "layers", None)
        if layers is not None:
            for lyr in layers:
                lyr.keys = lyr.keys.repeat_interleave(n, dim=0)
                lyr.values = lyr.values.repeat_interleave(n, dim=0)
        else:  # older transformers cache API
            for i in range(len(cache.key_cache)):
                cache.key_cache[i] = cache.key_cache[i].repeat_interleave(n, dim=0)
                cache.value_cache[i] = cache.value_cache[i].repeat_interleave(n, dim=0)

    # ⚠️ @torch.no_grad() IS LOAD-BEARING, not hygiene. `vlm.generate` carries its own
    # no_grad, so the rollout path never built a graph; a plain forward does, and retaining
    # activations for a 3073-token prefill plus 60 expert passes OOM'd a card with 57 GiB
    # free (it tried to allocate 50 MiB at 57.36 GiB in use). The metric runner does not wrap
    # its call site, so the decorator has to live here.
    @torch.no_grad()
    def sample_trajectories_prefill_only(
        self,
        data: dict[str, Any],
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        diffusion_kwargs: dict[str, Any] | None = None,
        return_action: bool = False,
        cache_hook: Any = None,
        step_probe: Any = None,
        **kwargs: Any,
    ):
        """Prefill the VLM once, hand that cache to the expert. No generation.

        ``cache_hook(cache, input_ids, tokenized_data) -> cache`` (default None) intercepts the
        prefill cache before it is batch-expanded, so a caller can SUBSTITUTE part of it. That
        is the only way to ask "which part of the cache does the expert actually need": the
        expert's inputs are exactly (noisy action embedding, VLM cache), and the action
        embedding is the shared action_in_proj of the same traj_data, so 100% of a student's
        trajectory gap is attributable to the cache. Swapping teacher K/V into chosen
        layers/positions and re-reading min_ade localises it. Left None the path is unchanged.

        **Why this is not an approximation of the rollout -- it is the same measurement.**
        ``sample_trajectories_from_data_with_vlm_rollout`` calls ``vlm.generate`` and then
        discards everything it produced, because the eval prompt ALREADY ends with
        ``<|traj_future_start|>``: ``get_component_str`` always emits ``start_str``, and
        ``components_prompt: [traj_future]`` makes that the only thing it emits. The rollout
        then takes the FIRST occurrence of that token in ``sequences`` (which include the
        prompt), so ``offset`` lands at the end of the prompt, and
        ``attention_mask[i, offset:-n_diffusion] = False`` masks the whole generated span
        while the action tokens' RoPE continues from ``rope_deltas + offset``. The expert
        therefore attends to the prompt prefill and nothing else -- exactly the cache training
        teaches, cropped at ``tfs_idx + 1`` on both sides.

        So the generation was pure cost: ``num_return_sequences=6`` expands the batch before
        prefill, making each iteration prefill 6x identical sequences (ViT included) and then
        autoregressively decode tokens that are masked out.

        Set ``STITCH_ROLLOUT=1`` to force the old path (for re-measuring an old number), and
        ``STITCH_PREFILL_SELFTEST=1`` to assert on real data that both paths build the same
        cache.
        """
        n_samples_total = num_traj_samples * num_traj_sets
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."

        tokenized_data = dict(data["tokenized_data"])
        input_ids = tokenized_data.pop("input_ids")
        input_ids = self.fuse_traj_tokens(
            input_ids,
            {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot},
        )
        device = input_ids.device

        prompt_cache, prefill_seq_len, rope_deltas = self._prefill_prompt_cache(
            input_ids, tokenized_data
        )
        # ⚠️ 28 VLM layers -> 36 expert slots, BEFORE cache_hook and _expand_cache. Before the
        # hook so a caller substituting teacher K/V per expert slot sees a cache whose depth
        # already matches the expert; before the expansion so the mix runs once per clip
        # rather than once per trajectory sample.
        # The keys here are POST-RoPE while training mixes PRE-RoPE ones. That is not a
        # discrepancy: cos/sin are per-position and shared across layers, and rotation is
        # linear in K, so a LAYER mix commutes with it exactly (models/layer_mix.py, and
        # tests/test_layer_mix.py::test_mix_commutes_with_rope).
        mixer = getattr(self, "layer_mixer", None)
        if mixer is not None:
            prompt_cache = mixer.mix_cache(prompt_cache)
        if cache_hook is not None:
            # BEFORE _expand_cache: the hook sees one row per clip, not one per traj sample.
            prompt_cache = cache_hook(prompt_cache, input_ids, tokenized_data)

        # Where does the prompt end? The same rule the rollout uses, applied per row to the
        # PROMPT: first <|traj_future_start|>, inclusive. The shared helper preserves either
        # left or right tokenizer padding and masks every cache position after that row's
        # handoff token.
        tfs_id = self.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
        tfs_mask = input_ids == tfs_id
        if not bool(tfs_mask.any(dim=1).all()):
            missing = (~tfs_mask.any(dim=1)).nonzero().flatten().tolist()
            raise RuntimeError(
                f"prompt rows {missing} contain no <traj_future_start>; the prefill-only path "
                "needs it to know where the expert's cache ends. Is components_prompt missing "
                "'traj_future', or generation_mode false?"
            )
        self._expand_cache(prompt_cache, n_samples_total)
        b_star = B * n_samples_total
        tfs_mask = tfs_mask.repeat_interleave(n_samples_total, dim=0)
        prompt_attention_mask = tokenized_data.get("attention_mask")
        if prompt_attention_mask is not None:
            prompt_attention_mask = prompt_attention_mask.repeat_interleave(
                n_samples_total, dim=0
            )
        rope_deltas = (
            rope_deltas.repeat_interleave(n_samples_total, dim=0)
            if torch.is_tensor(rope_deltas) and rope_deltas.numel() == B
            else rope_deltas
        )

        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
        conditioning = build_expert_conditioning(
            traj_future_start_mask=tfs_mask,
            tokenizer_attention_mask=prompt_attention_mask,
            rope_deltas=rope_deltas,
            n_action_tokens=n_diffusion_tokens,
            dtype=self.dtype,
            attention_implementation=getattr(
                self.expert.config, "_attn_implementation", None
            ),
            cache_len=prefill_seq_len,
        )
        position_ids = conditioning.position_ids
        attention_mask = conditioning.attention_mask

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        _step = {"i": 0}

        def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            # ⚠️ AUTOCAST here, not at the call site. `action_in_proj` (PerWaypointActionInProjV2)
            # forces `x.float()` internally, so its fp32 activations meet bf16 weights and raise
            # "mat1 and mat2 must have the same dtype". evaluate_hf happens to wrap its call in
            # autocast, which is why the eval path works -- but that makes this method silently
            # caller-dependent, and it broke the moment a script drove it directly.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                future_token_embeds = self.action_in_proj(x, t)
            future_token_embeds = future_token_embeds.to(self.dtype)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(x.shape[0], n_diffusion_tokens, -1)
            expert_out = self.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            # ⚠️ roll the action K/V back off the cache: the next denoising step appends its
            # own, and without this the cache grows by 64 tokens per step.
            prompt_cache.crop(prefill_seq_len)
            last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vel = self.action_out_proj(last_hidden).view(
                    -1, *self.action_space.get_action_space_dims()
                )
            # ``step_probe(i, t, x, last_hidden, vel)`` (default None) exposes the tensors on
            # BOTH sides of the action head at every Euler step: `last_hidden` is pre-head,
            # `vel` is post-head. Both have been used as training targets (L_block matches the
            # former, L_field the latter) but neither has ever been checked for whether it
            # PREDICTS per-clip ade -- which is what a proxy has to do to be worth optimising.
            if step_probe is not None:
                step_probe(_step["i"], t, x, last_hidden, vel)
                _step["i"] += 1
            return vel

        sampled_action = self.diffusion.sample(
            batch_size=b_star, step_fn=step_fn, device=device,
            return_all_steps=False, **(diffusion_kwargs or {}),
        )
        hist_xyz_rep = einops.repeat(ego_history_xyz[:, -1], "b ... -> (b n) ...",
                                     n=n_samples_total)
        hist_rot_rep = einops.repeat(ego_history_rot[:, -1], "b ... -> (b n) ...",
                                     n=n_samples_total)
        pred_xyz, pred_rot = self.action_space.action_to_traj(
            sampled_action, hist_xyz_rep, hist_rot_rep
        )
        pred_xyz = einops.rearrange(pred_xyz, "(b ns nj) ... -> b ns nj ...",
                                    ns=num_traj_sets, nj=num_traj_samples)
        pred_rot = einops.rearrange(pred_rot, "(b ns nj) ... -> b ns nj ...",
                                    ns=num_traj_sets, nj=num_traj_samples)
        if return_action:
            # ⚠️ The action the SAMPLER produced, not one recovered from the trajectory.
            # Inverting `action_to_traj` via `traj_to_action` is NOT a way to get this back:
            # that path runs `theta_smooth` plus three ridge-regularised solves
            # (`unicycle_accel_curvature.py:269-283`), so it returns a SMOOTHED fit and a
            # bounds check applied to it can pass on a trajectory that is not feasible.
            return pred_xyz, pred_rot, einops.rearrange(
                sampled_action, "(b ns nj) ... -> b ns nj ...",
                ns=num_traj_sets, nj=num_traj_samples)
        return pred_xyz, pred_rot

    def sample_trajectories_from_data(self, data: dict[str, Any], **kwargs: Any):  # type: ignore[override]
        """Route the metric runner to the EXPERT, not the trajectory-token head.

        ``MetricRunner`` calls ``sample_trajectories_from_data`` (``metric_api.py:113``),
        and ``AlpamayoR1`` does not override it -- so an unmodified AlpamayoR1 scores its
        *token* head. That is exactly why the existing 10B baselines in this tree are
        token-head numbers. Overriding here is what makes the harness measure the expert.

        Defaults to the PREFILL-ONLY path: the rollout's generation is masked out of the
        expert's attention anyway (see sample_trajectories_prefill_only), so it was ~6x of
        wasted prefill. STITCH_ROLLOUT=1 restores it.
        """
        # `last_component` / `traj_only_generation` are token-head knobs the rollout does
        # not take; dropping them here keeps MetricRunner's call site unchanged.
        for dead in ("last_component", "traj_only_generation", "return_extra"):
            kwargs.pop(dead, None)
        if os.environ.get("STITCH_ROLLOUT") == "1":
            if getattr(self, "layer_mixer", None) is not None:
                raise RuntimeError(
                    "STITCH_ROLLOUT=1 is incompatible with layer_mix: that path builds its "
                    "cache inside `generate` and would hand the expert the student's raw "
                    f"{self.layer_mixer.n_student}-layer cache for "
                    f"{len(self.expert.layers)} slots, unmixed."
                )
            return self.sample_trajectories_from_data_with_vlm_rollout(data=data, **kwargs)
        if os.environ.get("STITCH_PREFILL_SELFTEST") == "1" and not getattr(
            self, "_prefill_selftested", False
        ):
            self._prefill_selftested = True
            self._selftest_prefill_matches_rollout(data, dict(kwargs))
        kwargs.pop("max_generation_length", None)   # no generation happens here
        kwargs.pop("top_p", None)
        kwargs.pop("temperature", None)
        return self.sample_trajectories_prefill_only(data=data, **kwargs)

    @staticmethod
    def _cache_layers(cache):
        """``[(K, V), ...]`` from either cache API. The installed transformers exposes
        ``cache.layers``; older ones ``key_cache``/``value_cache``."""
        layers = getattr(cache, "layers", None)
        if layers is not None:
            return [(l.keys, l.values) for l in layers]
        return list(zip(cache.key_cache, cache.value_cache))

    @torch.no_grad()
    def _selftest_prefill_matches_rollout(self, data, kwargs) -> None:
        """Do both paths hand the expert the same K/V? Compared on real data, once.

        "The rollout's generated tokens are discarded" is an argument about masks and offsets;
        this is the measurement. Only the region the expert can attend to (rows 0..offset) is
        compared -- that is the only part either path uses.
        """
        n = kwargs.get("num_traj_samples", 6) * kwargs.get("num_traj_sets", 1)
        tfs_id = self.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))

        def fresh(d):
            # ⚠️ a FRESH nested dict: the rollout does tokenized_data.pop("input_ids"), so
            # reusing the batch for a second path dies with KeyError: 'input_ids'.
            out = dict(d)
            out["tokenized_data"] = dict(d["tokenized_data"])
            return out

        d0 = fresh(data)
        ids = self.fuse_traj_tokens(
            d0["tokenized_data"]["input_ids"],
            {"ego_history_xyz": d0["ego_history_xyz"], "ego_history_rot": d0["ego_history_rot"]},
        )
        off = ((ids == tfs_id).int().argmax(dim=1) + 1).repeat_interleave(n)

        store = {}

        def pre(_m, _a, kw):
            if "cache" not in store and kw.get("past_key_values") is not None:
                store["cache"] = [(k.clone(), v.clone())
                                  for k, v in self._cache_layers(kw["past_key_values"])]
            return None

        h = self.expert.register_forward_pre_hook(pre, with_kwargs=True)
        try:
            self.sample_trajectories_from_data_with_vlm_rollout(data=fresh(data), **kwargs)
        finally:
            h.remove()
        if "cache" not in store:
            raise RuntimeError("self-test could not capture the rollout's cache")
        roll = store["cache"]

        d = fresh(data)
        td = dict(d["tokenized_data"]); td.pop("input_ids")
        cache, _, _ = self._prefill_prompt_cache(ids, td)
        self._expand_cache(cache, n)
        pre_kv = self._cache_layers(cache)

        worst = 0.0
        for (kr, vr), (kp, vp) in zip(roll, pre_kv):
            for i in range(kr.shape[0]):
                m = int(off[i])
                worst = max(
                    worst,
                    float((kr[i, :, :m].float() - kp[i, :, :m].float()).abs().max()),
                    float((vr[i, :, :m].float() - vp[i, :, :m].float()).abs().max()),
                )
        print(f"[selftest] prefill vs rollout cache over the ATTENDED region: "
              f"max |diff| = {worst:.3e} across {len(roll)} layers, "
              f"{roll[0][0].shape[0]} rows, offsets {off[:3].tolist()}", flush=True)
        if worst > 1e-2:
            raise RuntimeError(
                f"prefill-only cache differs from the rollout's (max {worst:.3e}); the two "
                "paths are NOT the same measurement -- do not trust either number.")


class TrainableStitchedAlpamayoR1(StitchedAlpamayoR1):
    """Student VLM (frozen) + teacher action expert (TRAINABLE), on the flow-matching loss.

    The other half of the compounding fix. ``L_block`` pushes the student's cache toward the
    teacher's and plateaus, because teacher-forcing hides the error that accumulates when the
    expert consumes the student's own cache (measured: free-running error 32x the
    teacher-forced one, 72x at the deepest layers -- ``scripts/freerun_probe.py``). Rather
    than force the cache to match, this lets the EXPERT adapt to the cache the student
    actually produces.

    ``StitchedAlpamayoR1`` knows how to pair a Qwen3-VL student tower with the teacher's
    expert but extends ``AlpamayoR1`` directly, so it only has the inference forward. The
    action-target construction is borrowed from ``TrainableAlpamayoR1``. The forward
    uses the same per-row cache masks and positions as prefill-only inference and CD,
    excluding padding and tokens after each row's trajectory handoff.

    ⚠️ ``cotrain_vlm``/``stop_grad_from_vlm`` are set here because they are assigned in
    ``TrainableAlpamayoR1.__init__``, which this class does not run.
    """

    _process_traj_future_training = TrainableAlpamayoR1._process_traj_future_training

    def forward(
        self,
        tokenized_data: dict[str, Any],
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        ego_future_xyz: torch.Tensor | None = None,
        ego_future_rot: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ReasoningVLAOutput:
        tokenized_data = dict(tokenized_data)
        input_ids = tokenized_data.pop("input_ids")
        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)
        # ⚠️ CE is gated SEPARATELY from gradient flow. `cotrain_vlm` opens the gradient
        # path from the diffusion loss to the VLM through the cache; `cotrain_vlm_ce` adds
        # the VLM's OWN token-head CE on top. They are different heads (COMPARE_EVAL §0),
        # so bundling them would silently make a cache experiment a token-head experiment.
        ce = getattr(self, "cotrain_vlm_ce", False)
        labels = input_ids.clone() if ce else None
        if labels is not None and labels_mask is not None:
            labels = torch.where(labels_mask, labels, IGNORE_INDEX)
        with nullcontext() if self.cotrain_vlm else torch.no_grad():
            vlm_outputs = self.vlm(
                input_ids=input_ids, labels=labels, use_cache=True, **tokenized_data
            )

        future_traj_data = self._process_traj_future_training(traj_data)
        action_embeds = self.action_in_proj(
            future_traj_data["noisy_x"], future_traj_data["timesteps"]
        )
        conditioning = build_expert_conditioning(
            traj_future_start_mask=input_ids == self.config.traj_token_ids["future_start"],
            tokenizer_attention_mask=tokenized_data.get("attention_mask"),
            rope_deltas=vlm_outputs.rope_deltas,
            n_action_tokens=action_embeds.shape[1],
            dtype=action_embeds.dtype,
            attention_implementation=getattr(self.expert.config, "_attn_implementation", None),
        )
        cache = vlm_outputs.past_key_values
        cache.crop(conditioning.prefix_len)
        if self.stop_grad_from_vlm:
            for layer in cache.layers:
                layer.keys = layer.keys.detach()
                layer.values = layer.values.detach()
        expert_outputs = self.expert(
            inputs_embeds=action_embeds,
            position_ids=conditioning.position_ids,
            past_key_values=cache,
            attention_mask=conditioning.attention_mask,
            use_cache=True,
            **({"is_causal": False} if self.config.expert_non_causal_attention else {}),
        )
        hidden = expert_outputs.last_hidden_state[:, -action_embeds.shape[1]:]
        pred = self.action_out_proj(hidden).view(-1, *self.action_space.get_action_space_dims())
        loss = self.diffusion.compute_loss_from_pred(training_data=future_traj_data, pred=pred)
        if ce:
            loss = loss + vlm_outputs.loss
        return ReasoningVLAOutput(loss=loss)

    @classmethod
    def from_student(cls, checkpoint_path: str, vlm_name_or_path: str,
                     teacher_checkpoint_path: str, cotrain_vlm: bool = False,
                     cotrain_vlm_layers: str | None = None,
                     cotrain_vlm_ce: bool = False, **kw):
        """Student tower from `checkpoint_path`, teacher expert from `teacher_checkpoint_path`.

        ``cotrain_vlm_layers`` (e.g. ``"27-35"``, inclusive) trains ONLY those text layers of
        the VLM and freezes the rest. Full cotrain does not fit: this config runs plain DDP
        (``deepspeed: null``, deliberately -- DeepSpeed's bf16 breaks
        ``PerWaypointActionInProjV2``'s internal ``x.float()``), so optimizer state is
        REPLICATED per rank and ~6.8 B trainable needs ~81 GB before activations, on 80 GB
        cards, with ``gradient_checkpointing`` unavailable (KaVaExpertTeacher raises).
        Layers 27-35 are the measured choice: COMPARE_EVAL §5 puts 95% of the recoverable
        gap in the deep third and 0% (inside the noise floor) in layers 0-9.

        ⚠️ ``stop_grad_from_vlm`` MUST go False whenever any VLM parameter trains. The
        forward detaches every cache K/V when it is True, so the diffusion loss would never
        reach the VLM and the only signal left would be the token-head CE -- a DIFFERENT
        head (COMPARE_EVAL §0). That runs clean and trains the wrong thing.
        """
        model = cls.from_stitch(
            checkpoint_path=checkpoint_path, vlm_name_or_path=vlm_name_or_path,
            teacher_checkpoint_path=teacher_checkpoint_path, **kw)
        model.cotrain_vlm = bool(cotrain_vlm) or cotrain_vlm_layers is not None
        model.cotrain_vlm_ce = bool(cotrain_vlm_ce)
        model.stop_grad_from_vlm = not model.cotrain_vlm
        for prm in model.vlm.parameters():
            prm.requires_grad_(model.cotrain_vlm)
        if cotrain_vlm_layers is not None:
            text_layers = model.vlm.model.language_model.layers
            lo, _, hi = str(cotrain_vlm_layers).partition("-")
            lo, hi = int(lo), int(hi if hi else lo)
            if not 0 <= lo <= hi < len(text_layers):
                raise ValueError(
                    f"cotrain_vlm_layers={cotrain_vlm_layers!r} outside "
                    f"0..{len(text_layers) - 1}"
                )
            for prm in model.vlm.parameters():
                prm.requires_grad_(False)
            for i in range(lo, hi + 1):
                for prm in text_layers[i].parameters():
                    prm.requires_grad_(True)
            n_vlm = sum(q.numel() for q in model.vlm.parameters() if q.requires_grad)
            print(f"[stitch-train] PARTIAL cotrain: VLM text layers {lo}-{hi} of "
                  f"{len(text_layers)} trainable ({n_vlm / 1e9:.2f} B), rest frozen; "
                  f"stop_grad_from_vlm={model.stop_grad_from_vlm}, "
                  f"vlm_ce={model.cotrain_vlm_ce}", flush=True)

        mixer = getattr(model, "layer_mixer", None)
        if mixer is not None:
            # ⚠️ THE TRAINING FORWARD DOES NOT MIX ON ITS OWN. It masks and crops the VLM
            # cache, then hands it straight to the expert -- it never touches
            # sample_trajectories_prefill_only, which is where the EVAL path applies the mix.
            # Left alone, a 28-layer cache would reach a 36-layer expert and DynamicCache
            # would AUTO-EXTEND: slots 28..35 created on demand holding only the 64 action
            # tokens, no VLM prefix. That runs clean, reports finite losses, and trains a
            # model nobody intended -- verified against a real Qwen3-VL decoder.
            #
            # A pre-hook rather than a copy of the forward: duplicating ~80 lines of borrowed
            # method is exactly how the two paths drift apart.
            def _mix_pre(_mod, args, kwargs, _m=mixer):
                cache = kwargs.get("past_key_values")
                # Self-guarding on depth: the inference path has ALREADY mixed by the time it
                # reaches here, and re-mixing a 36-layer cache would raise. This makes the
                # hook a no-op there instead.
                if cache is not None and len(getattr(cache, "layers", [])) == _m.n_student:
                    kwargs["past_key_values"] = _m.mix_cache(cache)
                return args, kwargs

            model.expert.register_forward_pre_hook(_mix_pre, with_kwargs=True)
            # P belongs to the FROZEN student's cache pipeline here: this arm adapts the
            # EXPERT to the cache the student actually produces, so the cache must hold still.
            for prm in mixer.parameters():
                prm.requires_grad_(False)
            print(f"[stitch-train] layer mix ACTIVE in the training forward "
                  f"({mixer.n_student} -> {mixer.n_expert}), P frozen", flush=True)
        n_tr = sum(q.numel() for q in model.parameters() if q.requires_grad)
        # ⚠️ Keyed on the RESOLVED attribute, not the `cotrain_vlm` argument: a partial
        # cotrain leaves the argument False while VLM layers really do train, and the old
        # print then said "VLM FROZEN" on a run that was training 0.91 B of it.
        vlm_state = ("partial" if cotrain_vlm_layers is not None
                     else "TRAINABLE" if model.cotrain_vlm else "frozen")
        logger.warning("[stitch-train] VLM %s, trainable params %.2f B",
                       vlm_state, n_tr / 1e9)
        print(f"[stitch-train] VLM {vlm_state}, trainable {n_tr / 1e9:.2f} B", flush=True)
        print(
            "[stitch-train] expert conditioning: per-row padding/suffix masks and RoPE positions",
            flush=True,
        )
        return model
