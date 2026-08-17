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
from typing import Any

import torch
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from safetensors.torch import load_file

from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA, load_alpamayo1_vlm

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

    missing, unexpected = model.load_state_dict(state, strict=False)
    # `missing` is dominated by every vlm.* key, which is correct and expected here -- the
    # student's VLM weights are already in place and must NOT be overwritten. What must be
    # empty is `unexpected`, and no teacher-side key may be left unloaded.
    unloaded = sorted(set(wanted) - set(state))
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
        _load_teacher_non_vlm(teacher_checkpoint_path, model)
        return model

    @classmethod
    def from_teacher(
        cls,
        checkpoint_path: str,
        vlm_name_or_path: str,
        teacher_checkpoint_path: str | None = None,
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
        _load_teacher_non_vlm(teacher_checkpoint_path or checkpoint_path, model)
        _apply_expert_pruning(model)
        return model

    def sample_trajectories_from_data(self, data: dict[str, Any], **kwargs: Any):  # type: ignore[override]
        """Route the metric runner to the EXPERT, not the trajectory-token head.

        ``MetricRunner`` calls ``sample_trajectories_from_data`` (``metric_api.py:113``),
        and ``AlpamayoR1`` does not override it -- so an unmodified AlpamayoR1 scores its
        *token* head. That is exactly why the existing 10B baselines in this tree are
        token-head numbers. Overriding here is what makes the harness measure the expert.
        """
        # `last_component` / `traj_only_generation` are token-head knobs the rollout does
        # not take; dropping them here keeps MetricRunner's call site unchanged.
        for dead in ("last_component", "traj_only_generation", "return_extra"):
            kwargs.pop(dead, None)
        return self.sample_trajectories_from_data_with_vlm_rollout(data=data, **kwargs)


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
    training forward lives on ``TrainableAlpamayoR1``, a sibling under ``AlpamayoR1``.
    Borrowing the three methods is the same pattern ``KaVaExpertTeacher`` uses for
    ``generate_cot_prefix`` -- if either class starts depending on state the other lacks,
    this is the line that breaks, loudly.

    ⚠️ ``cotrain_vlm``/``stop_grad_from_vlm`` are set here because they are assigned in
    ``TrainableAlpamayoR1.__init__``, which this class does not run.
    """

    forward = TrainableAlpamayoR1.forward
    _process_traj_future_training = TrainableAlpamayoR1._process_traj_future_training
    _process_position_ids_qwen2_5_vl = TrainableAlpamayoR1._process_position_ids_qwen2_5_vl

    @classmethod
    def from_student(cls, checkpoint_path: str, vlm_name_or_path: str,
                     teacher_checkpoint_path: str, cotrain_vlm: bool = False, **kw):
        """Student tower from `checkpoint_path`, teacher expert from `teacher_checkpoint_path`."""
        model = cls.from_stitch(
            checkpoint_path=checkpoint_path, vlm_name_or_path=vlm_name_or_path,
            teacher_checkpoint_path=teacher_checkpoint_path, **kw)
        model.cotrain_vlm = bool(cotrain_vlm)
        model.stop_grad_from_vlm = True
        for prm in model.vlm.parameters():
            prm.requires_grad_(cotrain_vlm)
        n_tr = sum(q.numel() for q in model.parameters() if q.requires_grad)
        logger.warning("[stitch-train] VLM %s, trainable params %.2f B",
                       "TRAINABLE" if cotrain_vlm else "frozen", n_tr / 1e9)
        print(f"[stitch-train] VLM {'trainable' if cotrain_vlm else 'FROZEN'}, "
              f"trainable {n_tr / 1e9:.2f} B", flush=True)
        return model
