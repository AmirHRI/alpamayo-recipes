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

"""Qwen3-VL-4B student distilled from a co-resident Alpamayo-1.5-10B teacher.

``L = CE(gt traj) + lambda_kd * KD(traj logits) + lambda_kv * KV_align(all tokens, all layers)``

**Why this pairing removes the approximations KAVA had to live with.**  The 2B student
forced a 28 -> 36 layer map and distilled a 13-token CoT compressed to 8 slots by R-KV
eviction.  Every alignment was approximate: depth, token correspondence, basis.  Measured
on the real checkpoints before this file was written:

===========================  ==========================  ====================
property                     teacher (Cosmos-Reason2-8B) student (Qwen3-VL-4B)
===========================  ==========================  ====================
text layers                  36                          36
kv-heads x head_dim          8 x 128 = 1024              8 x 128 = 1024
vocab / traj_token_start_idx 155,697 / 151,669           155,697 / 151,669
===========================  ==========================  ====================

so ``build_layer_map(36, 36)`` is the identity, K/V need **no width projection**, and the
trajectory slice denotes the same tokens on both sides.  Dropping the CoT removes eviction
entirely: teacher and student sequences were verified **element-wise equal** on a real
clip (both ``[1, 3073]``), so KV alignment is position-exact rather than a compressed
proxy.

**The teacher is co-resident and online, with no cache.**  All-token/all-layer KV would be
455 MB/clip -> 17.4 TB over LCDrive-train.  The teacher has no CoT to generate, so its pass
is one teacher-forced prefill and a single frozen forward yields *both* the trajectory
logits and all-layer K/V.  Under data parallelism each rank holds different samples, so
running the teacher on every rank is the required work done locally -- not duplication.

**Student K/V come from a recompute, not from forward hooks.**  ``KaVaReasoningVLA``
forbids ``gradient_checkpointing`` because hooks capture detached tensors under it, and
without checkpointing a 4B student plus a frozen 8B teacher does not fit on an 80 GB card
(~90 GB against 80).  :func:`recompute_kv` reconstructs pre-RoPE K/V from
``output_hidden_states`` instead.  Verified against the hook path on all 36 layers of the
real 4B: **rel-L2 exactly 0.0** (it is the same ops in the same order), and the result
still carries ``grad_fn`` under checkpointing.  That is what brings the budget to ~64 GB
at 4 GPUs x bs=2.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any

import os

import torch
from transformers.cache_utils import DynamicCache
import torch.nn as nn
from transformers.utils import ModelOutput

from alpamayo_r1.models.base_model import IGNORE_INDEX
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA
from alpamayo1_5_distill.models.kd_losses import assert_kd_compatible, logit_kd_loss
from alpamayo1_5_distill.models.block_losses import (
    block_output_loss,
    block_output_only,
    block_span_output,
    rotate_keys,
)
from alpamayo1_5_distill.models.expert_conditioning import build_expert_conditioning
from alpamayo1_5_distill.models.expert_holder import FrozenExpert
from alpamayo1_5_distill.models.kv_distill import (
    KVProjectorBank,
    build_layer_map,
    kv_matching_loss,
)



class _Phase:
    """CUDA-synced wall-clock per forward phase, opt-in via KD_TIMERS=1.

    ⚠️ Synchronises, so it is a DIAGNOSTIC and must stay off in real runs: without the sync
    every phase but the last would report near-zero, since the kernels are still queued.
    Reports the mean over a window, because per-step numbers swing with sequence length.
    """

    on = os.environ.get("KD_TIMERS") == "1"
    acc: dict = {}
    n = 0

    @staticmethod
    @contextlib.contextmanager
    def t(name):
        if not _Phase.on:
            yield
            return
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            _Phase.acc[name] = _Phase.acc.get(name, 0.0) + (time.perf_counter() - t0)

    @staticmethod
    def report(every=24):
        if not _Phase.on:
            return
        _Phase.n += 1
        if _Phase.n % every:
            return
        tot = sum(_Phase.acc.values())
        parts = " ".join(f"{k} {v / _Phase.n:.3f}s ({100 * v / max(tot, 1e-9):.0f}%)"
                         for k, v in sorted(_Phase.acc.items(), key=lambda kv: -kv[1]))
        print(f"[timers] per micro-batch over {_Phase.n}: total {tot / _Phase.n:.3f}s | {parts}",
              flush=True)


@dataclass
class KDVLAOutput(ModelOutput):
    """Everything but ``loss`` is detached and exists so ``KDTrainer`` can log the terms.

    A single total tells you nothing about which objective is doing the work -- the KAVA
    runs showed a term can sit at 0.0% of the backbone gradient while its own curve looks
    perfectly healthy.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    ce_loss: torch.FloatTensor | None = None
    kd_loss: torch.FloatTensor | None = None
    kv_loss: torch.FloatTensor | None = None
    block_loss: torch.FloatTensor | None = None
    block_loss_mse: torch.FloatTensor | None = None
    block_loss_cosine: torch.FloatTensor | None = None
    freerun_loss: torch.FloatTensor | None = None
    field_loss: torch.FloatTensor | None = None
    roll_loss: torch.FloatTensor | None = None
    kv_ratio_k: torch.FloatTensor | None = None
    kv_ratio_v: torch.FloatTensor | None = None
    block_loss_tf: torch.FloatTensor | None = None
    block_loss_span: torch.FloatTensor | None = None
    block_loss_early: torch.FloatTensor | None = None
    block_loss_mid: torch.FloatTensor | None = None
    block_loss_deep: torch.FloatTensor | None = None
    kv_loss_vision: torch.FloatTensor | None = None
    kv_loss_text: torch.FloatTensor | None = None
    kv_loss_traj: torch.FloatTensor | None = None


def recompute_kv(
    text_model: nn.Module,
    hidden_states: tuple[torch.Tensor, ...],
    n_kv_heads: int,
    head_dim: int,
    layers: list[int] | None = None,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Reconstruct pre-RoPE K/V for every layer from ``output_hidden_states``.

    ``Qwen3VLTextDecoderLayer.forward`` is::

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, ...)

    and attention computes ``k_norm(k_proj(x).view(B, T, n_kv, D))``.  With
    ``output_hidden_states=True`` the tuple has ``n_layers + 1`` entries and
    ``hidden_states[l]`` is the **input** to layer ``l``, so replaying those two ops
    reproduces exactly what a ``k_norm`` / ``v_proj`` forward hook would have seen.

    Verified on the real Qwen3-VL-4B: rel-L2 **0.000e+00** against the hook path on all 36
    layers, and gradient reaches ``k_proj.weight`` with checkpointing enabled -- which the
    hook path cannot do, because checkpointing hands hooks detached tensors.

    Returns:
        ``{layer: (K, V)}`` with each tensor ``[B, n_kv_heads, T, head_dim]``, grad-carrying.
    """
    if len(hidden_states) < len(text_model.layers) + 1:
        raise ValueError(
            f"expected >= {len(text_model.layers) + 1} hidden states for "
            f"{len(text_model.layers)} layers, got {len(hidden_states)}. "
            "Was output_hidden_states=True set on the forward?"
        )
    idx = range(len(text_model.layers)) if layers is None else layers
    out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for i in idx:
        layer = text_model.layers[i]
        x = layer.input_layernorm(hidden_states[i])
        b, t, _ = x.shape
        k = layer.self_attn.k_norm(
            layer.self_attn.k_proj(x).view(b, t, n_kv_heads, head_dim)
        ).permute(0, 2, 1, 3)
        v = layer.self_attn.v_proj(x).view(b, t, n_kv_heads, head_dim).permute(0, 2, 1, 3)
        out[i] = (k, v)
    return out


def stack_teacher_kv(
    kv: dict[int, tuple[torch.Tensor, torch.Tensor]], n_layers: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``{layer: (K, V)}`` -> two ``[B, L, H, T, D]`` tensors, the layout ``kv_matching_loss`` wants."""
    ks = torch.stack([kv[i][0] for i in range(n_layers)], dim=1)
    vs = torch.stack([kv[i][1] for i in range(n_layers)], dim=1)
    return ks, vs


class KDReasoningVLA(TrainableReasoningVLA):
    """Student with a frozen co-resident teacher supplying logit and K/V targets.

    As with ``latent_proj`` in the sibling recipes, submodules are deliberately not class
    attributes -- a class attribute would shadow ``nn.Module.__getattr__`` and always read
    ``None``.  Reach them with ``getattr(self, name, None)``.
    """

    ce_weight: float = 1.0
    block_weight: float = 0.0
    block_timestep: str = "zero"
    block_freerun_weight: float = 0.0
    block_freerun_layers: int = 0
    field_weight: float = 0.0
    roll_weight: float = 0.0
    roll_steps: int = 2
    block_norm: str = "teacher"   # 'teacher' | 'cache'
    block_span: int = 1
    #: Per-layer weighting of the block loss: 'uniform' (every layer equal, the historical
    #: behaviour), 'ladder' (the MEASURED expert sensitivity profile), 'deep_only' (layers
    #: 19-27 only), or an explicit comma list of n_layers floats.
    block_layer_weights: str = "uniform"
    #: Combine BOTH block objectives: the teacher-forced per-layer loss (m=1) AND the span
    #: loss at this m. 0/1 disables the mix and `block_span` alone decides the path.
    block_span_mix: int = 0
    block_span_mix_weight: float = 1.0
    kd_weight: float = 0.0
    kd_temperature: float = 1.0
    kv_weight: float = 0.0
    kv_loss_type: str = "l1"
    kv_align: str = "direct"
    kv_layerwise_std: bool = True
    kv_layer_bands: list | None = None
    log_kv_regions: bool = True

    #: When True, ``forward`` also stashes the loss terms **with their graph attached** in
    #: :attr:`last_loss_terms`, so ``KaVaTrainer`` can take ``autograd.grad`` of each term
    #: separately and report its share of the backbone gradient. Off by default: holding
    #: those references pins the graph longer than the step needs.
    #:
    #: This is the check that matters most for a three-term loss. ``L_KV`` once ran at
    #: 0.0% of the backbone gradient for an entire run while its own curve looked
    #: perfectly healthy, and the loss log alone could never have shown it.
    keep_loss_terms: bool = False

    # ------------------------------------------------------------------ setup
    def init_kd(
        self,
        teacher: nn.Module | None = None,
        ce_weight: float = 1.0,
        block_weight: float = 0.0,
        block_timestep: str = "zero",
        block_freerun_weight: float = 0.0,
        block_freerun_layers: int = 0,
        field_weight: float = 0.0,
        roll_weight: float = 0.0,
        roll_steps: int = 2,
        block_norm: str = "teacher",
        block_span: int = 1,
        block_layer_weights: str = "uniform",
        block_span_mix: int = 0,
        block_span_mix_weight: float = 1.0,
        kd_weight: float = 0.0,
        kd_temperature: float = 1.0,
        kv_weight: float = 0.0,
        kv_loss_type: str = "l1",
        kv_align: str = "direct",
        kv_layerwise_std: bool = True,
        kv_layer_bands: list | None = None,
        log_kv_regions: bool = True,
    ) -> None:
        """Attach the frozen teacher and the (optional) K/V projector bank.

        Args:
            teacher: an already-built ``TrainableReasoningVLA`` holding the 10B VLM.  Kept
                **outside** ``nn.Module`` registration (see :meth:`teacher`).
            kv_align: ``"direct"`` is the default and the primary arm.  Widths already
                match at 1024 and, with eviction gone, positions correspond exactly -- so a
                learned projector now has a well-posed target *and* enough capacity to
                absorb the entire cross-model mismatch without the student's own cache
                moving.  That is the failure mode the KAVA work spent a long time chasing.
            kv_loss_type: ``"l1"``.  ``smooth_l1`` was measured to self-extinguish here --
                its gradient decays 9x as the residual shrinks, ``l1``'s is constant.
        """
        self.ce_weight = float(ce_weight)
        self.block_weight = float(block_weight)
        # "zero"  -> x ~ N(0,I) at t=0, the sampler's first step (the original L_block)
        # "beta"  -> t ~ the TEACHER'S OWN training schedule, Beta(1.5,1.0) rescaled by
        #            0.999 (flow_matching.py:54-58); mass concentrated at low t
        # "uniform" -> t ~ U(0,1)
        if block_timestep not in ("zero", "beta", "uniform"):
            raise ValueError(f"block_timestep must be zero|beta|uniform, got {block_timestep}")
        self.block_timestep = str(block_timestep)
        # L_freerun: match the student's OWN chain at the final layer, not the teacher-forced
        # per-layer output. Measured motivation (scripts/freerun_probe.py, n=32): the
        # teacher-forced error L_block trains on is flat at ~4.1e-04 across all 36 layers,
        # while the free-running error grows to 1.3e-02 -- 32x, and 72x at the deepest layers.
        # Epochs 2-3 cut the teacher-forced term 33% and bought only -0.06 min_ade, because
        # L_block has no gradient path to the compounded error. This term supplies one.
        self.block_freerun_weight = float(block_freerun_weight)
        self._pi_probed = False        # BLOCK_PI_PROBE fires on the first batch only
        # How many of the DEEPEST layers carry gradient in the free-running chain. 0 = all.
        # The chain always runs in full, so the compounded error still reaches the final
        # layer; this only bounds how far back the gradient travels. Justified by where the
        # error actually lives (freerun_probe, n=32): 3-6x the teacher-forced value at layers
        # 4-8, but 28x at layer 24 and 71x at layers 28-32.
        # ⚠️ The cost: the EARLY layers' cache errors are what CAUSE the compounding, and this
        # variant cannot correct them through this term -- only their consequences at depth.
        # L_block still supervises all 36 layers, though at ~4% of the gradient when
        # block_freerun_weight=1.0.
        self.block_freerun_layers = int(block_freerun_layers)
        self.field_weight = float(field_weight)
        self.roll_weight = float(roll_weight)
        self.roll_steps = int(roll_steps)
        self.block_norm = str(block_norm)
        self.block_span = int(block_span)
        self.block_layer_weights = str(block_layer_weights)
        self._blw_logged = False
        self.block_span_mix = int(block_span_mix)
        self.block_span_mix_weight = float(block_span_mix_weight)
        self._mix_logged = False
        self._beta_dist = (
            torch.distributions.beta.Beta(torch.tensor(1.5), torch.tensor(1.0))
            if block_timestep == "beta" else None
        )
        # ⚠️ Held OUTSIDE nn.Module registration, exactly like the teacher: a registered 2.28 B
        # frozen expert would enter the optimizer, the ZeRO shard and every 68 GB checkpoint.
        self._expert_holder: list = []
        self.kd_weight = float(kd_weight)
        self.kd_temperature = float(kd_temperature)
        self.kv_weight = float(kv_weight)
        self.kv_loss_type = str(kv_loss_type)
        self.kv_align = str(kv_align)
        self.kv_layerwise_std = bool(kv_layerwise_std)
        self.kv_layer_bands = list(kv_layer_bands) if kv_layer_bands else None
        self.log_kv_regions = bool(log_kv_regions)

        n_student = len(self._text_model().layers)
        self.kv_layer_map: list[int] | None = None

        if teacher is not None and self.kd_weight <= 0 and self.kv_weight <= 0:
            print(
                "[kd] WARNING a teacher was loaded but kd_weight and kv_weight are both 0, "
                "so it will never be used — ~16 GB of GPU held for nothing. For the CE-only "
                "control arm pass ++model.teacher_checkpoint_path=null.",
                flush=True,
            )
        if teacher is not None:
            assert_kd_compatible(self, teacher)
            teacher.eval()
            teacher.requires_grad_(False)
            n_teacher = len(teacher.vlm.model.language_model.layers)
            self.kv_layer_map = build_layer_map(n_student, n_teacher)
            if self.kv_layer_map != list(range(n_student)):
                print(
                    f"[kd] NOTE layer map is not the identity ({n_student} -> {n_teacher}); "
                    f"expected identity for the 4B/10B pairing. map={self.kv_layer_map[:6]}...",
                    flush=True,
                )
            # Held in a list so nn.Module never sees it: registering the teacher would put
            # ~8 B frozen parameters into the optimizer, the ZeRO-2 shard and every saved
            # checkpoint.
            self._teacher_holder = [teacher]

        if self.kv_weight > 0 and self.kv_align != "direct":
            ref = next(self.vlm.parameters())
            self.kv_projector = KVProjectorBank(
                n_student, kv_width=self._kv_width(), align=self.kv_align
            )
            self.kv_projector.to(device=ref.device, dtype=ref.dtype)

    @classmethod
    def from_pretrained_vlm(
        cls,
        vlm_name_or_path: str,
        alpamayo_config_path: str | None = None,
        checkpoint_path: str | None = None,
        teacher_checkpoint_path: str | None = None,
        teacher_vlm_name_or_path: str | None = None,
        kd: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "KDReasoningVLA":
        """Build the student, then the frozen teacher, then the KD state.

        Args:
            teacher_checkpoint_path: the 10B A1-format checkpoint.  Loaded through
                ``from_alpamayo_checkpoint``, which filters to the ``vlm.`` prefix -- so
                the action expert and diffusion tensors are never materialised and the
                teacher costs ~16 GB rather than ~21 GB.
            kd: the kwarg block forwarded to :meth:`init_kd`.  Omit it and this is exactly
                a ``TrainableReasoningVLA`` -- the CE-only control arm, with no teacher
                resident and no extra memory.

        ⚠️ ``teacher_checkpoint_path`` / ``teacher_vlm_name_or_path`` / ``kd`` are
        **explicit parameters, never ``**kwargs``**.  The parent does
        ``config_kwargs.update(kwargs)``, so anything left to ``**kwargs`` silently becomes
        a *config field* instead of reaching this method -- which is how the KAVA
        ``zero_slots`` ablation ran for 500 clips doing nothing at all, both arms scoring
        bit-identically.
        """
        model = super().from_pretrained_vlm(
            vlm_name_or_path,
            alpamayo_config_path=alpamayo_config_path,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )
        teacher = None
        if teacher_checkpoint_path is not None:
            if teacher_vlm_name_or_path is None:
                raise ValueError(
                    "teacher_checkpoint_path needs teacher_vlm_name_or_path (the backbone "
                    "whose processor/tokenizer the checkpoint was built against)"
                )
            print(f"[kd] loading frozen teacher (vlm.* only) from {teacher_checkpoint_path}", flush=True)
            teacher = TrainableReasoningVLA.from_alpamayo_checkpoint(
                checkpoint_path=teacher_checkpoint_path,
                vlm_name_or_path=teacher_vlm_name_or_path,
            )
        model.init_kd(teacher=teacher, **(kd or {}))
        # Where the frozen action expert comes from. Same checkpoint as the teacher
        # VLM; L_block loads only `expert.*` + `action_in_proj.*` from it (~2.28 B).
        model._block_ckpt = teacher_checkpoint_path
        return model

    @property
    def teacher(self) -> nn.Module | None:
        holder = getattr(self, "_teacher_holder", None)
        return holder[0] if holder else None

    def _teacher_text_model(self) -> nn.Module:
        return self.teacher.vlm.model.language_model

    def _place_teacher(self, device: torch.device, dtype: torch.dtype) -> None:
        """Move the teacher onto the student's device/dtype, once.

        ⚠️ Necessary precisely *because* the teacher is held outside ``nn.Module``
        registration. That keeps ~8 B frozen parameters out of the optimizer, out of the
        ZeRO-2 shard and out of every saved checkpoint -- but ``Trainer`` (and
        ``.to(device)``, and accelerate's preparation) only ever walk registered children,
        so nothing moves it. Left on CPU it fails at the first embedding lookup with
        "Expected all tensors to be on the same device, but got index is on cuda:0,
        different from other tensors on cpu".

        Done lazily on first use rather than in ``init_kd`` because at construction time
        the student itself is still on CPU -- the device is only known once a batch
        arrives.
        """
        t = self.teacher
        if t is None or getattr(self, "_teacher_placed", None) == (device, dtype):
            return
        t.to(device=device, dtype=dtype)
        t.eval()
        t.requires_grad_(False)
        self._teacher_placed = (device, dtype)
        print(f"[kd] teacher placed on {device} as {dtype}", flush=True)

    def _text_model(self) -> nn.Module:
        return self.vlm.model.language_model

    def _kv_shape(self) -> tuple[int, int]:
        cfg = getattr(self.vlm.config, "text_config", self.vlm.config)
        return int(cfg.num_key_value_heads), int(cfg.head_dim)

    def _kv_width(self) -> int:
        h, d = self._kv_shape()
        return h * d

    # ------------------------------------------------------------------ regions
    def _region_masks(
        self, input_ids: torch.Tensor, traj_mask: torch.Tensor, attn: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        """Split the sequence into vision / text / traj for per-region KV reporting.

        ~93% of positions hold vision embeddings.  Those K/V are still the *language
        model's* and are still read by the action expert, so they belong in the objective
        -- but if they converge differently from the text and trajectory regions a single
        scalar would hide it entirely.
        """
        valid = torch.ones_like(input_ids, dtype=torch.bool) if attn is None else attn.bool()
        img_id = self.special_token_ids.get("image_pad")
        if img_id is None:
            tok = getattr(self, "tokenizer", None)
            img_id = tok.convert_tokens_to_ids("<|image_pad|>") if tok is not None else -1
        vision = (input_ids == img_id) & valid
        traj = traj_mask & valid
        return {"vision": vision, "traj": traj, "text": valid & ~vision & ~traj}

    # ------------------------------------------------------------------- loss
    def _kv_loss(
        self,
        student_kv: dict[int, tuple[torch.Tensor, torch.Tensor]],
        teacher_k: torch.Tensor,
        teacher_v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        one = lambda sub_kv, sub_map: kv_matching_loss(
            sub_kv,
            teacher_k,
            teacher_v,
            sub_map,
            valid_mask=mask,
            projector=getattr(self, "kv_projector", None),
            kind=self.kv_loss_type,
            layerwise_std=self.kv_layerwise_std,
        )
        if not self.kv_layer_bands:
            return one(student_kv, self.kv_layer_map)

        # Depth-banded weighting, measured causally rather than assumed. Swapping ONE layer
        # of the student's cache for the teacher's and re-driving the frozen expert
        # (scripts/layer_importance.py, n=100) showed the recoverable gain is almost entirely
        # in the back third:
        #     layers 0-11  sum +0.0031 of a 2.6396 m gap  (0.1%)
        #     layers 12-23 sum +1.2871                    (49%)
        #     layers 24-35 sum +2.7790                    (105%)
        # Uniform weighting therefore spent a third of the gradient on layers worth 0.1% of
        # the outcome. BANDS, not the raw 36-vector: those are point estimates and layer 22
        # scoring 6.79 beside layer 21 at 1.28 is more likely noise than a real cliff.
        # ⚠️ Weights are renormalised to mean 1.0, so the TOTAL loss scale is unchanged and
        # "better weighting" cannot be confounded with "different effective kv_weight".
        n = len(self.kv_layer_map)
        edges = [0, n // 3, 2 * n // 3, n]
        total = None
        for w, lo, hi in zip(self.kv_layer_bands, edges[:-1], edges[1:]):
            if w <= 0:
                continue
            sub_kv = {i: student_kv[i] for i in range(lo, hi)}
            sub_map = self.kv_layer_map[lo:hi]
            # reindex: kv_matching_loss walks the map positionally against sub_kv's keys
            sub_kv = {j: sub_kv[i] for j, i in enumerate(range(lo, hi))}
            term = w * one(sub_kv, sub_map)
            total = term if total is None else total + term
        # Divide by the number of bands so the result stays a mean, matching the unbanded
        # scale; the weights already average to 1.0.
        return total / max(1, sum(1 for w in self.kv_layer_bands if w > 0))

    # ---------------------------------------------------------------- forward
    # ------------------------------------------------------------------ L_block
    @property
    def expert(self):
        return self._expert_holder[0] if self._expert_holder else None

    def _ensure_expert(self, checkpoint_path: str, device, dtype) -> None:
        if not self._expert_holder:
            self._expert_holder.append(
                FrozenExpert(checkpoint_path, self._text_model().config)
            )
        e = self._expert_holder[0]
        if next(e.parameters()).device != device:
            self._expert_holder[0] = e.to(device=device, dtype=dtype)

    def _capture_rope(self):
        """Grab the (cos, sin) the VLM used, instead of recomputing them.

        Your constraint: run the models regularly and the positions are intact by
        construction. The text model computes `position_embeddings` once per forward and
        hands them to every layer, so a forward hook on `rotary_emb` yields exactly the
        rotation the cached keys carry -- no mrope reconstruction, nothing to get wrong.
        """
        store = {}

        def hook(_m, _inp, out):
            store["cos"], store["sin"] = out[0], out[1]

        handle = self._text_model().rotary_emb.register_forward_hook(hook)
        return store, handle

    def _sample_block_t(self, b: int, device) -> torch.Tensor:
        """One timestep per batch element, on the teacher's own training schedule."""
        if self.block_timestep == "zero":
            return torch.zeros(b, device=device)
        if self.block_timestep == "uniform":
            return torch.rand(b, device=device)
        t = self._beta_dist.sample((b,)).to(device)
        return 0.999 - t * 0.999          # flow_matching.py:148-149, verbatim

    def _layer_weights(self, n_layers: int) -> list[float]:
        """Per-layer weight for the block loss, from the CACHE LADDER's measured sensitivity.

        The uniform mean the loss has always used implicitly claims every layer of the cache
        matters equally to the expert. Measured (cacheladder, n=300, noise floor 0.034, total
        recoverable gap -1.3452 min_ade): substituting the teacher's K/V into

            layers 0-9    ->  +0.0120   (INSIDE the noise floor: ten layers, worth nothing)
            layers 10-18  ->  -0.8946   (0.099 / layer)
            layers 19-27  ->  -1.2846   (0.143 / layer, and -0.173 / layer over the last 4)

        i.e. sensitivity rises monotonically with depth and is ~0 for the first third. The
        mechanism is the same one that made the span curriculum a null result: the expert's
        layer map is CONTRACTIVE, so an error in cache layer 0 has 27 layers of damping ahead
        of it while an error at layer 27 reaches action_out_proj almost directly.

        ⚠️ 'ladder' FLOORS the early layers rather than zeroing them. The ladder showed that
        CORRECTING layers 0-9 buys nothing; it did NOT show they can be left unsupervised --
        substituting the teacher's better values is not the same as letting the student's drift
        arbitrarily. 'deep_only' takes that stronger bet, and is the sharper test.
        """
        w = self.block_layer_weights.strip()
        if "," in w:
            vals = [float(x) for x in w.split(",") if x != ""]
            if len(vals) != n_layers:
                raise ValueError(
                    f"block_layer_weights lists {len(vals)} weights for {n_layers} layers")
            return vals
        if w == "uniform":
            return [1.0] * n_layers
        # bands are expressed as FRACTIONS of depth so the profile transfers to a student of
        # a different depth instead of silently mis-binning
        def band(l: int) -> float:
            f = l / max(n_layers - 1, 1)
            if w == "deep_only":
                return 1.0 if f >= 19 / 27 else 0.0
            if w == "ladder_add":
                # ADDITIVE variant: never below 1.0, so the deep layers gain pressure without
                # the early layers losing supervision outright. ⚠️ The weighted mean is
                # normalised by the WEIGHT SUM, so this still reallocates share -- just far
                # less: early layers go 1/28 -> 1/32.25, a 13% cut, against the 72% cut that
                # 'ladder' applied and that measurably HURT (min_ade +0.1487, z +7.84).
                if f < 19 / 27:
                    return 1.00
                return 1.25 if f < 24 / 27 else 1.75
            if w == "ladder":
                if f < 10 / 27:
                    return 0.25          # floored, not zeroed -- see the warning above
                if f < 19 / 27:
                    return 1.00
                if f < 24 / 27:
                    return 1.25
                return 1.75
            raise ValueError(f"unknown block_layer_weights={w!r}")
        return [band(l) for l in range(n_layers)]

    def _surviving_expert_layers(self):
        """Indices of expert layers NOT replaced by identity stand-ins, or None if unpruned.

        Reads the live module list rather than re-parsing PRUNE_EXPERT_LAYERS, so the mapping
        can never disagree with what was actually bypassed.
        """
        try:
            layers = self.expert.expert.layers
        except AttributeError:
            return None
        surv = [i for i, m in enumerate(layers)
                if type(m).__name__ != "_SkippedExpertLayer"]
        return None if len(surv) == len(layers) else surv

    def _block_loss(self, student_kv, teacher_kv, rope, traj_future_start_mask, attn_mask,
                    rope_deltas, traj_data=None):
        """L_block = mean_l [ || B_l(h_l^T; K_s,V_s) - sg B_l(h_l^T; K_t,V_t) ||^2 + (1 - cos angle) ].

        ⚠️ The per-layer re-run reuses the EXACT kwargs the expert's own forward handed each
        layer -- captured by hook, never reconstructed. The enclosing
        ``Qwen3VLTextModel.forward`` prepares the backend-specific mask and computes
        ``position_embeddings`` via mrope; a layer called directly with a reconstructed mask
        and self-computed cos/sin attends DIFFERENTLY. That bug made even the teacher's own
        cache fail to reproduce the teacher's own outputs (self-test identity 1.04e3 instead
        of ~0, and layer-shuffling
        the student *improved* the loss). Capturing removes the entire class of error.

        ⚠️ The action noise is drawn ONCE. Both sides consume the same ``h_l^T`` from that
        single draw, so noise and timestep are shared by construction. ``t = 0.0`` is the
        sampler's own first step (``flow_matching._euler``).
        """
        expert = self.expert
        cos, sin = rope["cos"], rope["sin"]
        n_layers = len(expert.expert.layers)

        b = teacher_kv[0][0].shape[0]
        device, dtype = teacher_kv[0][0].device, teacher_kv[0][0].dtype
        n_act = expert.n_action_tokens
        conditioning = build_expert_conditioning(
            traj_future_start_mask=traj_future_start_mask,
            tokenizer_attention_mask=attn_mask,
            rope_deltas=rope_deltas,
            n_action_tokens=n_act,
            dtype=dtype,
            attention_implementation=getattr(
                expert.expert.config, "_attn_implementation", None
            ),
        )
        prefix_len = conditioning.prefix_len
        e_mask = conditioning.attention_mask
        pos = conditioning.position_ids

        # One physical cache length is shared by the batch. Per-row left padding and any
        # positions after that row's <traj_future_start> are excluded by e_mask. The +1 in
        # build_expert_conditioning includes the handoff token, matching deployment.
        rot = lambda k: rotate_keys(
            k[:, :, :prefix_len], cos[:, :prefix_len], sin[:, :prefix_len]
        )
        t_k = [rot(teacher_kv[i][0]) for i in range(n_layers)]
        t_v = [teacher_kv[i][1][:, :, :prefix_len] for i in range(n_layers)]
        s_k = [rot(student_kv[i][0]) for i in range(n_layers)]
        s_v = [student_kv[i][1][:, :, :prefix_len] for i in range(n_layers)]

        captured: dict[int, dict] = {}

        def make_hook(idx):
            def hook(_m, args, kwargs, output):
                captured[idx] = {
                    "h_in": (args[0] if args else kwargs["hidden_states"]).detach(),
                    "y_out": (output[0] if isinstance(output, tuple) else output).detach(),
                    "kwargs": {k: v for k, v in kwargs.items() if k != "past_key_values"},
                }
            return hook

        with torch.no_grad():
            if self.block_timestep == "zero" or traj_data is None:
                embeds = expert.initial_action_embeds(b, device, dtype)
            else:
                t_s = self._sample_block_t(b, device)
                embeds = expert.noisy_action_embeds(traj_data, t_s, device, dtype)
            if embeds.shape[1] != n_act:
                raise RuntimeError(
                    f"expert produced {embeds.shape[1]} action tokens, expected {n_act}"
                )

            cache = DynamicCache()
            for i in range(n_layers):
                cache.update(t_k[i], t_v[i], i, {})
            handles = [
                expert.expert.layers[i].register_forward_hook(make_hook(i), with_kwargs=True)
                for i in range(n_layers)
            ]
            try:
                expert.expert(
                    inputs_embeds=embeds, attention_mask=e_mask, position_ids=pos,
                    past_key_values=cache, use_cache=True, is_causal=False,
                )
            finally:
                for h in handles:
                    h.remove()

        # Identity stand-ins only exist in the ABLATION path (whole-teacher, 36 slots, 8
        # bypassed). With a remapped shallow expert every slot is real, so this is empty.
        skip = set(range(n_layers)) - set(self._surviving_expert_layers() or range(n_layers))

        # ⚠️ COLLAPSE MONITOR, always on. ||K_s||/||K_t|| and ||V_s||/||V_t|| averaged over
        # layers, logged every step alongside the losses. The failure this watches for is the
        # student driving its cache toward zero so the expert's prefix attention vanishes and
        # the loss sits at a floor set by the teacher's own activations -- at which point a
        # falling curve means nothing. Measured trajectory on the 4-camera arm: K 3.25 -> 2.42
        # -> 2.32 and V 7.16 -> 2.87 -> 2.63 over 0/500/1000 steps, i.e. shrinking toward 1
        # (the teacher's scale) from ABOVE, which is convergence, not collapse. Ratios heading
        # for 0 -- especially with the loss still falling -- are the signature to stop on.
        with torch.no_grad():
            self._kv_ratio_k = float(sum(
                s_k[i].float().norm() / t_k[i].float().norm().clamp_min(1e-9)
                for i in range(n_layers)) / n_layers)
            self._kv_ratio_v = float(sum(
                s_v[i].float().norm() / t_v[i].float().norm().clamp_min(1e-9)
                for i in range(n_layers)) / n_layers)

        # ⚠️ block_norm='cache': the per-layer normaliser becomes the CACHE-ATTRIBUTABLE part of
        # the output, ||y_teacher - y_zero||^2, instead of ||y_teacher||^2. y_zero is the layer
        # driven by a ZERO cache -- measured at 0.0104 against a trained student's 0.0012, i.e.
        # 99% of the old normaliser is cache-independent. Computed ONCE per batch under no_grad
        # (n_layers extra single-block forwards, the same cost as one _sweep) and reused by every
        # _sweep call, including the self-test's identity/shuffled/zero controls.
        y_zero: dict[int, torch.Tensor] = {}
        if self.block_norm == "cache":
            with torch.no_grad():
                for l in range(n_layers):
                    if l in skip:
                        continue
                    c = captured[l]
                    z = torch.zeros_like(s_k[l])
                    y_zero[l] = block_output_only(
                        expert.expert.layers[l], c["h_in"], z, torch.zeros_like(s_v[l]),
                        dict(c["kwargs"]))

        def _span_sweep(k_list, v_list, m: int):
            """L_span(m): teacher-force the span ENTRY, chain m blocks on the student's cache,
            compare at the span EXIT.

                h^S_{l+1} = B_l(h^T_l ; K^S_l,V^S_l)
                h^S_{l+j+1} = B_{l+j}(h^S_{l+j} ; K^S_{l+j},V^S_{l+j})     j = 1..m-1
                L = D(h^S_{l+m}, h^T_{l+m})

            ⚠️ DISJOINT spans (stride m), so the total number of block forwards is n_layers
            regardless of m -- the schedule costs the same at m=1 and m=28. Overlapping spans
            would be m x the compute for the same coverage.
            ⚠️ m=1 must reduce EXACTLY to block_output_loss: same fresh single-layer cache, same
            captured kwargs, same normaliser. That equivalence is the self-test.
            ⚠️ CHECKPOINTED per layer, but ONLY from m >= SPAN_CKPT_MIN (default 14).
            Retaining activations for a 28-deep chain OOMed the L_freerun arm (75 -> >79 GB), so
            the long stages need it; its backward also produced non-finite gradients in bf16,
            which is why the finiteness guard below is not optional there.
            ⚠️ But checkpointing EVERY m > 1 was 4.2x slower for nothing at m=7. MEASURED on the
            curriculum: m=1 (uncheckpointed) ran 4.81 s/it, m=7 (checkpointed) 20 s/it, with GPU
            util at 25-35% -- the card idling while the backward serially recomputes all 28
            blocks. The span count is DISJOINT, so m=7 does the same 28 block-forwards as m=1;
            checkpointing was the only difference, and it doubles them. Memory said it was
            unnecessary too: m=7 checkpointed used 66 GiB where m=1 uncheckpointed used 68.3.
            Raise SPAN_CKPT_MIN if a mid-length stage OOMs; lower it to 2 for the old behaviour.
            """
            # Deep chains must trade compute for memory; short ones must not (see docstring).
            _span_ckpt = m >= int(os.environ.get("SPAN_CKPT_MIN", "14"))
            if not getattr(self, "_span_ckpt_logged", False):
                print(f"[span] m={m} checkpointing={'ON' if _span_ckpt else 'OFF'} "
                      f"(SPAN_CKPT_MIN={os.environ.get('SPAN_CKPT_MIN', '14')})", flush=True)
                self._span_ckpt_logged = True
            acc = None
            n_span = 0
            for l0 in range(0, n_layers - m + 1, m):
                idx = [l for l in range(l0, l0 + m) if l not in skip]
                if not idx:
                    continue

                def _one(h, li):
                    l = int(li)
                    c = DynamicCache()
                    c.update(k_list[l], v_list[l], 0, {})
                    blk = expert.expert.layers[l]
                    orig = blk.self_attn.layer_idx
                    blk.self_attn.layer_idx = 0
                    try:
                        o = blk(h, past_key_values=c, use_cache=True,
                                **dict(captured[l]["kwargs"]))
                    finally:
                        blk.self_attn.layer_idx = orig
                    return o[0] if isinstance(o, tuple) else o

                h = captured[idx[0]]["h_in"]
                for l in idx:
                    h = (torch.utils.checkpoint.checkpoint(
                            _one, h, torch.tensor(l), use_reentrant=False)
                         if _span_ckpt else _one(h, torch.tensor(l)))
                tgt = captured[idx[-1]]["y_out"].float().detach()
                num = (h.float() - tgt).pow(2).mean()
                if self.block_norm == "cache" and y_zero.get(idx[-1]) is not None:
                    den = (tgt - y_zero[idx[-1]].float()).pow(2).mean()
                else:
                    den = tgt.pow(2).mean()
                term = num / den.clamp_min(1e-6)
                if not torch.isfinite(term):
                    self._span_dropped = getattr(self, "_span_dropped", 0) + 1
                    print(f"[span] NON-FINITE span at l0={l0} m={m} "
                          f"(dropped {self._span_dropped})", flush=True)
                    continue
                acc = term if acc is None else acc + term
                n_span += 1
            # ⚠️ divide by m as well as by the span count, so the value is PER-LAYER
            # equivalent and the four curriculum stages arrive at the same magnitude. Measured
            # without it: m=1 1.21e-3, m=7 9.0x, m=14 15.4x, m=28 34.9x -- because the loss is
            # roughly the SUM of the m per-layer errors (times the ~1.1-1.3x amplification the
            # span probes measured), while the span count falls as 1/m. Feeding a 35x larger
            # gradient into a shared LR schedule at stage 4 is how L_roll at lambda=1 NaN'd.
            return acc / max(n_span, 1) / m if acc is not None else None

        lw = self._layer_weights(n_layers)
        if not self._blw_logged:
            self._blw_logged = True
            print(f"[blw] block_layer_weights={self.block_layer_weights} -> "
                  f"{[round(x, 2) for x in lw]}", flush=True)

        _bands = {"early": [None, 0], "mid": [None, 0], "deep": [None, 0]}

        def _sweep(k_list, v_list):
            """Returns ``(mse + cosine, mse_mean, cosine_mean)``, weighted mean over layers."""
            mse_acc = cos_acc = None
            w_used = 0.0
            for l in range(n_layers):
                if l in skip:      # identity stand-in: B_l(h) == h, term is trivially 0
                    continue
                w = lw[l]
                if w == 0.0:       # zero weight: skip the forward too, not just the term
                    continue
                c = captured[l]
                mse, cosine = block_output_loss(
                    expert.expert.layers[l], c["h_in"], c["y_out"],
                    k_list[l], v_list[l], dict(c["kwargs"]),
                    y_zero=y_zero.get(l),
                )
                mse_acc = w * mse if mse_acc is None else mse_acc + w * mse
                cos_acc = w * cosine if cos_acc is None else cos_acc + w * cosine
                w_used += w
                # per-band UNWEIGHTED means, so "did the deep layers actually improve" is
                # answerable from the curve. The scalar block_loss is a weighted average and
                # can fall by REALLOCATION alone, which would otherwise be invisible.
                f = l / max(n_layers - 1, 1)
                band = ("early" if f < 10 / 27 else
                        "mid" if f < 19 / 27 else "deep")
                add = mse.detach() + cosine.detach()   # already unweighted: report raw per-layer
                _bands[band][0] = add if _bands[band][0] is None else _bands[band][0] + add
                _bands[band][1] += 1
            # ⚠️ WEIGHTED mean, divided by the weight sum rather than the layer count, so the
            # loss MAGNITUDE is unchanged by reweighting. Dividing by n_used instead would
            # scale the loss (and therefore the effective LR) with the profile -- the same
            # confound the /m note on _span_sweep documents.
            mse_mean = mse_acc / max(w_used, 1e-8)
            cos_mean = cos_acc / max(w_used, 1e-8)
            return mse_mean + cos_mean, mse_mean, cos_mean

        # ⚠️ DIAGNOSTIC, opt-in via BLOCK_FREERUN=1. Answers whether teacher-forcing hides
        # compounding: L_block feeds each block the TEACHER's h_l, so per-layer error is
        # measured in isolation, while at inference layer l receives whatever the preceding
        # layers produced from the student's cache. Both curves here use the TRAINING path --
        # a prefill over the same token sequence, no CoT generation -- so the two caches have
        # identical length and differ only in values.
        #   teacher-forced  ||B_l(h^T_l; K^S,V^S) - h^T_{l+1}||^2 / ||h^T_{l+1}||^2
        #   free-running    ||h^S_l - h^T_l||^2 / ||h^T_l||^2
        if os.environ.get("BLOCK_FREERUN") == "1":
            with torch.no_grad():
                fr_cache = DynamicCache()
                for i in range(n_layers):
                    fr_cache.update(s_k[i], s_v[i], i, {})
                fr_h: dict[int, torch.Tensor] = {}

                def _fr_hook(idx):
                    def f(_m, args, kw, out):
                        fr_h[idx] = (args[0] if args else kw["hidden_states"]).detach()
                        if idx == n_layers - 1:
                            fr_h[n_layers] = (out[0] if isinstance(out, tuple) else out).detach()
                    return f

                fh = [expert.expert.layers[i].register_forward_hook(_fr_hook(i), with_kwargs=True)
                      for i in range(n_layers)]
                try:
                    expert.expert(inputs_embeds=embeds, attention_mask=e_mask,
                                  position_ids=pos, past_key_values=fr_cache, use_cache=True,
                                  is_causal=False)
                finally:
                    for h_ in fh:
                        h_.remove()
                rel = lambda a, b: float((a.float() - b.float()).pow(2).mean()
                                         / b.float().pow(2).mean().clamp_min(1e-9))
                # ⚠️ A THIRD chain, on a ZERO cache, so the FREE-RUNNING curve can be normalised
                # the same way the reconditioned block loss is: by the CACHE-ATTRIBUTABLE part of
                # the teacher's hidden state at that depth, ||h^T_l - h^0_l||^2, rather than by
                # ||h^T_l||^2 (which the residual stream dominates -- measured: a zero cache
                # perturbs block outputs by only ~1%). On this scale a layer reads 0 when the
                # student's own chain matches the teacher's and 1 when it is no better than
                # supplying no cache at all, so teacher-forced and free-running are directly
                # comparable per layer instead of differing by ~100x of normalisation.
                z_h: dict[int, torch.Tensor] = {}
                if self.block_norm == "cache":
                    z_cache = DynamicCache()
                    for i in range(n_layers):
                        z_cache.update(torch.zeros_like(s_k[i]), torch.zeros_like(s_v[i]), i, {})
                    zh_store: dict[int, torch.Tensor] = {}

                    def _z_hook(idx):
                        def f(_m, args, kw, out):
                            zh_store[idx] = (args[0] if args else kw["hidden_states"]).detach()
                            if idx == n_layers - 1:
                                zh_store[n_layers] = (
                                    out[0] if isinstance(out, tuple) else out).detach()
                        return f

                    zhs = [expert.expert.layers[i].register_forward_hook(_z_hook(i),
                                                                        with_kwargs=True)
                           for i in range(n_layers)]
                    try:
                        expert.expert(inputs_embeds=embeds, attention_mask=e_mask,
                                      position_ids=pos, past_key_values=z_cache, use_cache=True,
                                      is_causal=False)
                    finally:
                        for h_ in zhs:
                            h_.remove()
                    z_h = zh_store
                    y_zero_fr = {l: block_output_only(
                        expert.expert.layers[l], captured[l]["h_in"],
                        torch.zeros_like(s_k[l]), torch.zeros_like(s_v[l]),
                        dict(captured[l]["kwargs"])) for l in range(n_layers)}
                else:
                    y_zero_fr = {}

                def _tf_term(l):
                    m, c = block_output_loss(expert.expert.layers[l], captured[l]["h_in"],
                                              captured[l]["y_out"], s_k[l], s_v[l],
                                              dict(captured[l]["kwargs"]),
                                              y_zero=y_zero_fr.get(l))
                    return float(m + c)
                tf = [_tf_term(l) for l in range(n_layers)]
                if z_h:
                    # cache-attributable: ||h^S_l - h^T_l||^2 / ||h^T_l - h^0_l||^2
                    fr = [float((fr_h[l].float() - captured[l]["h_in"].float()).pow(2).mean()
                                / (captured[l]["h_in"].float()
                                   - z_h[l].float()).pow(2).mean().clamp_min(1e-9))
                          for l in range(n_layers)]
                else:
                    fr = [rel(fr_h[l], captured[l]["h_in"]) for l in range(n_layers)]
                fr.append(rel(fr_h[n_layers], captured[n_layers - 1]["y_out"]))
                # ⚠️ RAW magnitudes, not just the ratio. A large normalised value is ambiguous:
                # it can mean the student's error is big, or that the cache-attributable
                # DENOMINATOR at that layer is small (the cache barely matters there). Those
                # imply different fixes -- fix the student vs ignore the layer -- so print both
                # the numerator ||y_S - y_T||^2 and the span ||y_T - y_0||^2, plus the teacher
                # output's own magnitude for scale.
                if y_zero_fr:
                    print("[fr-raw] layer   ||y_S-y_T||^2   ||y_T-y_0||^2     ||y_T||^2"
                          "   span/||y_T||", flush=True)
                    for l in range(0, n_layers, 4):
                        c = captured[l]
                        yt = c["y_out"].float()
                        ys = block_output_only(expert.expert.layers[l], c["h_in"],
                                               s_k[l], s_v[l], dict(c["kwargs"])).float()
                        y0 = y_zero_fr[l].float()
                        num = float((ys - yt).pow(2).mean())
                        span = float((yt - y0).pow(2).mean())
                        mag = float(yt.pow(2).mean())
                        print(f"[fr-raw] {l:>5}   {num:>13.4e}   {span:>13.4e}   {mag:>11.4e}"
                              f"   {span / max(mag, 1e-12):>11.4e}", flush=True)
                print("[freerun] layer teacher_forced free_running", flush=True)
                for l in range(0, n_layers, 4):
                    print(f"[freerun] {l:>3} {tf[l]:.4e} {fr[l]:.4e} "
                          f"ratio {fr[l] / max(tf[l], 1e-12):.1f}", flush=True)
                # ⚠️ Same decomposition for the FREE-RUNNING chain -- the state that actually
                # reaches action_out_proj. Teacher-forced numbers describe one block in
                # isolation; these describe what the head is handed. Both normalisers are shown
                # because they weight depth differently and neither is obviously right:
                #   norm=teacher  ||h^S-h^T||^2 / ||h^T||^2
                #   norm=cache    ||h^S-h^T||^2 / ||h^T-h^0||^2
                if z_h:
                    print("[fr-state] layer   ||h_S-h_T||^2   ||h_T-h_0||^2      ||h_T||^2"
                          "   norm=teach    norm=cache", flush=True)
                    for l in range(0, n_layers, 4):
                        ht = captured[l]["h_in"].float()
                        num = float((fr_h[l].float() - ht).pow(2).mean())
                        span = float((ht - z_h[l].float()).pow(2).mean())
                        mag = float(ht.pow(2).mean())
                        print(f"[fr-state] {l:>5}   {num:>13.4e}   {span:>13.4e}   {mag:>12.4e}"
                              f"   {num / max(mag, 1e-12):>10.4e}   {num / max(span, 1e-12):>11.4e}",
                              flush=True)
                print(f"[freerun] FINAL tf_mean {sum(tf) / len(tf):.4e} "
                      f"fr_last {fr[-1]:.4e} ratio {fr[-1] / max(sum(tf) / len(tf), 1e-12):.1f}",
                      flush=True)

        # ⚠️ DIAGNOSTIC, opt-in via BLOCK_TSWEEP=1. Is the objective BLIND TO PART OF THE
        # DENOISING TRAJECTORY? Training draws one t per step from the teacher's Beta(1.5,1)
        # schedule (t = 0.999 - b*0.999, mass concentrated at low-to-mid t), while inference
        # walks t on a UNIFORM grid, linspace(0,1,n_steps+1). If the per-layer error is largest
        # where the training density is lowest -- near t -> 1, the final steps that actually set
        # the trajectory -- the loss is systematically under-weighting the steps that matter and
        # no amount of training on it will fix them.
        # Reports, at each t the sampler visits: the block loss, the ZERO-cache floor at that t
        # (the level below which the loss carries no information), and the training density.
        # ⚠️ Everything is recomputed per t: the teacher's h^T_l depend on t through the noisy
        # action embeds, so reusing the captured states from one t would compare against the
        # wrong target -- silently, since the shapes match.
        if os.environ.get("BLOCK_TSWEEP") == "1" and traj_data is not None:
            with torch.no_grad():
                n_steps = 10
                grid = [i / n_steps for i in range(n_steps + 1)]
                # Beta(1.5,1) density of the MAPPED t, for the same grid: b = 1 - t/0.999,
                # p(b) = 1.5 * b^0.5, and |db/dt| = 1/0.999 is constant so it cancels in a
                # ratio -- reported normalised to its own max, which is what "under-weighted"
                # has to be read against.
                dens = [1.5 * max(1.0 - t / 0.999, 0.0) ** 0.5 for t in grid]
                dmax = max(dens) or 1.0
                print("[tsweep]     t   block_loss   zero-cache   train_density", flush=True)
                for t_v_, dn in zip(grid, dens):
                    tt = torch.full((b,), float(t_v_), device=device)
                    emb_t = expert.noisy_action_embeds(traj_data, tt, device, dtype)
                    cap_t: dict[int, dict] = {}

                    def mk(idx):
                        def h(_m, a, kw, out):
                            cap_t[idx] = {
                                "h_in": (a[0] if a else kw["hidden_states"]).detach(),
                                "y_out": (out[0] if isinstance(out, tuple) else out).detach(),
                                "kwargs": {k: v for k, v in kw.items()
                                           if k != "past_key_values"}}
                        return h

                    c_t = DynamicCache()
                    for i in range(n_layers):
                        c_t.update(t_k[i], t_v[i], i, {})
                    hs = [expert.expert.layers[i].register_forward_hook(mk(i), with_kwargs=True)
                          for i in range(n_layers)]
                    try:
                        expert.expert(inputs_embeds=emb_t, attention_mask=e_mask,
                                      position_ids=pos, past_key_values=c_t, use_cache=True,
                                      is_causal=False)
                    finally:
                        for h_ in hs:
                            h_.remove()

                    def sweep_t(k_list, v_list):
                        acc = 0.0
                        for l in range(n_layers):
                            m, c = block_output_loss(
                                expert.expert.layers[l], cap_t[l]["h_in"], cap_t[l]["y_out"],
                                k_list[l], v_list[l], dict(cap_t[l]["kwargs"]))
                            acc += float(m + c)
                        return acc / n_layers

                    real_t = sweep_t(s_k, s_v)
                    zero_t = sweep_t([torch.zeros_like(k) for k in s_k],
                                     [torch.zeros_like(v) for v in s_v])
                    print(f"[tsweep] {t_v_:5.2f}   {real_t:.4e}   {zero_t:.4e}   "
                          f"{dn / dmax:.3f}", flush=True)

        # ⚠️ DIAGNOSTIC, opt-in via BLOCK_SPAN=n. Does an objective that teacher-forces only
        # every n-th layer SEE the compounding that L_block (n=1) is blind to?
        # For each disjoint span [l, l+n): drive block l with the teacher's h^T_l, chain the
        # next n-1 blocks on the STUDENT's cache, and compare to the teacher's h^T_{l+n}.
        # Read it against two references, both printed:
        #   sum1  = the n single-layer terms L_block already pays over the same layers.
        #           span >> sum1 means the chain amplifies -- error the objective cannot see.
        #   span == sum1 means errors merely add, and spans buy nothing over n=1.
        # ⚠️ self-test built in: BLOCK_SPAN=1 must reproduce the n=1 numbers exactly, since
        # block_span_output with one block IS block_output_loss.
        # BLOCK_SPAN takes a LIST ("1,2,4,7,14"): one model load, the whole curve. A single
        # span length cannot distinguish "errors add" from "errors amplify" -- only the trend
        # in n can, and loading the 10B teacher costs ~5 min per run.
        span_ns = [int(x) for x in os.environ.get("BLOCK_SPAN", "").split(",") if x.strip()]
        if span_ns:
            with torch.no_grad():
                rel = lambda a, b: float((a.float() - b.float()).pow(2).mean()
                                         / b.float().pow(2).mean().clamp_min(1e-9))
                def _single_term(l):
                    m, c = block_output_loss(
                        expert.expert.layers[l], captured[l]["h_in"], captured[l]["y_out"],
                        s_k[l], s_v[l], dict(captured[l]["kwargs"]))
                    return float(m + c)
                singles = [_single_term(l) for l in range(n_layers)]
                for span_n in span_ns:
                    print(f"[span{span_n}] start  span_err     sum1        span/sum1", flush=True)
                    tot_s = tot_1 = 0.0
                    for l0 in range(0, n_layers - span_n + 1, span_n):
                        idx = list(range(l0, l0 + span_n))
                        y = block_span_output(
                            [expert.expert.layers[i] for i in idx],
                            captured[l0]["h_in"],
                            [s_k[i] for i in idx], [s_v[i] for i in idx],
                            [dict(captured[i]["kwargs"]) for i in idx])
                        e_span = rel(y, captured[idx[-1]]["y_out"])
                        e_sum1 = sum(singles[i] for i in idx)
                        tot_s += e_span; tot_1 += e_sum1
                        print(f"[span{span_n}] {l0:>4}  {e_span:.4e}  {e_sum1:.4e}  "
                              f"{e_span / max(e_sum1, 1e-12):>8.2f}", flush=True)
                    print(f"[span{span_n}] TOTAL span {tot_s:.4e}  sum1 {tot_1:.4e}  "
                          f"amplification {tot_s / max(tot_1, 1e-12):.2f}", flush=True)

        # ⚠️ DIAGNOSTIC, opt-in via BLOCK_COSINE=1. Is the teacher-forced MSE dominated by a
        # SCALE error or a DIRECTION error? Exactly, not by intuition:
        #     ||a-b||^2/||b||^2 = 1 + r^2 - 2 r cos      with r = ||a||/||b||
        # so cos ~ 1 with r != 1 means the student's block output points the right way and is
        # mis-scaled (an MSE objective is then fighting a gain, and a cosine term adds
        # nothing); cos < 1 means the direction itself is wrong, which is what a cosine or
        # angular objective would target and relative MSE under-weights when r is small.
        # `pred` re-derives the MSE from (r, cos): it must match `mse` to ~1e-3, and if it
        # does not, one of the three is being computed on a different tensor than assumed.
        # Built-in control: the TEACHER's own cache must give cos = 1.000, r = 1.000, mse ~ 0.
        if os.environ.get("BLOCK_COSINE") == "1":
            with torch.no_grad():
                def stats(y_s, y_t):
                    a = y_s.float().flatten(0, -2)      # [B*T, hidden], per-token rows
                    b = y_t.float().flatten(0, -2)
                    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()
                    r = (a.norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-9)).mean()
                    mse = (a - b).pow(2).mean() / b.pow(2).mean().clamp_min(1e-9)
                    return float(mse), float(cos), float(r)

                def run(l, k_list, v_list):
                    return block_span_output(
                        [expert.expert.layers[l]], captured[l]["h_in"],
                        [k_list[l]], [v_list[l]], [dict(captured[l]["kwargs"])])

                print("[cos] layer      mse     cos      r    pred_mse", flush=True)
                acc = []
                for l in range(n_layers):
                    m, c, r = stats(run(l, s_k, s_v), captured[l]["y_out"])
                    acc.append((m, c, r))
                    if l % 4 == 0:
                        print(f"[cos] {l:>5} {m:.3e} {c:.5f} {r:.5f}  "
                              f"{1 + r * r - 2 * r * c:.3e}", flush=True)
                mm = sum(a[0] for a in acc) / len(acc)
                mc = sum(a[1] for a in acc) / len(acc)
                mr = sum(a[2] for a in acc) / len(acc)
                im, ic, ir = stats(run(0, t_k, t_v), captured[0]["y_out"])
                print(f"[cos] MEAN mse {mm:.4e} cos {mc:.5f} r {mr:.5f}", flush=True)
                print(f"[cos] CONTROL teacher-cache layer0: mse {im:.3e} cos {ic:.5f} "
                      f"r {ir:.5f}   (must be ~0 / 1 / 1)", flush=True)

        # ⚠️ DIAGNOSTIC, opt-in via BLOCK_ODE=1. The span probe asked whether error compounds
        # along LAYERS (answer: barely -- ~1.2x, and the deep half contracts). This asks the
        # same question along the DENOISING axis, where the mechanism is different: all 10
        # Euler steps read the SAME student cache, so a biased field error accumulates
        # coherently instead of cancelling.
        #   e_step[k] = field error at the TEACHER's x_k  (teacher-forced along the ODE)
        #   e_final   = ||x^S_final - x^T_final||^2 / ||x^T_final||^2  (student integrates itself)
        #   amplification = e_final / sum_k e_step
        # `bias` is the mean cosine between consecutive steps' error vectors: ~1 means the
        # error points the same way every step (integrates coherently, rollout training would
        # help), ~0 means it is step-to-step noise (it partly cancels, rollout buys little).
        if os.environ.get("BLOCK_ODE") == "1":
            with torch.no_grad():
                dif = expert.diffusion
                n_act = expert.n_action_tokens

                def field(x, t, k_list, v_list):
                    while torch.is_tensor(t) and t.dim() < x.dim():   # see field_at's note
                        t = t.unsqueeze(-1)
                    c = DynamicCache()
                    for i in range(n_layers):
                        c.update(k_list[i], v_list[i], i, {})
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        emb = expert.action_in_proj(x, t)
                    emb = emb.to(embeds.dtype)
                    if emb.dim() == 2:
                        emb = emb.view(x.shape[0], n_act, -1)
                    out = expert.expert(inputs_embeds=emb, attention_mask=e_mask,
                                        position_ids=pos, past_key_values=c, use_cache=True,
                                        is_causal=False)
                    h = out.last_hidden_state[:, -n_act:]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        return expert.action_out_proj(h).view(-1, *expert.x_dims)

                errs, e_step = [], []

                def teacher_step(x, t):
                    f_t = field(x, t, t_k, t_v)
                    f_s = field(x, t, s_k, s_v)
                    e_step.append(float((f_s.float() - f_t.float()).pow(2).mean()
                                        / f_t.float().pow(2).mean().clamp_min(1e-9)))
                    errs.append((f_s.float() - f_t.float()).flatten())
                    return f_t                       # integrate along the TEACHER's path

                b = embeds.shape[0]
                dev = embeds.device
                # ⚠️ return_all_steps=True on BOTH runs: the quantity asked for is the
                # per-step DIVERGENCE of the student's own trajectory from the teacher's, not
                # just the endpoint. Same seed before each call, so x_0 is identical and every
                # later difference is attributable to the cache alone.
                torch.manual_seed(1234)              # identical initial noise for both runs
                xs_t, ts = dif.sample(batch_size=b, step_fn=teacher_step, device=dev,
                                      return_all_steps=True)
                torch.manual_seed(1234)
                xs_s, _ = dif.sample(batch_size=b, step_fn=lambda x, t: field(x, t, s_k, s_v),
                                     device=dev, return_all_steps=True)
                rel_x = lambda a, c: float((a.float() - c.float()).pow(2).mean()
                                           / c.float().pow(2).mean().clamp_min(1e-9))
                print("[ode-freerun] step      t   field_err(tf)   x_divergence(fr)", flush=True)
                for k in range(xs_t.shape[1]):
                    tv = float(ts[k]) if k < len(ts) else float("nan")
                    fe = e_step[k] if k < len(e_step) else float("nan")
                    print(f"[ode-freerun] {k:>4} {tv:>6.2f}   {fe:>13.4e}   "
                          f"{rel_x(xs_s[:, k], xs_t[:, k]):>16.4e}", flush=True)
                x_t, x_s = xs_t[:, -1], xs_s[:, -1]
                e_final = rel_x(x_s, x_t)
                tot = sum(e_step)
                cs = [float(torch.nn.functional.cosine_similarity(a, b_, dim=0))
                      for a, b_ in zip(errs, errs[1:])]
                print(f"[ode] steps {len(e_step)}  per-step " +
                      " ".join(f"{e:.2e}" for e in e_step[:5]) + " ...", flush=True)
                print(f"[ode] sum_steps {tot:.4e}  e_final {e_final:.4e}  "
                      f"amplification {e_final / max(tot, 1e-12):.2f}  "
                      f"bias(cos between consecutive step errors) {sum(cs) / max(len(cs), 1):.3f}",
                      flush=True)

        # ⚠️ SELF-TEST for the span schedule, opt-in via SPAN_SELFTEST=1. Two claims:
        #   (a) _span_sweep(m=1) == _sweep  -- the curriculum's first stage is EXACTLY the
        #       existing block loss, so every prior number stays the baseline. Compared here
        #       rather than argued: the two take different code paths (checkpointing off vs on,
        #       loop vs chain) and only agree if the driving is identical.
        #   (b) every m in the schedule yields a FINITE loss. m=28 is the full free-run, whose
        #       bf16 backward went non-finite in the L_freerun arm, so this is the go/no-go for
        #       the last stage before committing an epoch to it.
        if os.environ.get("SPAN_SELFTEST") == "1":
            with torch.no_grad():
                base = float(_sweep(s_k, s_v))
                one = float(_span_sweep(s_k, s_v, 1))
                print(f"[span-selftest] m=1 via _span_sweep {one:.6e} vs _sweep {base:.6e}  "
                      f"rel diff {abs(one - base) / max(base, 1e-12):.2e}   (must be ~0)",
                      flush=True)
                for m in (7, 14, 28):
                    if m > n_layers:
                        continue
                    v = _span_sweep(s_k, s_v, m)
                    print(f"[span-selftest] m={m:<3} loss {float(v):.6e}  "
                          f"spans {len(range(0, n_layers - m + 1, m))}  "
                          f"x m=1 {float(v) / max(one, 1e-12):.2f}", flush=True)

        # ⚠️ SELF-TEST, opt-in via BLOCK_SELFTEST=1. A finite, falling loss is NOT evidence
        # that it measures cache fidelity -- that gap has produced retractions here. Feeding
        # the TEACHER's own cache must give ~0; a layer-shuffled cache must be clearly worse
        # than the real student's. If identity is not ~0 the objective is broken however
        # healthy the curve looks.
        if os.environ.get("BLOCK_SELFTEST") == "1":
            with torch.no_grad():
                ident = float(_sweep(t_k, t_v)[0])
                shuf = list(range(n_layers))[::-1]
                shuffled = float(_sweep([s_k[i] for i in shuf], [s_v[i] for i in shuf])[0])
                real = float(_sweep(s_k, s_v)[0])
                # ⚠️ THE COLLAPSE CONTROL. If the student learns to drive its cache to ~0, the
                # expert's attention over the prefix vanishes and every block output falls back
                # to whatever the action tokens alone produce -- a FLOOR that depends only on
                # the teacher, not on cache fidelity. Then a zero cache scores the same as the
                # student's, the loss has stopped measuring anything, and a flat curve is the
                # expected outcome rather than convergence.
                zeros = float(_sweep([torch.zeros_like(k) for k in s_k],
                                     [torch.zeros_like(v) for v in s_v])[0])
                # Norm ratios per layer: monotone decay toward 0 through the loss drop is the
                # signature of the same collapse, visible before the scores converge.
                nk = [float(s_k[i].float().norm() / t_k[i].float().norm().clamp_min(1e-9))
                      for i in range(n_layers)]
                nv = [float(s_v[i].float().norm() / t_v[i].float().norm().clamp_min(1e-9))
                      for i in range(n_layers)]
            print(
                f"[block-selftest] identity(teacher cache)={ident:.6e}  student={real:.4f}  "
                f"layer-shuffled={shuffled:.4f}  ZERO-cache={zeros:.4f}\n"
                f"[block-selftest]   healthy: identity ~0, student < shuffled, "
                f"student << ZERO   |   collapsed: student ~ shuffled ~ ZERO",
                flush=True,
            )
            print(f"[kvnorm] ||K_s||/||K_t|| mean {sum(nk) / len(nk):.4f} "
                  f"min {min(nk):.4f} max {max(nk):.4f} | per-layer "
                  + " ".join(f"{v:.3f}" for v in nk[::4]), flush=True)
            print(f"[kvnorm] ||V_s||/||V_t|| mean {sum(nv) / len(nv):.4f} "
                  f"min {min(nv):.4f} max {max(nv):.4f} | per-layer "
                  + " ".join(f"{v:.3f}" for v in nv[::4]), flush=True)

        fr_term = None
        if self.block_freerun_weight > 0:
            # ⚠️ WITH gradient, unlike the BLOCK_FREERUN diagnostic above. s_k/s_v carry grad
            # (recompute_kv -> rotate), the expert is frozen, and `embeds` is detached, so the
            # only path back is through the student's cache -- which is the point.
            fr_cache = DynamicCache()
            for i in range(n_layers):
                fr_cache.update(s_k[i], s_v[i], i, {})
            # ⚠️ CHECKPOINTED, one layer at a time. Retaining activations for all 36
            # layers of this second forward pushed the step from ~75 GB to over the 79 GB
            # card and OOMed (surfaced as an opaque NCCL "unhandled cuda error").
            # ⚠️ A FRESH single-layer cache inside the checkpointed function, seeded exactly
            # as block_output_loss does. A shared DynamicCache would be mutated twice --
            # once in the forward and again when checkpointing recomputes it -- appending the
            # action K/V a second time and silently changing what the layer attends to.
            def _fr_layer(h, l_idx):
                l = int(l_idx)
                cache = DynamicCache()
                cache.update(s_k[l], s_v[l], 0, {})
                blk = expert.expert.layers[l]
                orig = blk.self_attn.layer_idx
                blk.self_attn.layer_idx = 0
                try:
                    o = blk(h, past_key_values=cache, use_cache=True,
                            **dict(captured[l]["kwargs"]))
                finally:
                    blk.self_attn.layer_idx = orig
                return o[0] if isinstance(o, tuple) else o

            # ⚠️ fp32 for the free-running chain, opt-in via BLOCK_FR_FP32=1. Its backward
            # runs through 36 frozen expert blocks -- far deeper than L_block's single-block
            # terms -- and in bf16 that produced a NON-FINITE gradient on the very first
            # backward under plain zero2: deepspeed reported grad_norm as a constant 2.0
            # sentinel at step 0 while both losses were still finite, then everything went
            # nan at step 1. zero2_offload did NOT show it, because DeepSpeedCPUAdam works in
            # fp32 on the host -- which means that path may have been MASKING the overflow
            # rather than avoiding it.
            fr_fp32 = os.environ.get("BLOCK_FR_FP32") == "1"
            n_grad = (self.block_freerun_layers if 0 < self.block_freerun_layers < n_layers
                      else n_layers)
            cut = n_layers - n_grad
            h_s = embeds
            if cut:
                with torch.no_grad():          # chain still runs; no activations retained
                    for _l in range(cut):
                        h_s = _fr_layer(h_s, _l)
                h_s = h_s.detach()
            with torch.autocast("cuda", enabled=not fr_fp32):
                if fr_fp32:
                    h_s = h_s.float()
                for _l in range(cut, n_layers):
                    h_s = torch.utils.checkpoint.checkpoint(
                        _fr_layer, h_s, torch.tensor(_l), use_reentrant=False)
            tgt = captured[n_layers - 1]["y_out"]
            fr_term = ((h_s.float() - tgt.float()).pow(2).mean()
                       / tgt.float().pow(2).mean().clamp_min(1e-6))

        # ---- L_roll: UNROLL the sampler and match the teacher's velocity on each path ----
        # ⚠️ This is the only term that sees what `ade` actually measures. Every other loss here
        # is teacher-forced in x: the student is scored at a state the TEACHER produced, so the
        # divergence of its own trajectory is invisible. Measured with BLOCK_ODE on this arm,
        # that divergence grows x5.62 then x3.20 over the first steps and reaches 0.76 relative
        # (~87% RMS) by the end -- while the per-step field error stays ~0.1-0.28 and the block
        # loss reports 1.7e-3. `ade` 6.29 against min_ade 2.73 is the same story at the metric.
        #
        #   x_0 = GT-conditioned noisy state at a random grid point t_k  (scheduled sampling:
        #         the START is on the teacher's path, the REST is each model's own)
        #   for j in range(roll_steps):
        #       v_S = v(x^S_j, t_{k+j} ; K^S)      v_T = v(x^T_j, t_{k+j} ; K^T)   [no grad]
        #       loss += mse(v_S, sg v_T)
        #       x^S_{j+1} = x^S_j + dt * v_S       x^T_{j+1} = x^T_j + dt * v_T
        #
        # ⚠️ Each side advances with ITS OWN velocity, which is the whole point -- teacher-forcing
        # the second step would collapse this back into the per-step field loss.
        # ⚠️ Plain MSE, matching `FlowMatching.compute_loss_from_pred`; the relative form blew up
        # to 10.75 on a near-zero-velocity clip and inverts the weighting toward trivial clips.
        roll_term = None
        if self.roll_weight > 0 and traj_data is not None:
            dif = expert.diffusion
            n_inf = int(getattr(dif, "num_inference_steps", 10))
            dt = 1.0 / n_inf
            n_roll = max(1, min(self.roll_steps, n_inf))
            # a random start on the sampler's OWN grid, leaving room for n_roll steps
            k0 = int(torch.randint(0, max(1, n_inf - n_roll + 1), (1,)).item())

            def field_at(x, t_scalar, k_list, v_list):
                # ⚠️ t must be shaped LIKE x ([B,1,1]), not [B]. `action_in_proj` does
                # `timesteps[..., -1]` then `.repeat(1, T, 1)`, so a [B] timestep collapses to a
                # 0-dim scalar and the timestep features come out batch-1 -- which broadcasts
                # silently at B=1 and dies with "Expected size 8 but got size 1" at B=8. That is
                # why every bs=1 smoke passed. `noisy_action_embeds` unsqueezes for this reason.
                tt = torch.full((x.shape[0],), t_scalar, device=device, dtype=torch.float32)
                while tt.dim() < x.dim():
                    tt = tt.unsqueeze(-1)
                cache = DynamicCache()
                for i in range(n_layers):
                    cache.update(k_list[i], v_list[i], i, {})
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    emb = expert.action_in_proj(x, tt)
                emb = emb.to(dtype)
                if emb.dim() == 2:
                    emb = emb.view(x.shape[0], expert.n_action_tokens, -1)
                out = expert.expert(inputs_embeds=emb, attention_mask=e_mask, position_ids=pos,
                                    past_key_values=cache, use_cache=True, is_causal=False)
                h = out.last_hidden_state[:, -expert.n_action_tokens:]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return expert.velocity(h)

            # ⚠️ TRUNCATED unroll: the first n_roll-1 steps advance under no_grad and only the
            # LAST step is differentiated. Backpropagating through the whole chain of frozen
            # bf16 expert blocks produces NON-FINITE gradients -- measured twice now: the
            # L_freerun arm reported deepspeed's grad_norm sentinel then NaN at the next step,
            # and so did this term with a perfectly healthy grad_norm of 10.56 at lambda=2e-3.
            # The property that matters survives: the student is still evaluated at ITS OWN
            # state, reached by its own velocity, so the divergence `ade` punishes is in the
            # loss. What is given up is the gradient path THROUGH the earlier steps.
            with torch.no_grad():
                x_t0 = expert.noisy_x(traj_data, torch.full((b,), k0 * dt), device)
                x_s, x_tt = x_t0, x_t0
                for j in range(n_roll - 1):          # advance both, no graph
                    t_j = (k0 + j) * dt
                    x_s = x_s + dt * field_at(x_s, t_j, s_k, s_v).float()
                    x_tt = x_tt + dt * field_at(x_tt, t_j, t_k, t_v).float()
            t_last = (k0 + n_roll - 1) * dt
            v_s = torch.utils.checkpoint.checkpoint(
                field_at, x_s.detach(), t_last, s_k, s_v, use_reentrant=False)
            with torch.no_grad():
                v_t = field_at(x_tt, t_last, t_k, t_v)
            # ⚠️ STATE matching, not velocity matching. Matching v_S(x^S) to v_T(x^T) compares
            # two DIFFERENT points once the paths diverge, and the correct velocity at the
            # student's state is not the teacher's velocity at the teacher's state -- so that
            # form only stops the gap growing, it never closes the gap already there. Writing
            # d = x^S - x^T (detached under truncation),
            #     ||x^S_{k+1} - x^T_{k+1}||^2 = ||d + dt (v_S - v_T)||^2
            # is minimised at v_S = v_T - d/dt: the teacher's velocity PLUS a correction that
            # cancels the offset, i.e. steer back onto the teacher's trajectory. At n=1 the two
            # forms coincide (d=0, so state = dt^2 x velocity); they differ only for n>=2, which
            # is the whole point of rolling out. And x IS the action, so this is the trajectory
            # error in action space -- one action_to_traj away from `ade` itself.
            # ⚠️ HUBER, not MSE. The self-test showed this term at 0.62 when the rollout starts
            # at step 5 but 8.19 at step 8 -- a 13x spread, because a free step late in the
            # trajectory can land far off-manifold where the velocity is extreme. Squared error
            # turns those batches into a gradient spike; every MSE variant tried NaN'd on the
            # step after one (lambda 1 -> grad 5327, lambda 2e-3 -> 10.6, lambda 1e-2 -> 55,
            # all NaN next step). Huber keeps the same minimum with a bounded gradient.
            x_s_next = x_s.detach().float() + dt * v_s.float()
            x_t_next = (x_tt.float() + dt * v_t.float()).detach()
            roll_term = torch.nn.functional.huber_loss(x_s_next, x_t_next, delta=1.0)
            # ⚠️ and a finiteness guard: one bad batch must not kill a multi-hour run. Dropped
            # terms are counted so a silently-inert loss cannot masquerade as a healthy one.
            if not torch.isfinite(roll_term):
                self._roll_dropped = getattr(self, "_roll_dropped", 0) + 1
                print(f"[roll] NON-FINITE term dropped (total {self._roll_dropped}) at "
                      f"start step {k0}", flush=True)
                roll_term = v_s.float().sum() * 0.0
            if os.environ.get("BLOCK_SELFTEST") == "1":
                with torch.no_grad():
                    xi = x_t0
                    ident = []
                    for j in range(n_roll):
                        t_j = (k0 + j) * dt
                        vi = field_at(xi, t_j, t_k, t_v)
                        vt = field_at(x_t0 if j == 0 else xi, t_j, t_k, t_v)
                        ident.append(float(torch.nn.functional.huber_loss(
                            (xi.float() + dt * vi.float()),
                            (xi.float() + dt * vt.float()), delta=1.0)))
                        xi = xi + dt * vi.float()
                print(f"[roll-selftest] start step {k0}, {n_roll} steps | identity(teacher "
                      f"cache both sides) {sum(ident) / len(ident):.3e}  student "
                      f"{float(roll_term):.4f}   (identity must be ~0)", flush=True)

        # ---- L_field: match the VELOCITY, not just the hidden states --------------------
        # ⚠️ MEASURED motivation, not a hunch. The hidden states agree to 3.2% RMS per layer
        # while the velocity the trajectory actually integrates is 30% off (BLOCK_ODE probe),
        # because `action_out_proj` reads one narrow projection of the residual stream and
        # L_block spends its capacity uniformly over directions the head discards.
        # Same chained forward as the freerun arm -- checkpointed per layer, gradient reaching
        # the VLM only through the student's cache -- but scored after norm + out_proj.
        # ⚠️ Deep layers get this gradient directly while shallow ones receive it through the
        # CONTRACTIVE deep half (measured 0.79 across layers 14-27), so this is meant to run
        # ALONGSIDE L_block, which supplies the dense per-layer signal, not to replace it.
        field_term = None
        if self.field_weight > 0:
            fcache = [None] * n_layers

            def _field_layer(h, l_idx):
                l = int(l_idx)
                cache = DynamicCache()
                cache.update(s_k[l], s_v[l], 0, {})
                blk = expert.expert.layers[l]
                orig = blk.self_attn.layer_idx
                blk.self_attn.layer_idx = 0
                try:
                    o = blk(h, past_key_values=cache, use_cache=True,
                            **dict(captured[l]["kwargs"]))
                finally:
                    blk.self_attn.layer_idx = orig
                return o[0] if isinstance(o, tuple) else o

            h_f = embeds
            # ⚠️ same gate as _span_sweep: checkpointing a 28-deep chain was measured at 4.3x
            # slower with GPU util at 25-35%, for no memory benefit (peak was FLAT at 65.8 GiB
            # from m=1 to m=28). SPAN_CKPT_MIN>n_layers disables it here too.
            _f_ckpt = n_layers >= int(os.environ.get("SPAN_CKPT_MIN", "14"))
            with torch.autocast("cuda", enabled=True):
                for _l in range(n_layers):
                    h_f = (torch.utils.checkpoint.checkpoint(
                               _field_layer, h_f, torch.tensor(_l), use_reentrant=False)
                           if _f_ckpt else _field_layer(h_f, torch.tensor(_l)))
                v_s = expert.velocity(h_f)
                with torch.no_grad():
                    # ⚠️ FREE: the teacher's final block output is already captured, so the
                    # target costs one norm + one head call, not a second expert forward.
                    v_t = expert.velocity(captured[n_layers - 1]["y_out"])
            # ⚠️ PLAIN MSE, not the relative form the other block terms use, and not by
            # analogy -- `FlowMatching.compute_loss_from_pred` is
            # `mse_loss(x - noise, pred)`, so this is the scale the expert was actually
            # trained in, and the velocity target is O(1) by construction (x normalised,
            # noise ~ N(0,1)) so it needs no per-batch rescaling.
            # MEASURED reason to avoid the relative form: dividing by ||v_T||^2 exploded to
            # 10.75 on a batch where the teacher's velocity was near zero -- a stopped or
            # slow ego, i.e. precisely the clips where the trajectory is trivial and this
            # term should count for LEAST. Relative normalisation inverts that weighting.
            field_term = torch.nn.functional.mse_loss(
                v_s.float(), v_t.float().detach())

            # ⚠️ SELF-TEST: the same chain driven by the TEACHER's cache must give ~0. This is
            # what catches a norm/head convention error, which otherwise yields a finite,
            # falling loss that measures nothing -- exactly how freerun_loss sat at 0.999.
            if os.environ.get("BLOCK_SELFTEST") == "1":
                with torch.no_grad():
                    h_i = embeds
                    for _l in range(n_layers):
                        c = DynamicCache(); c.update(t_k[_l], t_v[_l], 0, {})
                        blk = expert.expert.layers[_l]
                        orig = blk.self_attn.layer_idx
                        blk.self_attn.layer_idx = 0
                        try:
                            o = blk(h_i, past_key_values=c, use_cache=True,
                                    **dict(captured[_l]["kwargs"]))
                        finally:
                            blk.self_attn.layer_idx = orig
                        h_i = o[0] if isinstance(o, tuple) else o
                    v_i = expert.velocity(h_i)
                    ident = float(torch.nn.functional.mse_loss(v_i.float(), v_t.float()))
                print(f"[field-selftest] identity(teacher cache) {ident:.3e} "
                      f"student {float(field_term):.4f}   (identity must be ~0)", flush=True)

        # ⚠️ m=1 routes to _sweep (the original per-layer path) rather than _span_sweep, so
        # the m=1 stage of the schedule is bit-identical to every block-loss number already in
        # this tree -- the curriculum's first epoch is not a new objective.
        # ⚠️ _span_sweep compares at SPAN EXITS, so a per-layer profile has no well-defined
        # meaning there -- applying it to the exit layer only would silently weight 1/m of the
        # layers. Refuse rather than half-apply.
        if self.block_span > 1 and self.block_layer_weights != "uniform":
            raise ValueError(
                f"block_layer_weights={self.block_layer_weights} needs block_span=1; spans "
                "compare at exits, so a per-layer profile cannot be applied inside one.")
        self._tf_term = self._span_term = None
        mse_mean = cos_mean = None
        if self.block_span_mix > 1:
            # BOTH objectives at once: teacher-forced per-layer (each layer graded in
            # isolation) AND the span at m (m blocks chained on the student's own cache).
            # They constrain different things -- per-layer fidelity vs. survival of a chain --
            # and the span curriculum showed the span term alone is nearly m-invariant, so it
            # adds little on its own; this asks whether it adds anything ON TOP of m=1.
            # ⚠️ Only the combined scalar, not the mse/cosine split -- this path blends TWO
            # different metrics (teacher-forced + span), so a clean per-component split of
            # the result doesn't mean the same thing as it does in the plain _sweep path.
            tf = _sweep(s_k, s_v)[0]
            sp = _span_sweep(s_k, s_v, self.block_span_mix)
            w = self.block_span_mix_weight
            if not self._mix_logged:
                self._mix_logged = True
                print(f"[mix] teacher-forced(m=1) + {w} x span(m={self.block_span_mix}), "
                      f"weighted MEAN", flush=True)
            self._tf_term = None if tf is None else tf.detach()
            self._span_term = None if sp is None else sp.detach()
            if tf is None:
                block_term = sp
            elif sp is None:
                block_term = tf          # a fully non-finite span must not kill the step
            else:
                # ⚠️ weighted MEAN, not sum. Both terms are already per-layer-normalised, so
                # summing them would double the gradient scale and silently change the
                # effective LR -- the confound the /m note on _span_sweep documents.
                block_term = (tf + w * sp) / (1.0 + w)
        elif self.block_weight == 0.0:
            # ⚠️ block_weight=0 (e.g. FIELD-ONLY): the per-layer sweep is a DIAGNOSTIC here,
            # not a loss -- and it is 28 block forwards, so building its graph and discarding
            # it is pure waste. Under no_grad it still reports whether training on the field
            # term also reduces the block term, which is the cross-check between the two
            # objectives, at a fraction of the cost.
            with torch.no_grad():
                if self.block_span <= 1:
                    block_term, mse_mean, cos_mean = _sweep(s_k, s_v)
                else:
                    block_term = _span_sweep(s_k, s_v, self.block_span)
        else:
            if self.block_span <= 1:
                block_term, mse_mean, cos_mean = _sweep(s_k, s_v)
            else:
                block_term = _span_sweep(s_k, s_v, self.block_span)
        self._last_bands = {k: (v[0] / v[1] if v[0] is not None and v[1] else None)
                            for k, v in _bands.items()}
        return block_term, mse_mean, cos_mean, fr_term, field_term, roll_term

    def forward(
        self,
        tokenized_data: dict[str, Any],
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        ego_future_xyz: torch.Tensor | None = None,
        ego_future_rot: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> KDVLAOutput:
        # ⚠️ These two gates must list EVERY term that needs the teacher, and they were the
        # last place block_weight stood in for "the block path is active". With block_weight=0
        # and roll_weight>0 the teacher never ran and hidden states were never requested, so
        # the block path produced nothing and the total came out a scalar with NO grad_fn --
        # which DeepSpeed reports as "loss must be a scalar tensor", naming the wrong half of
        # its own check (`numel()==1 and grad_fn is not None`).
        _block_family = (self.block_weight > 0 or self.field_weight > 0
                         or self.roll_weight > 0 or self.block_freerun_weight > 0)
        need_teacher = (
            self.kd_weight > 0 or self.kv_weight > 0 or _block_family
        ) and self.teacher is not None
        want_hidden = (self.kv_weight > 0 or _block_family) and self.teacher is not None

        tokenized_data = dict(tokenized_data)
        input_ids = tokenized_data.pop("input_ids")
        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)

        # The action expert consumes the VLM prefix through the exact handoff token. Keep this
        # boundary independent of labels_mask and of the broader trajectory-region mask below:
        # history and future coordinates share the same discretized vocabulary, so using a
        # generic "trajectory token" mask can select history when labels are unavailable.
        traj_future_start_mask = input_ids == self.special_token_ids["traj_future_start"]

        labels = input_ids.clone()
        if labels_mask is not None:
            labels = torch.where(labels_mask, labels, IGNORE_INDEX)

        # Hook the rotary embedding so L_block reuses the EXACT rotation the cached keys
        # carry, rather than reconstructing mrope positions. Registered only when needed.
        _need_block = _block_family
        rope, rope_handle = self._capture_rope() if _need_block else ({}, None)
        try:
            # ⚠️ Only materialise logits when something actually reads them. CE needs them
            # (ce_weight>0) and logit-KD needs them (kd_weight>0); the pure cache arms
            # (blockonly / blockrandt / kvonly) need NEITHER, and the lm_head projects
            # ~3k positions onto a ~155k vocab -- roughly a 1 GB bf16 tensor per sample plus
            # the matmul, every step, discarded. `logits_to_keep=1` keeps one position so the
            # output shape stays valid. Passing `labels` additionally makes HF compute its
            # OWN cross-entropy internally, which this forward then recomputes and discards.
            want_logits = self.ce_weight > 0 or self.kd_weight > 0
            with _Phase.t("student_prefill"):
              outputs = self.vlm(
                input_ids=input_ids,
                labels=labels if self.ce_weight > 0 else None,
                output_hidden_states=want_hidden,
                # ⚠️ use_cache=False, matching the teacher call below. This is a PREFILL --
                # nothing here generates, and every K/V the objectives use is recomputed
                # from `hidden_states` by `recompute_kv`. Leaving it on materialised a
                # ~450 MB per-sample cache (36 layers x 8 kv heads x ~3k positions x 128)
                # that was never read.
                use_cache=False,
                **({} if want_logits else {"logits_to_keep": 1}),
                **tokenized_data,
              )
        finally:
            if rope_handle is not None:
                rope_handle.remove()

        # ---- CE, byte-identical to TrainableReasoningVLA.forward -------------
        traj_mask = (
            (
                (labels >= self.future_token_start_idx)
                & (labels < self.future_token_start_idx + self.config.traj_vocab_size)
            )
            | (labels == self.special_token_ids["traj_future_start"])
            | (labels == self.special_token_ids["traj_future_end"])
        )
        # ⚠️ Skipped entirely when ce_weight==0: it is a cross-entropy over a ~155k vocab at
        # every position and it contributed exactly 0 to `total_loss` anyway. The cost of
        # skipping is the token-head drift diagnostic (ce_loss 31.6 -> 27.7 across an epoch,
        # which is how we know a CE-free arm destroys its own token head); set
        # KD_LOG_CE=1 to compute it for logging without putting it in the loss.
        want_ce = self.ce_weight > 0 or os.environ.get("KD_LOG_CE") == "1"
        if want_ce:
            losses = {"future_traj": self._compute_next_token_loss(outputs, labels, traj_mask)}
            ce_labels = labels.clone()
            ce_labels[traj_mask] = IGNORE_INDEX
            losses["others"] = self._compute_next_token_loss(
                outputs, ce_labels, ce_labels != IGNORE_INDEX
            )
            ce_loss = sum(losses.values())
        else:
            ce_loss = None
        # ⚠️ ce_weight=0 is a REAL arm, not a degenerate config: it asks whether the student
        # needs token supervision at all when the goal is a teacher-compatible cache read by
        # the teacher's expert. Safe for the stitched eval because `<|traj_future_start|>` is
        # part of the PROMPT under `components_prompt: [traj_future]`, not something the
        # student must learn to emit -- which is why all four stitched arms logged zero
        # "No <traj_future_start> token found" warnings. The student's own token head is of
        # course destroyed; that is the point of the arm, not a side effect.
        total_loss = (self.ce_weight * ce_loss) if ce_loss is not None \
            else torch.zeros((), device=input_ids.device, dtype=self.vlm.dtype)
        # Graph-carrying terms, keyed to match KaVaTrainer's `weights` dict so the
        # gradient probe can weight each one as it enters the total.
        attached: dict[str, torch.Tensor] = {} if ce_loss is None else {"ce": ce_loss}

        kd_loss = kv_loss = block_loss = fr_loss = field_loss = roll_loss = None
        block_loss_mse = block_loss_cosine = None
        region_losses: dict[str, torch.Tensor] = {}

        if need_teacher:
            self._place_teacher(input_ids.device, next(self.vlm.parameters()).dtype)
            with torch.no_grad(), _Phase.t("teacher_prefill"):
                t_out = self.teacher.vlm(
                    input_ids=input_ids,
                    output_hidden_states=want_hidden,
                    use_cache=False,
                    # Teacher logits feed logit-KD and nothing else.
                    **({} if self.kd_weight > 0 else {"logits_to_keep": 1}),
                    **tokenized_data,
                )

            # ---- lambda_kd: match the teacher's trajectory distribution -------
            if self.kd_weight > 0:
                kd_loss = logit_kd_loss(
                    outputs.logits,
                    t_out.logits,
                    traj_mask,
                    self.future_token_start_idx,
                    int(self.config.traj_vocab_size),
                    temperature=self.kd_temperature,
                )
                total_loss = total_loss + self.kd_weight * kd_loss
                attached["kd"] = kd_loss

            # ---- lambda_block: does the student cache drive the same block update? --
            if _need_block:
                # ⚠️ Both towers consumed the SAME input_ids and the same tokenized_data, so
                # positions, prompt and image tokens are identical by construction -- asserted
                # rather than assumed, because the whole objective is meaningless otherwise.
                h, d = self._kv_shape()
                student_rope_deltas = getattr(outputs, "rope_deltas", None)
                teacher_rope_deltas = getattr(t_out, "rope_deltas", None)
                if student_rope_deltas is None or teacher_rope_deltas is None:
                    raise RuntimeError(
                        "block-family losses need rope_deltas from both VLM prefills"
                    )
                if not torch.equal(
                    student_rope_deltas.to(teacher_rope_deltas.device),
                    teacher_rope_deltas,
                ):
                    raise RuntimeError(
                        "student and teacher rope_deltas differ for the same prompt; "
                        "their expert cache positions are not comparable"
                    )
                self._ensure_expert(
                    self._block_ckpt, input_ids.device, next(self.vlm.parameters()).dtype
                )
                with _Phase.t('recompute_kv'):
                    s_kv = recompute_kv(self._text_model(), outputs.hidden_states, h, d)
                with torch.no_grad():
                    t_kv_b = recompute_kv(self._teacher_text_model(), t_out.hidden_states, h, d)
                # ⚠️ Depth mismatch is EXPECTED with a shallow student: the expert's layer
                # count follows the VLM's text config, so a 28-layer 2B gives a 28-layer
                # expert while the teacher VLM still has 36 cache layers. The expert was
                # loaded with a remap (teacher expert layer pi(j) -> slot j, see
                # expert_holder._load), so the TEACHER's cache must be subset the same way:
                # expert slot j reads teacher cache pi(j), the layer that slot was trained on.
                # The student maps 1:1 -- its layer j is expert slot j.
                # Zipping the teacher's first 28 layers instead would pair every slot with a
                # SHALLOWER teacher layer than it expects; it runs clean and trains the wrong
                # target, so this raises rather than guessing.
                if len(s_kv) != len(t_kv_b):
                    pi = getattr(self.expert, "_pi", None)
                    if pi is None or len(pi) != len(s_kv):
                        raise RuntimeError(
                            f"student has {len(s_kv)} layers, teacher {len(t_kv_b)}, but the "
                            f"expert reports pi={'None' if pi is None else len(pi)}; set "
                            f"PRUNE_EXPERT_LAYERS so the remap leaves exactly {len(s_kv)}")
                    # BLOCK_PI_PROBE=1: score the alternatives ONCE, on real data, before
                    # committing 3 epochs to one of them. A wrong pairing does not crash --
                    # every shape matches -- so the only evidence that pi is the right map is
                    # that it beats the naive first-28 zip and a shuffled control.
                    if os.environ.get("BLOCK_PI_PROBE") == "1" and not self._pi_probed:
                        self._pi_probed = True
                        # ⚠️ recompute_kv returns a DICT keyed by layer index, so slicing it
                        # raises KeyError; every candidate is rebuilt as {slot: (K, V)}.
                        n = len(s_kv)
                        with torch.no_grad():
                            cands = {
                                "pi": {j: t_kv_b[i] for j, i in enumerate(pi)},
                                "first28": {j: t_kv_b[j] for j in range(n)},
                                "last28": {j: t_kv_b[j + len(t_kv_b) - n] for j in range(n)},
                                "reversed_pi": {j: t_kv_b[i]
                                                for j, i in enumerate(pi[::-1])},
                            }
                            for nm, tk in cands.items():
                                lo, *_ = self._block_loss(
                                    s_kv, tk, rope, traj_future_start_mask,
                                    tokenized_data.get("attention_mask"),
                                    student_rope_deltas, None,
                                )
                                print(f"[pi-probe] {nm:12s} block_loss {float(lo):.6f}",
                                      flush=True)
                    t_kv_b = {j: t_kv_b[i] for j, i in enumerate(pi)}
                _traj = None
                if self.block_timestep != "zero":
                    if ego_future_xyz is None:
                        raise RuntimeError(
                            "block_timestep != zero needs the GT trajectory; ego_future_xyz "
                            "is None. Sampling t without it would silently fall back to t=0."
                        )
                    _traj = {"ego_history_xyz": ego_history_xyz,
                             "ego_history_rot": ego_history_rot,
                             "ego_future_xyz": ego_future_xyz,
                             "ego_future_rot": ego_future_rot}
                with _Phase.t("block+field"):
                    (block_loss, block_loss_mse, block_loss_cosine,
                     fr_loss, field_loss, roll_loss) = self._block_loss(
                        s_kv, t_kv_b, rope, traj_future_start_mask,
                        tokenized_data.get("attention_mask"), student_rope_deltas, _traj)
                if self.block_weight > 0:
                    total_loss = total_loss + self.block_weight * block_loss
                attached["block"] = block_loss      # logged either way, as a free diagnostic
                if fr_loss is not None:
                    total_loss = total_loss + self.block_freerun_weight * fr_loss
                    attached["freerun"] = fr_loss
                if field_loss is not None:
                    total_loss = total_loss + self.field_weight * field_loss
                    attached["field"] = field_loss
                if roll_loss is not None:
                    total_loss = total_loss + self.roll_weight * roll_loss
                    attached["roll"] = roll_loss
                del s_kv, t_kv_b

            # ---- lambda_kv: match the LLM's K/V at every position -------------
            if self.kv_weight > 0:
                h, d = self._kv_shape()
                n_student = len(self._text_model().layers)
                student_kv = recompute_kv(self._text_model(), outputs.hidden_states, h, d)
                with torch.no_grad():
                    t_kv = recompute_kv(
                        self._teacher_text_model(), t_out.hidden_states, h, d
                    )
                    teacher_k, teacher_v = stack_teacher_kv(t_kv, len(t_kv))
                del t_kv

                attn = tokenized_data.get("attention_mask")
                regions = self._region_masks(input_ids, traj_mask, attn)
                full = regions["vision"] | regions["text"] | regions["traj"]
                kv_loss = self._kv_loss(student_kv, teacher_k, teacher_v, full)
                total_loss = total_loss + self.kv_weight * kv_loss
                attached["kv"] = kv_loss

                if self.log_kv_regions:
                    # Reporting only. Three extra masked passes rather than threading a
                    # region accumulator through kv_matching_loss, whose denom==0
                    # graph-attachment fix and [B,1,T,1] mask broadcast were both hard-won
                    # -- not worth destabilising for a logging feature.
                    with torch.no_grad():
                        for name, m in regions.items():
                            if bool(m.any()):
                                region_losses[name] = self._kv_loss(
                                    student_kv, teacher_k, teacher_v, m
                                ).detach()

        self.last_loss_terms = attached if self.keep_loss_terms else None

        _Phase.report()
        return KDVLAOutput(
            loss=total_loss,
            logits=outputs.logits,
            ce_loss=None if ce_loss is None else ce_loss.detach(),
            kd_loss=None if kd_loss is None else kd_loss.detach(),
            kv_loss=None if kv_loss is None else kv_loss.detach(),
            block_loss=None if block_loss is None else block_loss.detach(),
            block_loss_mse=None if block_loss_mse is None else block_loss_mse.detach(),
            block_loss_cosine=None if block_loss_cosine is None else block_loss_cosine.detach(),
            freerun_loss=None if fr_loss is None else fr_loss.detach(),
            field_loss=None if field_loss is None else field_loss.detach(),
            roll_loss=None if roll_loss is None else roll_loss.detach(),
            kv_ratio_k=(torch.tensor(self._kv_ratio_k)
                        if getattr(self, "_kv_ratio_k", None) is not None else None),
            kv_ratio_v=(torch.tensor(self._kv_ratio_v)
                        if getattr(self, "_kv_ratio_v", None) is not None else None),
            block_loss_tf=getattr(self, "_tf_term", None),
            block_loss_span=getattr(self, "_span_term", None),
            block_loss_early=getattr(self, "_last_bands", {}).get("early"),
            block_loss_mid=getattr(self, "_last_bands", {}).get("mid"),
            block_loss_deep=getattr(self, "_last_bands", {}).get("deep"),
            kv_loss_vision=region_losses.get("vision"),
            kv_loss_text=region_losses.get("text"),
            kv_loss_traj=region_losses.get("traj"),
        )
