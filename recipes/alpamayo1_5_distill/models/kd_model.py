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
from alpamayo1_5_distill.models.block_losses import block_output_loss, rotate_keys
from alpamayo1_5_distill.models.expert_holder import FrozenExpert
from alpamayo1_5_distill.models.kv_distill import (
    KVProjectorBank,
    build_layer_map,
    kv_matching_loss,
)


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

    def _block_loss(self, student_kv, teacher_kv, rope, traj_mask, attn_mask,
                    traj_data=None):
        """L_block = mean_l || B_l(h_l^T; K_s,V_s) - sg B_l(h_l^T; K_t,V_t) ||^2.

        ⚠️ The per-layer re-run reuses the EXACT kwargs the expert's own forward handed each
        layer -- captured by hook, never reconstructed. The enclosing
        ``Qwen3VLTextModel.forward`` converts the 2-D key mask into the causal 4-D form the
        attention implementation wants and computes ``position_embeddings`` via mrope; a
        layer called directly with a hand-built 2-D mask and self-computed cos/sin attends
        DIFFERENTLY. That bug made even the teacher's own cache fail to reproduce the
        teacher's own outputs (self-test identity 1.04e3 instead of ~0, and layer-shuffling
        the student *improved* the loss). Capturing removes the entire class of error.

        ⚠️ The action noise is drawn ONCE. Both sides consume the same ``h_l^T`` from that
        single draw, so noise and timestep are shared by construction. ``t = 0.0`` is the
        sampler's own first step (``flow_matching._euler``).
        """
        expert = self.expert
        cos, sin = rope["cos"], rope["sin"]
        n_layers = len(expert.expert.layers)

        # The expert reads the cache only up to <|traj_future_start|>, i.e. everything before
        # the first trajectory token -- cropping there reproduces the rollout's
        # `attention_mask[offset:-n_action] = False` without rebuilding that mask.
        first_traj = int(traj_mask[0].nonzero()[0].item())
        rot = lambda k: rotate_keys(
            k[:, :, :first_traj], cos[:, :first_traj], sin[:, :first_traj]
        )
        t_k = [rot(teacher_kv[i][0]) for i in range(n_layers)]
        t_v = [teacher_kv[i][1][:, :, :first_traj] for i in range(n_layers)]
        s_k = [rot(student_kv[i][0]) for i in range(n_layers)]
        s_v = [student_kv[i][1][:, :, :first_traj] for i in range(n_layers)]

        b = t_k[0].shape[0]
        device, dtype = t_k[0].device, t_k[0].dtype

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
            n_act = embeds.shape[1]
            pos = torch.arange(first_traj, first_traj + n_act, device=device)[None].expand(b, -1)
            e_mask = torch.ones((b, first_traj + n_act), dtype=torch.bool, device=device)

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
                    past_key_values=cache, use_cache=True,
                )
            finally:
                for h in handles:
                    h.remove()

        def _sweep(k_list, v_list):
            acc = None
            for l in range(n_layers):
                c = captured[l]
                term = block_output_loss(
                    expert.expert.layers[l], c["h_in"], c["y_out"],
                    k_list[l], v_list[l], dict(c["kwargs"]),
                )
                acc = term if acc is None else acc + term
            return acc / n_layers

        # ⚠️ SELF-TEST, opt-in via BLOCK_SELFTEST=1. A finite, falling loss is NOT evidence
        # that it measures cache fidelity -- that gap has produced retractions here. Feeding
        # the TEACHER's own cache must give ~0; a layer-shuffled cache must be clearly worse
        # than the real student's. If identity is not ~0 the objective is broken however
        # healthy the curve looks.
        if os.environ.get("BLOCK_SELFTEST") == "1":
            with torch.no_grad():
                ident = float(_sweep(t_k, t_v))
                shuf = list(range(n_layers))[::-1]
                shuffled = float(_sweep([s_k[i] for i in shuf], [s_v[i] for i in shuf]))
                real = float(_sweep(s_k, s_v))
            print(
                f"[block-selftest] identity(teacher cache)={ident:.6e}  student={real:.4f}  "
                f"layer-shuffled={shuffled:.4f}  | identity ~0 and student < shuffled",
                flush=True,
            )

        return _sweep(s_k, s_v)

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
        need_teacher = (
            self.kd_weight > 0 or self.kv_weight > 0 or self.block_weight > 0
        ) and self.teacher is not None
        want_hidden = (self.kv_weight > 0 or self.block_weight > 0) and self.teacher is not None

        tokenized_data = dict(tokenized_data)
        input_ids = tokenized_data.pop("input_ids")
        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)

        labels = input_ids.clone()
        if labels_mask is not None:
            labels = torch.where(labels_mask, labels, IGNORE_INDEX)

        # Hook the rotary embedding so L_block reuses the EXACT rotation the cached keys
        # carry, rather than reconstructing mrope positions. Registered only when needed.
        rope, rope_handle = self._capture_rope() if self.block_weight > 0 else ({}, None)
        try:
            outputs = self.vlm(
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=want_hidden,
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
        losses = {"future_traj": self._compute_next_token_loss(outputs, labels, traj_mask)}
        ce_labels = labels.clone()
        ce_labels[traj_mask] = IGNORE_INDEX
        losses["others"] = self._compute_next_token_loss(
            outputs, ce_labels, ce_labels != IGNORE_INDEX
        )
        ce_loss = sum(losses.values())
        # ⚠️ ce_weight=0 is a REAL arm, not a degenerate config: it asks whether the student
        # needs token supervision at all when the goal is a teacher-compatible cache read by
        # the teacher's expert. Safe for the stitched eval because `<|traj_future_start|>` is
        # part of the PROMPT under `components_prompt: [traj_future]`, not something the
        # student must learn to emit -- which is why all four stitched arms logged zero
        # "No <traj_future_start> token found" warnings. The student's own token head is of
        # course destroyed; that is the point of the arm, not a side effect.
        total_loss = self.ce_weight * ce_loss
        # Graph-carrying terms, keyed to match KaVaTrainer's `weights` dict so the
        # gradient probe can weight each one as it enters the total.
        attached: dict[str, torch.Tensor] = {"ce": ce_loss}

        kd_loss = kv_loss = block_loss = None
        region_losses: dict[str, torch.Tensor] = {}

        if need_teacher:
            self._place_teacher(input_ids.device, next(self.vlm.parameters()).dtype)
            with torch.no_grad():
                t_out = self.teacher.vlm(
                    input_ids=input_ids,
                    output_hidden_states=want_hidden,
                    use_cache=False,
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
            if self.block_weight > 0:
                # ⚠️ Both towers consumed the SAME input_ids and the same tokenized_data, so
                # positions, prompt and image tokens are identical by construction -- asserted
                # rather than assumed, because the whole objective is meaningless otherwise.
                h, d = self._kv_shape()
                self._ensure_expert(
                    self._block_ckpt, input_ids.device, next(self.vlm.parameters()).dtype
                )
                s_kv = recompute_kv(self._text_model(), outputs.hidden_states, h, d)
                with torch.no_grad():
                    t_kv_b = recompute_kv(self._teacher_text_model(), t_out.hidden_states, h, d)
                if len(s_kv) != len(t_kv_b):
                    raise RuntimeError(f"layer count differs: student {len(s_kv)} teacher {len(t_kv_b)}")
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
                block_loss = self._block_loss(s_kv, t_kv_b, rope, traj_mask,
                                              tokenized_data.get("attention_mask"), _traj)
                total_loss = total_loss + self.block_weight * block_loss
                attached["block"] = block_loss
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

        return KDVLAOutput(
            loss=total_loss,
            logits=outputs.logits,
            ce_loss=ce_loss.detach(),
            kd_loss=None if kd_loss is None else kd_loss.detach(),
            kv_loss=None if kv_loss is None else kv_loss.detach(),
            block_loss=None if block_loss is None else block_loss.detach(),
            kv_loss_vision=region_losses.get("vision"),
            kv_loss_text=region_losses.get("text"),
            kv_loss_traj=region_losses.get("traj"),
        )
