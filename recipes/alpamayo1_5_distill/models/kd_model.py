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

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from alpamayo_r1.models.base_model import IGNORE_INDEX
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA
from alpamayo1_5_distill.models.kd_losses import assert_kd_compatible, logit_kd_loss
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
    kd_weight: float = 0.0
    kd_temperature: float = 1.0
    kv_weight: float = 0.0
    kv_loss_type: str = "l1"
    kv_align: str = "direct"
    kv_layerwise_std: bool = True
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
        kd_weight: float = 0.0,
        kd_temperature: float = 1.0,
        kv_weight: float = 0.0,
        kv_loss_type: str = "l1",
        kv_align: str = "direct",
        kv_layerwise_std: bool = True,
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
        self.kd_weight = float(kd_weight)
        self.kd_temperature = float(kd_temperature)
        self.kv_weight = float(kv_weight)
        self.kv_loss_type = str(kv_loss_type)
        self.kv_align = str(kv_align)
        self.kv_layerwise_std = bool(kv_layerwise_std)
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
        return kv_matching_loss(
            student_kv,
            teacher_k,
            teacher_v,
            self.kv_layer_map,
            valid_mask=mask,
            projector=getattr(self, "kv_projector", None),
            kind=self.kv_loss_type,
            layerwise_std=self.kv_layerwise_std,
        )

    # ---------------------------------------------------------------- forward
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
        need_teacher = (self.kd_weight > 0 or self.kv_weight > 0) and self.teacher is not None
        want_hidden = self.kv_weight > 0 and self.teacher is not None

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

        outputs = self.vlm(
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=want_hidden,
            **tokenized_data,
        )

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

        kd_loss = kv_loss = None
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
            kv_loss_vision=region_losses.get("vision"),
            kv_loss_text=region_losses.get("text"),
            kv_loss_traj=region_losses.get("traj"),
        )
