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

"""Latent-reasoning distillation model for Alpamayo-1.5 Stage-1.

``DistillReasoningVLA`` extends the Stage-1 student
(:class:`alpamayo1_5_sft.models.sft_base_model.TrainableReasoningVLA`) with a
single extra objective: match the student's hidden state at the
``<traj_future_start>`` token to a *cached* teacher hidden state (the teacher run
offline with chain-of-thought in context).

Why the ``<traj_future_start>`` position?  At inference the action expert
consumes the VLM KV cache cropped at exactly ``future_start_idx + 1`` (see
``alpamayo1_5_sft.models.sft_alpamayo_r1.TrainableAlpamayoR1.forward``), i.e.
everything **up to and including** that token.  The last-layer hidden state at
that position is therefore the single vector that best summarises the context
the expert is handed.  Distilling it compiles the teacher's reasoning into the
KV the expert reads, so the student never has to emit reasoning tokens.

The same class serves as the *teacher* at cache-generation time: call
:meth:`extract_tfs_hidden` (no projector, no loss) to read the raw teacher
hidden at ``<traj_future_start>``.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers.utils import ModelOutput

from alpamayo_r1.models.base_model import IGNORE_INDEX
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA


@dataclass
class DistillVLAOutput(ModelOutput):
    """Output of :class:`DistillReasoningVLA`.

    ``ce_loss`` / ``latent_loss`` are detached scalars exposed for logging; the
    total (back-proppable) objective is ``loss``.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    ce_loss: torch.FloatTensor | None = None
    latent_loss: torch.FloatTensor | None = None


class DistillReasoningVLA(TrainableReasoningVLA):
    """Stage-1 student with an added latent-reasoning distillation loss.

    Class-level defaults keep instances built through inherited factories that
    do not call :meth:`init_distillation` (e.g. the offline *teacher* loaded via
    ``from_alpamayo_checkpoint``) in a well-defined, projector-free state.

    Note ``latent_proj`` is deliberately NOT a class attribute: a submodule is
    stored in ``nn.Module._modules`` and reached via ``__getattr__``, which only
    fires when normal lookup fails — a class attribute would shadow it and always
    read ``None``. Access it with ``getattr(self, "latent_proj", None)``.
    """

    latent_loss_weight: float = 1.0
    latent_cosine_weight: float = 0.1
    teacher_hidden_dim: int | None = None

    # ------------------------------------------------------------------ setup
    def init_distillation(
        self,
        teacher_hidden_dim: int,
        latent_loss_weight: float = 1.0,
        latent_cosine_weight: float = 0.1,
    ) -> None:
        """Attach the student→teacher projector and store loss weights.

        Registered as a submodule so it is picked up by the optimizer and saved
        in checkpoints.  It maps the student hidden width to the (larger) teacher
        width and is unused / discardable at inference.
        """
        student_hidden = self._student_hidden_size()
        self.teacher_hidden_dim = int(teacher_hidden_dim)
        self.latent_loss_weight = float(latent_loss_weight)
        self.latent_cosine_weight = float(latent_cosine_weight)
        self.latent_proj = torch.nn.Linear(student_hidden, self.teacher_hidden_dim)
        # match the model's parameter dtype/device (VLM is bf16)
        ref = next(self.vlm.parameters())
        self.latent_proj.to(device=ref.device, dtype=ref.dtype)

    def _student_hidden_size(self) -> int:
        cfg = self.vlm.config
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and getattr(text_cfg, "hidden_size", None) is not None:
            return int(text_cfg.hidden_size)
        return int(cfg.hidden_size)

    @classmethod
    def from_pretrained_vlm(
        cls,
        vlm_name_or_path: str,
        alpamayo_config_path: str | None = None,
        checkpoint_path: str | None = None,
        teacher_hidden_dim: int | None = None,
        latent_loss_weight: float = 1.0,
        latent_cosine_weight: float = 0.1,
        **kwargs: Any,
    ) -> "DistillReasoningVLA":
        """Build the student VLM, then attach the distillation projector.

        Mirrors :meth:`TrainableReasoningVLA.from_pretrained_vlm` and adds the
        distillation kwargs.  ``teacher_hidden_dim`` must equal the width of the
        cached teacher hidden vectors (recorded in the cache metadata by
        ``scripts/generate_teacher_features.py``).
        """
        model = super().from_pretrained_vlm(
            vlm_name_or_path,
            alpamayo_config_path=alpamayo_config_path,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )
        if teacher_hidden_dim is not None:
            model.init_distillation(
                teacher_hidden_dim=teacher_hidden_dim,
                latent_loss_weight=latent_loss_weight,
                latent_cosine_weight=latent_cosine_weight,
            )
        return model

    # ------------------------------------------------------------- tfs helper
    def _traj_future_start_columns(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return, per row, the column of the *last* ``<traj_future_start>`` token.

        Left-padded batches put the token at different columns per row, so we
        locate it by value rather than a fixed index.  "Last occurrence" matches
        the Stage-2 handoff convention (``nonzero()[-1]`` in
        ``TrainableAlpamayoR1.forward``).
        """
        tfs_id = self.special_token_ids["traj_future_start"]
        mask = input_ids == tfs_id  # [B, L]
        cols = torch.arange(input_ids.shape[1], device=input_ids.device)
        masked = torch.where(mask, cols.unsqueeze(0), torch.full_like(mask, -1, dtype=torch.long))
        last_col = masked.max(dim=1).values  # [B]
        if (last_col < 0).any():
            missing = int((last_col < 0).sum())
            raise ValueError(
                f"{missing} sample(s) in the batch have no <traj_future_start> token; "
                "cannot locate the distillation position."
            )
        return last_col

    def _gather_tfs_hidden(
        self, input_ids: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Gather the last-layer hidden state at ``<traj_future_start>``. -> [B, H]."""
        last_col = self._traj_future_start_columns(input_ids)
        batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
        return hidden_states[batch_idx, last_col]

    @torch.no_grad()
    def extract_tfs_hidden(
        self,
        tokenized_data: dict[str, Any],
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        ego_future_xyz: torch.Tensor | None = None,
        ego_future_rot: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Teacher-side: return the raw hidden at ``<traj_future_start>``. -> [B, H].

        No projector, no CE loss — just one VLM forward with hidden states.  The
        tokens after ``<traj_future_start>`` cannot affect its causal hidden, so
        whether the future trajectory is fused in is immaterial; we fuse anyway
        to keep the sequence identical to training.
        """
        tokenized_data = dict(tokenized_data)
        input_ids = tokenized_data.pop("input_ids")
        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)
        outputs = self.vlm(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True,
            **tokenized_data,
        )
        return self._gather_tfs_hidden(input_ids, outputs.hidden_states[-1])

    # -------------------------------------------------------------- KD losses
    def _latent_loss(self, h_student: torch.Tensor, h_teacher: torch.Tensor) -> torch.Tensor:
        """Smooth-L1 + cosine on the projected student vs cached teacher hidden."""
        h_student = h_student.float()
        h_teacher = h_teacher.float()
        smooth_l1 = F.smooth_l1_loss(h_student, h_teacher)
        cosine = F.cosine_similarity(h_student, h_teacher, dim=-1).mean()
        return smooth_l1 + self.latent_cosine_weight * (1.0 - cosine)

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        tokenized_data: dict[str, Any],
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        ego_future_xyz: torch.Tensor | None = None,
        ego_future_rot: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        teacher_tfs_hidden: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> DistillVLAOutput:
        """Stage-1 forward with next-token CE + optional latent-reasoning KD.

        When ``teacher_tfs_hidden`` is absent (or no projector attached) this is
        exactly the base Stage-1 objective, so eval / generation are unaffected.
        """
        # 1. tokenize trajectory and fuse into input_ids (mirrors the parent)
        input_ids = tokenized_data.pop("input_ids")
        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)

        # 2. labels
        labels = input_ids.clone()
        if labels_mask is not None:
            labels = torch.where(labels_mask, labels, IGNORE_INDEX)

        # 3. VLM forward (with hidden states for the latent loss)
        outputs = self.vlm(
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
            **tokenized_data,
        )

        # 4. next-token CE (identical split to TrainableReasoningVLA.forward)
        losses: dict[str, torch.Tensor] = {}
        traj_mask = (
            (
                (labels >= self.future_token_start_idx)
                & (labels < self.future_token_start_idx + self.config.traj_vocab_size)
            )
            | (labels == self.special_token_ids["traj_future_start"])
            | (labels == self.special_token_ids["traj_future_end"])
        )
        losses["future_traj"] = self._compute_next_token_loss(outputs, labels, traj_mask)
        labels[traj_mask] = IGNORE_INDEX
        losses["others"] = self._compute_next_token_loss(outputs, labels, labels != IGNORE_INDEX)
        ce_loss = sum(losses.values())

        # 5. latent-reasoning distillation
        total_loss = ce_loss
        latent_loss = None
        latent_proj = getattr(self, "latent_proj", None)
        if teacher_tfs_hidden is not None and latent_proj is not None:
            h_student = self._gather_tfs_hidden(input_ids, outputs.hidden_states[-1])
            h_student = latent_proj(h_student)
            latent_loss = self._latent_loss(h_student, teacher_tfs_hidden.to(h_student.device))
            total_loss = total_loss + self.latent_loss_weight * latent_loss

        return DistillVLAOutput(
            loss=total_loss,
            logits=outputs.logits,
            ce_loss=ce_loss.detach(),
            latent_loss=None if latent_loss is None else latent_loss.detach(),
        )
