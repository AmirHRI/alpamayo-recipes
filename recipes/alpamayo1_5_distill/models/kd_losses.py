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

"""Ordinary logit distillation over the trajectory-token slice.

The companion to :mod:`kv_distill`, and much simpler: match the teacher's *output
distribution* over trajectory tokens rather than its internal K/V.

**Why the slice and not the full vocabulary.**  The extended vocabulary is 155,697
entries, of which the 4000 trajectory tokens ``<i0>..<i3999>`` occupy a contiguous run
starting at ``traj_token_start_idx``.  Everything outside that run is text the model is
not being asked to produce at these positions -- the released rollout even masks it out
explicitly (``ExpertLogitsProcessor``).  Restricting the KL to the slice therefore loses
nothing and costs ~39x less memory: a dense fp32 KL over 155,697 entries at 128 positions
is ~80 MB per sample per side, against ~2 MB for the slice.

Verified on the real checkpoints before this was written: student and teacher agree on
``vocab_size`` (155,697) and ``traj_token_start_idx`` (151,669), so the slice denotes the
*same* tokens in the *same* order on both sides.  :func:`assert_kd_compatible` pins that
-- nothing else in the stack enforces it, and a backbone with a different base tokenizer
would silently distil against permuted classes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def assert_kd_compatible(student: object, teacher: object) -> None:
    """Refuse to distil across mismatched vocabularies.

    ``traj_token_start_idx`` is derived per model as ``convert_tokens_to_ids("<i0>")``,
    i.e. it is wherever the *base* tokenizer happened to end.  It is 151,669 for both
    Cosmos-Reason2-8B and Qwen3-VL-4B because both have 26 added tokens ending at 151,668
    -- a coincidence of those two checkpoints, not a guarantee.  Swap in a backbone with a
    different base tokenizer and every KD target silently refers to a different token.
    """
    s_cfg, t_cfg = student.config, teacher.config
    pairs = [
        ("vocab_size", int(s_cfg.vocab_size), int(t_cfg.vocab_size)),
        ("traj_token_start_idx", int(s_cfg.traj_token_start_idx), int(t_cfg.traj_token_start_idx)),
        ("traj_vocab_size", int(s_cfg.traj_vocab_size), int(t_cfg.traj_vocab_size)),
    ]
    bad = [(n, a, b) for n, a, b in pairs if a != b]
    if bad:
        raise ValueError(
            "student and teacher disagree on the trajectory vocabulary, so logit-KD would "
            "match different tokens: "
            + ", ".join(f"{n} student={a} teacher={b}" for n, a, b in bad)
        )


def logit_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    position_mask: torch.Tensor,
    traj_start: int,
    traj_vocab_size: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """``T^2 * KL(teacher || student)`` over the trajectory slice, at supervised positions.

    Args:
        student_logits: ``[B, T, V]``, grad-carrying.
        teacher_logits: ``[B, T, V]``.  Detached internally, so the caller need not.
        position_mask: ``[B, T]`` bool over the *label* positions to supervise -- pass the
            same trajectory mask the CE term uses, so the two objectives cover exactly the
            same tokens.
        traj_start: ``future_token_start_idx``; the slice is
            ``[traj_start, traj_start + traj_vocab_size)``.
        temperature: softmax temperature.  The ``T^2`` factor keeps the gradient magnitude
            comparable to CE as ``T`` varies (Hinton et al.), so ``lambda_kd`` does not
            have to be retuned for every temperature.

    Returns:
        Scalar.  When no position is supervised the value is 0.0 but the tensor stays
        **attached to the graph** -- see the warning below.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    # Same next-token shift as `_compute_next_token_loss`: logits at position p predict
    # the token at p+1, so a mask over label positions selects logits at p-1.
    s = student_logits[:, :-1, traj_start : traj_start + traj_vocab_size]
    t = teacher_logits[:, :-1, traj_start : traj_start + traj_vocab_size].detach()
    sel = position_mask[:, 1:]

    s = s[sel].float()
    t = t[sel].float()

    if s.numel() == 0:
        # ⚠️ MUST stay graph-connected. A detached `torch.zeros(())` here gives the LM head
        # no gradient on THIS RANK ONLY; under ZeRO-2 the ranks then reduce different
        # parameter sets, their NCCL sequences shift by one, and the next size-mismatched
        # collective deadlocks. That exact bug cost two runs in the KAVA work -- see the
        # `denom == 0` branch of kv_distill.kv_matching_loss. `s.sum() * 0.0` is exactly
        # zero and still carries the graph.
        return student_logits.sum() * 0.0

    loss = F.kl_div(
        F.log_softmax(s / temperature, dim=-1),
        F.log_softmax(t / temperature, dim=-1),
        reduction="batchmean",
        log_target=True,
    )
    return loss * (temperature**2)
