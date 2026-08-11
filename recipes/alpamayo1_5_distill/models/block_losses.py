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

r"""Teacher-forced block-output matching: does the student cache drive the same update?

`L_KV` (L1 on K and V) is indifferent to direction -- an error along an axis no expert query
reads costs as much as one that dominates the softmax.  Matching attention MAPS fixes the
direction problem but supervises K only, because ``A = softmax(QK^T)`` contains no V.  This
objective fixes both: run each frozen expert block twice on the SAME teacher action state,
once with the teacher's VLM cache and once with the student's, and match the outputs.

.. math::
    L_{block} = \frac{1}{L}\sum_l \big\| B_l(h^T_l;\,K^S_l, V^S_l)
                                    - \mathrm{sg}\,B_l(h^T_l;\,K^T_l, V^T_l) \big\|_2^2

**Why teacher-forcing ``h^T_l`` is the whole trick.**  The block weights are identical, the
action input ``h^T_l`` is identical, and the timestep conditioning is identical -- therefore
the action-side tensors are identical *by construction*:

    Q^{S|T}_l = Q^T_l,   K^{S|T}_{a,l} = K^T_{a,l},   V^{S|T}_{a,l} = V^T_{a,l}

so the ONLY difference between the two evaluations is the VLM cache.  Each layer's term is
independently well-posed, with no drift compounding across depth, and the loss answers the
question that actually matters: *does the student's cache produce the same update when
consumed by the real frozen action block?*

**The teacher side is free.**  In a standard decoder ``h_{l+1} = B_l(h_l)``, so
``y^T_l = h^T_{l+1}`` -- one expert forward with ``output_hidden_states=True`` on the teacher
cache yields every ``h^T_l`` AND every ``y^T_l``.  Only the student side needs re-running,
and that is 36 single-block forwards over ~128 action tokens.

⚠️ **The cache is POST-RoPE.**  ``modeling_qwen3_vl.py:433`` rotates K before
``past_key_values.update``.  ``L_KV`` matches PRE-RoPE K, which is harmless there (RoPE is
the same per-position orthogonal rotation on both models, and L2 is rotation-invariant), but
here the student's K enters a real attention computation against teacher-derived queries, so
it must be rotated with the SAME cos/sin first.  :func:`rotate_keys` does that.  Passing
unrotated K yields a perfectly well-behaved and entirely wrong objective.
"""

from __future__ import annotations

import contextlib

import torch
from transformers.cache_utils import DynamicCache


def rotate_keys(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply the VLM's RoPE to pre-RoPE keys ``[B, H_kv, T, D]``.

    Mirrors ``apply_rotary_pos_emb`` for the key side.  ``cos``/``sin`` must be the ones the
    VLM used for these positions; teacher and student share them exactly (identical prompts,
    identical ``rope_theta`` and ``mrope_section``), which is what makes the comparison valid.
    """
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    half = k.shape[-1] // 2
    return k * cos + torch.cat((-k[..., half:], k[..., :half]), dim=-1) * sin


@contextlib.contextmanager
def _as_layer0(block):
    """Run ``block`` against a single-layer cache.

    ``self_attn.layer_idx`` is what the block uses to index ``past_key_values``.  Building a
    36-layer cache just to populate one slot would allocate 36x the memory for no reason, so
    the index is borrowed for the duration of one call and restored.
    """
    attn = block.self_attn
    orig = attn.layer_idx
    attn.layer_idx = 0
    try:
        yield
    finally:
        attn.layer_idx = orig


def block_output_loss(
    block: torch.nn.Module,
    h_in: torch.Tensor,
    y_teacher: torch.Tensor,
    student_k: torch.Tensor,
    student_v: torch.Tensor,
    layer_kwargs: dict,
    normalize: bool = True,
) -> torch.Tensor:
    """One layer's ``|| B(h; K_s,V_s) - sg B(h; K_t,V_t) ||^2``.

    Args:
        block: the frozen expert decoder layer ``B_l``.
        h_in: ``h^T_l`` -- the teacher's action state entering this block, detached.
            Teacher-forcing it is what holds Q, K_a and V_a fixed, so the cache is the only
            difference between the two evaluations.
        y_teacher: ``y^T_l`` as the real forward produced it, detached.
        student_k / student_v: student PREFIX cache, K already rotated. These carry the grad.
        layer_kwargs: ⚠️ the EXACT kwargs the enclosing model handed this layer, captured by
            hook -- causal mask, position_embeddings, position_ids, cache_position. Do NOT
            rebuild these: the model's forward converts the 2-D key mask to the causal 4-D
            form and computes mrope embeddings, and a hand-built substitute attends
            differently. ``past_key_values`` is supplied here and must be absent from this
            dict.
        normalize: divide by the teacher output's mean square so per-layer terms are
            commensurable. Expert activations span orders of magnitude across depth, so
            without this the largest layers own the gradient.

    Returns:
        Scalar.
    """
    cache = DynamicCache()
    # Seed the prefix as the rollout's prompt_cache does; the block appends its own
    # action-token K/V on top, identically in both evaluations.
    cache.update(student_k, student_v, 0, {})
    with _as_layer0(block):
        out = block(h_in, past_key_values=cache, use_cache=True, **layer_kwargs)
    y_student = out[0] if isinstance(out, tuple) else out

    diff = (y_student.float() - y_teacher.float().detach()).pow(2).mean()
    if normalize:
        # Guarded: a layer whose teacher output is ~0 would divide by ~0 and produce an inf
        # that poisons the whole sum.
        scale = y_teacher.float().detach().pow(2).mean().clamp_min(1e-6)
        diff = diff / scale
    return diff
