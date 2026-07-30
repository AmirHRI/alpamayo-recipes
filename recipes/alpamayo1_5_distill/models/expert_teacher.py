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

"""Score the teacher's CoT by the **action expert's** cross-attention.

R-KV's importance term asks which cached key/value pairs the reader actually uses.
In KAVA the reader is the answer *text*; here it is the flow-matching action expert,
which is handed the VLM KV cache cropped at ``future_start_idx + 1`` and cross-attends
it layer for layer (expert layer *i* reads VLM cache layer *i*).  So this is the most
faithful importance signal available for a driving VLA — these are literally the
queries that read the cache at inference — and it needs no text-answer proxy.

Two things make this non-trivial, and both are handled here rather than papered over:

**The expert runs non-causal.** ``expert_non_causal_attention: true`` reaches the
attention as an ``is_causal=False`` forward kwarg, which only the *sdpa* path honours
— ``eager_attention_forward`` ignores ``is_causal`` entirely and applies whatever mask
``create_causal_mask`` built.  Capturing weights via ``output_attentions=True`` (which
requires eager) would therefore hand back **causal** attention that the expert never
computes.  So the attention is reconstructed directly instead: post-RoPE queries from
a ``q_norm`` hook against the post-RoPE keys the forward left in the cache, softmaxed
over the full key axis with no mask. That is exactly what sdpa computes for
``is_causal=False, attention_mask=None``, and it is also cheaper — one layer's logits
at a time instead of 36 layers of attention matrices held at once.

**The expert's input is noisy.** ``noisy_x`` and ``timesteps`` are sampled randomly
during training, which an offline cache cannot tolerate. Here the timesteps are a
fixed grid and the noise comes from a seeded CPU generator, so the same clip always
produces the same score. The grid spans the denoising trajectory: ``t = 1`` is the
clean action, the KAVA-faithful "answer" end, while ``t = 0`` is pure noise, the state
the expert actually starts from at inference. Averaging over the grid scores a cache
entry by how much the expert reads it *across* denoising, not at one arbitrary point.
"""

from contextlib import contextmanager
from typing import Any, Iterator

import einops
import torch

from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1
from alpamayo1_5_distill.models.distill_base_model import DistillReasoningVLA

#: Default timestep grid: the two endpoints of the flow plus the midpoint. Each entry
#: costs one expert forward, so this is deliberately short; widen it with
#: ``expert_timesteps=`` if the score looks unstable across t.
DEFAULT_EXPERT_TIMESTEPS = (0.0, 0.5, 1.0)


class KaVaExpertTeacher(TrainableAlpamayoR1):
    """A teacher that carries the action expert *and* can generate its own CoT.

    ``TrainableAlpamayoR1`` (expert, action space, diffusion) and
    ``DistillReasoningVLA`` (CoT generation) are siblings under ``ReasoningVLA``, not
    a chain, so neither alone can build this cache. ``generate_cot_prefix`` only
    touches the shared ``ReasoningVLA`` surface — ``vlm``, ``tokenizer``, ``config``,
    ``future_token_start_idx``, ``fuse_traj_tokens`` — so it is borrowed directly
    rather than duplicated. If that method ever starts using
    ``TrainableReasoningVLA``-only state, this line is what breaks, loudly.
    """

    generate_cot_prefix = DistillReasoningVLA.generate_cot_prefix


@contextmanager
def _capture_expert_queries(
    expert: Any, n_q_heads: int, head_dim: int
) -> Iterator[dict[int, torch.Tensor]]:
    """Collect each expert layer's pre-RoPE queries via a ``q_norm`` hook.

    ``q_norm``'s output is ``[B, T, n_q_heads, head_dim]`` — the transpose to
    ``[B, H, T, D]`` and the rotation both happen after it in the attention module.
    """
    store: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx: int):
        def hook(_m: Any, _a: Any, out: torch.Tensor) -> None:
            store[idx] = out.detach().view(out.shape[0], out.shape[1], n_q_heads, head_dim)

        return hook

    for idx, layer in enumerate(expert.layers):
        handles.append(layer.self_attn.q_norm.register_forward_hook(make_hook(idx)))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()


def build_noisy_action(
    model: Any,
    traj_data: dict[str, torch.Tensor],
    timestep: float,
    noise: torch.Tensor | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic ``(noisy_x, timesteps, noise)`` at one point on the flow.

    Reimplements ``FlowMatching.construct_training_data``'s interpolation
    (``noisy_x = t * x + (1 - t) * noise``) with ``t`` pinned to a grid point instead
    of Beta-sampled and the noise drawn from a seeded **CPU** generator, so the result
    is identical across devices and runs. The diffusion type is asserted rather than
    duck-typed: a different scheme would have a different interpolation and this would
    silently score the wrong input.
    """
    from alpamayo_r1.diffusion.flow_matching import FlowMatching

    if not isinstance(model.diffusion, FlowMatching):
        raise NotImplementedError(
            f"expert importance assumes FlowMatching's noisy_x = t*x + (1-t)*noise; "
            f"got {type(model.diffusion).__name__}. Add the right interpolation before "
            "using importance_source='expert' with this diffusion."
        )

    action = model.action_space.traj_to_action(
        traj_history_xyz=traj_data["ego_history_xyz"],
        traj_history_rot=traj_data["ego_history_rot"],
        traj_future_xyz=traj_data["ego_future_xyz"],
        traj_future_rot=traj_data["ego_future_rot"],
    )
    action = action.reshape(-1, *model.action_space.get_action_space_dims())

    if noise is None:
        generator = torch.Generator().manual_seed(int(seed))
        noise = torch.randn(action.shape, generator=generator, dtype=torch.float32)
        noise = noise.to(device=action.device, dtype=action.dtype)

    t = torch.full((action.shape[0],), float(timestep), device=action.device, dtype=action.dtype)
    while t.dim() < action.dim():
        t = t.unsqueeze(-1)
    return t * action + (1.0 - t) * noise, t, noise


def score_from_qk(
    q: torch.Tensor,
    keys: torch.Tensor,
    n_rep: int,
    scaling: float,
    lo: int,
    hi: int,
) -> torch.Tensor:
    """Reconstruct one layer's attention onto the CoT span. -> ``[B, n_kv, N_C]``.

    Equivalent to ``eager_attention_forward`` with ``attention_mask=None`` — which is
    what sdpa computes for the expert's ``is_causal=False`` — followed by R-KV's GQA
    MaxPool and an average over the query positions.

    Queries are *grouped* rather than keys *repeated*: ``repeat_kv`` lays query head
    ``i`` against kv head ``i // n_rep``, so reshaping to ``[B, n_kv, n_rep, T_q, D]``
    gives the same pairing at a fraction of the memory.

    Args:
        q: ``[B, n_q_heads, T_q, D]`` post-RoPE queries.
        keys: ``[B, n_kv_heads, T_k, D]`` post-RoPE keys (the whole key axis, so the
            softmax denominator matches the real forward).
        n_rep: ``n_q_heads // n_kv_heads``.
        scaling: ``head_dim ** -0.5``.
        lo, hi: the CoT column range within ``T_k``.
    """
    batch, n_q_heads, t_q, head_dim = q.shape
    n_kv_heads = n_q_heads // n_rep
    grouped = q.reshape(batch, n_kv_heads, n_rep, t_q, head_dim).float()
    logits = torch.einsum("bhrqd,bhkd->bhrqk", grouped, keys.float()) * scaling
    probs = logits.softmax(dim=-1)  # over the FULL key axis, no mask
    pooled = probs.max(dim=2).values  # MaxPool the GQA group -> [B, n_kv, T_q, T_k]
    return pooled[..., lo:hi].mean(dim=-2)  # -> [B, n_kv, N_C]


@torch.no_grad()
def expert_cot_importance(
    model: Any,
    cache: Any,
    lo: int,
    hi: int,
    crop_len: int,
    traj_data: dict[str, torch.Tensor],
    rope_deltas: torch.Tensor,
    timesteps: tuple[float, ...] = DEFAULT_EXPERT_TIMESTEPS,
    seed: int = 0,
) -> torch.Tensor:
    """Attention mass the action expert puts on each CoT token. -> ``[L, H, N_C]``.

    Args:
        model: a :class:`KaVaExpertTeacher` (needs ``expert``, ``action_in_proj``,
            ``action_space``, ``diffusion``).
        cache: the VLM cache, already cropped to the handoff point — i.e. exactly what
            ``TrainableAlpamayoR1.forward`` hands the expert.
        lo, hi: the CoT span's column range within that cache.
        crop_len: the cache length at the handoff; the cache is rolled back to it after
            every timestep, mirroring the inference loop's
            ``crop_cache(prompt_cache, prefill_seq_len)`` between Euler steps.
        traj_data: ``ego_{history,future}_{xyz,rot}`` for the clip.
        rope_deltas: from the VLM forward, for the expert's M-RoPE offset.
        timesteps: the grid to average over.
        seed: noise seed; one draw is shared across the grid so the only thing varying
            along it is ``t``.

    Returns:
        ``[L_expert, n_kv_heads, N_C]`` non-negative attention mass, already MaxPooled
        over each GQA query group (several queries share one cached pair, and a pair
        matters if *any* of them needs it) and averaged over the action queries.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    expert = model.expert
    n_vlm_layers = len(model.vlm.model.language_model.layers)
    if len(expert.layers) != n_vlm_layers:
        raise ValueError(
            f"expert depth {len(expert.layers)} != VLM depth {n_vlm_layers}; the "
            "expert-layer-i-reads-cache-layer-i correspondence this score relies on "
            "does not hold."
        )

    expert_cfg = expert.config
    n_q_heads = int(expert_cfg.num_attention_heads)
    n_kv_heads = int(expert_cfg.num_key_value_heads)
    head_dim = int(expert_cfg.head_dim)
    n_rep = n_q_heads // n_kv_heads
    scaling = head_dim**-0.5

    forward_kwargs: dict[str, Any] = {}
    if getattr(model.config, "expert_non_causal_attention", False):
        forward_kwargs["is_causal"] = False

    _, _, noise = build_noisy_action(model, traj_data, timesteps[0], seed=seed)
    total: torch.Tensor | None = None

    for timestep in timesteps:
        noisy_x, t_tensor, _ = build_noisy_action(
            model, traj_data, timestep, noise=noise, seed=seed
        )
        action_embeds = model.action_in_proj(noisy_x, t_tensor)
        batch, n_action = action_embeds.shape[0], action_embeds.shape[1]

        position_ids = torch.arange(n_action, device=action_embeds.device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=batch).clone()
        position_ids += (rope_deltas.to(position_ids.device) + crop_len)

        with _capture_expert_queries(expert, n_q_heads, head_dim) as queries:
            expert(
                inputs_embeds=action_embeds,
                position_ids=position_ids,
                past_key_values=cache,
                attention_mask=None,
                use_cache=True,
                **forward_kwargs,
            )

        cos, sin = expert.rotary_emb(action_embeds, position_ids)
        per_layer = []
        # Explicitly outside autocast: einsum is on autocast's cast-to-bf16 list, so a
        # bare `.float()` on the operands would be silently undone and the softmax
        # would see bf16 logits. Reconstructing attention is the one place in this
        # cache job where that precision actually matters.
        with torch.autocast("cuda", enabled=False):
            for layer_idx in range(n_vlm_layers):
                q = queries[layer_idx].permute(0, 2, 1, 3)  # [B, n_q, T_a, D]
                q, _ = apply_rotary_pos_emb(q, q, cos, sin)
                keys = cache.layers[layer_idx].keys  # [B, n_kv, crop_len+T_a, D] post-RoPE
                per_layer.append(score_from_qk(q, keys, n_rep, scaling, lo, hi))
                del q

        step_importance = torch.cat(per_layer, dim=0)  # [L, H, N_C] at batch size 1
        total = step_importance if total is None else total + step_importance
        cache.crop(crop_len)  # roll back the expert's own K/V before the next timestep

    return total / float(len(timesteps))
