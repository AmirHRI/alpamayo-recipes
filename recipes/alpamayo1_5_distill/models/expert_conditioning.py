# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared prompt-cache mask and position construction for the action expert."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExpertConditioning:
    """Per-row conditioning needed to append action queries to a prompt cache."""

    offsets: torch.Tensor
    prefix_len: int
    key_padding_mask: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor


def build_expert_conditioning(
    *,
    traj_future_start_mask: torch.Tensor,
    tokenizer_attention_mask: torch.Tensor | None,
    rope_deltas: torch.Tensor | None,
    n_action_tokens: int,
    dtype: torch.dtype,
    attention_implementation: str | None,
    cache_len: int | None = None,
) -> ExpertConditioning:
    """Build batch-safe expert masks and positions.

    ``traj_future_start_mask`` identifies the handoff token in each collated row.  The
    physical cache has one shared length, but validity is per row: left padding and tokens
    after that row's handoff are masked.  The handoff token itself is included.

    FlashAttention consumes a 2-D key-padding mask.  SDPA/eager need an explicit 4-D
    non-causal additive mask so all action queries can attend all action keys.
    """
    if traj_future_start_mask.ndim != 2:
        raise ValueError(
            "traj_future_start_mask must have shape [B, L], got "
            f"{tuple(traj_future_start_mask.shape)}"
        )
    start_mask = traj_future_start_mask.bool()
    present = start_mask.any(dim=1)
    if not bool(present.all()):
        missing = (~present).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"rows {missing} contain no <traj_future_start> token")

    batch, sequence_len = start_mask.shape
    # The prompt cache includes <traj_future_start>, hence +1.
    offsets = start_mask.int().argmax(dim=1) + 1
    required_len = int(offsets.max().item())
    prefix_len = required_len if cache_len is None else int(cache_len)
    if prefix_len < required_len or prefix_len > sequence_len:
        raise ValueError(
            f"cache_len={prefix_len} must be in [{required_len}, {sequence_len}]"
        )

    if tokenizer_attention_mask is None:
        valid_tokens = torch.ones_like(start_mask)
    else:
        if tokenizer_attention_mask.shape != start_mask.shape:
            raise ValueError(
                "tokenizer_attention_mask must match traj_future_start_mask, got "
                f"{tuple(tokenizer_attention_mask.shape)} vs {tuple(start_mask.shape)}"
            )
        valid_tokens = tokenizer_attention_mask.bool()

    columns = torch.arange(prefix_len, device=start_mask.device).unsqueeze(0)
    prefix_valid = valid_tokens[:, :prefix_len] & (columns < offsets[:, None])
    action_valid = torch.ones(
        (batch, int(n_action_tokens)), dtype=torch.bool, device=start_mask.device
    )
    key_padding_mask = torch.cat((prefix_valid, action_valid), dim=1)

    if rope_deltas is None:
        raise ValueError("rope_deltas are required for multimodal expert action positions")
    deltas = rope_deltas.to(device=start_mask.device).reshape(batch, -1)
    if deltas.shape[1] != 1:
        raise ValueError(f"expected one rope delta per row, got {tuple(rope_deltas.shape)}")
    action_start = offsets + deltas[:, 0].to(offsets.dtype)
    action_positions = torch.arange(
        int(n_action_tokens), device=start_mask.device, dtype=offsets.dtype
    )
    position_ids = action_positions.view(1, 1, -1) + action_start.view(1, batch, 1)
    position_ids = position_ids.expand(3, -1, -1)

    implementation = attention_implementation or "eager"
    if implementation in {"flash_attention_2", "flash_attention_3"}:
        attention_mask = key_padding_mask
    else:
        if not dtype.is_floating_point:
            raise ValueError(f"additive expert attention mask needs floating dtype, got {dtype}")
        attention_mask = torch.zeros(
            (batch, 1, int(n_action_tokens), prefix_len + int(n_action_tokens)),
            dtype=dtype,
            device=start_mask.device,
        )
        invalid = ~key_padding_mask[:, None, None, :]
        attention_mask.masked_fill_(invalid, torch.finfo(dtype).min)

    return ExpertConditioning(
        offsets=offsets,
        prefix_len=prefix_len,
        key_padding_mask=key_padding_mask,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
