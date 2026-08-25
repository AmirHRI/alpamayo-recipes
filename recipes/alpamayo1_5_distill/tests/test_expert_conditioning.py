# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for batch-safe action-expert cache conditioning."""

import pytest
import torch

from alpamayo1_5_distill.models.expert_conditioning import build_expert_conditioning


def _mixed_masks():
    start = torch.zeros(2, 8, dtype=torch.bool)
    start[0, 4] = True
    start[1, 5] = True
    attention = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    return start, attention


def test_expert_conditioning_masks_left_padding_and_row_suffixes_for_flash():
    start, attention = _mixed_masks()
    out = build_expert_conditioning(
        traj_future_start_mask=start,
        tokenizer_attention_mask=attention,
        rope_deltas=torch.tensor([[-3], [-1]]),
        n_action_tokens=2,
        dtype=torch.bfloat16,
        attention_implementation="flash_attention_2",
    )

    assert out.offsets.tolist() == [5, 6]
    assert out.prefix_len == 6
    # Row 0: left padding is invalid, tfs at column 4 is included, and column 5
    # (after this row's tfs) is invalid. Both appended action keys are valid.
    assert out.key_padding_mask.tolist() == [
        [False, False, True, True, True, False, True, True],
        [True, True, True, True, True, True, True, True],
    ]
    assert torch.equal(out.attention_mask, out.key_padding_mask)
    assert out.position_ids.shape == (3, 2, 2)
    assert out.position_ids[0].tolist() == [[2, 3], [5, 6]]
    assert torch.equal(out.position_ids[0], out.position_ids[1])
    assert torch.equal(out.position_ids[1], out.position_ids[2])


def test_expert_conditioning_builds_noncausal_additive_mask_for_sdpa():
    start, attention = _mixed_masks()
    out = build_expert_conditioning(
        traj_future_start_mask=start,
        tokenizer_attention_mask=attention,
        rope_deltas=torch.tensor([[-3], [-1]]),
        n_action_tokens=2,
        dtype=torch.float32,
        attention_implementation="sdpa",
    )

    assert out.attention_mask.shape == (2, 1, 2, 8)
    minimum = torch.finfo(torch.float32).min
    # Every action query sees the same valid prompt keys and both action keys.
    assert out.attention_mask[0, 0, 0].tolist() == [
        minimum, minimum, 0.0, 0.0, 0.0, minimum, 0.0, 0.0
    ]
    assert torch.equal(out.attention_mask[0, 0, 0], out.attention_mask[0, 0, 1])


def _run_batch_invariance(device, attention_implementation, dtype):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    torch.manual_seed(7)
    config = Qwen3VLTextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
    )
    config._attn_implementation = attention_implementation
    model = Qwen3VLTextModel(config).to(device=device, dtype=dtype).eval()
    width = config.hidden_size

    short = torch.randn(1, 2, width, device=device, dtype=dtype)
    long = torch.randn(1, 5, width, device=device, dtype=dtype)
    pads = torch.randn(1, 3, width, device=device, dtype=dtype)
    mixed = torch.cat((torch.cat((pads, short), dim=1), long), dim=0)
    mixed_prompt_mask = torch.tensor(
        [[0, 0, 0, 1, 1], [1, 1, 1, 1, 1]], device=device
    )
    mixed_prompt_pos = (mixed_prompt_mask.cumsum(-1) - 1).clamp_min(0)
    mixed_prompt_pos = mixed_prompt_pos.unsqueeze(0).expand(3, -1, -1)
    single_prompt_mask = torch.ones(1, 2, dtype=torch.long, device=device)
    single_prompt_pos = (
        torch.tensor([[0, 1]], device=device).unsqueeze(0).expand(3, -1, -1)
    )

    single_start = torch.tensor([[False, True]], device=device)
    mixed_start = torch.tensor(
        [[False, False, False, False, True], [False, False, False, False, True]],
        device=device,
    )
    single_cond = build_expert_conditioning(
        traj_future_start_mask=single_start,
        tokenizer_attention_mask=single_prompt_mask,
        rope_deltas=torch.tensor([[0]], device=device),
        n_action_tokens=2,
        dtype=dtype,
        attention_implementation=attention_implementation,
        cache_len=2,
    )
    mixed_cond = build_expert_conditioning(
        traj_future_start_mask=mixed_start,
        tokenizer_attention_mask=mixed_prompt_mask,
        rope_deltas=torch.tensor([[-3], [0]], device=device),
        n_action_tokens=2,
        dtype=dtype,
        attention_implementation=attention_implementation,
        cache_len=5,
    )

    action = torch.randn(1, 2, width, device=device, dtype=dtype)
    mixed_action = torch.cat(
        (action, torch.randn(1, 2, width, device=device, dtype=dtype)), dim=0
    )

    def run(prefix, prompt_mask, prompt_pos, action_embeds, conditioning):
        with torch.no_grad():
            prefill = model(
                inputs_embeds=prefix,
                attention_mask=prompt_mask,
                position_ids=prompt_pos,
                use_cache=True,
            )
            return model(
                inputs_embeds=action_embeds,
                attention_mask=conditioning.attention_mask,
                position_ids=conditioning.position_ids,
                past_key_values=prefill.past_key_values,
                use_cache=True,
                is_causal=False,
            ).last_hidden_state

    alone = run(short, single_prompt_mask, single_prompt_pos, action, single_cond)
    batched = run(mixed, mixed_prompt_mask, mixed_prompt_pos, mixed_action, mixed_cond)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(batched[:1], alone, rtol=tolerance, atol=tolerance / 10)

    # Demonstrate that exposing the three padding K/V entries breaks the invariant.
    if mixed_cond.attention_mask.ndim == 2:
        wrong = torch.ones_like(mixed_cond.attention_mask)
    else:
        wrong = torch.zeros_like(mixed_cond.attention_mask)
    wrong_cond = type(mixed_cond)(
        offsets=mixed_cond.offsets,
        prefix_len=mixed_cond.prefix_len,
        key_padding_mask=torch.ones_like(mixed_cond.key_padding_mask),
        attention_mask=wrong,
        position_ids=mixed_cond.position_ids,
    )
    wrong_batched = run(mixed, mixed_prompt_mask, mixed_prompt_pos, mixed_action, wrong_cond)
    correct_rel = ((batched[:1].float() - alone.float()).norm() / alone.float().norm()).item()
    wrong_rel = (
        (wrong_batched[:1].float() - alone.float()).norm() / alone.float().norm()
    ).item()
    assert wrong_rel > max(correct_rel * 2, 1e-4)
    return correct_rel, wrong_rel


def test_same_prompt_is_batch_invariant_with_correct_expert_mask():
    """The same short prompt must not change when batched beside a longer prompt."""
    _run_batch_invariance(torch.device("cpu"), "sdpa", torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_same_prompt_is_batch_invariant_with_flash_attention_on_cuda():
    correct_rel, wrong_rel = _run_batch_invariance(
        torch.device("cuda"), "flash_attention_2", torch.bfloat16
    )
    print(f"FlashAttention batch relative error: correct={correct_rel:.3e}, wrong={wrong_rel:.3e}")
