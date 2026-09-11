# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for batch-safe action-expert cache conditioning."""

from types import SimpleNamespace

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


def test_expert_conditioning_uses_last_handoff_occurrence():
    start = torch.tensor([[False, True, False, False, True, False]])
    out = build_expert_conditioning(
        traj_future_start_mask=start,
        tokenizer_attention_mask=torch.ones_like(start),
        rope_deltas=torch.tensor([[0]]),
        n_action_tokens=2,
        dtype=torch.float32,
        attention_implementation="sdpa",
    )

    assert out.offsets.tolist() == [5]
    assert out.prefix_len == 5
    assert out.key_padding_mask.tolist() == [[True, True, True, True, True, True, True]]


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


@pytest.mark.parametrize("reverse_rows", [False, True])
def test_eos_forward_masks_padding_and_future_suffixes(reverse_rows):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    from alpamayo1_5_distill.models.stitched_model import TrainableStitchedAlpamayoR1

    torch.manual_seed(19)

    def text_model():
        config = Qwen3VLTextConfig(
            vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=128,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
        )
        config._attn_implementation = "sdpa"
        return Qwen3VLTextModel(config).eval()

    vlm = text_model()
    expert = text_model()
    action_in = torch.nn.Linear(2, 32)
    action_out = torch.nn.Linear(32, 2)
    captured = {}

    def prefill(input_ids, attention_mask, labels, use_cache):
        assert labels is None
        positions = (attention_mask.cumsum(-1) - 1).clamp_min(0)
        output = vlm(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=positions.unsqueeze(0).expand(3, -1, -1), use_cache=use_cache,
        )
        output.rope_deltas = attention_mask.sum(-1, keepdim=True) - input_ids.shape[1]
        return output

    def capture_expert(module, args, kwargs):
        captured["mask"] = kwargs["attention_mask"].detach().clone()
        captured["positions"] = kwargs["position_ids"].detach().clone()
        captured["prefix_len"] = kwargs["past_key_values"].get_seq_length()

    def compute_loss(training_data, pred):
        captured["pred"] = pred.detach().clone()
        return pred.square().mean()

    expert.register_forward_pre_hook(capture_expert, with_kwargs=True)
    harness = SimpleNamespace(
        cotrain_vlm=False, stop_grad_from_vlm=True,
        config=SimpleNamespace(traj_token_ids={"future_start": 31}, expert_non_causal_attention=True),
        fuse_traj_tokens=lambda input_ids, traj_data: input_ids,
        vlm=prefill, expert=expert,
        action_in_proj=lambda noisy, timesteps: action_in(noisy),
        action_out_proj=action_out,
        action_space=SimpleNamespace(get_action_space_dims=lambda: (2, 2)),
        diffusion=SimpleNamespace(compute_loss_from_pred=compute_loss),
        _process_traj_future_training=lambda data: {
            "noisy_x": data["ego_future_xyz"], "timesteps": torch.zeros(1),
        },
    )
    action = torch.randn(1, 2, 2)
    single = {"input_ids": torch.tensor([[4, 31, 10, 11, 12]]),
              "attention_mask": torch.ones(1, 5, dtype=torch.long)}
    TrainableStitchedAlpamayoR1.forward(harness, single, ego_future_xyz=action)
    expected = captured["pred"].clone()
    mixed = {
        "input_ids": torch.tensor([[0, 0, 4, 31, 10, 11, 12], [2, 3, 4, 5, 31, 10, 11]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]),
    }
    actions = torch.cat((action, torch.randn_like(action)))
    if reverse_rows:
        mixed = {key: value.flip(0) for key, value in mixed.items()}
        actions = actions.flip(0)
    output = TrainableStitchedAlpamayoR1.forward(harness, mixed, ego_future_xyz=actions)
    short_row = int(reverse_rows)
    torch.testing.assert_close(captured["pred"][short_row:short_row + 1], expected)
    assert captured["prefix_len"] == 5
    assert captured["mask"][short_row, 0, 0].eq(0).tolist() == [
        False, False, True, True, False, True, True,
    ]
    assert captured["positions"][0, short_row].tolist() == [2, 3]
    assert "input_ids" in mixed
    output.loss.backward()
    assert all(parameter.grad is None for parameter in vlm.parameters())
    for parameter in (expert.layers[0].self_attn.q_proj.weight, action_in.weight, action_out.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_same_prompt_is_batch_invariant_with_flash_attention_on_cuda():
    correct_rel, wrong_rel = _run_batch_invariance(
        torch.device("cuda"), "flash_attention_2", torch.bfloat16
    )
    print(f"FlashAttention batch relative error: correct={correct_rel:.3e}, wrong={wrong_rel:.3e}")
