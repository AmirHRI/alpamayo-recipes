# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file as save_safetensors_file
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from alpamayo1_5_sft_qwen3_5.models.sft_alpamayo_r1 import TrainableAlpamayoR1
from alpamayo1_5_sft_qwen3_5.models.sft_base_model import load_alpamayo1_vlm
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


def _write_vlm_checkpoint(checkpoint_dir: Path, state_dict: dict[str, torch.Tensor]) -> None:
    shard_name = "model-00001-of-00001.safetensors"
    save_safetensors_file(state_dict, checkpoint_dir / shard_name)
    index = {"weight_map": {key: shard_name for key in state_dict}}
    (checkpoint_dir / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )


def _tiny_qwen3_5() -> Qwen3_5ForConditionalGeneration:
    # 4 text layers -> layer_types = [linear, linear, linear, full] (interval 4),
    # covering both the Gated-DeltaNet and full-attention code paths.
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 64,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 8,
        },
        image_token_id=28,
        video_token_id=29,
        vision_start_token_id=30,
        vision_end_token_id=31,
    )
    return Qwen3_5ForConditionalGeneration(config)


def test_load_alpamayo1_vlm_strips_prefix_for_nested_vlm(tmp_path: Path) -> None:
    model = _tiny_qwen3_5()
    key = "model.language_model.embed_tokens.weight"
    replacement = torch.full_like(model.state_dict()[key], 2)
    _write_vlm_checkpoint(tmp_path, {f"vlm.{key}": replacement})

    load_alpamayo1_vlm(str(tmp_path), model)

    torch.testing.assert_close(model.state_dict()[key], replacement)


def test_load_alpamayo1_vlm_accepts_unprefixed_nested_vlm(tmp_path: Path) -> None:
    model = _tiny_qwen3_5()
    key = "model.language_model.embed_tokens.weight"
    replacement = torch.full_like(model.state_dict()[key], 3)
    _write_vlm_checkpoint(tmp_path, {key: replacement})

    load_alpamayo1_vlm(str(tmp_path), model)

    torch.testing.assert_close(model.state_dict()[key], replacement)


def test_load_alpamayo1_vlm_keeps_prefix_for_full_model(tmp_path: Path) -> None:
    model = torch.nn.Module()
    model.vlm = torch.nn.Linear(1, 1, bias=False)
    replacement = torch.full_like(model.vlm.weight, 3)
    _write_vlm_checkpoint(tmp_path, {"vlm.weight": replacement})

    load_alpamayo1_vlm(str(tmp_path), model)

    torch.testing.assert_close(model.vlm.weight, replacement)


def test_load_alpamayo1_vlm_rejects_unmatched_keys(tmp_path: Path) -> None:
    _write_vlm_checkpoint(tmp_path, {"vlm.unknown": torch.ones(1)})

    with pytest.raises(ValueError, match="do not match the target model"):
        load_alpamayo1_vlm(str(tmp_path), torch.nn.Linear(1, 1))


def test_from_pretrained_applies_stage1_after_parent_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replacement = torch.full((1, 1), 2.0)
    _write_vlm_checkpoint(tmp_path, {"vlm.weight": replacement})

    def fake_parent_from_pretrained(
        cls: type[TrainableAlpamayoR1],
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> TrainableAlpamayoR1:
        assert pretrained_model_name_or_path == "base-checkpoint"
        assert "stage1_vlm_checkpoint_path" not in kwargs
        model = object.__new__(cls)
        torch.nn.Module.__init__(model)
        model.vlm = torch.nn.Linear(1, 1, bias=False, dtype=torch.float16)
        model.vlm.weight.data.fill_(1)
        model.cotrain_vlm = kwargs["cotrain_vlm"]
        return model

    monkeypatch.setattr(AlpamayoR1, "from_pretrained", classmethod(fake_parent_from_pretrained))

    model = TrainableAlpamayoR1.from_pretrained(
        "base-checkpoint",
        stage1_vlm_checkpoint_path=str(tmp_path),
        cotrain_vlm=False,
    )

    torch.testing.assert_close(model.vlm.weight, replacement.to(model.vlm.weight))
    assert model.vlm.weight.dtype == torch.float16
    assert not model.vlm.weight.requires_grad


def test_constructor_rejects_stage1_checkpoint() -> None:
    with pytest.raises(ValueError, match="only supported by.*from_pretrained"):
        TrainableAlpamayoR1(None, stage1_vlm_checkpoint_path="stage1-checkpoint")  # type: ignore[arg-type]


def test_resolve_vlm_backend_detects_qwen3_5(tmp_path: Path) -> None:
    """The upstream dispatch patch should auto-detect a Qwen 3.5 checkpoint's
    model_type and route it to the Qwen 3.5 backend, without requiring an
    explicit `vlm_backend` override."""
    from alpamayo_r1.models.base_model import ReasoningVLA, ReasoningVLAConfig

    model = _tiny_qwen3_5()
    model.save_pretrained(tmp_path)

    config = object.__new__(ReasoningVLAConfig)
    config.vlm_backend = "qwenvl3"  # the default every existing checkpoint carries
    config.vlm_name_or_path = str(tmp_path)

    assert ReasoningVLA._resolve_vlm_backend(config) == "qwen3_5"
