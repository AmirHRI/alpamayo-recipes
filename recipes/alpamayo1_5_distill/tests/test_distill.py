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

"""GPU-free unit tests for the distillation plumbing.

Covers the parts that are easy to get subtly wrong: the cache key format /
round-trip, and the left-padding-robust ``<traj_future_start>`` column finder
(which must pick the *last* occurrence per row and error if the token is
missing). The heavier forward/train path is validated by the recipe's smoke
runs on real data (see README).
"""

import types

import numpy as np
import torch

from alpamayo1_5_distill.data.distill_dataset import (
    load_teacher_cache_metadata,
    sample_key,
    save_teacher_cache,
)
from alpamayo1_5_distill.models.distill_base_model import DistillReasoningVLA


def test_sample_key_normalises_timestamp_types():
    assert sample_key("clip-abc", 123) == "clip-abc::123"
    assert sample_key("clip-abc", 123.0) == "clip-abc::123"
    assert sample_key("clip-abc", np.int64(123)) == "clip-abc::123"


def test_cache_roundtrip(tmp_path):
    path = str(tmp_path / "feats.safetensors")
    feats = {
        sample_key("c1", 100): torch.randn(4096),
        sample_key("c2", 200): torch.randn(4096),
    }
    meta = {"teacher_hidden_dim": 4096, "n_samples": 2, "format": "alpamayo_distill_teacher_v1"}
    save_teacher_cache(feats, meta, path)

    from safetensors.torch import load_file

    loaded = load_file(path)
    assert set(loaded) == set(feats)
    for k in feats:
        assert torch.allclose(loaded[k], feats[k].to(torch.float32))

    assert load_teacher_cache_metadata(path)["teacher_hidden_dim"] == 4096


def _stub(tfs_id=7):
    """A minimal stand-in exposing only what the column finder reads."""
    return types.SimpleNamespace(special_token_ids={"traj_future_start": tfs_id})


def test_traj_future_start_columns_last_occurrence_and_padding():
    tfs = 7
    stub = _stub(tfs)
    # row 0: left-padded (pad=0), tfs at col 4
    # row 1: tfs appears twice (cols 1 and 5) -> must pick the last (5)
    input_ids = torch.tensor(
        [
            [0, 0, 3, 3, tfs, 9, 9, 9],
            [3, tfs, 4, 4, 4, tfs, 9, 9],
        ]
    )
    cols = DistillReasoningVLA._traj_future_start_columns(stub, input_ids)
    assert cols.tolist() == [4, 5]


def test_traj_future_start_columns_missing_raises():
    stub = _stub(7)
    input_ids = torch.tensor([[1, 2, 3, 4]])  # no tfs token
    try:
        DistillReasoningVLA._traj_future_start_columns(stub, input_ids)
    except ValueError as e:
        assert "traj_future_start" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for missing <traj_future_start>")


def test_gather_tfs_hidden_selects_right_vectors():
    tfs = 7
    stub = _stub(tfs)
    # _gather_tfs_hidden calls self._traj_future_start_columns internally; bind it.
    stub._traj_future_start_columns = types.MethodType(
        DistillReasoningVLA._traj_future_start_columns, stub
    )
    B, L, H = 2, 6, 5
    input_ids = torch.tensor(
        [
            [0, 0, 3, tfs, 9, 9],  # col 3
            [3, 3, 3, 3, 3, tfs],  # col 5
        ]
    )
    hidden = torch.arange(B * L * H, dtype=torch.float32).reshape(B, L, H)
    out = DistillReasoningVLA._gather_tfs_hidden(stub, input_ids, hidden)
    assert out.shape == (B, H)
    assert torch.equal(out[0], hidden[0, 3])
    assert torch.equal(out[1], hidden[1, 5])
