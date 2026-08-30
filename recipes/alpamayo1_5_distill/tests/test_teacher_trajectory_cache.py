# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from alpamayo1_5_distill.data import teacher_trajectory_io
from alpamayo1_5_distill.scripts.generate_teacher_trajectories import seed_for_key


def test_teacher_trajectory_cache_round_trip_is_float32_and_atomic(tmp_path):
    key = "12345678-abcd::5100000"
    states = torch.randn(6, 11, 64, 2, dtype=torch.bfloat16)

    path = teacher_trajectory_io.save_states(
        tmp_path, key, states, metadata={"teacher": "full"}
    )

    assert path == teacher_trajectory_io.entry_path(tmp_path, key)
    assert path.exists() and not path.with_suffix(".safetensors.tmp").exists()
    assert teacher_trajectory_io.has_entry(tmp_path, key)
    loaded = teacher_trajectory_io.load_states(tmp_path, key)
    assert loaded.dtype == torch.float32 and loaded.shape == states.shape
    torch.testing.assert_close(loaded, states.float())


def test_teacher_noise_seed_is_stable_and_sample_specific():
    a = seed_for_key("clip-a::1", 1234)
    assert a == seed_for_key("clip-a::1", 1234)
    assert a != seed_for_key("clip-b::1", 1234)
    assert a != seed_for_key("clip-a::1", 1235)
