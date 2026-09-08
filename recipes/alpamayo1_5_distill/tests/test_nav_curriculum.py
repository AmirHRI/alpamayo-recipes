# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free checks for the direction-only nav input and epoch span curriculum."""

from types import SimpleNamespace

import pytest

from alpamayo1_5_distill.callbacks import BlockSpanScheduleCallback
from alpamayo1_5_distill.data.camera_subset import strip_turn_distance


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Turn left in 29m", "Turn left"),
        ("Turn right in 7m", "Turn right"),
        ("Turn left", "Turn left"),
        ("Continue straight", "Continue straight"),
        ("Reverse", "Reverse"),
    ],
)
def test_strip_turn_distance(raw, expected):
    assert strip_turn_distance(raw) == expected


def test_strip_turn_distance_rejects_unknown_turn_grammar():
    with pytest.raises(ValueError, match="unsupported nav turn instruction"):
        strip_turn_distance("Turn slightly left in twenty meters")


def test_block_span_schedule_selects_requested_epoch_and_unwraps_model():
    base = SimpleNamespace(
        block_span=99,
        block_span_mix=0,
        _text_model=lambda: SimpleNamespace(layers=[object()] * 36),
    )
    wrapped = SimpleNamespace(module=SimpleNamespace(module=base))
    callback = BlockSpanScheduleCallback([1, 9, 18, 36])
    args = SimpleNamespace(num_train_epochs=4)
    control = object()

    for epoch, expected in enumerate((1, 9, 18, 36)):
        state = SimpleNamespace(epoch=float(epoch), is_world_process_zero=False)
        assert callback.on_epoch_begin(args, state, control, model=wrapped) is control
        assert base.block_span == expected
        assert base._span_ckpt_logged is False


def test_block_span_schedule_rejects_epoch_count_mismatch():
    callback = BlockSpanScheduleCallback([1, 9, 18, 36])
    with pytest.raises(ValueError, match="block schedule has 4 stages"):
        callback.on_train_begin(
            SimpleNamespace(num_train_epochs=5),
            SimpleNamespace(epoch=0.0),
            object(),
            model=SimpleNamespace(block_span=1, block_span_mix=0),
        )
