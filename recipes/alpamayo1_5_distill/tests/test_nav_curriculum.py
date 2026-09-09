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


def test_block_span_mix_schedule_keeps_teacher_forced_term_every_epoch():
    """target='block_span_mix' rides the span ALONGSIDE m=1 instead of replacing it.

    The distinction is invisible in the scheduled number -- both targets walk [1,9,18,36] --
    so this pins the field that actually moves and the one that must stay put. m=1 in stage 1
    is degenerate on purpose: block_span_mix <= 1 disables the mix, leaving the pure
    teacher-forced loss, which is what makes stage 1 identical to the plain-span arm's stage 1.
    """
    base = SimpleNamespace(
        block_span=1,
        block_span_mix=0,
        _text_model=lambda: SimpleNamespace(layers=[object()] * 36),
    )
    callback = BlockSpanScheduleCallback([1, 9, 18, 36], target="block_span_mix")
    args = SimpleNamespace(num_train_epochs=4)
    control = object()

    for epoch, expected in enumerate((1, 9, 18, 36)):
        state = SimpleNamespace(epoch=float(epoch), is_world_process_zero=False)
        assert callback.on_epoch_begin(args, state, control, model=base) is control
        assert base.block_span_mix == expected
        # block_span is what the mix path's teacher-forced term reads; it must never move.
        assert base.block_span == 1
        # Both once-only logs reset per stage, so the run log proves each stage went live.
        assert base._span_ckpt_logged is False
        assert base._mix_logged is False


def test_block_span_schedule_rejects_companion_field_that_would_change_the_objective():
    """A stray companion field silently selects a DIFFERENT loss than the one scheduled."""
    span_sched = BlockSpanScheduleCallback([1, 9], target="block_span")
    with pytest.raises(ValueError, match="block_span_mix=9 must be <= 1"):
        span_sched.on_epoch_begin(
            SimpleNamespace(num_train_epochs=2),
            SimpleNamespace(epoch=0.0, is_world_process_zero=False),
            object(),
            model=SimpleNamespace(block_span=1, block_span_mix=9),
        )

    mix_sched = BlockSpanScheduleCallback([1, 9], target="block_span_mix")
    with pytest.raises(ValueError, match="block_span=9 must be <= 1"):
        mix_sched.on_epoch_begin(
            SimpleNamespace(num_train_epochs=2),
            SimpleNamespace(epoch=0.0, is_world_process_zero=False),
            object(),
            model=SimpleNamespace(block_span=9, block_span_mix=1),
        )


def test_block_span_schedule_rejects_unknown_target():
    with pytest.raises(ValueError, match="target must be one of"):
        BlockSpanScheduleCallback([1, 9], target="block_freerun")


def test_block_span_schedule_bounds_by_expert_depth_not_student_depth():
    """A layer-mix student is SHALLOWER than the expert whose blocks the span chains.

    28-layer 2B student, 36-layer expert with synthesised cache slots: m=36 is legal and is
    the arm's final stage. Bounding by the student's 28 rejects it -- which is exactly how job
    20777 died at its epoch-4 boundary after 7.5 h, having already written epochs 1-3.
    """
    # The real shape of a layer-mix 2B: the student VLM is 28 deep, the expert MODULE is also
    # 28 (AlpamayoR1 sizes it from the VLM config), and only the mixer knows the 36 synthesised
    # slots the span actually chains. Both module reads say 28, so only layer_mix_expert_layers
    # / layer_mixer.n_expert can authorise m=36.
    base = SimpleNamespace(
        block_span=1,
        block_span_mix=0,
        layer_mix=True,
        layer_mix_expert_layers=36,
        layer_mixer=SimpleNamespace(n_expert=36),
        expert=SimpleNamespace(expert=SimpleNamespace(layers=[object()] * 28)),
        _text_model=lambda: SimpleNamespace(layers=[object()] * 28),
    )
    callback = BlockSpanScheduleCallback([1, 9, 18, 36])
    for epoch, expected in enumerate((1, 9, 18, 36)):
        callback.on_epoch_begin(
            SimpleNamespace(num_train_epochs=4),
            SimpleNamespace(epoch=float(epoch), is_world_process_zero=False),
            object(),
            model=base,
        )
        assert base.block_span == expected


def test_block_span_schedule_still_rejects_span_past_the_expert():
    base = SimpleNamespace(
        block_span=1,
        block_span_mix=0,
        layer_mix=True,
        layer_mix_expert_layers=36,
        layer_mixer=SimpleNamespace(n_expert=36),
    )
    callback = BlockSpanScheduleCallback([1, 48])
    with pytest.raises(ValueError, match="m=48 exceeds expert depth 36"):
        callback.on_epoch_begin(
            SimpleNamespace(num_train_epochs=2),
            SimpleNamespace(epoch=1.0, is_world_process_zero=False),
            object(),
            model=base,
        )
