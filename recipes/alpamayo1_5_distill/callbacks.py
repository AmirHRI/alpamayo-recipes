# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer callbacks used by distillation curricula."""

from __future__ import annotations

import math
from typing import Any

from transformers import TrainerCallback


class BlockSpanScheduleCallback(TrainerCallback):
    """Select one block-loss span per epoch without restarting the LR scheduler.

    Keeping the curriculum in one Trainer run is important: a chain of one-epoch resume jobs
    would restore a scheduler that had already reached the end of its originally configured
    horizon. This callback changes only the scheduled field; optimizer, scheduler, sampler and
    checkpoint state remain continuous.

    ``target`` picks WHICH field the schedule drives, and the two are different objectives:

    ``block_span`` (default)
        One span per epoch, and that span is the WHOLE loss. m=1 is the teacher-forced
        per-layer loss; m>1 replaces it with the chained span. ``block_span_mix`` must stay
        disabled or the model would silently run the mix path instead of the schedule.

    ``block_span_mix``
        The teacher-forced m=1 loss is kept in EVERY epoch and the schedule drives the span
        that rides alongside it (weighted mean, see ``block_span_mix_weight``). m=1 is
        degenerate by construction -- ``block_span_mix <= 1`` disables the mix -- so stage 1
        is pure teacher-forced and stages 2+ are "m=1 plus span m". ``block_span`` must be 1,
        because it is what the mix path's teacher-forced term reads.
    """

    #: Fields this callback knows how to drive, and the field that must be pinned for each so
    #: the scheduled objective is the one that actually runs.
    _TARGETS = {"block_span": ("block_span_mix", 1), "block_span_mix": ("block_span", 1)}

    def __init__(
        self,
        spans: list[int],
        strict_num_train_epochs: bool = True,
        target: str = "block_span",
    ) -> None:
        self.spans = [int(span) for span in spans]
        if not self.spans or any(span < 1 for span in self.spans):
            raise ValueError(f"block span schedule must contain positive integers, got {spans}")
        if target not in self._TARGETS:
            raise ValueError(
                f"target must be one of {sorted(self._TARGETS)}, got {target!r}"
            )
        self.target = target
        self.strict_num_train_epochs = bool(strict_num_train_epochs)
        self._last_epoch_index: int | None = None

    @staticmethod
    def _unwrap(model: Any) -> Any:
        seen: set[int] = set()
        while hasattr(model, "module") and id(model) not in seen:
            seen.add(id(model))
            model = model.module
        return model

    def _apply(self, state: Any, model: Any) -> None:
        if model is None:
            raise RuntimeError("BlockSpanScheduleCallback did not receive the training model")
        epoch = float(state.epoch or 0.0)
        epoch_index = max(0, min(int(math.floor(epoch + 1e-8)), len(self.spans) - 1))
        base = self._unwrap(model)
        if not hasattr(base, "block_span"):
            raise TypeError("block span schedule requires a KDReasoningVLA model")
        # The companion field has to be pinned, or the model would run a DIFFERENT objective
        # than the one being scheduled: a stray block_span_mix>1 sends every epoch down the mix
        # path regardless of block_span, and a block_span>1 under a mix schedule would change
        # what the mix's "teacher-forced" term even means.
        pinned_name, pinned_value = self._TARGETS[self.target]
        pinned = int(getattr(base, pinned_name, pinned_value))
        if pinned > pinned_value:
            raise ValueError(
                f"{pinned_name}={pinned} must be <= {pinned_value} when {self.target} is "
                f"epoch-scheduled, or the scheduled objective is not the one that runs"
            )
        span = self.spans[epoch_index]
        # ⚠️ Bound the span by the number of slots the loss actually chains, and under layer
        # mixing that is NEITHER the student's depth NOR the expert module's.
        # `AlpamayoR1.__init__` sizes the expert from `vlm.config.text_config`, so a 28-layer
        # 2B builds a 28-layer expert module; the mixer then SYNTHESISES
        # `layer_mix_expert_layers` (36) cache slots on top of it, and the span sweep runs over
        # those. Reading either module therefore reports 28 and rejects the arm's final stage:
        # job 20777 died on "m=36 exceeds student depth 28" and its resume 20779 died again on
        # "exceeds expert depth 28" when I only half-fixed this.
        n_layers = None
        if bool(getattr(base, "layer_mix", False)):
            mixer = getattr(base, "layer_mixer", None)
            n_layers = int(getattr(mixer, "n_expert", 0)) or int(
                getattr(base, "layer_mix_expert_layers", 0)
            ) or None
        if n_layers is None:
            try:
                n_layers = len(base.expert.expert.layers)
            except (AttributeError, TypeError):
                try:
                    n_layers = len(base._text_model().layers)
                except (AttributeError, TypeError):
                    n_layers = None
        if n_layers is not None and span > n_layers:
            raise ValueError(f"scheduled block span m={span} exceeds expert depth {n_layers}")
        setattr(base, self.target, span)
        # The model logs checkpointing once, and the mix path logs its composition once. Reset
        # both at a stage boundary so the log proves every requested stage became active.
        base._span_ckpt_logged = False
        base._mix_logged = False
        if epoch_index != self._last_epoch_index:
            self._last_epoch_index = epoch_index
            if getattr(state, "is_world_process_zero", True):
                # Name the target, not just the number: "m=9" alone reads identically whether
                # it replaced the teacher-forced term or was added on top of it.
                how = (
                    f"span m={span} (replaces m=1)"
                    if self.target == "block_span"
                    else (
                        "teacher-forced m=1 only (mix degenerate at m=1)"
                        if span <= 1
                        else f"teacher-forced m=1 + span m={span}"
                    )
                )
                print(
                    f"[block-schedule] epoch {epoch_index + 1}/{len(self.spans)} "
                    f"-> {self.target}={span}: {how}",
                    flush=True,
                )

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.strict_num_train_epochs and int(args.num_train_epochs) != len(self.spans):
            raise ValueError(
                f"num_train_epochs={args.num_train_epochs} but block schedule has "
                f"{len(self.spans)} stages: {self.spans}"
            )
        self._apply(state, model)
        return control

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        self._apply(state, model)
        return control
