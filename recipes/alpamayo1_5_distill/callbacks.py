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
    horizon. This callback changes only ``block_span``; optimizer, scheduler, sampler and
    checkpoint state remain continuous.
    """

    def __init__(self, spans: list[int], strict_num_train_epochs: bool = True) -> None:
        self.spans = [int(span) for span in spans]
        if not self.spans or any(span < 1 for span in self.spans):
            raise ValueError(f"block span schedule must contain positive integers, got {spans}")
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
        if int(getattr(base, "block_span_mix", 0)) > 1:
            raise ValueError("block_span_mix must be disabled when block_span is epoch-scheduled")
        span = self.spans[epoch_index]
        try:
            n_layers = len(base._text_model().layers)
        except (AttributeError, TypeError):
            n_layers = None
        if n_layers is not None and span > n_layers:
            raise ValueError(f"scheduled block span m={span} exceeds student depth {n_layers}")
        base.block_span = span
        # The model logs checkpointing once. Reset at a stage boundary so the log proves all
        # four requested spans actually became active.
        base._span_ckpt_logged = False
        if epoch_index != self._last_epoch_index:
            self._last_epoch_index = epoch_index
            if getattr(state, "is_world_process_zero", True):
                print(
                    f"[block-schedule] epoch {epoch_index + 1}/{len(self.spans)} -> span m={span}",
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
