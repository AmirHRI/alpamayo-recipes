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

"""Trainer for KAVA runs: logs the loss terms, and keeps decay off the soft prompt.

``DistillVLAOutput`` has exposed detached ``ce_loss`` / ``latent_loss`` for logging
since the recipe was written, but nothing ever read them — HF ``Trainer`` only
consumes ``outputs["loss"]``, so the split was invisible in every run so far.  With
a third term (``L_KV``) that stops being a cosmetic gap: "total went down" cannot
tell you whether the KV objective did anything, and a dead ``L_KV`` is precisely the
failure this recipe has to detect.
"""

from typing import Any

from alpamayo1_5_sft.trainer import ReasoningVLA_Trainer

#: Detached scalars on ``KaVaVLAOutput`` that get averaged into the training logs.
AUX_LOSS_KEYS = ("ce_loss", "latent_loss", "kv_loss", "n_valid_slots")

#: Parameters excluded from weight decay on top of HF's own bias/norm exclusions.
#: ``slot_embeddings`` is a soft prompt: decaying it pulls the slots back toward the
#: origin and directly fights the measured vocabulary initialisation
#: (``reasoning-setup-2b.md`` §9.1 C3), which is the whole reason those values start
#: where they do.
NO_DECAY_PARAMS = ("slot_embeddings",)


class KaVaTrainer(ReasoningVLA_Trainer):
    """``ReasoningVLA_Trainer`` plus per-term loss logging.

    Terms are summed across ``compute_loss`` calls (so gradient accumulation is
    handled) and averaged on each ``log``, matching how HF averages ``loss`` itself.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._aux_sums: dict[str, float] = {}
        self._aux_count: int = 0

    def get_decay_parameter_names(self, model: Any) -> list[str]:
        decay = super().get_decay_parameter_names(model)
        return [name for name in decay if not any(part in name for part in NO_DECAY_PARAMS)]

    def compute_loss(self, model: Any, inputs: Any, return_outputs: bool = False, **kwargs: Any):
        result = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        loss, outputs = result if isinstance(result, tuple) else (result, None)
        self._stash(outputs)
        return (loss, outputs) if return_outputs else loss

    def _stash(self, outputs: Any) -> None:
        if outputs is None:
            return
        seen = False
        for key in AUX_LOSS_KEYS:
            value = outputs.get(key) if hasattr(outputs, "get") else getattr(outputs, key, None)
            if value is None:
                continue
            self._aux_sums[key] = self._aux_sums.get(key, 0.0) + float(value.detach().item())
            seen = True
        if seen:
            self._aux_count += 1

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        if self._aux_count:
            for key, total in self._aux_sums.items():
                logs[key] = round(total / self._aux_count, 6)
            self._aux_sums = {}
            self._aux_count = 0
        super().log(logs, *args, **kwargs)
