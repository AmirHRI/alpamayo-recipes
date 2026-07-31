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

import os
from typing import Any

import torch

from alpamayo1_5_sft.trainer import ReasoningVLA_Trainer

#: Detached scalars on ``KaVaVLAOutput`` that get averaged into the training logs.
AUX_LOSS_KEYS = ("ce_loss", "latent_loss", "kv_loss", "n_valid_slots")

#: Parameters excluded from weight decay on top of HF's own bias/norm exclusions.
#: ``slot_embeddings`` is a soft prompt: decaying it pulls the slots back toward the
#: origin and directly fights the measured vocabulary initialisation
#: (``reasoning-setup-2b.md`` §9.1 C3), which is the whole reason those values start
#: where they do.
NO_DECAY_PARAMS = ("slot_embeddings",)

#: How often to re-measure each loss term's share of the backbone gradient. 0 disables.
#: Cheap at this interval (three partial backwards over one layer, on the graph the
#: step already built), and it is the standing check that no term has gone inert.
#: Settable via ``KAVA_GRAD_PROBE_STEPS`` — an env var rather than a config key so it
#: needs no field on the shared ``TrainingArguments`` owned by the sft recipe.
GRAD_PROBE_STEPS = int(os.environ.get("KAVA_GRAD_PROBE_STEPS", 200))

#: The slice of the backbone the probe differentiates. One mid-stack layer is enough to
#: rank the terms and keeps the probe to a few ms; layer 14 of the student's 28.
PROBE_LAYER_PREFIX = "vlm.model.language_model.layers.14."


class KaVaTrainer(ReasoningVLA_Trainer):
    """``ReasoningVLA_Trainer`` plus per-term loss logging.

    Terms are summed across ``compute_loss`` calls (so gradient accumulation is
    handled) and averaged on each ``log``, matching how HF averages ``loss`` itself.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._aux_sums: dict[str, float] = {}
        self._aux_count: int = 0
        self._last_probe_step: int = -1
        self._probe_failed: bool = False

    def get_decay_parameter_names(self, model: Any) -> list[str]:
        decay = super().get_decay_parameter_names(model)
        return [name for name in decay if not any(part in name for part in NO_DECAY_PARAMS)]

    def compute_loss(self, model: Any, inputs: Any, return_outputs: bool = False, **kwargs: Any):
        base = self._unwrapped(model)
        probing = self._should_probe()
        if probing:
            base.keep_loss_terms = True
        try:
            result = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        finally:
            if probing:
                base.keep_loss_terms = False
        loss, outputs = result if isinstance(result, tuple) else (result, None)
        self._stash(outputs)
        if probing:
            self._probe_gradient_shares(base)
        return (loss, outputs) if return_outputs else loss

    # -------------------------------------------------------- gradient probe
    def _unwrapped(self, model: Any) -> Any:
        return getattr(self, "accelerator", None) and self.accelerator.unwrap_model(model) or model

    def _should_probe(self) -> bool:
        if self._probe_failed or GRAD_PROBE_STEPS <= 0:
            return False
        step = int(getattr(self.state, "global_step", 0))
        if step == self._last_probe_step:
            return False  # once per optimizer step, not once per accumulation micro-batch
        return step % GRAD_PROBE_STEPS == 0

    def _probe_gradient_shares(self, base: Any) -> None:
        """Log each loss term's share of the *backbone* gradient.

        A term can be finite, decreasing, and still steer nothing — which is exactly
        what happened to ``L_KV`` at its first ``lambda_2``: it normalises over every
        (layer, head, slot, dim) element, so the shipped weight left it contributing
        0.0% of the backbone gradient while looking perfectly healthy in the loss log.
        A weight calibrated once by measurement is only safe if something keeps
        checking it, so this re-measures on one batch every ``GRAD_PROBE_STEPS``.

        Uses ``autograd.grad`` on the *existing* graph (before HF's backward), so the
        cost is three partial backwards over one mid-stack layer rather than three
        extra forwards. Nothing is written to ``.grad``, so training is unaffected.

        The probe is best-effort: under ZeRO-3 the parameters are sharded and this
        cannot work, so a failure disables it for the rest of the run rather than
        taking the job down.
        """
        terms = getattr(base, "last_loss_terms", None)
        if not terms:
            return
        probe = [
            p
            for name, p in base.named_parameters()
            if name.startswith(PROBE_LAYER_PREFIX) and p.requires_grad
        ]
        if not probe:
            self._probe_failed = True
            return

        weights = {
            "ce": 1.0,
            "latent": float(getattr(base, "latent_loss_weight", 0.0)),
            "kv": float(getattr(base, "kv_loss_weight", 0.0)),
        }
        try:
            norms = {}
            for name, term in terms.items():
                grads = torch.autograd.grad(
                    term, probe, retain_graph=True, allow_unused=True, materialize_grads=True
                )
                norms[name] = weights.get(name, 1.0) * float(
                    torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
                )
        except Exception as ex:  # ZeRO-3 sharding, a freed graph, ...
            print(f"[kava] gradient probe disabled: {ex}", flush=True)
            self._probe_failed = True
            return

        self._last_probe_step = int(getattr(self.state, "global_step", 0))
        reference = norms.get("ce", 0.0)
        summary = "  ".join(
            f"{k}={v:.3e}" + (f" ({v / reference:.0%} of ce)" if reference and k != "ce" else "")
            for k, v in norms.items()
        )
        print(f"[kava] step {self._last_probe_step} weighted backbone grad: {summary}", flush=True)
        for name, value in norms.items():
            if name != "ce" and reference:
                self._aux_sums[f"gradshare_{name}"] = value / reference
                self._aux_count = max(self._aux_count, 1)

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
