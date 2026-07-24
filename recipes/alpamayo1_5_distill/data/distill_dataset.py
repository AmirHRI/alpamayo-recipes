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

"""Datasets for latent-reasoning distillation.

Two wrappers over the shared PAI datasets add exactly what distillation needs:

* the ability to attach a *cached teacher hidden state* (``teacher_tfs_hidden``)
  to each sample so it rides into the model's ``forward`` via the collator, and
* (nav variant) injecting the annotation ``cot`` into the sample dict *before*
  the VLA preprocessor runs, so a teacher-side processor with ``cot`` in
  ``components_order`` can teacher-force it.

The cache is keyed by ``f"{clip_id}::{t0_us}"``.  The **same** ``_sample_key``
is used by the offline cache builder and by the training-time loader, so the
teacher run (with CoT) and the student run (CoT-free) line up sample-for-sample
as long as they iterate the identical dataset config (same clips / keyframes).
"""

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as _load_safetensors
from safetensors.torch import save_file as _save_safetensors

from alpamayo.data.pai import PAIDataset
from alpamayo.data.pai_nav import PAIDatasetWithNav
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset


def sample_key(clip_id: Any, t0_us: Any) -> str:
    """Canonical cache key for a (clip_id, keyframe-timestamp) pair."""
    return f"{clip_id}::{int(t0_us)}"


def save_teacher_cache(
    features: dict[str, torch.Tensor], metadata: dict[str, Any], out_path: str
) -> None:
    """Persist the teacher feature cache as safetensors + a ``.meta.json`` sidecar.

    ``features`` maps :func:`sample_key` -> a 1-D teacher hidden vector.
    safetensors metadata must be ``str -> str``; the richer metadata (dims,
    counts, provenance) is written next to it as JSON.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    contiguous = {k: v.contiguous().to(torch.float32).cpu() for k, v in features.items()}
    _save_safetensors(contiguous, str(out), metadata={"format": "alpamayo_distill_teacher_v1"})
    with out.with_suffix(out.suffix + ".meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_teacher_cache_metadata(cache_path: str) -> dict[str, Any]:
    """Load the ``.meta.json`` sidecar for a teacher cache (or ``{}`` if absent)."""
    meta = Path(cache_path).with_suffix(Path(cache_path).suffix + ".meta.json")
    if meta.exists():
        with meta.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


class _TeacherFeatureMixin:
    """Attach cached teacher hidden vectors to samples.

    Mix in *before* the concrete dataset in the MRO and call
    :meth:`_setup_teacher_cache` from ``__init__``.
    """

    def _setup_teacher_cache(self, teacher_cache_path: str | None) -> None:
        self.teacher_cache_path = teacher_cache_path
        self._teacher_cache: dict[str, torch.Tensor] | None = None

    def _cache(self) -> dict[str, torch.Tensor]:
        if self._teacher_cache is None:
            self._teacher_cache = _load_safetensors(self.teacher_cache_path)
        return self._teacher_cache

    def _attach_teacher_feature(
        self, sample: dict[str, Any] | None, key: str
    ) -> dict[str, Any] | None:
        if sample is None or self.teacher_cache_path is None:
            return sample
        cache = self._cache()
        if key not in cache:
            raise KeyError(
                f"teacher feature for key '{key}' not found in cache "
                f"'{self.teacher_cache_path}' ({len(cache)} entries). Re-run "
                "scripts/generate_teacher_features.py against the same dataset config."
            )
        sample["teacher_tfs_hidden"] = cache[key]
        return sample

    def _sample_key(self, idx: int) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


class DistillPAIDataset(_TeacherFeatureMixin, PAIDataset):
    """PAIDataset that (optionally) attaches cached teacher hidden states.

    CoT is provided automatically by :class:`PAIDataset` when
    ``reasoning_metadata`` is set (via ``get_reasoning_data``), so a teacher-side
    processor with ``cot`` in ``components_order`` works out of the box on this
    class; the student-side (CoT-free) processor simply ignores it.
    """

    def __init__(self, *args: Any, teacher_cache_path: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._setup_teacher_cache(teacher_cache_path)

    def _sample_key(self, idx: int) -> str:
        clip_id = self.clip_ids[idx]
        t0_us = (
            self.DEFAULT_T0_US
            if self.use_default_keyframe
            else self.avdi.get_clip_key_frame(clip_id)
        )
        return sample_key(clip_id, t0_us)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        sample = super().__getitem__(idx)
        return self._attach_teacher_feature(sample, self._sample_key(idx))


class DistillNavDataset(_TeacherFeatureMixin, PAIDatasetWithNav):
    """Annotation-driven distillation dataset.

    Injects the annotation ``cot`` into the sample dict *before* preprocessing
    (so a teacher-side processor with ``cot`` in ``components_order`` can
    teacher-force it) and attaches cached teacher hidden states.  The parent
    ``__getitem__`` cannot be reused directly because it builds ``tokenized_data``
    at the end and would run before ``cot`` is available.
    """

    def __init__(
        self,
        *args: Any,
        teacher_cache_path: str | None = None,
        inject_cot: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._setup_teacher_cache(teacher_cache_path)
        self.inject_cot = inject_cot

    def _sample_key(self, idx: int) -> str:
        entry = self._samples[idx]
        return sample_key(entry["clip_id"], entry["t0_relative"])

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        entry = self._samples[idx]
        clip_id = entry["clip_id"]
        t0_us = int(entry["t0_relative"])

        sample_data = load_physical_aiavdataset(
            clip_id,
            t0_us=t0_us,
            avdi=self.avdi,
            num_history_steps=self.num_history_steps,
            num_future_steps=self.num_future_steps,
            time_step=self.time_step,
        )

        sample_data["nav_text"] = entry["nav_text"]
        # Inject CoT whenever the annotation carries the field (even if empty) so
        # a teacher-side processor with `cot` in components_order never trips
        # construct_cot's `"cot" in data` assert.
        if self.inject_cot and "cot" in entry:
            sample_data["cot"] = entry["cot"]

        for key in list(sample_data.keys()):
            if key.startswith("ego_"):
                sample_data[key] = sample_data[key].squeeze(0)

        if self.vla_preprocess_func is not None:
            sample_data["tokenized_data"] = self.vla_preprocess_func(data=sample_data)

        return self._attach_teacher_feature(sample_data, sample_key(clip_id, t0_us))
