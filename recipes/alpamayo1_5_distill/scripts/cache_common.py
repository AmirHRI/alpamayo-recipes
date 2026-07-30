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

"""Shared plumbing for the offline teacher-cache builders.

Extracted from ``generate_teacher_features.py`` so the single-vector and the
per-layer-KV builders share one copy of the parts that are easy to get wrong or
slow: the ``no_init_weights`` load trick, building the processor exactly once, the
shard slice, and a dataloader that overlaps frame decoding with the GPU.
"""

import os
import sys
import time
from typing import Any, Iterator

import torch


def parse_argv(argv: list[str]) -> dict[str, str]:
    """Parse the builders' ``key=value`` CLI into a dict."""
    return dict(a.split("=", 1) for a in argv if "=" in a)


def compose_config(config_name: str, argv: dict[str, str], config_dir: str | None = None) -> Any:
    """Compose the recipe's hydra config, translating the ``teacher=`` shorthand.

    Only keys containing ``.`` or ``@`` are treated as hydra overrides — the rest are
    the builder's own flags — and ``teacher=X`` expands to ``models@model=X`` so the
    common case stays short.
    """
    from hydra import compose, initialize_config_dir

    overrides = [f"{k}={v}" for k, v in argv.items() if ("." in k or "@" in k)]
    if "teacher" in argv:
        overrides.append(f"models@model={argv['teacher']}")

    if config_dir is None:
        config_dir = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")
        )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def build_teacher(cfg: Any, device: torch.device) -> torch.nn.Module:
    """Instantiate the frozen teacher on ``device``.

    The teacher's 8B VLM skeleton is random-initialised on CPU by
    ``Qwen3VLForConditionalGeneration(config)`` and then FULLY overwritten by the
    checkpoint (``load_alpamayo1_vlm``, ``assign=True``).  Skipping that throwaway
    init cuts the load from ~200 s to well under a minute.
    """
    import hydra.utils as hyu
    from transformers.modeling_utils import no_init_weights

    print("[cache] instantiating teacher ...", flush=True)
    started = time.time()
    with no_init_weights():
        model = hyu.instantiate(cfg.model, _convert_="partial")
    model = model.to(device).eval()
    model.requires_grad_(False)
    print(f"[cache] teacher built in {time.time() - started:.1f}s", flush=True)
    return model


def build_processor(model: torch.nn.Module) -> Any:
    """Build ONE reusable ``QwenProcessor`` for collation.

    ``collate_fn_from_model_config`` rebuilds a processor
    (``AutoProcessor.from_pretrained`` + 4000 added tokens) on every call — a few
    seconds per sample.  Building it once here removes that from the hot loop.
    """
    from alpamayo.processor.qwen_processor import QwenProcessor

    processor = QwenProcessor(
        vlm_name_or_path=model.config.vlm_name_or_path,
        traj_vocab_size=model.config.traj_vocab_size,
        min_pixels=model.config.min_pixels,
        max_pixels=model.config.max_pixels,
        chat_template_version="r1_5",
    )
    processor.build_processor()
    return processor


def to_device(x: Any, device: torch.device) -> Any:
    """Recursively move tensors (including those nested in dicts) to ``device``."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def shard_indices(n_samples: int, num_shards: int, shard: int) -> list[int]:
    """The modulo slice this shard owns — N shards on N GPUs never overlap."""
    return [i for i in range(n_samples) if i % num_shards == shard]


def sample_loader(
    dataset: Any, indices: list[int], processor: Any, num_workers: int = 8
) -> Iterator[tuple[int, dict[str, Any] | None]]:
    """Yield ``(idx, collated-batch-of-1)``, decoding frames in worker processes.

    Workers only touch CPU (frame load + tokenize) so disk and CPU overlap the GPU
    forward, which stays in the main process.  A per-sample ``try/except`` means one
    bad clip logs and is skipped rather than killing a 10 h job.
    """
    from torch.utils.data import DataLoader

    class _IdxDataset:
        def __len__(self) -> int:
            return len(indices)

        def __getitem__(self, j: int) -> tuple[int, Any]:
            i = indices[j]
            try:
                return i, dataset[i]
            except Exception as ex:  # a bad clip shouldn't kill the whole run
                print(f"[cache] idx={i} load error: {ex}", flush=True)
                return i, None

    def _collate_one(items: list[tuple[int, Any]]) -> tuple[int, Any]:
        i, s = items[0]
        return (i, None) if s is None else (i, processor.collate_fn([s]))

    loader = DataLoader(
        _IdxDataset(),
        batch_size=1,
        num_workers=num_workers,
        collate_fn=_collate_one,
        prefetch_factor=4 if num_workers else None,
    )
    return iter(loader)


class RateReporter:
    """Throughput / ETA printer for the long cache runs."""

    def __init__(self, total: int, label: str = "cache") -> None:
        self.total = total
        self.label = label
        self.started = time.time()

    def report(self, done: int, scanned: int, note: str = "") -> None:
        elapsed = max(time.time() - self.started, 1e-9)
        rate = done / elapsed
        remaining = (self.total - scanned) / rate if rate else 0.0
        print(
            f"[{self.label}] {done} new ({scanned}/{self.total} scanned)  "
            f"{rate:.2f}/s  ~{remaining / 3600:.1f}h left  {note}",
            flush=True,
        )


def main_argv() -> dict[str, str]:
    return parse_argv(sys.argv[1:])
