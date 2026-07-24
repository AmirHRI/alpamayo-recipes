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

"""Offline teacher-feature cache builder for latent-reasoning distillation.

Runs the frozen teacher (default: Alpamayo-1.5-10B) once over the dataset with
chain-of-thought in context, and caches the teacher's last-layer hidden state at
``<traj_future_start>`` — the single vector the action expert conditions on.
The student later matches this via ``DistillReasoningVLA``'s latent loss, with
no teacher present in the training loop.

Usage (from the recipe dir, with the alpamayo1_5_sft venv + PYTHONPATH=recipes)::

    python -m alpamayo1_5_distill.scripts.generate_teacher_features \
        config=cache_teacher_features \
        out=/path/to/teacher_features.safetensors \
        model=/models/teacher_ar1_5_10b            # or the default 2B teacher

Any dotted ``key=value`` is forwarded as a Hydra override, e.g.
``data.cache_dataset.chunk_ids=[2368]`` or ``limit=5`` (non-dotted keys
``config``/``out``/``limit`` are consumed by this script).
"""

import os
import sys
import time

import hydra.utils as hyu
import torch
from hydra import compose, initialize_config_dir

from alpamayo1_5_distill.data.distill_dataset import save_teacher_cache


def _to_device(x, device):
    """Recursively move tensors (incl. those nested in dicts) to ``device``."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x


def main() -> None:
    argv = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    config_name = argv.get("config", "cache_teacher_features")
    out_path = argv.get("out", "teacher_features.safetensors")
    limit = int(argv.get("limit", 0))  # 0 == all samples

    # Forward any dotted or @-packaged key=value as a Hydra override; the script
    # keys (config/out/limit/teacher) contain neither "." nor "@".
    overrides = [f"{k}={v}" for k, v in argv.items() if ("." in k or "@" in k)]
    # Convenience: `teacher=<option>` swaps the @model config-group default,
    # e.g. `teacher=teacher_ar1_5_10b` (the real 10B) or `teacher=teacher_cosmos2b`.
    if "teacher" in argv:
        overrides.append(f"models@model={argv['teacher']}")

    cfg_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")
    )
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    device = torch.device("cuda")

    print("[cache] instantiating teacher ...", flush=True)
    t0 = time.time()
    model = hyu.instantiate(cfg.model, _convert_="partial").to(device).eval()
    model.requires_grad_(False)
    print(f"[cache] teacher built in {time.time() - t0:.1f}s", flush=True)

    dataset = hyu.instantiate(
        cfg.data.cache_dataset, _convert_="partial", model_config=model.config
    )
    collate = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=model.config)

    n_total = len(dataset)
    n = min(n_total, limit) if limit else n_total
    print(f"[cache] extracting teacher <traj_future_start> hidden for {n}/{n_total} samples", flush=True)

    features: dict[str, torch.Tensor] = {}
    teacher_hidden_dim: int | None = None
    t_start = time.time()
    for idx in range(n):
        sample = dataset[idx]
        if sample is None:
            print(f"[cache] idx={idx}: sample is None, skipping", flush=True)
            continue
        key = dataset._sample_key(idx)
        batch = _to_device(collate([sample]), device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.extract_tfs_hidden(**batch)  # [1, H]
        vec = hidden[0].detach().float().cpu()
        teacher_hidden_dim = int(vec.shape[-1])
        features[key] = vec
        if (idx + 1) % 5 == 0 or idx + 1 == n:
            rate = (idx + 1) / (time.time() - t_start)
            print(f"[cache] {idx + 1}/{n}  ({rate:.2f} samples/s)  key={key}", flush=True)

    if not features:
        raise RuntimeError("No teacher features were extracted; check the dataset config.")

    metadata = {
        "format": "alpamayo_distill_teacher_v1",
        "teacher_hidden_dim": teacher_hidden_dim,
        "n_samples": len(features),
        "config_name": config_name,
        "teacher_target": cfg.model.get("_target_"),
        "teacher_checkpoint": cfg.model.get("checkpoint_path", cfg.model.get("vlm_name_or_path")),
    }
    save_teacher_cache(features, metadata, out_path)
    print(
        f"[cache] wrote {len(features)} features (dim={teacher_hidden_dim}) -> {out_path}",
        flush=True,
    )
    print(f"[cache] metadata: {metadata}", flush=True)
    print(
        f"[cache] set `model.teacher_hidden_dim={teacher_hidden_dim}` in the student config.",
        flush=True,
    )


if __name__ == "__main__":
    main()
