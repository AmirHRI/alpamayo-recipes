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

Runs the frozen teacher (default: Alpamayo-1.5-10B) over the dataset and caches
its last-layer hidden state at ``<traj_future_start>`` — the vector the action
expert conditions on — keyed by ``(clip_id, t0_us)``. The student later matches
this via ``DistillReasoningVLA``'s latent loss, with no teacher in the loop.

Two extraction modes:

* ``mode=generate`` (default, recommended): the teacher **generates its own CoT**
  (masking the trajectory vocab, stopping at ``<traj_future_start>``) exactly as
  it does at deployment, so the cached hidden is reasoning-conditioned WITHOUT
  any ground-truth CoT. Works on any clip. Needs a ``generation_mode=true``
  dataset with a CoT-free processor (the teacher produces the CoT itself).
* ``mode=teacher_force``: a single forward with ground-truth CoT teacher-forced
  in context (needs a ``generation_mode=false`` dataset + a ``cot``-in-order
  processor, e.g. ``distill_teacher``). Use only where GT CoT exists.

Sharding + resume (for the LCDrive-scale run): ``num_shards``/``shard`` process
the modulo slice of the dataset; an existing ``out`` file is loaded and its keys
skipped, and the cache is re-written every ``save_every`` samples, so the job
survives interruption and can be run as N parallel shards on N GPUs.

Usage::

    python -m alpamayo1_5_distill.scripts.generate_teacher_features \
        config=cache_teacher_features_lcdrive teacher=teacher_ar1_5_10b \
        mode=generate out=.../teacher_lcdrive.shard0.safetensors \
        num_shards=4 shard=0
"""

import os
import sys
import time

import hydra.utils as hyu
import torch
from hydra import compose, initialize_config_dir
from safetensors.torch import load_file as _load_safetensors

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
    limit = int(argv.get("limit", 0))  # 0 == all
    mode = argv.get("mode", "generate")  # "generate" | "teacher_force"
    num_shards = int(argv.get("num_shards", 1))
    shard = int(argv.get("shard", 0))
    num_workers = int(argv.get("num_workers", 8))  # dataloader prefetch (frame decode)
    save_every = int(argv.get("save_every", 200))
    log_texts = int(argv.get("log_texts", 3))  # print this many generated CoTs
    max_new_tokens = int(argv["max_new_tokens"]) if "max_new_tokens" in argv else None
    do_sample = argv.get("do_sample", "false").lower() in ("1", "true", "yes")

    overrides = [f"{k}={v}" for k, v in argv.items() if ("." in k or "@" in k)]
    if "teacher" in argv:
        overrides.append(f"models@model={argv['teacher']}")

    cfg_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")
    )
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    device = torch.device("cuda")

    print(f"[cache] mode={mode} shard={shard}/{num_shards} out={out_path}", flush=True)
    print("[cache] instantiating teacher ...", flush=True)
    t0 = time.time()
    # The teacher's 8B VLM skeleton is random-initialised on CPU by
    # Qwen3VLForConditionalGeneration(config) and then FULLY overwritten by the
    # checkpoint (load_alpamayo1_vlm, assign=True). Skipping that throwaway init
    # cuts the load from ~200s to well under a minute.
    from transformers.modeling_utils import no_init_weights

    with no_init_weights():
        model = hyu.instantiate(cfg.model, _convert_="partial")
    model = model.to(device).eval()
    model.requires_grad_(False)
    print(f"[cache] teacher built in {time.time() - t0:.1f}s", flush=True)

    dataset = hyu.instantiate(
        cfg.data.cache_dataset, _convert_="partial", model_config=model.config
    )

    # Build ONE reusable processor for collation. collate_fn_from_model_config
    # rebuilds a QwenProcessor (AutoProcessor.from_pretrained + add 4000 tokens)
    # on every call — a few seconds per sample. Building it once here removes that.
    from alpamayo.processor.qwen_processor import QwenProcessor

    _qp = QwenProcessor(
        vlm_name_or_path=model.config.vlm_name_or_path,
        traj_vocab_size=model.config.traj_vocab_size,
        min_pixels=model.config.min_pixels,
        max_pixels=model.config.max_pixels,
        chat_template_version="r1_5",
    )
    _qp.build_processor()

    # Resume: load any partial cache for this shard and skip its keys.
    features: dict[str, torch.Tensor] = {}
    if os.path.exists(out_path):
        features = dict(_load_safetensors(out_path))
        print(f"[cache] resuming: {len(features)} features already present in {out_path}", flush=True)

    shard_idx = [i for i in range(len(dataset)) if i % num_shards == shard]
    todo = [i for i in shard_idx if dataset._sample_key(i) not in features]
    if limit:
        todo = todo[:limit]
    print(
        f"[cache] dataset={len(dataset)} shard-assigned={len(shard_idx)} to-do={len(todo)} "
        f"(workers={num_workers})",
        flush=True,
    )

    # DataLoader prefetches + decodes camera frames in parallel workers so disk/CPU
    # overlaps the GPU. Workers only touch CPU (frame load + tokenize); the GPU
    # forward stays in the main process. Each item is (idx, collated-batch-of-1).
    from torch.utils.data import DataLoader

    class _IdxDataset:
        def __init__(self, base, idxs):
            self.base = base
            self.idxs = idxs

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, j):
            i = self.idxs[j]
            try:
                s = self.base[i]
            except Exception as ex:  # a bad clip shouldn't kill the whole run
                print(f"[cache] idx={i} load error: {ex}", flush=True)
                s = None
            return i, s

    def _collate_one(items):
        i, s = items[0]
        return (i, None) if s is None else (i, _qp.collate_fn([s]))

    loader = DataLoader(
        _IdxDataset(dataset, todo),
        batch_size=1,
        num_workers=num_workers,
        collate_fn=_collate_one,
        prefetch_factor=4 if num_workers else None,
    )

    teacher_hidden_dim: int | None = None
    done = 0
    printed = 0
    t_start = time.time()
    for count, (idx, batch) in enumerate(loader):
        if batch is None:
            continue
        key = dataset._sample_key(idx)
        batch = _to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if mode == "generate":
                want_text = printed < log_texts
                res = model.extract_tfs_hidden_generated(
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    return_text=want_text,
                    **batch,
                )
                if want_text:
                    hidden, text = res
                    print(f"[cache] idx={idx} generated CoT: {text!r}", flush=True)
                    printed += 1
                else:
                    hidden = res
            else:
                hidden = model.extract_tfs_hidden(**batch)
        vec = hidden[0].detach().float().cpu()
        teacher_hidden_dim = int(vec.shape[-1])
        features[key] = vec
        done += 1
        if done % save_every == 0:
            _save(features, teacher_hidden_dim, config_name, mode, cfg, out_path)
            rate = done / (time.time() - t_start)
            remaining = (len(todo) - count - 1) / rate if rate else 0
            print(
                f"[cache] {done} new ({count + 1}/{len(todo)} scanned)  "
                f"{rate:.2f}/s  ~{remaining / 3600:.1f}h left  [checkpointed]",
                flush=True,
            )

    if teacher_hidden_dim is None and features:
        teacher_hidden_dim = int(next(iter(features.values())).shape[-1])
    _save(features, teacher_hidden_dim, config_name, mode, cfg, out_path)
    print(
        f"[cache] DONE: {len(features)} features (dim={teacher_hidden_dim}) -> {out_path}",
        flush=True,
    )
    if teacher_hidden_dim:
        print(f"[cache] set `model.teacher_hidden_dim={teacher_hidden_dim}` in the student config.", flush=True)


def _save(features, teacher_hidden_dim, config_name, mode, cfg, out_path):
    metadata = {
        "format": "alpamayo_distill_teacher_v1",
        "teacher_hidden_dim": teacher_hidden_dim,
        "n_samples": len(features),
        "config_name": config_name,
        "mode": mode,
        "teacher_target": cfg.model.get("_target_"),
        "teacher_checkpoint": cfg.model.get("checkpoint_path", cfg.model.get("vlm_name_or_path")),
    }
    save_teacher_cache(features, metadata, out_path)


if __name__ == "__main__":
    main()
