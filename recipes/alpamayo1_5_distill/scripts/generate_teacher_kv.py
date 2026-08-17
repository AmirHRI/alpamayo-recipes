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

"""Offline teacher **KV-cache** builder for KAVA distillation.

Runs the frozen 10B teacher once per clip and records, per sample:

* ``full/`` — the uncompressed per-layer CoT cache (**pre-RoPE** keys and values)
  plus both R-KV scores, so any ``(M, lambda, method)`` tier can be re-derived later
  with ``compress_teacher_kv.py`` and no teacher re-run,
* ``compressed_<tag>/`` — the tier training actually reads, and
* ``cot_text.shard<N>.jsonl`` — the teacher's reasoning as text, which is also the
  cheapest early warning that the prompt is wrong (see below).

Running the teacher offline rather than in the training loop is not a small win: the
CoT is generated autoregressively, so an online teacher would dominate step time by
an order of magnitude, the 10B's ~21 GB of weights would crowd out the student on the
same card, and every sweep point over ``M``/``T``/``lambda`` re-pays the cost. Offline
it is paid once — valid here because the PAI path is fully deterministic (no
augmentation, no random frame sampling, constant ``DEFAULT_T0_US``) and generation is
pinned greedy.

⚠️ The teacher only reasons if the assistant turn **ends** at ``<|cot_start|>``, which
means ``cot`` must be **last** in the processor's ``components_order``
(``vla_processor=distill_teacher_generate``).  With a ``traj_future``-last prompt the
``<|traj_future_start|>`` token is pre-filled and the teacher emits an *empty* CoT;
this script raises on an empty span rather than caching zero-length targets.

Sharding + resume: ``num_shards``/``shard`` take the modulo slice, and because every
sample is its own file, resume is a plain existence check — N shards on N GPUs never
touch each other's data.

Usage::

    python -m alpamayo1_5_distill.scripts.generate_teacher_kv \
        config=cache_teacher_kv_lcdrive teacher=teacher_ar1_5_10b \
        cache_root=/temp/achahe/.../teacher_kv_lcdrive \
        m=16 lam=0.1 eviction=rkv mode=generate \
        num_shards=8 shard=0
"""

import hydra.utils as hyu
import torch

from alpamayo1_5_distill.data import kv_cache_io
from alpamayo1_5_distill.models.kv_distill import evict_teacher_cache
from alpamayo1_5_distill.models.teacher_kv import extract_teacher_kv
from alpamayo1_5_distill.scripts import cache_common


def main() -> None:
    argv = cache_common.main_argv()
    config_name = argv.get("config", "cache_teacher_kv_lcdrive")
    cache_root = argv.get("cache_root")
    if not cache_root:
        raise SystemExit("cache_root=<dir> is required (put it on /data, not /home)")

    mode = argv.get("mode", "generate")  # "generate" | "teacher_force"
    importance_source = argv.get("importance_source", "vlm_post_cot")
    budget = int(argv.get("m", 16))
    lam = float(argv.get("lam", 0.1))
    eviction = argv.get("eviction", "rkv")
    limit = int(argv.get("limit", 0))  # 0 == all
    num_shards = int(argv.get("num_shards", 1))
    shard = int(argv.get("shard", 0))
    num_workers = int(argv.get("num_workers", 8))
    log_every = int(argv.get("log_every", 50))
    log_texts = int(argv.get("log_texts", 3))
    max_new_tokens = int(argv["max_new_tokens"]) if "max_new_tokens" in argv else None
    do_sample = argv.get("do_sample", "false").lower() in ("1", "true", "yes")
    write_full = argv.get("write_full", "true").lower() in ("1", "true", "yes")
    store_redundancy = argv.get("store_redundancy", "true").lower() in ("1", "true", "yes")
    expert_timesteps = (
        tuple(float(v) for v in argv["expert_timesteps"].split(","))
        if "expert_timesteps" in argv
        else None
    )
    expert_noise_seed = int(argv.get("expert_noise_seed", 0))

    tag = kv_cache_io.compressed_tag(budget, lam, eviction)
    cfg = cache_common.compose_config(config_name, argv)
    device = torch.device("cuda")

    print(
        f"[kv-cache] mode={mode} importance={importance_source} tier={tag} "
        f"shard={shard}/{num_shards} full={write_full} root={cache_root}",
        flush=True,
    )

    model = cache_common.build_teacher(cfg, device)
    dataset = hyu.instantiate(
        cfg.data.cache_dataset, _convert_="partial", model_config=model.config
    )
    processor = cache_common.build_processor(model)

    teacher_layers = len(model.vlm.model.language_model.layers)
    text_cfg = getattr(model.vlm.config, "text_config", model.vlm.config)

    def is_done(index: int) -> bool:
        key = dataset._sample_key(index)
        if not kv_cache_io.has_entry(cache_root, tag, key):
            return False
        return not write_full or kv_cache_io.has_entry(cache_root, "full", key)

    owned = cache_common.shard_indices(len(dataset), num_shards, shard)
    todo = [i for i in owned if not is_done(i)]
    if limit:
        todo = todo[:limit]
    print(
        f"[kv-cache] dataset={len(dataset)} shard-assigned={len(owned)} to-do={len(todo)} "
        f"(workers={num_workers}, teacher_layers={teacher_layers})",
        flush=True,
    )

    provenance = {
        "format": kv_cache_io.FORMAT_FULL,
        "config_name": config_name,
        "mode": mode,
        "importance_source": importance_source,
        "expert_timesteps": list(expert_timesteps) if expert_timesteps else None,
        "expert_noise_seed": expert_noise_seed if importance_source == "expert" else None,
        "eviction": eviction,
        "lam": lam,
        "m": budget,
        "teacher_layers": teacher_layers,
        "teacher_kv_heads": int(text_cfg.num_key_value_heads),
        "teacher_head_dim": int(text_cfg.head_dim),
        "teacher_hidden_dim": int(text_cfg.hidden_size),
        "teacher_target": cfg.model.get("_target_"),
        "teacher_checkpoint": cfg.model.get("checkpoint_path", cfg.model.get("vlm_name_or_path")),
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
    }

    n_cot_index: dict[str, int] = {}
    reporter = cache_common.RateReporter(len(todo), label="kv-cache")
    loader = cache_common.sample_loader(dataset, todo, processor, num_workers=num_workers)
    done = 0
    printed = 0

    with kv_cache_io.CotTextLog(cache_root, shard=shard) as cot_log:
        for scanned, (idx, batch) in enumerate(loader, start=1):
            if batch is None:
                continue
            key = dataset._sample_key(idx)
            batch = cache_common.to_device(batch, device)

            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    sample = extract_teacher_kv(
                        model,
                        mode=mode,
                        importance_source=importance_source,
                        store_redundancy=store_redundancy,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        expert_timesteps=expert_timesteps,
                        expert_noise_seed=expert_noise_seed,
                        **batch,
                    )
            except (ValueError, RuntimeError) as ex:
                print(f"[kv-cache] idx={idx} key={key} capture failed: {ex}", flush=True)
                continue

            if printed < log_texts:
                print(
                    f"[kv-cache] idx={idx} N_C={sample.n_cot} CoT: {sample.cot_text!r}",
                    flush=True,
                )
                printed += 1

            if write_full:
                kv_cache_io.save_full_entry(
                    cache_root,
                    key,
                    sample.k_pre,
                    sample.v,
                    imp=sample.imp,
                    red=sample.red,
                    tfs_hidden=sample.tfs_hidden,
                    tfs_hidden_all=sample.tfs_hidden_all,
                    metadata=provenance,
                )

            k_sel, v_sel, sel_idx = evict_teacher_cache(
                sample.k_pre.float(),
                sample.v.float(),
                m=budget,
                n_valid=sample.n_cot,
                imp=sample.imp,
                red=sample.red,
                lam=lam,
                method=eviction,
            )
            kv_cache_io.save_compressed_entry(
                cache_root,
                tag,
                key,
                k_sel,
                v_sel,
                sel_idx=sel_idx,
                n_valid=k_sel.shape[-2],
                tfs_hidden=sample.tfs_hidden,
                tfs_hidden_all=sample.tfs_hidden_all,
                metadata={**provenance, "format": kv_cache_io.FORMAT_COMPRESSED},
            )

            cot_log.append(key, sample.cot_text, sample.n_cot)
            n_cot_index[key] = sample.n_cot
            done += 1
            if done % log_every == 0:
                kv_cache_io.write_index(cache_root, n_cot_index, provenance, shard=shard)
                reporter.report(done, scanned, "[index written]")

    kv_cache_io.write_index(cache_root, n_cot_index, provenance, shard=shard)
    _report_stats(cache_root, tag, write_full, n_cot_index, len(dataset))


def _report_stats(
    cache_root: str, tag: str, write_full: bool, n_cot: dict[str, int], dataset_size: int
) -> None:
    """Print per-tier size and extrapolate the full run — the pilot's whole purpose.

    ``N_C`` is what decides the disk bill, and it is the one number that cannot be
    predicted from configs: the teacher's CoT length is capped at ``max_new_tokens``
    but its typical value is an empirical property of the checkpoint and the scene.
    """
    print("", flush=True)
    for tier in (["full"] if write_full else []) + [tag]:
        stats = kv_cache_io.tier_stats(cache_root, tier)
        projected = stats["mb_per_entry"] * dataset_size / 1024
        print(
            f"[kv-cache] {tier}: {stats['n_entries']} entries, "
            f"{stats['mb_per_entry']:.2f} MB/entry -> {projected:.1f} GB for "
            f"{dataset_size} clips",
            flush=True,
        )
    if n_cot:
        lengths = sorted(n_cot.values())
        mid = lengths[len(lengths) // 2]
        print(
            f"[kv-cache] N_C over {len(lengths)} samples: min={lengths[0]} "
            f"median={mid} max={lengths[-1]}  (mean={sum(lengths) / len(lengths):.1f})",
            flush=True,
        )
        print(
            "[kv-cache] set the student config's kava.num_slots and the tier name to "
            "match; slots beyond N_C are masked out of L_KV.",
            flush=True,
        )


if __name__ == "__main__":
    main()
