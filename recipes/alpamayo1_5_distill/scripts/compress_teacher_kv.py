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

"""Derive a new compressed tier from the ``full/`` tier — **no teacher, no GPU**.

This is the reason the uncompressed CoT cache is stored at all.  Every point in the
``(M, lambda, eviction method)`` sweep is a different eviction of the *same* teacher
forward, so re-tuning any of them costs a few minutes of CPU here instead of a ~10 h
re-run of the 10B.

Usage::

    python -m alpamayo1_5_distill.scripts.compress_teacher_kv \
        cache_root=/data/achahe/.../teacher_kv_lcdrive m=32 lam=0.1 eviction=rkv

    # the paper's eviction ablations, all from one cache:
    ... m=16 eviction=cosine     # diversity only  (lambda = 0)
    ... m=16 eviction=attn       # importance only (lambda = 1)
    ... m=16 eviction=crop       # keep the first M, the naive baseline
"""

import sys
import time

import torch

from alpamayo1_5_distill.data import kv_cache_io
from alpamayo1_5_distill.models.kv_distill import evict_teacher_cache
from alpamayo1_5_distill.scripts.cache_common import parse_argv


def main() -> None:
    argv = parse_argv(sys.argv[1:])
    cache_root = argv.get("cache_root")
    if not cache_root:
        raise SystemExit("cache_root=<dir> is required")
    budget = int(argv.get("m", 16))
    lam = float(argv.get("lam", 0.1))
    eviction = argv.get("eviction", "rkv")
    limit = int(argv.get("limit", 0))
    num_shards = int(argv.get("num_shards", 1))
    shard = int(argv.get("shard", 0))
    overwrite = argv.get("overwrite", "false").lower() in ("1", "true", "yes")

    tag = kv_cache_io.compressed_tag(budget, lam, eviction)
    index = kv_cache_io.read_index(cache_root)
    provenance = dict(index.get("metadata", {}))
    provenance.update(
        {
            "format": kv_cache_io.FORMAT_COMPRESSED,
            "m": budget,
            "lam": lam,
            "eviction": eviction,
            "derived_from": "full",
        }
    )

    keys = sorted(kv_cache_io.iter_keys(cache_root, "full"))
    if not keys:
        raise SystemExit(
            f"no 'full' tier under {cache_root}. Run generate_teacher_kv.py with "
            "write_full=true first (it is the default)."
        )
    keys = [k for i, k in enumerate(keys) if i % num_shards == shard]
    if not overwrite:
        keys = [k for k in keys if not kv_cache_io.has_entry(cache_root, tag, k)]
    if limit:
        keys = keys[:limit]

    print(
        f"[compress] tier={tag} from full/ ({len(keys)} to write, shard {shard}/{num_shards})",
        flush=True,
    )

    # 'cosine' and 'crop' never read the importance score, so skip those pages
    # entirely — it is ~150 KiB per sample of pure I/O otherwise.
    needs_imp = eviction in ("rkv", "attn") and lam > 0.0
    names = ("k_pre", "v", "red", "tfs_hidden", "tfs_hidden_all") + (
        ("imp",) if needs_imp else ()
    )

    started = time.time()
    n_cot_index: dict[str, int] = {}
    for count, key in enumerate(keys, start=1):
        entry = kv_cache_io.load_entry(cache_root, "full", key, names=names)
        k_pre = entry["k_pre"].float()
        n_cot = int(index.get("n_cot", {}).get(key, k_pre.shape[-2]))

        k_sel, v_sel, sel_idx = evict_teacher_cache(
            k_pre,
            entry["v"].float(),
            m=budget,
            n_valid=n_cot,
            imp=entry.get("imp"),
            red=entry.get("red"),
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
            tfs_hidden=entry.get("tfs_hidden"),
            tfs_hidden_all=entry.get("tfs_hidden_all"),
            metadata=provenance,
        )
        n_cot_index[key] = n_cot
        if count % 200 == 0:
            rate = count / max(time.time() - started, 1e-9)
            print(f"[compress] {count}/{len(keys)}  {rate:.1f}/s", flush=True)

    stats = kv_cache_io.tier_stats(cache_root, tag)
    print(
        f"[compress] DONE {tag}: {stats['n_entries']} entries, "
        f"{stats['mb_per_entry']:.2f} MB/entry, {stats['bytes'] / 1024**3:.1f} GB total",
        flush=True,
    )
    print(
        "[compress] point the student config at this tier:\n"
        f"  data.train_dataset.kv_tier={tag}\n"
        f"  model.kava.num_slots={budget}",
        flush=True,
    )
    if torch.cuda.is_available():  # nothing here used the GPU; say so explicitly
        print("[compress] (ran entirely on CPU — no teacher, no GPU)", flush=True)


if __name__ == "__main__":
    main()
