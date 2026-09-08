# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit the layer-mix result tables as markdown, straight from the per-clip eval JSONs.

MIX_PRUNING.md quotes numbers from four training arms across five epochs on two metrics.
Transcribing those by hand is how a table drifts from the files it claims to summarise, so
the tables are GENERATED and the doc is regenerated whenever an eval lands:

    python -m alpamayo1_5_distill.scripts.mix_results_table > /tmp/tables.md

⚠️ Every comparison is PAIRED on clip id. The arms share the 1k event-anchored nav val subset
and a per-batch diffusion seed, so the per-clip difference removes clip difficulty -- which is
most of the variance. Comparing two means with their separate SEMs understates significance
by a large factor and is not what this design supports.
"""

from __future__ import annotations

import glob
import json
import re

import numpy as np

TRAIN = "/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training"
STEPS_PER_EPOCH = 1563

#: label -> per-clip result glob. Order is the reading order of every table below.
ARMS = {
    "pruned (28-layer expert)":
        f"{TRAIN}/stitch_2b_nav2bmix_m7w1.0_e5_clean_maskfix_checkpoint-*_cam13_nav*.json",
    "mix 10x (P fixed)":
        f"{TRAIN}/stitch_2b_mix2bnav_m9w1.0_checkpoint-*_cam13_nav.json",
    "mix 100x (P learned)":
        f"{TRAIN}/stitch_2b_mix2bnav_m9w1.0_plr100.0_checkpoint-*_cam13_nav.json",
    "pinned 100x (ends fixed)":
        f"{TRAIN}/stitch_2b_mixpin2bnav_m9w1.0_plr100.0_checkpoint-*_cam13_nav.json",
}


def _load(path: str) -> dict:
    d = json.load(open(path))
    rows = d if isinstance(d, list) else d.get("per_clip", d)
    if isinstance(rows, dict):
        rows = list(rows.values())
    return {r["clip_id"]: r for r in rows
            if isinstance(r, dict) and "min_ade" in r and "clip_id" in r}


def _arm(pattern: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for f in sorted(glob.glob(pattern)):
        out.setdefault(int(re.search(r"checkpoint-(\d+)", f).group(1)), _load(f))
    return out


def paired(a: dict, b: dict, key: str):
    """Mean paired difference a - b, its SEM, z, and the fraction of clips where a wins."""
    ids = sorted(set(a) & set(b))
    if len(ids) < 2:
        return None
    d = np.array([a[i][key] - b[i][key] for i in ids], dtype=np.float64)
    sem = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), sem, (d.mean() / sem if sem else float("inf")), float((d < 0).mean()), len(d)


def main() -> None:
    arms = {k: _arm(v) for k, v in ARMS.items()}
    epochs = range(1, 6)

    for key in ("min_ade", "ade"):
        print(f"\n### {key}\n")
        print("| arm | " + " | ".join(f"epoch {e}" for e in epochs) + " | best |")
        print("|---" * (len(list(epochs)) + 2) + "|")
        for nm, a in arms.items():
            cells, vals = [], []
            for e in epochs:
                st = e * STEPS_PER_EPOCH
                if st in a:
                    v = float(np.mean([r[key] for r in a[st].values()]))
                    vals.append(v)
                    cells.append(f"{v:.4f}")
                else:
                    cells.append("—")
            best = f"**{min(vals):.4f}**" if vals else "—"
            print(f"| {nm} | " + " | ".join(cells) + f" | {best} |")

    ref_pairs = [("mix 10x (P fixed)", "pruned (28-layer expert)"),
                 ("mix 100x (P learned)", "mix 10x (P fixed)"),
                 ("pinned 100x (ends fixed)", "mix 100x (P learned)")]
    for key in ("min_ade", "ade"):
        print(f"\n### paired deltas, {key} (negative = first arm better)\n")
        print("| comparison | " + " | ".join(f"epoch {e}" for e in epochs) + " |")
        print("|---" * (len(list(epochs)) + 1) + "|")
        for x, y in ref_pairs:
            cells = []
            for e in epochs:
                st = e * STEPS_PER_EPOCH
                r = (paired(arms[x][st], arms[y][st], key)
                     if st in arms[x] and st in arms[y] else None)
                cells.append(f"{r[0]:+.4f} (z {r[2]:+.2f})" if r else "—")
            print(f"| {x} vs {y} | " + " | ".join(cells) + " |")

    # best-vs-best, the headline
    print("\n### best checkpoint vs best checkpoint\n")
    print("| comparison | metric | a | b | delta | z | a wins on | n |")
    print("|---|---|---|---|---|---|---|---|")
    for x, y in [("mix 10x (P fixed)", "pruned (28-layer expert)")]:
        for key in ("min_ade", "ade"):
            bx = min(arms[x], key=lambda s: np.mean([r[key] for r in arms[x][s].values()]))
            by = min(arms[y], key=lambda s: np.mean([r[key] for r in arms[y][s].values()]))
            r = paired(arms[x][bx], arms[y][by], key)
            av = np.mean([v[key] for v in arms[x][bx].values()])
            bv = np.mean([v[key] for v in arms[y][by].values()])
            print(f"| {x} e{bx // STEPS_PER_EPOCH} vs {y} e{by // STEPS_PER_EPOCH} | {key} | "
                  f"{av:.4f} | {bv:.4f} | {r[0]:+.4f} | {r[2]:+.2f} | {100 * r[3]:.1f}% | {r[4]} |")


if __name__ == "__main__":
    main()
