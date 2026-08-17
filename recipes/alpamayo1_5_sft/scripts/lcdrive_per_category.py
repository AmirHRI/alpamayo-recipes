#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Group LCDrive per-clip eval metrics by scenario category (paper Table 2 style).

Joins the per-clip metrics dumped by ``evaluate_hf.py``
(``lcdrive_val_per_clip_metrics.json``) with the LCDrive validation scenario
manifest (``lcdrive_val_primary_scenario_for_table2.csv``), then reports the
mean of every metric per ``scenario_category_paper`` plus an overall row.

Example:
    python scripts/lcdrive_per_category.py \
        --per-clip /data/.../output_stage1_cosmos2b_lcdrive/lcdrive_val_per_clip_metrics.json \
        --category-csv /temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests/lcdrive_val_primary_scenario_for_table2.csv \
        --out-csv /data/.../output_stage1_cosmos2b_lcdrive/lcdrive_val_by_category.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict


def load_categories(csv_path: str) -> dict[str, str]:
    """clip_uuid -> scenario_category_paper."""
    mapping: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uuid = row.get("clip_uuid") or row.get("clip_id")
            cat = row.get("scenario_category_paper") or row.get("scenario_category")
            if uuid and cat:
                mapping[uuid.strip()] = cat.strip()
    return mapping


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-clip", required=True, help="per-clip metrics JSON from evaluate_hf.py")
    p.add_argument("--category-csv", required=True, help="lcdrive_val_primary_scenario_for_table2.csv")
    p.add_argument("--out-csv", default=None, help="optional path to write the per-category table")
    p.add_argument(
        "--metric",
        default="min_ade",
        help="metric to sort the printed table by (default: min_ade)",
    )
    args = p.parse_args()

    with open(args.per_clip, encoding="utf-8") as f:
        records = json.load(f)
    if not records:
        raise SystemExit(f"No records in {args.per_clip}")

    cat_map = load_categories(args.category_csv)

    # Discover metric keys (everything except clip_id).
    metric_keys = [k for k in records[0].keys() if k != "clip_id"]

    # Accumulate sums/counts per category (+ overall) per metric.
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    clip_counts: dict[str, int] = defaultdict(int)
    n_unmatched = 0

    for rec in records:
        cid = rec["clip_id"]
        cat = cat_map.get(cid)
        if cat is None:
            n_unmatched += 1
            cat = "<uncategorized>"
        clip_counts[cat] += 1
        clip_counts["ALL"] += 1
        for mk in metric_keys:
            v = rec.get(mk)
            if v is None:
                continue
            sums[cat][mk] += float(v)
            counts[cat][mk] += 1
            sums["ALL"][mk] += float(v)
            counts["ALL"][mk] += 1

    def mean(cat: str, mk: str) -> float:
        c = counts[cat][mk]
        return sums[cat][mk] / c if c else float("nan")

    # Order: categories by chosen metric (desc), ALL last.
    cats = [c for c in clip_counts if c != "ALL"]
    cats.sort(key=lambda c: mean(c, args.metric), reverse=True)
    ordered = cats + ["ALL"]

    # Print table.
    col_w = max(28, *(len(c) for c in ordered))
    header = f"{'category':<{col_w}} {'n':>7}  " + "  ".join(f"{mk:>16}" for mk in metric_keys)
    print(header)
    print("-" * len(header))
    for c in ordered:
        row = f"{c:<{col_w}} {clip_counts[c]:>7}  " + "  ".join(
            f"{mean(c, mk):>16.4f}" for mk in metric_keys
        )
        print(row)
    if n_unmatched:
        print(f"\n[warn] {n_unmatched} clips had no category match (shown as <uncategorized>).")

    # Optional CSV.
    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["category", "n_clips", *metric_keys])
            for c in ordered:
                w.writerow([c, clip_counts[c], *(f"{mean(c, mk):.6f}" for mk in metric_keys)])
        print(f"\nWrote per-category table to {args.out_csv}")


if __name__ == "__main__":
    main()
