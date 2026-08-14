#!/usr/bin/env python
"""Create a stratified subset of the LCDrive val scenario-category manifest.

Samples a fixed total number of clips (default: 2000) from
`lcdrive_val_primary_scenario_for_table2.csv`, keeping each
`scenario_category_paper` category's proportion roughly the same as in the
full val split. Only clips that are present in a given per-clip metrics JSON
(e.g. produced by `evaluate_hf.py`) are eligible, since some clips in the
manifest may be missing from the actual dataset on disk.

Usage:
    python scripts/make_lcdrive_val_subset.py \
        --category-csv /path/to/lcdrive_val_primary_scenario_for_table2.csv \
        --per-clip-json /path/to/lcdrive_val_per_clip_metrics.json \
        --out-csv /path/to/lcdrive_val_primary_scenario_mysubset.csv \
        --n-total 2000 \
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import math

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category-csv",
        type=str,
        default="/home/achahe/alpamayo-recipes/lcdrive_physicalai_av_manifests/"
        "lcdrive_val_primary_scenario_for_table2.csv",
        help="Full val scenario-category manifest CSV.",
    )
    parser.add_argument(
        "--per-clip-json",
        type=str,
        default="/data/01/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/training/"
        "output_stage1_cosmos2b_lcdrive_4gpu_10ep/lcdrive_val_per_clip_metrics.json",
        help="Per-clip metrics JSON (list of {'clip_id': ...}) used to check clip existence.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="/home/achahe/alpamayo-recipes/lcdrive_physicalai_av_manifests/"
        "lcdrive_val_primary_scenario_mysubset.csv",
        help="Output CSV path for the sampled subset (same schema as --category-csv).",
    )
    parser.add_argument("--n-total", type=int, default=2000, help="Total number of clips to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument(
        "--category-col",
        type=str,
        default="scenario_category_paper",
        help="Column to stratify sampling by.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.category_csv)
    n_manifest = len(df)
    print(f"Loaded manifest: {n_manifest} rows from {args.category_csv}")

    with open(args.per_clip_json, "r", encoding="utf-8") as f:
        per_clip_records = json.load(f)
    available_clip_ids = {rec["clip_id"] for rec in per_clip_records}
    print(f"Loaded per-clip metrics: {len(available_clip_ids)} unique clip_ids from {args.per_clip_json}")

    df_avail = df[df["clip_uuid"].isin(available_clip_ids)].copy()
    n_avail = len(df_avail)
    n_missing = n_manifest - n_avail
    print(f"Clips available in per-clip metrics: {n_avail} (missing from data: {n_missing})")

    if args.n_total > n_avail:
        raise ValueError(
            f"Requested n_total={args.n_total} exceeds available clips ({n_avail})."
        )

    # Stratified sampling: allocate n_total proportionally across categories,
    # using largest-remainder method so the counts sum exactly to n_total.
    counts = df_avail[args.category_col].value_counts()
    raw_alloc = counts / n_avail * args.n_total
    base_alloc = raw_alloc.apply(math.floor).astype(int)
    remainder = args.n_total - int(base_alloc.sum())

    # Distribute remaining slots to categories with the largest fractional part.
    fractional = (raw_alloc - base_alloc).sort_values(ascending=False)
    alloc = base_alloc.copy()
    for cat in fractional.index[:remainder]:
        alloc[cat] += 1

    assert int(alloc.sum()) == args.n_total, (int(alloc.sum()), args.n_total)

    sampled_parts = []
    rng_state = args.seed
    for cat, n_sample in alloc.items():
        cat_df = df_avail[df_avail[args.category_col] == cat]
        n_sample = min(n_sample, len(cat_df))
        sampled_parts.append(cat_df.sample(n=n_sample, random_state=rng_state))
        rng_state += 1

    subset = pd.concat(sampled_parts, ignore_index=True)
    subset = subset.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    subset.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(subset)} clips to {args.out_csv}")

    print("\nPer-category counts (subset vs full-available, ratio):")
    full_counts = df_avail[args.category_col].value_counts()
    subset_counts = subset[args.category_col].value_counts()
    for cat in full_counts.index:
        full_n = full_counts.get(cat, 0)
        sub_n = subset_counts.get(cat, 0)
        full_ratio = full_n / n_avail
        sub_ratio = sub_n / len(subset)
        print(
            f"  {cat:<35s} full={full_n:6d} ({full_ratio:6.2%})  "
            f"subset={sub_n:5d} ({sub_ratio:6.2%})"
        )


if __name__ == "__main__":
    main()
