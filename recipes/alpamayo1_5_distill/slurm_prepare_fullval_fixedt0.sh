#!/bin/bash
#SBATCH --job-name=cm_fullval_nav
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/fullval_nav_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/fullval_nav_%j.err
set -euo pipefail
REPO=/home/achahe/alpamayo-recipes
PYTHON="$REPO/recipes/alpamayo1_5_sft/.venv/bin/python"
MANIFEST_DIR=/temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests
export PYTHONPATH="$REPO/recipes:$REPO/src"
if [[ -z "${RAW_ANNOTATIONS:-}" ]]; then
    WORK_DIR="/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/fullval_fixedt0_${SLURM_JOB_ID}"
    mkdir "$WORK_DIR"
    RAW_ANNOTATIONS="$WORK_DIR/raw.json"
    "$PYTHON" -m alpamayo1_5_distill.scripts.gen_nav_annotations \
    --clip-list "$MANIFEST_DIR/lcdrive_val_clip_uuids.txt" \
    --local-dir /temp/achahe/physical_ai_av/ --chunk-ids 0-3146 \
    --t0-us 5100000 --n-future 64 --time-step 0.1 --workers 8 \
    --out "$RAW_ANNOTATIONS"
fi
"$PYTHON" - "$RAW_ANNOTATIONS" "$MANIFEST_DIR" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from alpamayo1_5_distill.data.camera_subset import strip_turn_distance
from alpamayo1_5_distill.scripts.gen_nav_annotations import _avdi

raw_path, directory = map(Path, sys.argv[1:])
expected = set((directory / "lcdrive_val_clip_uuids.txt").read_text().split())
index = pd.read_parquet("/temp/achahe/physical_ai_av/clip_index.parquet")
excluded = expected - set(index.index.astype(str))
if len(expected) != 23758 or len(excluded) != 427:
    raise RuntimeError("Validation/index coverage changed; review the exclusion set before proceeding")
expected -= excluded
rows = json.loads(raw_path.read_text())
counts = Counter(row["clip_id"] for row in rows)
if len(expected) != 23331 or set(counts) != expected or any(count != 1 for count in counts.values()):
    raise RuntimeError(f"Incomplete available validation coverage: {len(rows)} rows, {len(expected - set(counts))} additional missing clips")
available = set(_avdi("/temp/achahe/physical_ai_av/", "0-3146").get_all_clip_ids())
if expected - available:
    raise RuntimeError(f"{len(expected - available)} validation clips unavailable in configured chunks")
for row in rows:
    if row["t0_relative"] != 5100000:
        raise RuntimeError("Non-default timestamp in full validation manifest")
    row["nav_text"] = strip_turn_distance(row["nav_text"])
    if row["nav_text"] not in {"Continue straight", "Turn left", "Turn right", "Reverse"}:
        raise RuntimeError(f"Unexpected navigation instruction: {row['nav_text']!r}")
rows.sort(key=lambda row: row["clip_id"])
destination = directory / "nav_lcdrive_val_available_fixedt0_23331_stripped.json"
if destination.exists():
    if json.loads(destination.read_text()) != rows:
        raise RuntimeError(f"Refusing to replace different manifest: {destination}")
else:
    temporary = destination.with_suffix(f".{raw_path.parent.name}.tmp")
    temporary.write_text(json.dumps(rows, indent=2) + "\n")
    temporary.rename(destination)
for filename, clip_ids in (
    ("lcdrive_val_available_23331_clip_uuids.txt", expected),
    ("lcdrive_val_excluded_missing_index_427_clip_uuids.txt", excluded),
):
    path = directory / filename
    content = "\n".join(sorted(clip_ids)) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"Refusing to replace different clip list: {path}")
    else:
        path.write_text(content)
print(f"Validated {len(rows)} clips, t0=5.1s, stripped navigation -> {destination}")
print(f"Excluded exactly {len(excluded)} clips absent from local index; no other clips dropped")
print(dict(Counter(row["nav_text"] for row in rows)))
PY