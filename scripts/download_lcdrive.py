"""Download only the chunks containing the LCDrive subset clips.

Companion to NVIDIA's scripts/download_pai.py. Same sensor args, plus:
  --lcdrive-manifest  path to a CSV with a clip_uuid column (or a .txt of UUIDs)

It maps the manifest's clips -> chunk IDs via clip_index.parquet, checks each chunk
actually has files in the repo, warns + skips any that don't, then downloads the rest.

Place this file in the same scripts/ folder as download_pai.py (or add that folder
to sys.path) so the import below resolves.

Example (your sensor set):
  python scripts/download_lcdrive.py \
    --lcdrive-manifest ./lcdrive_manifest/lcdrive_train_val_split_clip_uuids.csv \
    --camera camera_front_wide_120fov camera_cross_left_120fov camera_cross_right_120fov camera_front_tele_30fov \
    --calibration camera_intrinsics sensor_extrinsics \
    --labels egomotion \
    --output-dir /data/datasets/physical_ai_av
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, snapshot_download

# import sys; sys.path.append("/path/to/physical_ai_av/scripts")  # if not co-located
from download_pai import build_allow_patterns, parse_component_subparts, DEFAULT_REPO_ID

CHUNK_RE = re.compile(r"\.chunk_(\d{4})\.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download LCDrive-subset chunks from PhysicalAI-AV.")
    p.add_argument("--lcdrive-manifest", type=Path, required=True, dest="lcdrive_manifest",
                   help="CSV with a clip_uuid column, or a .txt list of UUIDs.")
    p.add_argument("--camera", nargs="+", default=None)
    p.add_argument("--calibration", nargs="+", default=None)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--lidar", nargs="+", default=None)
    p.add_argument("--radar", nargs="+", default=None)
    p.add_argument("--reasoning", nargs="+", default=None)
    p.add_argument("--output-dir", type=Path, default=Path("nvidia/PhysicalAI-Autonomous-Vehicles"),
                   dest="output_dir")
    p.add_argument("--revision", default=None,
                   help="Dataset revision/commit SHA (e.g. the 310,895-row one). Default: main.")
    return p.parse_args()


def read_manifest_uuids(path: Path) -> list[str]:
    if path.suffix.lower() == ".txt":
        return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    df = pd.read_csv(path)
    col = next((c for c in df.columns if "uuid" in c.lower() or c.lower() == "clip_id"),
               df.columns[0])
    return df[col].astype(str).tolist()


def manifest_to_chunks(output_dir: Path, uuids: list[str], revision: str | None) -> list[int]:
    # Ensure clip_index is present (small), then map clips -> chunks.
    snapshot_download(repo_id=DEFAULT_REPO_ID, repo_type="dataset",
                      local_dir=str(output_dir), local_dir_use_symlinks=False,
                      allow_patterns=["clip_index.parquet"], revision=revision)
    ci = pd.read_parquet(output_dir / "clip_index.parquet")
    ci.index = ci.index.astype(str)

    want = pd.Index(pd.unique(pd.Series(uuids, dtype=str)))
    have = ci.index.intersection(want)
    missing = want.difference(ci.index)
    if len(missing):
        print(f"[warn] {len(missing)} manifest clip(s) absent from clip_index "
              f"(withdrawn or wrong revision); skipping. e.g. {list(missing[:3])}")
    chunks = sorted(int(c) for c in ci.loc[have, "chunk"].unique())
    print(f"[manifest] {len(have)} clip(s) -> {len(chunks)} chunk(s)")
    return chunks


def keep_existing_chunks(component_pairs, chunk_ids, revision) -> list[int]:
    """Warn + drop chunk IDs that have no files in the repo for the requested sensors."""
    repo_files = HfApi().list_repo_files(DEFAULT_REPO_ID, repo_type="dataset", revision=revision)

    # One pass: which chunk ints exist for each requested (component, subpart).
    prefixes = {(c, s): f"{c}/{s}/{s}." for c, s in component_pairs}
    present: dict[tuple, set[int]] = {k: set() for k in prefixes}
    for rf in repo_files:
        m = CHUNK_RE.search(rf)
        if not m:
            continue
        chunk = int(m.group(1))
        for key, pref in prefixes.items():
            if rf.startswith(pref):
                present[key].add(chunk)
                break

    existing_any = set().union(*present.values()) if present else set()

    keep = []
    for chunk in chunk_ids:
        if chunk in existing_any:
            keep.append(chunk)
            # Note (not skip) sensors that lack this particular chunk.
            absent_for = [f"{c}/{s}" for (c, s), got in present.items() if chunk not in got]
            if absent_for:
                print(f"[warn] chunk {chunk:04d} missing for: {', '.join(absent_for)}")
        else:
            print(f"[warn] chunk {chunk:04d} not present in repo for any requested sensor; skipping.")
    print(f"[chunks] keeping {len(keep)}/{len(chunk_ids)} chunk(s)")
    return keep


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    uuids = read_manifest_uuids(args.lcdrive_manifest)
    component_pairs = parse_component_subparts(args)          # official helper
    if not component_pairs:
        raise SystemExit("No sensors requested (need at least one --camera/--labels/...).")

    chunk_ids = manifest_to_chunks(args.output_dir, uuids, args.revision)
    chunk_ids = keep_existing_chunks(component_pairs, chunk_ids, args.revision)
    if not chunk_ids:
        raise SystemExit("No downloadable chunks after existence check.")

    allow = build_allow_patterns(component_pairs, chunk_ids)  # official builder + mandatory files
    print(f"[download] {len(allow)} allow-patterns across {len(chunk_ids)} chunk(s)")
    snapshot_download(repo_id=DEFAULT_REPO_ID, repo_type="dataset",
                      local_dir=str(args.output_dir), local_dir_use_symlinks=False,
                      allow_patterns=allow, revision=args.revision)
    print("done")


if __name__ == "__main__":
    main()