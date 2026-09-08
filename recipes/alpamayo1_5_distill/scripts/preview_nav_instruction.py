# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a nav manifest and print one exact route component before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpamayo.chat_template.components import construct_route

from alpamayo1_5_distill.data.camera_subset import strip_turn_distance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{args.manifest} must contain a non-empty JSON list")

    changed = 0
    preview = None
    for index, row in enumerate(rows):
        missing = {"clip_id", "t0_relative", "nav_text"} - set(row)
        if missing:
            raise ValueError(f"manifest row {index} is missing {sorted(missing)}")
        raw = row["nav_text"]
        normalized = strip_turn_distance(raw)
        changed += normalized != raw
        if preview is None and normalized in {"Turn left", "Turn right"}:
            preview = (index, row, raw, normalized)
    if preview is None:
        raise ValueError("manifest contains no left/right turn instruction to preview")

    index, row, raw, normalized = preview
    route = construct_route({"nav_text": normalized})[0]["text"]
    if " in " in normalized or "m" in normalized:
        raise RuntimeError(f"distance leaked into normalized instruction: {normalized!r}")
    print(f"[preflight] manifest: {args.manifest}")
    print(f"[preflight] rows: {len(rows)}; distance-qualified turns normalized: {changed}")
    print(
        f"[preflight] sample row {index}: clip={row['clip_id']} "
        f"t0_relative={row['t0_relative']}"
    )
    print(f"[preflight] raw nav_text: {raw}")
    print(f"[preflight] normalized instruction: {normalized}")
    print(f"[preflight] model-visible route component: {route}")


if __name__ == "__main__":
    main()
