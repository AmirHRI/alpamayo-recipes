#!/usr/bin/env python3
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

r"""Render what the model actually receives, from the frame cache, for eyeballing.

Every panel is the frame AFTER the processor's own resize, so what is on screen is the ViT's
input up to normalisation -- not the 1080p the cache stores. The geometry is taken from the
processor's ``image_grid_thw`` rather than assumed, and printed alongside, so a mismatch
between "what I drew" and "what the model tokenised" cannot hide.

Each sheet shows one anchor: rows are cameras in the loader's order, columns are the four
frames at ``[t0-0.3s, t0-0.2s, t0-0.1s, t0]``, captioned with each frame's real decoded
timestamp. The route component is printed verbatim as the model sees it, delimiters included,
because "the nav text is in the JSON" and "the nav text reached the prompt" are different
claims -- the second is the one that matters.

Usage::

    preview_frame_cache.py --out <dir> [--n 3] [--nav-contains "Turn left"]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import types

import numpy as np
from PIL import Image, ImageDraw

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (os.path.join(_REPO, "src"), os.path.join(_REPO, "recipes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MANIFEST = (
    "/temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests/"
    "nav_lcdrive_train_anchors_all.json"
)
FRAME_CACHE = "/temp/achahe/physical_ai_av/framecache_nav2cam_1080p"
ALPAMAYO = "/temp/achahe/hf_cache/hub/models--nvidia--Alpamayo-1.5-10B-A1-format"
COSMOS = (
    "/temp/achahe/hf_cache/hub/models--nvidia--Cosmos-Reason2-8B/snapshots/"
    "a9fae2cf89dc64db96b12860417f0eb403013bb9"
)
#: Loader camera index -> the label the prompt uses, from camera_subset's module docstring.
CAMERA_LABELS = {0: "Front left", 1: "Front (wide 120)", 2: "Front right", 3: "Front telephoto (30)"}

PRE_ARGS = {
    "_target_": "alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config",
    "chat_template_version": "r1_5",
    "components_order": ["image", "traj_history", "route", "prompt", "traj_future"],
    "components_prompt": ["traj_future"],
    "label_components": ["traj_future"],
    "include_camera_ids": True,
    "include_frame_nums": True,
    "generation_mode": True,
}


def model_config():
    with open(os.path.join(ALPAMAYO, "config.json")) as handle:
        raw = json.load(handle)
    cfg = types.SimpleNamespace(
        **{
            k: raw[k]
            for k in (
                "traj_vocab_size",
                "min_pixels",
                "max_pixels",
                "tokens_per_history_traj",
                "tokens_per_future_traj",
            )
        }
    )
    cfg.vlm_name_or_path = COSMOS
    return cfg


def smart_resize(height: int, width: int, max_pixels: int, min_pixels: int, factor: int = 32):
    """Qwen's geometry: keep aspect, land inside [min_pixels, max_pixels], snap to ``factor``."""
    import math

    h, w = height, width
    if h * w > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        h, w = math.floor(h / beta / factor) * factor, math.floor(w / beta / factor) * factor
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h, w = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return h, w


def route_component(text: str) -> str:
    match = re.search(r"<\|route_start\|>.*?<\|route_end\|>", text, flags=re.S)
    return match.group(0) if match else "<< NO ROUTE COMPONENT IN PROMPT >>"


def draw_sheet(sample, cameras, cfg, path: str) -> tuple[int, int]:
    images = sample["image_frames"]  # (n_cam, n_frames, 3, H, W) uint8
    n_cam, n_frames = images.shape[0], images.shape[1]
    src_h, src_w = int(images.shape[3]), int(images.shape[4])
    tgt_h, tgt_w = smart_resize(src_h, src_w, cfg.max_pixels, cfg.min_pixels)

    pad, top, cap = 12, 132, 30
    sheet = Image.new(
        "RGB",
        (pad + n_frames * (tgt_w + pad), top + n_cam * (tgt_h + cap + pad)),
        (250, 250, 250),
    )
    draw = ImageDraw.Draw(sheet)

    text = sample["tokenized_data"]["text"]
    grid = sample["tokenized_data"]["image_grid_thw"]
    draw.text((pad, 10), f"clip {sample['clip_id']}   t0_relative={sample['t0_us']} us",
              fill=(0, 0, 0))
    draw.text((pad, 30), f"nav_text (post-strip): {sample['nav_text']!r}", fill=(0, 0, 0))
    draw.text((pad, 50), f"model-visible route:   {route_component(text)}", fill=(150, 0, 0))
    draw.text(
        (pad, 70),
        f"panels drawn at {tgt_w}x{tgt_h} (processor smart_resize of {src_w}x{src_h}); "
        f"image_grid_thw={grid.tolist()}",
        fill=(0, 0, 140),
    )
    draw.text(
        (pad, 90),
        f"grid implies {int(grid[0][1]) * 16}x{int(grid[0][2]) * 16} px per image "
        f"=> {'MATCH' if int(grid[0][1]) * 16 == tgt_h and int(grid[0][2]) * 16 == tgt_w else 'MISMATCH'}"
        f"   |   {n_cam} cameras x {n_frames} frames = {n_cam * n_frames} images",
        fill=(0, 110, 0),
    )
    draw.text((pad, 110), "columns: t0-0.3s, t0-0.2s, t0-0.1s, t0 (left to right)",
              fill=(90, 90, 90))

    stamps = sample.get("absolute_timestamps")
    for r in range(n_cam):
        label = CAMERA_LABELS.get(int(cameras[r]), f"camera idx {cameras[r]}")
        for c in range(n_frames):
            frame = images[r, c].permute(1, 2, 0).numpy()
            panel = Image.fromarray(frame).resize((tgt_w, tgt_h), Image.Resampling.BICUBIC)
            x = pad + c * (tgt_w + pad)
            y = top + r * (tgt_h + cap + pad)
            sheet.paste(panel, (x, y))
            ts = int(stamps[r][c]) if stamps is not None else None
            draw.text(
                (x, y + tgt_h + 6),
                f"{label} | frame {c} | t={ts} us"
                + (f" ({(ts - sample['t0_us']) / 1e6:+.2f}s)" if ts is not None else ""),
                fill=(40, 40, 40),
            )
    sheet.save(path, quality=92)
    return tgt_w, tgt_h


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--frame-cache", default=FRAME_CACHE)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument(
        "--nav-contains",
        action="append",
        default=None,
        help="pick a sample whose nav_text contains this; repeatable, one per sheet",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from alpamayo1_5_distill.data.camera_subset import CameraSubsetPAIDataset

    cfg = model_config()
    ds = CameraSubsetPAIDataset(
        cameras=[1, 3],
        annotations_path=args.manifest,
        frame_cache_root=args.frame_cache,
        strip_nav_turn_distance=True,
        vla_preprocess_args=PRE_ARGS,
        model_config=cfg,
    )

    # Pick deliberately different instructions: a sheet of three "Continue straight" anchors
    # would not exercise the route component at all.
    wanted = args.nav_contains or ["Turn left", "Turn right", "Continue straight"]
    rng = random.Random(args.seed)
    chosen: list[tuple[int, str]] = []
    for phrase in wanted[: args.n]:
        hits = [
            i for i, s in enumerate(ds.base._samples) if phrase.lower() in s["nav_text"].lower()
        ]
        if not hits:
            print(f"[preview] no anchor whose nav_text contains {phrase!r}; skipping", flush=True)
            continue
        chosen.append((rng.choice(hits), phrase))

    print(f"[preview] {len(chosen)} sheets -> {args.out}", flush=True)
    for n, (idx, phrase) in enumerate(chosen, 1):
        sample = ds[idx]
        slug = re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")
        path = os.path.join(args.out, f"sample{n}_{slug}.jpg")
        w, h = draw_sheet(sample, ds.cameras, cfg, path)
        print(
            f"[preview] {path}\n"
            f"          clip={sample['clip_id']} t0={sample['t0_us']} "
            f"nav={sample['nav_text']!r}\n"
            f"          route={route_component(sample['tokenized_data']['text'])}\n"
            f"          panels {w}x{h}, grid={sample['tokenized_data']['image_grid_thw'].tolist()}",
            flush=True,
        )

    # The prompt is the other half of "is everything right", and it is text, so print one in
    # full rather than making it something only a picture could show.
    if chosen:
        sample = ds[chosen[0][0]]
        text = sample["tokenized_data"]["text"]
        compact = re.sub(r"(<\|vision_start\|>)(<\|image_pad\|>)+(<\|vision_end\|>)",
                         r"\1<|image_pad|> x N\3", text)
        path = os.path.join(args.out, "prompt_sample1.txt")
        with open(path, "w") as handle:
            handle.write(text)
        print(f"\n[preview] full prompt written to {path}\n", flush=True)
        print(compact[:2000], flush=True)


if __name__ == "__main__":
    main()
