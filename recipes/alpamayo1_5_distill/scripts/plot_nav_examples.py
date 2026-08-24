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

r"""One representative anchor per ``meta_action``: front camera at t0 + BEV trajectory + label.

Picks the MODAL nav_text outcome for each category rather than a turn example, so each row
shows what that category typically looks like. ``steer_left`` therefore appears as "Continue
straight" -- that is the honest picture (only 10.7% of its anchors classify as a turn), and
cherry-picking a turn would misrepresent why the category has low yield.

The BEV panel draws the classifier's 5 m and 40 m rings, because those bounds are what the
label actually depends on: ``route_to_nav_text`` only inspects waypoints between them, so a
trajectory that leaves the outer ring before bending is labelled straight no matter what it
does afterwards.

Usage::

    python -m alpamayo1_5_distill.scripts.plot_nav_examples \
        --annotations .../nav_lcdrive_train_anchors_all.json --out .../nav_examples.png
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

ORDER = ["maintain_speed", "go_straight", "gentle_deceleration", "gentle_acceleration",
         "steer_right", "steer_left", "strong_deceleration", "stop", "strong_acceleration",
         "sharp_steer_right", "sharp_steer_left", "reverse", "reverse_left", "reverse_right"]


def bucket(nav: str) -> str:
    """straight | left | right | reverse -- "Reverse" is one word, so no blind split()[1]."""
    if nav.startswith("Turn"):
        return nav.split()[1]
    return "reverse" if nav.startswith("Reverse") else "straight"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--local-dir", default="/data/datasets/physical_ai_av/")
    ap.add_argument("--chunk-ids", default="0-3146")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cols", type=int, default=2, help="category blocks per row")
    ap.add_argument("--fixed-span", type=float, default=0.0,
                    help="use one BEV half-width (m) for every panel, so panels are "
                         "directly comparable and both rings are always in view. Default 0 "
                         "scales each panel to its own trajectory, which shows shape better "
                         "but means the rings differ in apparent size panel to panel")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    from alpamayo.data.pai_utils import PhysicalAIAVDatasetLocalInterface
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    with open(a.annotations) as fh:
        data = json.load(fh)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for e in data:
        for c in e["meta_action"]:
            by_cat[c].append(e)
    cats = [c for c in ORDER if c in by_cat] + sorted(set(by_cat) - set(ORDER))

    avdi = PhysicalAIAVDatasetLocalInterface(
        local_dir=a.local_dir, chunk_ids=a.chunk_ids,
        features_metadata="features.csv", clip_index_metadata="clip_index.parquet")
    front = avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV

    picks = []
    for c in cats:
        pool = by_cat[c]
        mode = Counter(bucket(e["nav_text"]) for e in pool).most_common(1)[0][0]
        cand = [e for e in pool if bucket(e["nav_text"]) == mode]
        rate = 100 * sum(bucket(e["nav_text"]) in ("left", "right") for e in pool) / len(pool)
        picks.append((c, rng.choice(cand), len(pool), rate))

    nrow = int(np.ceil(len(picks) / a.cols))
    fig, axes = plt.subplots(nrow, 2 * a.cols, figsize=(7.2 * a.cols, 2.55 * nrow),
                             gridspec_kw={"width_ratios": [1.75, 1] * a.cols})
    axes = np.atleast_2d(axes)

    for i, (cat, e, n, rate) in enumerate(picks):
        r, cblk = divmod(i, a.cols)
        ax_im, ax_bev = axes[r, 2 * cblk], axes[r, 2 * cblk + 1]
        try:
            s = load_physical_aiavdataset(
                e["clip_id"], t0_us=int(e["t0_relative"]), avdi=avdi,
                num_history_steps=16, num_future_steps=64, time_step=0.1,
                camera_features=[front], num_frames=1)
        except Exception as exc:                      # a bad clip must not kill the figure
            ax_im.text(0.5, 0.5, f"{cat}\nload failed\n{exc}", ha="center", va="center",
                       fontsize=7, transform=ax_im.transAxes)
            ax_im.axis("off"); ax_bev.axis("off")
            continue

        img = s["image_frames"][0, 0].permute(1, 2, 0).numpy()
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 1) if img.max() <= 1.0 else np.clip(img / 255.0, 0, 1)
        ax_im.imshow(img)
        ax_im.axis("off")
        turn = e["nav_text"].startswith("Turn")
        rev_t = e["nav_text"].startswith("Reverse")
        ax_im.set_title(f"{cat}   (n={n}, {rate:.0f}% turn)\n"
                        f'"{e["nav_text"]}"   t0={e["t0_relative"] / 1e6:.1f}s',
                        fontsize=8.5,
                        color="crimson" if turn else ("darkorange" if rev_t else "black"),
                        fontweight="bold" if (turn or rev_t) else "normal", loc="left")

        h = s["ego_history_xyz"][0, 0].numpy()
        f = s["ego_future_xyz"][0, 0].numpy()
        # +y is LEFT in the rig frame, so the y axis is inverted to put left on the left
        ax_bev.plot(h[:, 1], h[:, 0], "-", color="0.55", lw=1.6, label="history 1.6s")
        rev = e["nav_text"].startswith("Reverse")
        ax_bev.plot(f[:, 1], f[:, 0], "-o", ms=1.8, lw=1.5,
                    color="crimson" if turn else ("darkorange" if rev else "steelblue"),
                    label="future 6.4s")
        ax_bev.plot(0, 0, "k^", ms=6)
        span = (a.fixed_span if a.fixed_span else
                max(12.0, np.abs(f[:, 0]).max(), np.abs(f[:, 1]).max() * 1.4) * 1.15)
        # only draw a ring that is actually in view -- an off-plot or clipped ring makes the
        # panel look like a different scale than it is
        for rad, st in ((5, ":"), (40, "--")):            # the classifier's inspection window
            if rad <= span * 1.25:
                ax_bev.add_patch(Circle((0, 0), rad, fill=False, ec="0.7", ls=st, lw=0.8))
        path = float(np.linalg.norm(np.diff(f[:, :2], axis=0), axis=1).sum())
        ax_bev.text(0.03, 0.955, f"{path:.0f} m in 6.4 s", transform=ax_bev.transAxes,
                    fontsize=6.5, va="top", color="0.25")
        ax_bev.set_xlim(span, -span)                      # inverted: left turn bends left
        ax_bev.set_ylim(-span * 0.25, span * 1.25)
        ax_bev.set_aspect("equal")
        ax_bev.tick_params(labelsize=6)
        ax_bev.grid(alpha=0.25, lw=0.4)
        ax_bev.set_xlabel("y (m)  ←left   right→", fontsize=6.5)
        if i == 0:
            # lower left: upper left holds the path-length annotation
            ax_bev.legend(fontsize=5.5, loc="lower left", framealpha=0.85)

    for j in range(len(picks), nrow * a.cols):            # blank any unused block
        r, cblk = divmod(j, a.cols)
        axes[r, 2 * cblk].axis("off")
        axes[r, 2 * cblk + 1].axis("off")

    scale_note = (f"all panels at a common ±{a.fixed_span:.0f} m scale"
                  if a.fixed_span else "each panel scaled to its own trajectory")
    fig.suptitle("Nav-text examples by meta_action — modal outcome per category; "
                 "front-wide camera at t0, trajectory in the t0 rig frame\n"
                 "dotted/dashed rings = the classifier's 5 m / 40 m inspection window "
                 f"({scale_note})", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(a.out, dpi=125, bbox_inches="tight")
    print(f"[plot] wrote {a.out}  ({len(picks)} categories)")
    for cat, e, n, rate in picks:
        print(f"[plot]   {cat:<20} {e['clip_id'][:8]}  t0={e['t0_relative'] / 1e6:>5.1f}s  "
              f'"{e["nav_text"]}"')


if __name__ == "__main__":
    main()
