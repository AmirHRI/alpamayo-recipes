# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""BEV of the GT futures clustered by ``meta_action``, split into its two real families.

⚠️ ``meta_action`` IS NOT ONE VOCABULARY. Its 14 tokens are two disjoint families -- a
LATERAL one (go_straight, steer_*, sharp_steer_*, reverse*) and a LONGITUDINAL one
(maintain_speed, gentle_/strong_ acceleration/deceleration, stop) -- and 3,674 of the
109,997 anchors carry exactly one of each. Bucketing all 14 as if they were mutually
exclusive double-counts those anchors and silently mixes "which way" with "how fast".
Hence one row per family, and the panel shares are within-family.

⚠️ Every anchor appears in AT MOST ONE panel per row, but an anchor with only a speed label
appears in no lateral panel (and vice versa) -- so the row shares do not reach 100%. The
"(no lateral label)" / "(no speed label)" counts are printed rather than hidden.

⚠️ ``reverse*`` is 32 anchors of 109,997 and travels BACKWARDS, so it gets its own panel
scale; 32 short paths on a 150 m axis is an empty box. The title says so.

Usage::

    python -m alpamayo1_5_distill.scripts.plot_train_traj_by_meta \
        npz=<from extract_train_trajectories.py> out=<png>
"""

from __future__ import annotations

import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK_2, INK_3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8a87"
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE)

LATERAL = ["go_straight", "steer_left", "steer_right",
           "sharp_steer_left", "sharp_steer_right", "reverse*"]
SPEED = ["strong_acceleration", "gentle_acceleration", "maintain_speed",
         "gentle_deceleration", "strong_deceleration", "stop"]
REVERSE = ("reverse", "reverse_left", "reverse_right")


def _spines(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_2, labelsize=6.5, length=2.5, width=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_3); ax.spines[s].set_linewidth(0.6)
    ax.grid(True, color=INK_3, lw=0.4, alpha=0.25)
    ax.set_axisbelow(True)


def _bev(ax, traj, extent, bins, vmax, main, sub):
    im = None
    if len(traj) >= 200:
        h, _, _ = np.histogram2d(traj[..., 1].ravel(), traj[..., 0].ravel(), bins=bins,
                                 range=[[extent[0], extent[1]], [extent[2], extent[3]]])
        with np.errstate(divide="ignore"):
            h = np.log10(h)
        h[np.isneginf(h)] = np.nan
        im = ax.imshow(h.T, origin="lower", extent=extent, aspect="equal", cmap=CMAP,
                       vmin=0, vmax=vmax, interpolation="nearest")
    elif len(traj):
        ax.plot(traj[:, :, 1].T, traj[:, :, 0].T, color=BLUE[4], lw=1.0, alpha=0.6,
                solid_capstyle="round")
        ax.set_aspect("equal")
    ax.plot([0], [0], marker="^", ms=5, color=INK_2, zorder=5, clip_on=False)
    ax.axvline(0, color=INK_3, lw=0.6, ls=(0, (4, 4)), zorder=1)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.invert_xaxis()
    _spines(ax)
    ax.set_title(f"{main}\n{sub}", fontsize=8, color=INK, pad=4, linespacing=1.4)
    return im


def main() -> None:
    a = dict(q.split("=", 1) for q in sys.argv[1:] if "=" in q)
    d = np.load(a["npz"], allow_pickle=True)
    ok = d["ok"]
    fut = d["future_xyz"][ok]
    ma = d["meta_action"].astype(str)[ok]
    n = len(fut)
    toks = [set(t for t in s.split(",") if t) for s in ma]

    def sel(name):
        if name == "reverse*":
            return np.array([bool(t & set(REVERSE)) for t in toks])
        return np.array([name in t for t in toks])

    fwd = float(np.ceil(np.percentile(fut[..., 0], 99.0) / 10) * 10)
    lat = float(np.ceil(np.percentile(np.abs(fut[..., 1]), 99.9) / 10) * 10)
    extent = (-lat, lat, -10.0, fwd)
    bins = (int(2 * lat / 0.75), int((fwd + 10) / 0.75))
    hh, _, _ = np.histogram2d(fut[..., 1].ravel(), fut[..., 0].ravel(), bins=bins,
                              range=[[extent[0], extent[1]], [extent[2], extent[3]]])
    vmax = float(np.log10(max(hh.max(), 10)))

    fig = plt.figure(figsize=(17.0, 8.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 6, hspace=0.36, wspace=0.24,
                          left=0.045, right=0.925, top=0.80, bottom=0.06)

    im = None
    for r, (fam, names) in enumerate((("LATERAL", LATERAL), ("LONGITUDINAL", SPEED))):
        covered = np.zeros(n, bool)
        first_ax = None
        for c, nm in enumerate(names):
            s = sel(nm); covered |= s
            ax = fig.add_subplot(gs[r, c])
            if c == 0:
                first_ax = ax
            t = fut[s]
            if nm == "reverse*" and len(t):
                # own scale: 32 anchors travelling backwards would be invisible otherwise
                rl = max(3.0, float(np.ceil(np.abs(t[..., 1]).max())))
                ext = (-rl, rl, float(np.floor(t[..., 0].min() - 1)),
                       max(3.0, float(np.ceil(t[..., 0].max() + 1))))
                _bev(ax, t, ext, bins, vmax, f"{nm}  ⚠️ own scale",
                     f"{int(s.sum()):,} · {100*s.mean():.3f}%")
            else:
                got = _bev(ax, t, extent, bins, vmax, nm,
                           f"{int(s.sum()):,} · {100*s.mean():.2f}%")
                im = im or got
            if c == 0:
                ax.set_ylabel("forward from ego (m)", fontsize=8, color=INK_2, labelpad=1)
            if r == 1:
                ax.set_xlabel("lateral y (m)  ·  + = LEFT", fontsize=7.5, color=INK_2)
        pos = first_ax.get_position()
        fig.text(0.008, pos.y0 + pos.height / 2, fam, rotation=90, fontsize=11,
                 color=INK, ha="center", va="center")
        print(f"[meta] {fam:<13} covered {100*covered.mean():.2f}% of anchors "
              f"({int((~covered).sum()):,} have no label in this family)", flush=True)

    cax = fig.add_axes([0.935, 0.30, 0.008, 0.34])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("waypoints per 0.75 m cell (log$_{10}$)", fontsize=7.5, color=INK_2)
    cb.ax.tick_params(colors=INK_2, labelsize=6.5, length=3, width=0.6)
    cb.outline.set_visible(False)

    fig.suptitle("LCDrive anchors — GT futures in BEV clustered by meta_action",
                 fontsize=13, color=INK, x=0.045, ha="left", y=0.955)
    fig.text(0.045, 0.905,
             f"{n:,} anchors · meta_action is TWO disjoint families, so one row each and "
             f"shares are within-family · 3,674 anchors carry one label from each · "
             f"ego frame at $t_0$, heading up · colour is log$_{{10}}$ count",
             fontsize=9, color=INK_2, ha="left")
    fig.savefig(a["out"], dpi=165, facecolor=SURFACE)
    print(f"[meta] wrote {a['out']}", flush=True)
    print("DONE_META", flush=True)


if __name__ == "__main__":
    main()
