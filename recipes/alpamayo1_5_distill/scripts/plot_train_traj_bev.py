# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""BEV distribution of the GT ego futures in a nav-anchor training manifest.

Answers "what does the distillation set actually contain": the spatial envelope of the 64-step
(6.4 s) futures, in the ego frame at t0, over every anchor.

FORM. The job is spatial MAGNITUDE, so the primary panels are log-density heatmaps on a single
sequential hue (light -> dark), not a categorical scatter. The maneuver breakdown is SMALL
MULTIPLES on the same ramp rather than coloured series, which sidesteps the categorical
all-pairs series cap entirely -- identity comes from the facet title, never from hue.

⚠️ LOG COUNT. 90% of the mass is a straight line ahead; on a linear ramp the turns vanish
completely. The colourbar is explicitly labelled as log10 so the compression is not silent.

⚠️ TURNS ARE CLASSIFIED BY GEOMETRY, not by ``meta_action``. That field mixes steering and
speed labels and is often a combination ("maintain_speed,steer_left"), so bucketing on it
double-counts. ``final_yaw`` -- the net heading change over the 6.4 s -- is unambiguous, and
its sign was verified against the manifest's own ``nav_text``: "Turn left" anchors average
+53.7 deg / +21.1 m lateral, "Turn right" -65.0 deg / -19.6 m. Positive y is LEFT.

Usage::

    python -m alpamayo1_5_distill.scripts.plot_train_traj_bev \
        npz=<from extract_train_trajectories.py> out=<png> [turn_deg=15] [lines=4000]
"""

from __future__ import annotations

import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#: Sequential blue ramp, steps 100->700 of the reference palette. One hue, light -> dark.
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK_2, INK_3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8a87"
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE)


def _panel(ax, y, x, extent, bins, vmax, title, n, *, cmap=CMAP):
    """One BEV log-density heatmap. Lateral horizontal, forward vertical."""
    h, _, _ = np.histogram2d(
        y, x, bins=bins, range=[[extent[0], extent[1]], [extent[2], extent[3]]]
    )
    with np.errstate(divide="ignore"):
        h = np.log10(h)
    h[np.isneginf(h)] = np.nan
    im = ax.imshow(
        h.T, origin="lower", extent=extent, aspect="equal", cmap=cmap,
        vmin=0, vmax=vmax, interpolation="nearest",
    )
    _dress(ax, extent, title, n)
    return im


def _dress(ax, extent, title, n):
    ax.set_facecolor(SURFACE)
    # ego at the origin, pointing up -- the only annotation that is not data
    ax.plot([0], [0], marker="^", ms=6, color=INK_2, zorder=5, clip_on=False)
    ax.axvline(0, color=INK_3, lw=0.6, ls=(0, (4, 4)), zorder=1)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.invert_xaxis()
    ax.set_title(f"{title}\n{n:,} anchors", fontsize=9, color=INK, pad=6, linespacing=1.5)
    ax.tick_params(colors=INK_2, labelsize=7.5, length=3, width=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_3); ax.spines[s].set_linewidth(0.6)
    ax.grid(True, color=INK_3, lw=0.4, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    a = dict(q.split("=", 1) for q in sys.argv[1:] if "=" in q)
    d = np.load(a["npz"], allow_pickle=True)
    out = a["out"]
    turn_deg = float(a.get("turn_deg", 15))
    n_lines = int(a.get("lines", 4000))

    ok = d["ok"]
    fut = d["future_xyz"][ok]                       # [N, 64, 3] ego frame at t0
    yaw = np.degrees(d["final_yaw"][ok])
    n = len(fut)
    x, y = fut[..., 0].ravel(), fut[..., 1].ravel()

    # View: equal aspect, so the geometry is honest -- these futures really do reach much
    # further forward than sideways. Cropped at the 99th pct of forward reach because a
    # handful of highway clips run to 200 m and would leave a third of every panel empty.
    fwd_hi = float(np.ceil(np.percentile(fut[..., 0], 99.0) / 10) * 10)
    lat = float(np.ceil(np.percentile(np.abs(fut[..., 1]), 99.9) / 10) * 10)
    frac_shown = float(
        ((fut[..., 0] <= fwd_hi) & (np.abs(fut[..., 1]) <= lat)).mean()
    )
    extent = (-lat, lat, -10.0, fwd_hi)
    bins = (int(2 * lat / 0.5), int((fwd_hi + 10) / 0.5))       # 0.5 m cells

    hh, _, _ = np.histogram2d(y, x, bins=bins,
                              range=[[extent[0], extent[1]], [extent[2], extent[3]]])
    vmax = float(np.log10(max(hh.max(), 10)))

    fig = plt.figure(figsize=(13.5, 8.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.55, 1.0], width_ratios=[1, 1, 1, 1],
                          hspace=0.30, wspace=0.28, left=0.06, right=0.90,
                          top=0.88, bottom=0.07)

    # (a) every point, log density
    ax0 = fig.add_subplot(gs[0, 0:2])
    im = _panel(ax0, y, x, extent, bins, vmax, "All futures — log density", n)
    ax0.set_ylabel("forward from ego at $t_0$  (m)", fontsize=8.5, color=INK_2)
    ax0.set_xlabel("lateral y (m)  ·  + = LEFT", fontsize=8.5, color=INK_2)

    # (b) the same set as individual paths, so single-trajectory shape is visible
    ax1 = fig.add_subplot(gs[0, 2:4])
    rng = np.random.default_rng(0)
    pick = rng.choice(n, size=min(n_lines, n), replace=False)
    # ⚠️ Alpha must SCALE with the count or the panel is either invisible or a solid block:
    # accumulated opacity goes as n*alpha, so hold that roughly constant.
    alpha = float(np.clip(140.0 / max(len(pick), 1), 0.012, 0.30))
    ax1.plot(fut[pick, :, 1].T, fut[pick, :, 0].T, color=BLUE[4], lw=0.5, alpha=alpha,
             solid_capstyle="round")
    _dress(ax1, extent, f"Individual paths — {len(pick):,} of {n:,} drawn", n)
    del alpha
    ax1.set_xlabel("lateral y (m)  ·  + = LEFT", fontsize=8.5, color=INK_2)

    # (c-e) small multiples by net heading change. Facet title carries identity, not hue.
    cats = [
        (f"Straight  |Δψ| < {turn_deg:g}°", np.abs(yaw) < turn_deg),
        (f"Left turn  Δψ ≥ {turn_deg:g}°", yaw >= turn_deg),
        (f"Right turn  Δψ ≤ −{turn_deg:g}°", yaw <= -turn_deg),
    ]
    for i, (title, sel) in enumerate(cats):
        ax = fig.add_subplot(gs[1, i])
        s = fut[sel]
        if len(s):
            _panel(ax, s[..., 1].ravel(), s[..., 0].ravel(), extent, bins, vmax,
                   f"{title}   {100 * sel.mean():.1f}%", int(sel.sum()))
        ax.tick_params(labelsize=6.5)
        if i == 0:
            ax.set_ylabel("forward (m)", fontsize=8, color=INK_2)

    # heading histogram -- the "is the turn subset actually preserved" read-out
    axh = fig.add_subplot(gs[1, 3])
    axh.hist(np.clip(yaw, -120, 120), bins=80, color=BLUE[3], edgecolor="none")
    axh.set_yscale("log")
    axh.axvline(turn_deg, color=INK_3, lw=0.7, ls=(0, (3, 3)))
    axh.axvline(-turn_deg, color=INK_3, lw=0.7, ls=(0, (3, 3)))
    axh.set_facecolor(SURFACE)
    axh.set_title(f"Net heading change over 6.4 s\nmedian |Δψ| = "
                  f"{np.median(np.abs(yaw)):.1f}°", fontsize=9, color=INK, pad=6,
                  linespacing=1.5)
    axh.set_xlabel("Δψ  (deg, clipped to ±120)", fontsize=8, color=INK_2)
    axh.set_ylabel("anchors (log)", fontsize=8, color=INK_2)
    axh.tick_params(colors=INK_2, labelsize=6.5, length=3, width=0.6)
    for s_ in ("top", "right"):
        axh.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        axh.spines[s_].set_color(INK_3); axh.spines[s_].set_linewidth(0.6)
    axh.grid(True, axis="y", color=INK_3, lw=0.4, alpha=0.25)
    axh.set_axisbelow(True)

    cax = fig.add_axes([0.915, 0.42, 0.012, 0.44])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("waypoints per 0.5 m cell  (log$_{10}$)", fontsize=8, color=INK_2)
    cb.ax.tick_params(colors=INK_2, labelsize=7, length=3, width=0.6)
    cb.outline.set_visible(False)

    fig.suptitle(
        "LCDrive 50k turn-preserved training anchors — ground-truth ego futures in BEV",
        fontsize=13, color=INK, x=0.06, ha="left", y=0.965,
    )
    fig.text(0.06, 0.925,
             f"{n:,} of {len(ok):,} anchors resolved · 64 steps at 10 Hz (6.4 s) · ego frame "
             f"at $t_0$, heading up · colour is log$_{{10}}$ count so the turns stay visible · "
             f"view holds {100 * frac_shown:.1f}% of waypoints",
             fontsize=9, color=INK_2, ha="left")
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"[bev] wrote {out}", flush=True)
    print(f"[bev] n={n}  straight {100*cats[0][1].mean():.1f}%  "
          f"left {100*cats[1][1].mean():.1f}%  right {100*cats[2][1].mean():.1f}%", flush=True)
    print("DONE_PLOT", flush=True)


if __name__ == "__main__":
    main()
