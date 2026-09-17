# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Speed / acceleration distribution of a nav-anchor training manifest.

Companion to ``plot_train_traj_bev.py``, which showed the BEV envelope is dominated by SPEED
spread rather than steering. This quantifies that axis.

FORM. Four one-dimensional distributions get histograms; the two questions that are really
"distribution of a distribution" (speed vs time, speed vs turn angle) get 2-D log-density
heatmaps on the sequential blue ramp. The maneuver comparison is the one place identity
matters, so it is three CATEGORICAL series -- validated all-pairs on the light surface
(worst CVD ΔE 9.2, normal-vision 24.0). Aqua sits at 2.74:1 on this surface, under the 3:1
bar, so the relief rule applies: every series carries a visible direct label, not colour alone.

⚠️ GROUND SPEED, XY only. Measured over this manifest the XY and full 3-D speeds differ by
0.001 m/s on average, so the z (grade) component is noise here -- but the label says XY so
nobody later assumes otherwise.

⚠️ ``t0`` speed is a CENTRAL difference across the history/future join (``history[-1]`` IS
t0, ``future[0]`` is t0+0.1 s). A forward difference is noisier and biases low at the stop.

Usage::

    python -m alpamayo1_5_distill.scripts.plot_train_speed_dist \
        npz=<from extract_train_trajectories.py> out=<png> [turn_deg=15]
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
#: Categorical slots 1-3. Fixed order, never cycled.
SERIES = {"Straight": "#2a78d6", "Left turn": "#eb6834", "Right turn": "#1baf7a"}
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE)
DT = 0.1


def _spines(ax, *, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_2, labelsize=7.5, length=3, width=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_3); ax.spines[s].set_linewidth(0.6)
    ax.grid(True, axis=grid_axis, color=INK_3, lw=0.4, alpha=0.25)
    ax.set_axisbelow(True)


def _title(ax, main, sub):
    ax.set_title(f"{main}\n{sub}", fontsize=9.5, color=INK, pad=6, linespacing=1.5)


def _kmh_axis(ax):
    """Same measure, second unit -- NOT a dual-measure axis."""
    top = ax.secondary_xaxis("top", functions=(lambda v: v * 3.6, lambda v: v / 3.6))
    top.set_xlabel("km/h", fontsize=7.5, color=INK_3, labelpad=3)
    top.tick_params(colors=INK_3, labelsize=6.5, length=2.5, width=0.5)
    top.spines["top"].set_color(INK_3); top.spines["top"].set_linewidth(0.6)


def _density(ax, x, y, xr, yr, bins, cmap=CMAP, *, per_column=False, vmax=None):
    """2-D density. ``per_column`` normalises each x-column to a distribution.

    ⚠️ A raw log-count map SATURATES whenever every column holds the whole sample -- the
    speed-vs-time panel has all N anchors in each of its 64 time bins, so counts are
    near-uniform and it renders as a solid block. Normalising per column is what the
    question ("how does the distribution evolve") actually asks for.
    """
    h, _, _ = np.histogram2d(x, y, bins=bins, range=[xr, yr])
    if per_column:
        tot = h.sum(axis=1, keepdims=True)
        h = np.divide(h, tot, out=np.zeros_like(h), where=tot > 0)
        h[h == 0] = np.nan
    else:
        with np.errstate(divide="ignore"):
            h = np.log10(h)
        h[np.isneginf(h)] = np.nan
    im = ax.imshow(h.T, origin="lower", extent=(*xr, *yr), aspect="auto", cmap=cmap,
                   vmin=0, vmax=vmax, interpolation="nearest")
    _spines(ax, grid_axis="both")
    return im


def main() -> None:
    a = dict(q.split("=", 1) for q in sys.argv[1:] if "=" in q)
    d = np.load(a["npz"], allow_pickle=True)
    turn_deg = float(a.get("turn_deg", 15))

    ok = d["ok"]
    hist, fut = d["history_xyz"][ok], d["future_xyz"][ok]
    yaw = np.degrees(d["final_yaw"][ok])
    n = len(fut)

    full = np.concatenate([hist, fut], axis=1)          # [N, 80, 3]; index 15 is t0
    sp = np.linalg.norm(np.diff(full[..., :2], axis=1), axis=-1) / DT
    v0 = sp[:, 14]                                      # central difference at t0
    vfut = sp[:, 15:]                                   # 64 future speeds
    acc = np.diff(sp, axis=1) / DT
    path = np.linalg.norm(np.diff(fut[..., :2], axis=1), axis=-1).sum(1)

    cats = {"Straight": np.abs(yaw) < turn_deg,
            "Left turn": yaw >= turn_deg,
            "Right turn": yaw <= -turn_deg}

    v_hi = float(np.ceil(np.percentile(v0, 99.5)))
    fig = plt.figure(figsize=(14.0, 7.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.26,
                          left=0.055, right=0.985, top=0.82, bottom=0.09)

    # (a) speed at t0
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(np.clip(v0, 0, v_hi), bins=90, color=BLUE[3], edgecolor="none")
    stopped = 100 * (v0 < 0.5).mean()
    ax.axvline(np.median(v0), color=INK_2, lw=1.0, ls=(0, (3, 2)))
    ax.annotate(f"median {np.median(v0):.1f} m/s\n({np.median(v0) * 3.6:.0f} km/h)",
                (np.median(v0), ax.get_ylim()[1] * 0.94), xytext=(6, 0),
                textcoords="offset points", fontsize=7.5, color=INK_2, va="top")
    _spines(ax); _kmh_axis(ax)
    _title(ax, "Speed at $t_0$", f"{stopped:.1f}% stationary (< 0.5 m/s)")
    ax.set_xlabel("m/s", fontsize=8.5, color=INK_2)
    ax.set_ylabel("anchors", fontsize=8.5, color=INK_2)

    # (b) how speed evolves across the horizon the model predicts
    ax = fig.add_subplot(gs[0, 1])
    t = np.repeat(np.arange(1, 65) * DT, n)
    im_col = _density(ax, t, np.clip(vfut.T.ravel(), 0, v_hi), (DT, 6.4), (0, v_hi), (64, 60),
                      per_column=True, vmax=float(np.percentile(
                          np.histogram2d(t, np.clip(vfut.T.ravel(), 0, v_hi), bins=(64, 60),
                                         range=[(DT, 6.4), (0, v_hi)])[0]
                          / n, 99.5)))
    med = np.median(vfut, axis=0)
    q1, q3 = np.percentile(vfut, [10, 90], axis=0)
    tt = np.arange(1, 65) * DT
    ax.plot(tt, med, color=INK, lw=1.8, solid_capstyle="round", zorder=4)
    ax.plot(tt, q1, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=4)
    ax.plot(tt, q3, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=4)
    for lbl, arr in (("median", med), ("10th", q1), ("90th", q3)):
        ax.annotate(lbl, (6.4, arr[-1]), xytext=(-3, 5), textcoords="offset points",
                    fontsize=7, color=INK, ha="right")
    _title(ax, "Speed across the 6.4 s horizon",
           "each time column normalised to a distribution")
    ax.set_xlabel("time after $t_0$  (s)", fontsize=8.5, color=INK_2)
    ax.set_ylabel("m/s", fontsize=8.5, color=INK_2)

    # (c) path length -- the BEV envelope's driver
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(np.clip(path, 0, 220), bins=90, color=BLUE[3], edgecolor="none")
    ax.axvline(np.median(path), color=INK_2, lw=1.0, ls=(0, (3, 2)))
    ax.annotate(f"median {np.median(path):.0f} m", (np.median(path), ax.get_ylim()[1] * 0.94),
                xytext=(6, 0), textcoords="offset points", fontsize=7.5, color=INK_2, va="top")
    _spines(ax)
    _title(ax, "Distance travelled in 6.4 s", "the BEV envelope's forward reach")
    ax.set_xlabel("path length  (m)", fontsize=8.5, color=INK_2)
    ax.set_ylabel("anchors", fontsize=8.5, color=INK_2)

    # (d) THE comparison -- three categorical series, each directly labelled (relief rule)
    ax = fig.add_subplot(gs[1, 0])
    grid = np.linspace(0, v_hi, 400)
    for k, (name, sel) in enumerate(cats.items()):
        v = np.sort(v0[sel])
        cdf = np.searchsorted(v, grid, side="right") / len(v)
        ax.plot(grid, cdf, color=SERIES[name], lw=2.0, solid_capstyle="round", label=name)
        med = float(np.median(v))
        # Direct label at a distinct height per series -- free space, no collision, and
        # required rather than optional: aqua is 2.74:1 on this surface (relief rule).
        ax.annotate(f"{name} · median {med:.1f} m/s", (v_hi, 0.30 - 0.09 * k),
                    xytext=(-4, 0), textcoords="offset points", fontsize=7.5,
                    color=INK_2, ha="right", va="center")
    ax.axhline(0.5, color=INK_3, lw=0.6, ls=(0, (4, 4)))
    ax.set_ylim(0, 1.02)
    _spines(ax); _kmh_axis(ax)
    leg = ax.legend(frameon=False, fontsize=7.5, loc="upper left", labelcolor=INK_2)
    for txt in leg.get_texts():
        txt.set_color(INK_2)
    _title(ax, "Speed at $t_0$ by maneuver",
           "cumulative, so the 84/7/8 split cannot hide the turns")
    ax.set_xlabel("m/s", fontsize=8.5, color=INK_2)
    ax.set_ylabel("fraction of anchors ≤ x", fontsize=8.5, color=INK_2)

    # (e) speed vs turn magnitude
    ax = fig.add_subplot(gs[1, 1])
    im = _density(ax, np.clip(np.abs(yaw), 0, 120), np.clip(v0, 0, v_hi), (0, 120),
                  (0, v_hi), (60, 60))
    ax.axvline(turn_deg, color=INK_2, lw=0.8, ls=(0, (3, 3)))
    _title(ax, "Speed at $t_0$ vs turn magnitude", "log density · sharper turns are slower")
    ax.set_xlabel("|Δψ| over 6.4 s  (deg)", fontsize=8.5, color=INK_2)
    ax.set_ylabel("m/s", fontsize=8.5, color=INK_2)

    # (f) longitudinal acceleration
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(np.clip(acc.ravel(), -4, 4), bins=120, color=BLUE[3], edgecolor="none")
    ax.set_yscale("log")
    _spines(ax)
    p1, p99 = np.percentile(acc, [1, 99])
    _title(ax, "Longitudinal acceleration",
           f"1st–99th pct  {p1:+.2f} to {p99:+.2f} m/s²  ·  {100 * (np.abs(acc) > 5).mean():.2f}% "
           f"beyond ±5")
    ax.set_xlabel("m/s²  (clipped to ±4)", fontsize=8.5, color=INK_2)
    ax.set_ylabel("waypoints (log)", fontsize=8.5, color=INK_2)

    for mappable, box, lab in (
        (im_col, [0.395, 0.868, 0.115, 0.012], "share of anchors per cell"),
        (im, [0.700, 0.868, 0.115, 0.012], "log$_{10}$ count per cell"),
    ):
        cax = fig.add_axes(box)
        cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cb.set_label(lab, fontsize=7, color=INK_2, labelpad=3)
        cb.ax.tick_params(colors=INK_2, labelsize=6, length=2.5, width=0.5)
        cb.ax.xaxis.set_label_position("top"); cb.ax.xaxis.set_ticks_position("bottom")
        cb.outline.set_visible(False)

    fig.suptitle("LCDrive 50k turn-preserved training anchors — speed & acceleration",
                 fontsize=13, color=INK, x=0.055, ha="left", y=0.965)
    fig.text(0.055, 0.925,
             f"{n:,} anchors · XY ground speed from the 10 Hz ego poses · $t_0$ speed is a "
             f"central difference across the history/future join",
             fontsize=9, color=INK_2, ha="left")
    fig.savefig(a["out"], dpi=170, facecolor=SURFACE)
    print(f"[speed] wrote {a['out']}", flush=True)
    for name, sel in cats.items():
        print(f"[speed] {name:11s} n={int(sel.sum()):6d}  median v0 {np.median(v0[sel]):5.2f} m/s "
              f"({np.median(v0[sel]) * 3.6:5.1f} km/h)", flush=True)
    print("DONE_SPEED", flush=True)


if __name__ == "__main__":
    main()
