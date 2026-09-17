# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Deceleration / braking events in a nav-anchor training manifest.

"The ego braked" is not one event, so this splits it on the speed series rather than on
``meta_action`` (which mixes steering and speed labels and is often a combination):

    brake_to_stop    v0 >= 2 m/s AND min(future speed) < 0.5 m/s      3,850   7.70%
    decel_no_stop    v0 >= 2 m/s, ends <= half of v0, never stops      3,757   7.51%
    standing_start   v0 < 0.5 m/s and moves off                        1,654   3.31%
    stays_stopped    v0 < 0.5 m/s and never moves                      2,248   4.50%
    cruising         everything else                                  38,491  76.98%

⚠️ SHARED, ZOOMED BEV SCALE. These are SHORT paths -- braking to a stop covers a median
12.2 m against 64.7 m for cruising -- so the 160 m axis of ``plot_train_traj_bev.py`` would
render them as three dots. The three braking panels share one zoomed extent so they stay
comparable to each other; they are NOT comparable to the main BEV figure's axes.

⚠️ ``stays_stopped`` has no BEV panel on purpose: median path length is 0.0 m, so its
"trajectory" is a single point and a density map of it is a pixel. It appears in the class
breakdown only.

Usage::

    python -m alpamayo1_5_distill.scripts.plot_train_braking \
        npz=<from extract_train_trajectories.py> out=<png> [moving=2.0] [stop=0.5]
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
DT = 0.1


def _spines(ax, grid_axis="both"):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_2, labelsize=7, length=3, width=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_3); ax.spines[s].set_linewidth(0.6)
    ax.grid(True, axis=grid_axis, color=INK_3, lw=0.4, alpha=0.25)
    ax.set_axisbelow(True)


def _title(ax, main, sub):
    ax.set_title(f"{main}\n{sub}", fontsize=9, color=INK, pad=6, linespacing=1.5)


def main() -> None:
    a = dict(q.split("=", 1) for q in sys.argv[1:] if "=" in q)
    d = np.load(a["npz"], allow_pickle=True)
    MOVING, STOP = float(a.get("moving", 2.0)), float(a.get("stop", 0.5))

    ok = d["ok"]
    hist, fut = d["history_xyz"][ok], d["future_xyz"][ok]
    nav = d["nav_text"].astype(str)[ok]
    n = len(fut)

    full = np.concatenate([hist, fut], axis=1)
    sp = np.linalg.norm(np.diff(full[..., :2], axis=1), axis=-1) / DT
    v0, vfut = sp[:, 14], sp[:, 15:]
    acc = np.diff(sp, axis=1) / DT
    path = np.linalg.norm(np.diff(fut[..., :2], axis=1), axis=-1).sum(1)

    brake = (v0 >= MOVING) & (vfut.min(1) < STOP)
    decel = (v0 >= MOVING) & (~brake) & (vfut[:, -1] <= 0.5 * v0)
    start = (v0 < STOP) & (vfut.max(1) >= STOP)
    stopped = (v0 < STOP) & (vfut.max(1) < STOP)
    cruise = ~(brake | decel | start | stopped)

    # when / how far into the horizon the stop lands
    first = np.argmax(vfut[brake] < STOP, axis=1)
    t_stop = (first + 1) * DT
    fb = fut[brake]
    d_stop = np.array([
        np.linalg.norm(np.diff(fb[i, : max(first[i] + 1, 2), :2], axis=0), axis=-1).sum()
        for i in range(len(fb))])
    peak = acc[brake].min(1)

    # shared, zoomed extent over the three braking-related classes only
    sub3 = fut[brake | decel | start]
    fwd_hi = float(np.ceil(np.percentile(sub3[..., 0], 99.0) / 5) * 5)
    lat = float(np.ceil(np.percentile(np.abs(sub3[..., 1]), 99.5) / 5) * 5)
    extent = (-lat, lat, -5.0, fwd_hi)
    bins = (int(2 * lat / 0.5), int((fwd_hi + 5) / 0.5))
    hh, _, _ = np.histogram2d(sub3[..., 1].ravel(), sub3[..., 0].ravel(), bins=bins,
                              range=[[extent[0], extent[1]], [extent[2], extent[3]]])
    vmax = float(np.log10(max(hh.max(), 10)))

    fig = plt.figure(figsize=(14.6, 9.0), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1.0], hspace=0.42, wspace=0.30,
                          left=0.05, right=0.935, top=0.80, bottom=0.08)

    panels = [
        ("brake to stop", brake, f"reaches < {STOP:g} m/s inside 6.4 s"),
        ("decelerate, no stop", decel, "ends at ≤ half its $t_0$ speed"),
        ("standing start", start, f"starts < {STOP:g} m/s, moves off"),
    ]
    im = None
    for i, (name, sel, why) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        t = fut[sel]
        h, _, _ = np.histogram2d(t[..., 1].ravel(), t[..., 0].ravel(), bins=bins,
                                 range=[[extent[0], extent[1]], [extent[2], extent[3]]])
        with np.errstate(divide="ignore"):
            h = np.log10(h)
        h[np.isneginf(h)] = np.nan
        im = ax.imshow(h.T, origin="lower", extent=extent, aspect="equal", cmap=CMAP,
                       vmin=0, vmax=vmax, interpolation="nearest")
        ax.plot([0], [0], marker="^", ms=6, color=INK_2, zorder=5, clip_on=False)
        ax.axvline(0, color=INK_3, lw=0.6, ls=(0, (4, 4)), zorder=1)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.invert_xaxis()
        _spines(ax)
        _title(ax, name, f"{int(sel.sum()):,} · {100 * sel.mean():.2f}% · median path "
                         f"{np.median(path[sel]):.1f} m\n{why}")
        ax.set_xlabel("lateral y (m)  ·  + = LEFT", fontsize=8, color=INK_2)
        if i == 0:
            ax.set_ylabel("forward from ego at $t_0$  (m)", fontsize=8.5, color=INK_2)

    # class breakdown -- one series, values labelled directly
    ax = fig.add_subplot(gs[0, 3])
    names = ["cruising", "brake to stop", "decel, no stop", "stays stopped", "standing start"]
    vals = [100 * s.mean() for s in (cruise, brake, decel, stopped, start)]
    cols = [BLUE[1], BLUE[5], BLUE[4], BLUE[2], BLUE[3]]
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, vals, height=0.62, color=cols, edgecolor="none")
    for yv, v in zip(ypos, vals):
        ax.annotate(f"{v:.2f}%", (v, yv), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK_2, va="center")
    ax.set_yticks(ypos, names, fontsize=8, color=INK_2)
    ax.set_xlim(0, 92)
    _spines(ax, "x")
    _title(ax, "Class breakdown",
           f"braking of any kind: {100 * (brake | decel).mean():.1f}% of anchors")
    ax.set_xlabel("% of anchors", fontsize=8.5, color=INK_2)

    # -- row 2 (a) the deceleration profile itself
    ax = fig.add_subplot(gs[1, 0])
    tt = np.arange(1, 65) * DT
    vb = vfut[brake]
    v_hi = float(np.ceil(np.percentile(vb, 99.5)))
    hh2, _, _ = np.histogram2d(np.repeat(tt, len(vb)), np.clip(vb.T.ravel(), 0, v_hi),
                               bins=(64, 50), range=[(DT, 6.4), (0, v_hi)])
    tot = hh2.sum(axis=1, keepdims=True)
    hh2 = np.divide(hh2, tot, out=np.zeros_like(hh2), where=tot > 0)
    hh2[hh2 == 0] = np.nan
    im2 = ax.imshow(hh2.T, origin="lower", extent=(DT, 6.4, 0, v_hi), aspect="auto",
                    cmap=CMAP, vmin=0, vmax=float(np.nanpercentile(hh2, 99)))
    med = np.median(vb, axis=0)
    ax.plot(tt, med, color=INK, lw=1.8, zorder=4)
    ax.annotate("median", (DT, med[0]), xytext=(4, 5), textcoords="offset points",
                fontsize=7, color=INK, ha="left")
    for q, lab in ((10, "10th"), (90, "90th")):
        pc = np.percentile(vb, q, axis=0)
        ax.plot(tt, pc, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=4)
        ax.annotate(lab, (DT, pc[0]), xytext=(4, 5), textcoords="offset points",
                    fontsize=7, color=INK, ha="left")
    _spines(ax)
    # The 90th percentile turning back UP past ~5.5 s is real, not an artefact: a slice of
    # these anchors stop and then move off again inside the horizon (stop-and-go).
    p90 = np.percentile(vb, 90, axis=0)
    resumed = 100 * ((vb.min(1) < STOP) & (vb[:, -1] > 1.0)).mean()
    _title(ax, "Braking profile (brake to stop)",
           f"column-normalised · {resumed:.1f}% stop then move off again")
    ax.set_xlabel("time after $t_0$  (s)", fontsize=8.5, color=INK_2)
    ax.set_ylabel("m/s", fontsize=8.5, color=INK_2)

    # -- row 2 (b) when the stop happens
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(t_stop, bins=np.arange(0, 6.5, 0.2), color=BLUE[3], edgecolor="none")
    ax.axvline(np.median(t_stop), color=INK_2, lw=1.0, ls=(0, (3, 2)))
    ax.annotate(f"median {np.median(t_stop):.1f} s", (np.median(t_stop), ax.get_ylim()[1] * 0.95),
                xytext=(6, 0), textcoords="offset points", fontsize=7.5, color=INK_2, va="top")
    _spines(ax, "y")
    _title(ax, "Time to reach standstill", f"{100 * (t_stop <= 6.4).mean():.0f}% inside the horizon "
                                           "by construction")
    ax.set_xlabel("s after $t_0$", fontsize=8.5, color=INK_2)
    ax.set_ylabel("anchors", fontsize=8.5, color=INK_2)

    # -- row 2 (c) how far it travels first
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(np.clip(d_stop, 0, 60), bins=60, color=BLUE[3], edgecolor="none")
    ax.axvline(np.median(d_stop), color=INK_2, lw=1.0, ls=(0, (3, 2)))
    ax.annotate(f"median {np.median(d_stop):.1f} m", (np.median(d_stop), ax.get_ylim()[1] * 0.95),
                xytext=(6, 0), textcoords="offset points", fontsize=7.5, color=INK_2, va="top")
    _spines(ax, "y")
    _title(ax, "Distance travelled before stopping",
           f"peak decel: median {np.median(peak):.2f}, 10th pct {np.percentile(peak, 10):.2f} m/s²")
    ax.set_xlabel("m (clipped at 60)", fontsize=8.5, color=INK_2)
    ax.set_ylabel("anchors", fontsize=8.5, color=INK_2)

    # -- row 2 (d) does the nav command predict braking?
    ax = fig.add_subplot(gs[1, 3])
    cmd = np.where(np.char.startswith(nav, "Continue straight"), "Continue straight",
          np.where(np.char.startswith(nav, "Turn left"), "Turn left",
          np.where(np.char.startswith(nav, "Turn right"), "Turn right", "Reverse")))
    order = ["Turn left", "Turn right", "Continue straight", "Reverse"]
    rate = [100 * (brake & (cmd == c)).sum() / max((cmd == c).sum(), 1) for c in order]
    cnt = [int((cmd == c).sum()) for c in order]
    yp = np.arange(len(order))[::-1]
    ax.barh(yp, rate, height=0.6, color=BLUE[4], edgecolor="none")
    for y_, r_, c_ in zip(yp, rate, cnt):
        ax.annotate(f"{r_:.1f}%   (n={c_:,})", (r_, y_), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=INK_2, va="center")
    ax.set_yticks(yp, order, fontsize=8, color=INK_2)
    ax.set_xlim(0, 17)
    _spines(ax, "x")
    _title(ax, "Brake-to-stop rate by nav command", "turns brake to a stop ~1.4× as often")
    ax.set_xlabel("% of that command's anchors", fontsize=8.5, color=INK_2)

    for mappable, box, lab in ((im, [0.945, 0.50, 0.009, 0.28], "waypoints per 0.5 m cell (log$_{10}$)"),
                               (im2, [0.945, 0.14, 0.009, 0.20], "share of anchors per cell")):
        cax = fig.add_axes(box)
        cb = fig.colorbar(mappable, cax=cax)
        cb.set_label(lab, fontsize=7, color=INK_2)
        cb.ax.tick_params(colors=INK_2, labelsize=6.5, length=3, width=0.6)
        cb.outline.set_visible(False)

    fig.suptitle("LCDrive 50k turn-preserved anchors — deceleration and braking events",
                 fontsize=13, color=INK, x=0.05, ha="left", y=0.965)
    fig.text(0.05, 0.915,
             f"{n:,} anchors · classes from the 10 Hz speed series (moving ≥ {MOVING:g} m/s, "
             f"stopped < {STOP:g} m/s) · ⚠️ BEV panels share a ZOOMED extent, not the main "
             f"figure's 160 m axis", fontsize=9, color=INK_2, ha="left")
    fig.savefig(a["out"], dpi=170, facecolor=SURFACE)
    print(f"[brake] wrote {a['out']}", flush=True)
    for name, sel in (("brake_to_stop", brake), ("decel_no_stop", decel),
                      ("standing_start", start), ("stays_stopped", stopped),
                      ("cruising", cruise)):
        print(f"[brake] {name:15s} n={int(sel.sum()):6d} {100*sel.mean():6.2f}%  "
              f"median path {np.median(path[sel]):6.1f} m", flush=True)
    print("DONE_BRAKE", flush=True)


if __name__ == "__main__":
    main()
