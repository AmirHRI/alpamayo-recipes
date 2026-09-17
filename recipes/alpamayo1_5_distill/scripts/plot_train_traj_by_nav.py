# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Trajectories clustered by the NAV INSTRUCTION, and how often the instruction is obeyed.

The manifest carries exactly one nav command per anchor. Collapsing the trailing distance
("Turn left in 13m" -> "Turn left in Nm") leaves a vocabulary of exactly FOUR:

    Continue straight   45,371  90.74%
    Turn right in N m    2,338   4.68%
    Turn left in N m     2,270   4.54%
    Reverse                 21   0.04%

⚠️ NAV COMMAND != REALIZED MANEUVER, and the gap is the point of this figure. The command is
what the route planner asked for; the geometry is what the ego did inside the 6.4 s horizon.
They agree ~90% of the time, which means ~3,540 "Continue straight" anchors DO turn >=15 deg
(lane changes, curved roads, roundabouts) and ~9% of turn commands do not turn yet.

⚠️ REVERSE GETS ITS OWN PANEL AND ITS OWN SCALE. 21 anchors on a 160 m axis is an empty box,
and reverse travels NEGATIVE forward (min -7.1 m), so it is a different phenomenon rather
than a fourth small multiple. The label says so. Its yaw-based classification is also
meaningless -- heading change while backing up does not mean "turned left" -- so it is
excluded from the confusion matrix's column logic rather than silently mislabelled.

Usage::

    python -m alpamayo1_5_distill.scripts.plot_train_traj_by_nav \
        npz=<from extract_train_trajectories.py> out=<png> [turn_deg=15]
"""

from __future__ import annotations

import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK_2, INK_3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8a87"
SERIES = {"straight": "#2a78d6", "left": "#eb6834", "right": "#1baf7a"}
LABEL = {"straight": "Continue straight", "left": "Turn left", "right": "Turn right",
         "reverse": "Reverse"}
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


def _bev(ax, traj, extent, bins, vmax, main, sub):
    """Log-density if there is enough mass; individual paths when there is not."""
    y, x = traj[..., 1].ravel(), traj[..., 0].ravel()
    if len(traj) >= 200:
        h, _, _ = np.histogram2d(y, x, bins=bins,
                                 range=[[extent[0], extent[1]], [extent[2], extent[3]]])
        with np.errstate(divide="ignore"):
            h = np.log10(h)
        h[np.isneginf(h)] = np.nan
        im = ax.imshow(h.T, origin="lower", extent=extent, aspect="equal", cmap=CMAP,
                       vmin=0, vmax=vmax, interpolation="nearest")
    else:
        im = None
        ax.plot(traj[:, :, 1].T, traj[:, :, 0].T, color=BLUE[4], lw=1.0, alpha=0.55,
                solid_capstyle="round")
        ax.set_aspect("equal")
    ax.plot([0], [0], marker="^", ms=6, color=INK_2, zorder=5, clip_on=False)
    ax.axvline(0, color=INK_3, lw=0.6, ls=(0, (4, 4)), zorder=1)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.invert_xaxis()
    _spines(ax)
    _title(ax, main, sub)
    return im


def main() -> None:
    a = dict(q.split("=", 1) for q in sys.argv[1:] if "=" in q)
    d = np.load(a["npz"], allow_pickle=True)
    turn_deg = float(a.get("turn_deg", 15))

    ok = d["ok"]
    fut = d["future_xyz"][ok]
    hist = d["history_xyz"][ok]
    yaw = np.degrees(d["final_yaw"][ok])
    nav = d["nav_text"].astype(str)[ok]
    n = len(fut)

    cmd = np.where(np.char.startswith(nav, "Continue straight"), "straight",
          np.where(np.char.startswith(nav, "Turn left"), "left",
          np.where(np.char.startswith(nav, "Turn right"), "right",
          np.where(np.char.startswith(nav, "Reverse"), "reverse", "?"))))
    assert (cmd != "?").all(), f"unparsed nav_text: {set(nav[cmd == '?'])}"
    geo = np.where(np.abs(yaw) < turn_deg, "straight", np.where(yaw >= turn_deg, "left", "right"))
    dist = np.array([float(m.group(1)) if (m := re.search(r"in (\d+)m", s)) else np.nan
                     for s in nav])

    full = np.concatenate([hist, fut], axis=1)
    v0 = np.linalg.norm(np.diff(full[..., :2], axis=1), axis=-1)[:, 14] / DT

    fwd_hi = float(np.ceil(np.percentile(fut[..., 0], 99.0) / 10) * 10)
    lat = float(np.ceil(np.percentile(np.abs(fut[..., 1]), 99.9) / 10) * 10)
    extent = (-lat, lat, -10.0, fwd_hi)
    bins = (int(2 * lat / 0.5), int((fwd_hi + 10) / 0.5))
    hh, _, _ = np.histogram2d(fut[..., 1].ravel(), fut[..., 0].ravel(), bins=bins,
                              range=[[extent[0], extent[1]], [extent[2], extent[3]]])
    vmax = float(np.log10(max(hh.max(), 10)))

    fig = plt.figure(figsize=(14.6, 9.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.5, 1.0], hspace=0.34, wspace=0.30,
                          left=0.05, right=0.93, top=0.855, bottom=0.075)

    # -- row 1: the three high-count commands, SHARED scale (true small multiples)
    im = None
    for i, c in enumerate(("straight", "left", "right")):
        ax = fig.add_subplot(gs[0, i])
        sel = cmd == c
        got = _bev(ax, fut[sel], extent, bins, vmax, f'"{LABEL[c]}"',
                   f"{int(sel.sum()):,} anchors · {100 * sel.mean():.2f}%")
        im = im or got
        ax.set_xlabel("lateral y (m)  ·  + = LEFT", fontsize=8, color=INK_2)
        if i == 0:
            ax.set_ylabel("forward from ego at $t_0$  (m)", fontsize=8.5, color=INK_2)

    # -- obedience: does the command match what the ego actually did?
    ax = fig.add_subplot(gs[0, 3])
    rows = ("straight", "left", "right")
    mat = np.array([[100 * ((geo == g) & (cmd == c)).sum() / max((cmd == c).sum(), 1)
                     for g in rows] for c in rows])
    ax.imshow(mat, cmap=CMAP, vmin=0, vmax=100, aspect="auto")
    for r in range(3):
        for q in range(3):
            ax.text(q, r, f"{mat[r, q]:.1f}%", ha="center", va="center", fontsize=8.5,
                    color=INK if mat[r, q] < 55 else SURFACE)
    ax.set_xticks(range(3), [LABEL[g].replace("Continue ", "").replace("Turn ", "")
                             for g in rows], fontsize=7.5, color=INK_2)
    ax.set_yticks(range(3), [f'"{LABEL[c]}"' for c in rows], fontsize=7.5, color=INK_2)
    ax.set_xlabel(f"REALIZED in 6.4 s  (|Δψ| ≥ {turn_deg:g}°)", fontsize=8, color=INK_2)
    # No y-label: the row ticks already name the command, and a rotated label here
    # overruns the neighbouring BEV panel.
    _title(ax, "Is the instruction obeyed?",
           "rows = nav COMMAND, row-normalised · diagonal = agreement")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.grid(False)
    ax.tick_params(length=0)

    # -- row 2 (a): Reverse, own scale (21 anchors on a 160 m axis is an empty box)
    ax = fig.add_subplot(gs[1, 0])
    sel = cmd == "reverse"
    rt = fut[sel]
    r_lat = max(2.0, float(np.ceil(np.abs(rt[..., 1]).max())))
    r_ext = (-r_lat, r_lat, float(np.floor(rt[..., 0].min() - 1)),
             max(2.0, float(np.ceil(rt[..., 0].max() + 1))))
    _bev(ax, rt, r_ext, bins, vmax, '"Reverse"  ⚠️ own scale',
         f"{int(sel.sum())} anchors · {100 * sel.mean():.2f}% · travels backwards")
    ax.set_xlabel("lateral y (m)  ·  + = LEFT", fontsize=8, color=INK_2)
    ax.set_ylabel("forward (m)", fontsize=8.5, color=INK_2)

    # -- row 2 (b): where the commanded turn is, and whether it lands inside the horizon
    ax = fig.add_subplot(gs[1, 1])
    turn = np.isin(cmd, ("left", "right"))
    edges = np.arange(0, 45, 2.5)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    frac = np.array([
        100 * ((np.abs(yaw) >= turn_deg) & turn & (dist >= lo) & (dist < hi)).sum()
        / max((turn & (dist >= lo) & (dist < hi)).sum(), 1)
        for lo, hi in zip(edges[:-1], edges[1:])])
    cnt = np.array([(turn & (dist >= lo) & (dist < hi)).sum()
                    for lo, hi in zip(edges[:-1], edges[1:])])
    # ⚠️ NO TWIN AXIS. An earlier revision drew the count as bars and the realized RATE on
    # a second y-scale -- two measures, two scales, the single worst chart mistake. Stacking
    # the same bars by realized/not-yet puts both on ONE axis: the height is the count and
    # the split IS the rate, read directly off the segment boundary.
    done = np.array([((np.abs(yaw) >= turn_deg) & turn & (dist >= lo) & (dist < hi)).sum()
                     for lo, hi in zip(edges[:-1], edges[1:])])
    ax.bar(ctr, done, width=2.2, color=BLUE[4], edgecolor=SURFACE, linewidth=1.0,
           label="turn happens within 6.4 s")
    ax.bar(ctr, cnt - done, width=2.2, bottom=done, color=BLUE[1], edgecolor=SURFACE,
           linewidth=1.0, label="not yet at the horizon")
    _spines(ax, "y")
    ax.set_ylabel("turn commands", fontsize=8.5, color=INK_2)
    ax.set_xlabel('distance in the instruction ("in N m")', fontsize=8, color=INK_2)
    leg = ax.legend(frameon=False, fontsize=7, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK_2)
    overall = 100 * done.sum() / max(cnt.sum(), 1)
    _title(ax, "Commanded turn distance",
           f"{overall:.1f}% of commanded turns land inside the horizon")

    # -- row 2 (c): speed by command. Three series; reverse excluded (n=21) and said so.
    ax = fig.add_subplot(gs[1, 2])
    v_hi = float(np.ceil(np.percentile(v0, 99.5)))
    grid = np.linspace(0, v_hi, 400)
    for k, c in enumerate(("straight", "left", "right")):
        v = np.sort(v0[cmd == c])
        ax.plot(grid, np.searchsorted(v, grid, side="right") / len(v), color=SERIES[c],
                lw=2.0, solid_capstyle="round", label=f'"{LABEL[c]}"')
        ax.annotate(f"{LABEL[c]} · {np.median(v):.1f} m/s", (v_hi, 0.30 - 0.09 * k),
                    xytext=(-4, 0), textcoords="offset points", fontsize=7.5,
                    color=INK_2, ha="right", va="center")
    ax.axhline(0.5, color=INK_3, lw=0.6, ls=(0, (4, 4)))
    ax.set_ylim(0, 1.02)
    _spines(ax)
    leg = ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK_2)
    _title(ax, "Speed at $t_0$ by command", "cumulative · Reverse omitted (n=21)")
    ax.set_xlabel("m/s", fontsize=8.5, color=INK_2)
    ax.set_ylabel("fraction ≤ x", fontsize=8.5, color=INK_2)

    # -- row 2 (d): realized heading change per command
    ax = fig.add_subplot(gs[1, 3])
    for k, c in enumerate(("straight", "left", "right")):
        y = np.sort(yaw[cmd == c])
        ax.plot(y, np.arange(1, len(y) + 1) / len(y), color=SERIES[c], lw=2.0,
                solid_capstyle="round")
        ax.annotate(f"{LABEL[c]}", (118, 0.30 - 0.09 * k), xytext=(-4, 0),
                    textcoords="offset points", fontsize=7.5, color=INK_2,
                    ha="right", va="center")
    for v in (-turn_deg, turn_deg):
        ax.axvline(v, color=INK_3, lw=0.7, ls=(0, (3, 3)))
    ax.set_xlim(-120, 120); ax.set_ylim(0, 1.02)
    _spines(ax)
    _title(ax, "Realized Δψ by command", f"dashed = ±{turn_deg:g}° turn threshold")
    ax.set_xlabel("Δψ over 6.4 s  (deg)", fontsize=8.5, color=INK_2)
    ax.set_ylabel("fraction ≤ x", fontsize=8.5, color=INK_2)

    cax = fig.add_axes([0.945, 0.50, 0.010, 0.30])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("waypoints per 0.5 m cell (log$_{10}$)", fontsize=7.5, color=INK_2)
    cb.ax.tick_params(colors=INK_2, labelsize=6.5, length=3, width=0.6)
    cb.outline.set_visible(False)

    fig.suptitle("LCDrive 50k turn-preserved anchors — trajectories clustered by nav instruction",
                 fontsize=13, color=INK, x=0.05, ha="left", y=0.965)
    fig.text(0.05, 0.915,
             f"{n:,} anchors, exactly one command each · four commands in the vocabulary · "
             f"ego frame at $t_0$, heading up · colour is log$_{{10}}$ count",
             fontsize=9, color=INK_2, ha="left")
    fig.savefig(a["out"], dpi=170, facecolor=SURFACE)
    print(f"[nav] wrote {a['out']}", flush=True)
    for c in ("straight", "left", "right", "reverse"):
        s = cmd == c
        agree = 100 * (geo[s] == c).sum() / max(s.sum(), 1) if c != "reverse" else float("nan")
        print(f"[nav] {LABEL[c]:18s} n={int(s.sum()):6d}  {100*s.mean():6.2f}%  "
              f"obeyed {agree:5.1f}%  median v0 {np.median(v0[s]):5.2f} m/s", flush=True)
    print("DONE_NAV", flush=True)


if __name__ == "__main__":
    main()
