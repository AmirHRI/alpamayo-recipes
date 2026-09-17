# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Does the 50k training subset represent the full 110k anchor set?

Both 50k manifests are STRICT SUBSETS of ``nav_lcdrive_train_anchors_all.json`` (verified:
zero keys outside it), and all three span the same 32,022 clips. So one extraction over the
110k covers everything and the subsets are selected by ``(clip_id, t0)`` key -- no second
egomotion pass, and no risk of the three runs disagreeing because of a data-loading
difference.

⚠️ THE DIFFERENCE MAPS ARE SAMPLING RATE, NOT DENSITY RATIO. A subset holding 50,000 of
109,997 anchors is everywhere ~0.4545x the parent by construction, so a raw ratio map is
uniformly negative and says nothing. Dividing by that expected rate puts 0 at "sampled
exactly as often as the parent", which is the question actually being asked.

⚠️ Diverging palette for those maps (blue <-> red, neutral gray midpoint) because the
quantity has POLARITY -- over- vs under-represented. The marginals are three categorical
series (slots 1-3, validated all-pairs on this surface: worst CVD ΔE 9.2, normal-vision
24.0); aqua is 2.74:1 here, under the 3:1 bar, so every series is directly labelled.

Usage::

    python -m alpamayo1_5_distill.scripts.plot_subset_vs_all \
        npz=<110k extraction> out=<png> \
        all_manifest=<...anchors_all.json> \
        subsets=<50k_turnpreserved.json,50k.json>
"""

from __future__ import annotations

import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
#: Red arm of the diverging pair, stepped to mirror BLUE's lightness profile. The palette
#: publishes red only as categorical slot 8 (#e34948) plus "equal step count per arm", so
#: the intermediate steps are interpolated around that anchor rather than invented freehand.
RED = ["#fbdad9", "#f4a9a8", "#e97d7c", "#e34948", "#c22f2e", "#99201f", "#6b1414"]
GRAY_MID = "#f0efec"
SURFACE, INK, INK_2, INK_3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8a87"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE)
DIV = LinearSegmentedColormap.from_list("div_br", BLUE[::-1] + [GRAY_MID] + RED)
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


def _cdf(ax, vals, lo, hi, labels, medians_at):
    grid = np.linspace(lo, hi, 500)
    for k, (name, v) in enumerate(zip(labels, vals)):
        sv = np.sort(v)
        ax.plot(grid, np.searchsorted(sv, grid, side="right") / len(sv), color=SERIES[k],
                lw=2.0, solid_capstyle="round", label=name)
        ax.annotate(f"{name} · med {np.median(sv):.4g}", (hi, medians_at - 0.09 * k),
                    xytext=(-4, 0), textcoords="offset points", fontsize=7,
                    color=INK_2, ha="right", va="center")
    ax.axhline(0.5, color=INK_3, lw=0.6, ls=(0, (4, 4)))
    ax.set_ylim(0, 1.02)
    _spines(ax)


def main() -> None:
    a = dict(q.split("=", 1) for q in sys.argv[1:] if "=" in q)
    d = np.load(a["npz"], allow_pickle=True)
    ok = d["ok"]
    fut, hist = d["future_xyz"][ok], d["history_xyz"][ok]
    yaw = np.degrees(d["final_yaw"][ok])
    nav = d["nav_text"].astype(str)[ok]
    clip, t0s = d["clip_ids"][ok].astype(str), d["t0_relative"][ok]
    n_all = len(fut)

    sp = np.linalg.norm(np.diff(np.concatenate([hist, fut], axis=1)[..., :2], axis=1),
                        axis=-1) / DT
    v0, vfut = sp[:, 14], sp[:, 15:]
    path = np.linalg.norm(np.diff(fut[..., :2], axis=1), axis=-1).sum(1)

    pos = {(c, int(t)): i for i, (c, t) in enumerate(zip(clip, t0s))}
    names, masks = ["all 110k"], [np.ones(n_all, bool)]
    for p in a["subsets"].split(","):
        keys = [(e["clip_id"], int(e["t0_relative"])) for e in json.load(open(p))]
        m = np.zeros(n_all, bool)
        miss = 0
        for k in keys:
            j = pos.get(k)
            if j is None:
                miss += 1
            else:
                m[j] = True
        lbl = "50k turn-pres." if "turnpreserved" in p else "50k naive"
        print(f"[cmp] {lbl}: matched {int(m.sum()):,}/{len(keys):,} (missing {miss})", flush=True)
        names.append(lbl); masks.append(m)

    fwd_hi = float(np.ceil(np.percentile(fut[..., 0], 99.0) / 10) * 10)
    lat = float(np.ceil(np.percentile(np.abs(fut[..., 1]), 99.9) / 10) * 10)
    extent = (-lat, lat, -10.0, fwd_hi)
    bins = (int(2 * lat / 1.0), int((fwd_hi + 10) / 1.0))            # 1 m cells: ratios need counts

    def hist2(mask):
        t = fut[mask]
        h, _, _ = np.histogram2d(t[..., 1].ravel(), t[..., 0].ravel(), bins=bins,
                                 range=[[extent[0], extent[1]], [extent[2], extent[3]]])
        return h

    H = [hist2(m) for m in masks]

    fig = plt.figure(figsize=(15.2, 9.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.5, 1.0], hspace=0.34, wspace=0.30,
                          left=0.05, right=0.925, top=0.845, bottom=0.075)

    with np.errstate(divide="ignore"):
        lg = np.log10(H[0]); lg[np.isneginf(lg)] = np.nan
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(lg.T, origin="lower", extent=extent, aspect="equal", cmap=CMAP, vmin=0,
                   interpolation="nearest")
    ax.plot([0], [0], marker="^", ms=6, color=INK_2, zorder=5, clip_on=False)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.invert_xaxis(); _spines(ax)
    _title(ax, "all 110k — log density", f"{n_all:,} anchors · the parent set")
    ax.set_ylabel("forward from ego at $t_0$  (m)", fontsize=8.5, color=INK_2)
    ax.set_xlabel("lateral y (m)  ·  + = LEFT", fontsize=8, color=INK_2)

    imd = None
    for c, k in enumerate((1, 2), start=1):
        ax = fig.add_subplot(gs[0, c])
        exp = masks[k].sum() / n_all                    # the rate the subset SHOULD show
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.log2((H[k] / np.maximum(H[0], 1e-12)) / exp)
        rel[H[0] < 30] = np.nan                         # too few parent waypoints to be a rate
        imd = ax.imshow(rel.T, origin="lower", extent=extent, aspect="equal", cmap=DIV,
                        vmin=-1.0, vmax=1.0, interpolation="nearest")
        ax.plot([0], [0], marker="^", ms=6, color=INK_2, zorder=5, clip_on=False)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.invert_xaxis(); _spines(ax)
        frac = 100 * np.nanmean(np.abs(rel) > np.log2(1.25))
        _title(ax, f"{names[k]} vs parent",
               f"log$_2$ sampling rate ÷ {exp:.4f}\n{frac:.1f}% of cells off by >25%")
        ax.set_xlabel("lateral y (m)  ·  + = LEFT", fontsize=8, color=INK_2)

    # nav-command and driving-class shares, three series side by side
    ax = fig.add_subplot(gs[0, 3])
    cmd = np.where(np.char.startswith(nav, "Continue straight"), "straight",
          np.where(np.char.startswith(nav, "Turn left"), "left",
          np.where(np.char.startswith(nav, "Turn right"), "right", "reverse")))
    brake = (v0 >= 2.0) & (vfut.min(1) < 0.5)
    stopped = (v0 < 0.5) & (vfut.max(1) < 0.5)
    rows = [("turn cmd", np.isin(cmd, ("left", "right"))),
            ("|Δψ| ≥ 15°", np.abs(yaw) >= 15),
            ("brake to stop", brake),
            ("stays stopped", stopped),
            ("v₀ < 0.5 m/s", v0 < 0.5)]
    yp = np.arange(len(rows))[::-1]
    hgt = 0.24
    for k, m in enumerate(masks):
        vals = [100 * (r & m).sum() / m.sum() for _, r in rows]
        ax.barh(yp + (1 - k) * hgt, vals, height=hgt, color=SERIES[k], edgecolor=SURFACE,
                linewidth=0.8, label=names[k])
        for y_, v_ in zip(yp + (1 - k) * hgt, vals):
            ax.annotate(f"{v_:.2f}", (v_, y_), xytext=(3, 0), textcoords="offset points",
                        fontsize=6.5, color=INK_2, va="center")
    ax.set_yticks(yp, [r for r, _ in rows], fontsize=8, color=INK_2)
    ax.set_xlim(0, 19.5)
    _spines(ax, "x")
    leg = ax.legend(frameon=False, fontsize=7, loc="lower right")
    for t_ in leg.get_texts():
        t_.set_color(INK_2)
    _title(ax, "Composition", "% of each set · labels are the values")
    ax.set_xlabel("% of set", fontsize=8.5, color=INK_2)

    # marginals
    for c, (vals, lo, hi, main, xlab) in enumerate((
        ([yaw[m] for m in masks], -120, 120, "Net heading change", "Δψ over 6.4 s (deg)"),
        ([v0[m] for m in masks], 0, 33, "Speed at $t_0$", "m/s"),
        ([path[m] for m in masks], 0, 200, "Distance in 6.4 s", "path length (m)"),
    )):
        ax = fig.add_subplot(gs[1, c])
        _cdf(ax, vals, lo, hi, names, 0.32)
        ks = [float(np.max(np.abs(
            np.searchsorted(np.sort(vals[0]), g, side="right") / len(vals[0])
            - np.searchsorted(np.sort(vals[k]), g, side="right") / len(vals[k]))))
            for k in (1, 2) for g in [np.linspace(lo, hi, 500)]]
        _title(ax, main, f"CDF · KS vs parent: tp {ks[0]:.4f}, naive {ks[1]:.4f}")
        ax.set_xlabel(xlab, fontsize=8.5, color=INK_2)
        if c == 0:
            ax.set_ylabel("fraction ≤ x", fontsize=8.5, color=INK_2)
            leg = ax.legend(frameon=False, fontsize=7, loc="upper left")
            for t_ in leg.get_texts():
                t_.set_color(INK_2)

    # per-command turn rate, the thing "turn-preserved" is named for
    ax = fig.add_subplot(gs[1, 3])
    labs = ["Continue\nstraight", "Turn\nleft", "Turn\nright", "Reverse"]
    xp = np.arange(4)
    for k, m in enumerate(masks):
        vals = [100 * ((cmd == c) & m).sum() / m.sum()
                for c in ("straight", "left", "right", "reverse")]
        ax.bar(xp + (k - 1) * 0.26, vals, width=0.26, color=SERIES[k], edgecolor=SURFACE,
               linewidth=0.8)
        for x_, v_ in zip(xp + (k - 1) * 0.26, vals):
            ax.annotate(f"{v_:.2f}", (x_, v_), xytext=(0, 2), textcoords="offset points",
                        fontsize=6.5, color=INK_2, ha="center")
    ax.set_yscale("log"); ax.set_ylim(0.02, 300)
    ax.set_xticks(xp, labs, fontsize=7.5, color=INK_2)
    _spines(ax, "y")
    _title(ax, "Nav-command mix", "log scale so Reverse stays visible")
    ax.set_ylabel("% of set (log)", fontsize=8.5, color=INK_2)

    for mp, box, lab in ((im, [0.935, 0.60, 0.009, 0.20], "waypoints per 1 m cell (log$_{10}$)"),
                         (imd, [0.935, 0.36, 0.009, 0.18],
                          "log$_2$ over/under-sampling\n(0 = representative)")):
        cax = fig.add_axes(box)
        cb = fig.colorbar(mp, cax=cax)
        cb.set_label(lab, fontsize=7, color=INK_2)
        cb.ax.tick_params(colors=INK_2, labelsize=6.5, length=3, width=0.6)
        cb.outline.set_visible(False)

    fig.suptitle("Does the 50k training subset represent the full 110k? — LCDrive nav anchors",
                 fontsize=13, color=INK, x=0.05, ha="left", y=0.965)
    fig.text(0.05, 0.905,
             f"both 50k manifests are strict subsets of the 110k (verified) · same 32,022 "
             f"clips throughout · difference maps are LOCAL SAMPLING RATE ÷ the set's overall "
             f"rate, so 0 = representative", fontsize=9, color=INK_2, ha="left")
    fig.savefig(a["out"], dpi=170, facecolor=SURFACE)
    print(f"[cmp] wrote {a['out']}", flush=True)
    print("DONE_CMP", flush=True)


if __name__ == "__main__":
    main()
