# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""BEV of the GT futures on the JOINT lateral x longitudinal maneuver grid.

⚠️ ``meta_action`` CANNOT support this plot. Its lateral and longitudinal tokens are disjoint
families and only 3,674 of 109,997 anchors (3.3%) carry one of each, so a grid of label PAIRS
would be 97% empty. Both axes are therefore derived from the GEOMETRY, which classifies every
anchor by the same rule, and the thresholds are calibrated against meta_action:

    lateral   |dpsi| over 6.4 s        go_straight p50 1.1 deg, steer_* p50 4.4-5.0 (p90 ~45),
                                       sharp_steer_* p10 38-43   ->  cuts at 3 and 40 deg
    longitud. dv = v_end - v0          strong_acc p50 +7.0, gentle_acc +2.3, maintain +0.1,
                                       gentle_dec -2.4, strong_dec -5.8  ->  cuts at +-1, +-4
                                       'stop' is separate: 100% of those anchors reach < 0.5
                                       m/s while dv ~ 0 (they start AND end near rest)

The printed confusion against meta_action is the check that these cuts mean what they claim.

⚠️ ``stop`` takes precedence over the dv bins: reaching standstill is the physically distinct
event, and 44.6% of meta_action's own ``strong_deceleration`` anchors do it too.

⚠️ x axis INVERTED so the ego's LEFT is on the LEFT (ego frame is right-handed, x forward,
z up, so +y is LEFT). Tick values are true ego-frame y and increase leftward.

Usage::

    python -m alpamayo1_5_distill.scripts.plot_train_traj_joint \
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
DT = 0.1
STRAIGHT_DEG, SHARP_DEG = 3.0, 40.0
DV_GENTLE, DV_STRONG = 1.0, 4.0
STOP_MS = 0.5


def main() -> None:
    a = dict(q.split("=", 1) for q in sys.argv[1:] if "=" in q)
    d = np.load(a["npz"], allow_pickle=True)
    ok = d["ok"]
    fut, hist = d["future_xyz"][ok], d["history_xyz"][ok]
    yaw = np.degrees(d["final_yaw"][ok])
    ma = d["meta_action"].astype(str)[ok]
    n = len(fut)

    sp = np.linalg.norm(np.diff(np.concatenate([hist, fut], 1)[..., :2], axis=1), axis=-1) / DT
    v0, vfut = sp[:, 14], sp[:, 15:]
    dv = vfut[:, -1] - v0
    reaches_stop = vfut.min(1) < STOP_MS

    # ---- lateral classes, ordered LEFT -> RIGHT to match the inverted axis
    lat_cls = [
        (f"sharp left\n(Δψ > {SHARP_DEG:g}°)", yaw > SHARP_DEG),
        (f"left\n({STRAIGHT_DEG:g}–{SHARP_DEG:g}°)", (yaw > STRAIGHT_DEG) & (yaw <= SHARP_DEG)),
        (f"straight\n(|Δψ| ≤ {STRAIGHT_DEG:g}°)", np.abs(yaw) <= STRAIGHT_DEG),
        (f"right\n(−{SHARP_DEG:g}–−{STRAIGHT_DEG:g}°)",
         (yaw < -STRAIGHT_DEG) & (yaw >= -SHARP_DEG)),
        (f"sharp right\n(Δψ < −{SHARP_DEG:g}°)", yaw < -SHARP_DEG),
    ]
    # ---- longitudinal classes; `stop` wins over the dv bins
    m = ~reaches_stop
    lon_cls = [
        (f"strong accel\n(Δv ≥ +{DV_STRONG:g})", m & (dv >= DV_STRONG)),
        (f"gentle accel\n(+{DV_GENTLE:g}…+{DV_STRONG:g})", m & (dv >= DV_GENTLE) & (dv < DV_STRONG)),
        (f"maintain\n(|Δv| < {DV_GENTLE:g})", m & (np.abs(dv) < DV_GENTLE)),
        (f"gentle decel\n(−{DV_STRONG:g}…−{DV_GENTLE:g})", m & (dv <= -DV_GENTLE) & (dv > -DV_STRONG)),
        (f"strong decel\n(Δv ≤ −{DV_STRONG:g})", m & (dv <= -DV_STRONG)),
        (f"stop\n(reaches < {STOP_MS:g} m/s)", reaches_stop),
    ]
    assert sum(int(s.sum()) for _, s in lat_cls) == n, "lateral classes must partition"
    assert sum(int(s.sum()) for _, s in lon_cls) == n, "longitudinal classes must partition"

    # validate the cuts against meta_action where a label exists
    tok = [set(t for t in s.split(",") if t) for s in ma]
    print("[joint] threshold check vs meta_action (share of each token landing in each class):")
    for t, names in (("lateral", ("go_straight", "steer_left", "sharp_steer_left")),
                     ("longitudinal", ("strong_acceleration", "maintain_speed", "stop"))):
        cls = lat_cls if t == "lateral" else lon_cls
        for nm in names:
            s = np.array([nm in x for x in tok])
            best = max(cls, key=lambda c: (c[1] & s).sum())
            print(f"    {nm:<20} -> {best[0].splitlines()[0]:<13} "
                  f"{100*(best[1] & s).sum()/max(s.sum(),1):5.1f}%", flush=True)

    fwd = float(np.ceil(np.percentile(fut[..., 0], 97.0) / 10) * 10)
    lat = float(np.ceil(np.percentile(np.abs(fut[..., 1]), 99.5) / 10) * 10)
    extent = (-lat, lat, -8.0, fwd)
    bins = (int(2 * lat / 1.0), int((fwd + 8) / 1.0))
    hh, _, _ = np.histogram2d(fut[..., 1].ravel(), fut[..., 0].ravel(), bins=bins,
                              range=[[extent[0], extent[1]], [extent[2], extent[3]]])
    vmax = float(np.log10(max(hh.max(), 10)))

    nr, nc = len(lon_cls), len(lat_cls)
    fig = plt.figure(figsize=(2.55 * nc + 1.6, 3.15 * nr + 1.3), facecolor=SURFACE)
    gs = fig.add_gridspec(nr, nc, hspace=0.16, wspace=0.12,
                          left=0.075, right=0.925, top=0.885, bottom=0.035)
    im = None
    for r, (lon_name, lon_sel) in enumerate(lon_cls):
        for c, (lat_name, lat_sel) in enumerate(lat_cls):
            s = lon_sel & lat_sel
            ax = fig.add_subplot(gs[r, c])
            t = fut[s]
            if len(t) >= 60:
                h, _, _ = np.histogram2d(t[..., 1].ravel(), t[..., 0].ravel(), bins=bins,
                                         range=[[extent[0], extent[1]], [extent[2], extent[3]]])
                with np.errstate(divide="ignore"):
                    h = np.log10(h)
                h[np.isneginf(h)] = np.nan
                got = ax.imshow(h.T, origin="lower", extent=extent, aspect="equal", cmap=CMAP,
                                vmin=0, vmax=vmax, interpolation="nearest")
                im = im or got
            elif len(t):
                ax.plot(t[:, :, 1].T, t[:, :, 0].T, color=BLUE[4], lw=0.8, alpha=0.5)
                ax.set_aspect("equal")
            else:
                ax.set_aspect("equal")
            ax.plot([0], [0], marker="^", ms=4, color=INK_2, zorder=5, clip_on=False)
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            ax.invert_xaxis()
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=INK_3, lw=0.35, alpha=0.22); ax.set_axisbelow(True)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
            for sp_ in ("left", "bottom"):
                ax.spines[sp_].set_color(INK_3); ax.spines[sp_].set_linewidth(0.5)
            ax.set_title(f"{int(s.sum()):,}  ·  {100*s.mean():.2f}%", fontsize=7.5,
                         color=INK if s.sum() else INK_3, pad=3)
            ax.tick_params(colors=INK_2, labelsize=5.5, length=2, width=0.5)
            if r != nr - 1:
                ax.set_xticklabels([])
            if c != 0:
                ax.set_yticklabels([])
            if r == 0:
                ax.text(0.5, 1.20, lat_name, transform=ax.transAxes, fontsize=9, color=INK,
                        ha="center", va="bottom", linespacing=1.4)
            if c == 0:
                ax.text(-0.34, 0.5, lon_name, transform=ax.transAxes, fontsize=9, color=INK,
                        ha="center", va="center", rotation=90, linespacing=1.4)
    fig.text(0.5, 0.012, "lateral y (m)  ·  + = LEFT", fontsize=9, color=INK_2, ha="center")
    fig.text(0.022, 0.5, "forward from ego at $t_0$  (m)", fontsize=9, color=INK_2,
             rotation=90, va="center")
    cax = fig.add_axes([0.935, 0.36, 0.008, 0.28])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("waypoints per 1 m cell (log$_{10}$)", fontsize=7.5, color=INK_2)
    cb.ax.tick_params(colors=INK_2, labelsize=6.5, length=3, width=0.6)
    cb.outline.set_visible(False)
    fig.suptitle("LCDrive anchors — GT futures on the joint lateral × longitudinal grid",
                 fontsize=13.5, color=INK, x=0.075, ha="left", y=0.990)
    fig.text(0.075, 0.966,
             f"{n:,} anchors, every one classified · both axes GEOMETRIC (meta_action pairs "
             f"cover only 3.3%) · thresholds calibrated to meta_action · shares are of the "
             f"whole set, so the 30 cells sum to 100%",
             fontsize=9, color=INK_2, ha="left")
    fig.savefig(a["out"], dpi=150, facecolor=SURFACE)
    print(f"[joint] wrote {a['out']}", flush=True)
    print("[joint] marginals  lateral:", {nm.splitlines()[0]: f"{100*s.mean():.1f}%"
                                          for nm, s in lat_cls}, flush=True)
    print("[joint] marginals  longitud:", {nm.splitlines()[0]: f"{100*s.mean():.1f}%"
                                           for nm, s in lon_cls}, flush=True)
    print("DONE_JOINT", flush=True)


if __name__ == "__main__":
    main()
