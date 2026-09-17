# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Front camera at t0 and t0+3 s, plus the BEV future, for two samples of every nav command.

One row per command in the manifest's four-word vocabulary, two samples each, and per sample:
the front-wide frame the model sees at ``t0``, the same camera 3 s later (i.e. what the
commanded maneuver actually led to), and the GT future in BEV with the ``t0+3 s`` waypoint
marked so the picture and the curve are tied to the same instant.

⚠️ ``load_physical_aiavdataset`` CANNOT supply the t0+3 s frame -- it decodes a fixed
``[t0-0.3, t0-0.2, t0-0.1, t0]`` window. This calls ``decode_images_from_timestamps``
directly. The decoder returns the nearest keyframe-seekable frame, off by up to one frame
interval (~33 ms at 30 fps), and the actual timestamp is printed in each panel title rather
than assumed.

⚠️ SAMPLES ARE PICKED TO BE LEGIBLE, NOT RANDOM: turns are drawn from |Δψ| >= 45 deg and
straights from |Δψ| < 5 deg with v0 > 6 m/s, so the pictures show the maneuver rather than
an ambiguous 16-degree drift. These are ILLUSTRATIONS; every distribution claim belongs to
the other figures in this set.

Usage::

    python -m alpamayo1_5_distill.scripts.plot_nav_samples \
        npz=<from extract_train_trajectories.py> out=<png> [seed=0] [dt_s=3.0]
"""

from __future__ import annotations

import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK_2, INK_3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8a87"
TRAJ, MARK = "#2a78d6", "#eb6834"
LABEL = {"straight": "Continue straight", "left": "Turn left", "right": "Turn right",
         "reverse": "Reverse"}


def main() -> None:
    a = dict(q.split("=", 1) for q in sys.argv[1:] if "=" in q)
    d = np.load(a["npz"], allow_pickle=True)
    out, seed, dt_s = a["out"], int(a.get("seed", 0)), float(a.get("dt_s", 3.0))
    k3 = int(round(dt_s / 0.1)) - 1          # future[i] is t0 + (i+1)*0.1 s

    from alpamayo.data.pai_utils import PhysicalAIAVDatasetLocalInterface
    avdi = PhysicalAIAVDatasetLocalInterface(
        local_dir=a.get("local_dir", "/data/datasets/physical_ai_av/"),
        chunk_ids=a.get("chunk_ids", "0-3146"))
    CAM = avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV

    ok = d["ok"]
    idx_all = np.flatnonzero(ok)
    fut = d["future_xyz"][ok]
    hist = d["history_xyz"][ok]
    yaw = np.degrees(d["final_yaw"][ok])
    nav = d["nav_text"].astype(str)[ok]
    clip = d["clip_ids"][ok].astype(str)
    t0s = d["t0_relative"][ok]
    v0 = np.linalg.norm(
        np.diff(np.concatenate([hist, fut], axis=1)[..., :2], axis=1), axis=-1)[:, 14] / 0.1

    cmd = np.where(np.char.startswith(nav, "Continue straight"), "straight",
          np.where(np.char.startswith(nav, "Turn left"), "left",
          np.where(np.char.startswith(nav, "Turn right"), "right", "reverse")))
    pools = {
        "straight": np.flatnonzero((cmd == "straight") & (np.abs(yaw) < 5) & (v0 > 6)),
        "left": np.flatnonzero((cmd == "left") & (yaw >= 45)),
        "right": np.flatnonzero((cmd == "right") & (yaw <= -45)),
        "reverse": np.flatnonzero(cmd == "reverse"),
    }
    rng = np.random.default_rng(seed)

    def fetch(i):
        """(frame_t0, frame_t0+dt, actual_us) or None if this clip will not decode."""
        try:
            cam = avdi.get_clip_feature(clip[i], CAM)
            want = np.array([int(t0s[i]), int(t0s[i] + dt_s * 1e6)], dtype=np.int64)
            fr, got = cam.decode_images_from_timestamps(want)
            if fr is None or len(fr) != 2:
                return None
            return fr[0], fr[1], got
        except Exception as ex:                                   # noqa: BLE001
            print(f"[samples] decode failed {clip[i][:8]} @{t0s[i]}: {type(ex).__name__}",
                  flush=True)
            return None

    picks = {}
    for key, pool in pools.items():
        order = rng.permutation(pool)
        chosen = []
        for i in order:
            got = fetch(int(i))
            if got is not None:
                chosen.append((int(i), got))
            if len(chosen) == 2:
                break
        picks[key] = chosen
        print(f"[samples] {LABEL[key]:18s} pool={len(pool):5d}  picked={len(chosen)}", flush=True)

    keys = [k for k in ("straight", "left", "right", "reverse") if picks[k]]
    fig = plt.figure(figsize=(16.4, 3.05 * len(keys)), facecolor=SURFACE)
    gs = fig.add_gridspec(len(keys), 6, width_ratios=[1.55, 1.55, 1.0, 1.55, 1.55, 1.0],
                          hspace=0.42, wspace=0.16, left=0.035, right=0.995,
                          top=0.885, bottom=0.02)

    for r, key in enumerate(keys):
        for c, (i, (f0, f3, got)) in enumerate(picks[key]):
            base = 3 * c
            for j, (frame, when) in enumerate(((f0, 0.0), (f3, dt_s))):
                ax = fig.add_subplot(gs[r, base + j])
                ax.imshow(frame[::2, ::2])                        # 2x down for file size
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color(INK_3); s.set_linewidth(0.6)
                off = (got[j] - int(t0s[i])) / 1e6
                ax.set_title(f"$t_0$" if when == 0 else f"$t_0$ + {dt_s:g} s",
                             fontsize=8.5, color=INK, pad=3)
                ax.set_xlabel(f"actual {off:+.2f} s", fontsize=6.5, color=INK_3, labelpad=2)

            # BEV: the GT future, with the instant the second photo was taken marked
            ax = fig.add_subplot(gs[r, base + 2])
            t = fut[i]
            ax.plot(t[:, 1], t[:, 0], color=TRAJ, lw=2.0, solid_capstyle="round", zorder=3)
            ax.plot(hist[i][:, 1], hist[i][:, 0], color=INK_3, lw=1.2, ls=(0, (2, 2)),
                    zorder=2)
            ax.plot([0], [0], marker="^", ms=7, color=INK_2, zorder=5)
            ax.plot([t[k3, 1]], [t[k3, 0]], marker="o", ms=7, mfc=MARK, mec=SURFACE,
                    mew=1.2, zorder=6)
            ax.annotate(f"+{dt_s:g} s", (t[k3, 1], t[k3, 0]), xytext=(7, -2),
                        textcoords="offset points", fontsize=7, color=INK_2)
            span = max(float(np.abs(t[:, :2]).max()) * 1.15, 6.0)
            ax.set_xlim(-span, span); ax.set_ylim(-span * 0.35, span * 1.35)
            # ⚠️ INVERTED: +y is LEFT in the ego frame, so plotting +y rightward mirrors
            # every turn relative to what the camera panel beside it shows.
            ax.invert_xaxis()
            ax.set_aspect("equal")
            ax.set_facecolor(SURFACE)
            ax.tick_params(colors=INK_2, labelsize=6.5, length=2.5, width=0.5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(INK_3); ax.spines[s].set_linewidth(0.6)
            ax.grid(True, color=INK_3, lw=0.4, alpha=0.25)
            ax.set_axisbelow(True)
            ax.set_title(f'"{nav[i]}"\nΔψ {yaw[i]:+.0f}°  ·  $v_0$ {v0[i]:.1f} m/s',
                         fontsize=8, color=INK, pad=3, linespacing=1.4)

        # row label, left of the first image
        ax0 = fig.add_subplot(gs[r, 0])
        ax0.axis("off")
        fig.text(0.006, ax0.get_position().y0 + ax0.get_position().height / 2,
                 LABEL[key], rotation=90, fontsize=11, color=INK, ha="center", va="center")

    fig.suptitle("LCDrive 50k turn-preserved anchors — front camera and GT future, "
                 "two samples per nav command",
                 fontsize=13, color=INK, x=0.035, ha="left", y=0.985)
    fig.text(0.035, 0.945,
             f"front-wide 120° camera · BEV: solid = GT future (6.4 s), dashed = history, "
             f"orange dot = the {dt_s:g} s instant shown in the right-hand photo · "
             f"samples chosen for legibility, not at random",
             fontsize=9, color=INK_2, ha="left")
    fig.savefig(out, dpi=135, facecolor=SURFACE)
    print(f"[samples] wrote {out}", flush=True)
    print("DONE_SAMPLES", flush=True)


if __name__ == "__main__":
    main()
