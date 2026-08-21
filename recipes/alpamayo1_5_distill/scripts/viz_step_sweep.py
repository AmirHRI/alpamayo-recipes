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

r"""Qualitative view of Gate C: what the proposal DISTRIBUTION looks like at each step count.

The Gate C table says diversity falls 1.588 -> 0.649 m as the sampler goes 10 -> 1 Euler
steps.  This draws that: one small-multiple panel per step count, all six sampled modes in
each, on shared axes so the fan widths are directly comparable, beside the front-wide camera
the model actually saw.

**No GPU and no model.**  Every trajectory here is read from the `.npz` files
`eval_step_sweep.py` already wrote, so this cannot disagree with the table -- it is the same
tensors.  The dataset is opened WITHOUT preprocessing purely to fetch the image and the
camera calibration.

⚠️ The BEV panels share one set of limits, unlike ``viz_arms.py`` which scales each panel to
its own data.  Comparing fan widths across panels is the entire point here, and per-panel
limits would rescale exactly the quantity being compared.  Axes are still independently
scaled in x and y (lateral stretched), so curvature is exaggerated -- the camera panel is the
undistorted reference.

⚠️ Trajectories were generated from a TWO-camera prompt (front-wide + front-tele), matching
the Gate C run.  The photo is front-wide, one of the two the model saw.

Usage::

    python -m alpamayo1_5_distill.scripts.viz_step_sweep \
      --config-path pkg://alpamayo1_5_distill/configs \
      --config-name sft_eval_stitched_4b_lcdrive ++viz.n_clips=3
"""

from __future__ import annotations

import os
import tempfile

import hydra
import hydra.utils as hyu
import matplotlib
import numpy as np
import scipy.spatial.transform as spt

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from alpamayo.visualization.viz import project_waypoints_ftheta  # noqa: E402

SWEEP = "/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/stepsweep"
STEPS = [1, 2, 3, 5, 10]

#: Step count is an ORDINAL variable, so it takes a one-hue ramp light->dark, not categorical
#: hues. Validated as an ordinal ramp (monotone L, adjacent dL >= 0.06, light end 2.06:1 vs
#: surface, hue spread 3 deg). Ground truth is a different ENTITY, so it takes a categorical
#: slot: orange, which separates from every ramp step at CVD dE 23-38 (floor 8).
RAMP = {1: "#86b6ef", 2: "#5598e7", 3: "#2a78d6", 5: "#1c5cab", 10: "#104281"}
GT = "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"

#: front_wide is index 1 in the loader's camera_features order, frame 3 is t0 (the last of 4).
CAM_FW, FRAME_T0 = 1, 3


def load_sweep():
    """{steps: {clip: [6, 64, 2]}} plus {clip: gt[64, 2]}, aligned across all step counts."""
    per, gt = {}, {}
    for s in STEPS:
        d = np.load(os.path.join(SWEEP, f"teacher_k{s}.npz"), allow_pickle=True)
        ids = list(map(str, d["clip_ids"]))
        # ⚠️ keep 3D: the f-theta projection needs z. BEV slices [..., :2] at use.
        per[s] = {c: d["pred_xyz"][i].astype(np.float32)[0] for i, c in enumerate(ids)}
        if not gt:
            gt = {c: d["gt_xyz"][i].astype(np.float32) for i, c in enumerate(ids)}
    common = sorted(set.intersection(*(set(v) for v in per.values())))
    return {s: {c: per[s][c] for c in common} for s in STEPS}, {c: gt[c] for c in common}, common


def spread(modes):
    """Mean pairwise L2 between the 6 modes -- the same quantity the table calls `diversity`."""
    n = len(modes)
    m2 = modes[..., :2]
    d = np.linalg.norm(m2[:, None] - m2[None, :], axis=-1).mean(-1)
    iu = np.triu_indices(n, 1)
    return float(d[iu].mean())


def pick(per, gt, common, n):
    """Clips where the step count visibly matters, plus a typical one for contrast."""
    coll = sorted(common, key=lambda c: spread(per[10][c]) - spread(per[1][c]), reverse=True)
    wide = sorted(common, key=lambda c: spread(per[10][c]), reverse=True)
    out = [wide[0], coll[0], common[len(common) // 2]]
    seen, uniq = set(), []
    for c in out + wide:
        if c not in seen:
            seen.add(c); uniq.append(c)
        if len(uniq) == n:
            break
    return uniq


def draw(clip, image, gt_xy, per, intr, extr, out_dir):
    fig = plt.figure(figsize=(5.6 + 2.7 * len(STEPS), 4.9))
    gs = fig.add_gridspec(1, len(STEPS) + 1, width_ratios=[1.9] + [1] * len(STEPS),
                      wspace=0.30, left=0.005, right=0.995, top=0.80, bottom=0.13)
    fig.suptitle(f"Proposal distribution vs denoising steps — clip {clip[:8]}   "
                 f"(teacher, 2 cameras, 6 samples, seeded)", fontsize=12, color=INK)

    cam_i, cam_e = intr.loc["camera_front_wide_120fov"], extr.loc["camera_front_wide_120fov"]
    K = [cam_i[k] for k in ("width", "height", "cx", "cy",
                            "fw_poly_0", "fw_poly_1", "fw_poly_2", "fw_poly_3", "fw_poly_4")]
    rot = spt.Rotation.from_quat([cam_e["qx"], cam_e["qy"], cam_e["qz"], cam_e["qw"]]).as_matrix()
    trans = np.array([cam_e["x"], cam_e["y"], cam_e["z"]], dtype=np.float64)

    # Camera: only the two EXTREMES overlaid. All 30 curves on a photo is unreadable, and the
    # 1-vs-10 contrast is the finding.
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image)
    for s in (1, 10):
        for i, m in enumerate(per[s][clip]):
            uv = project_waypoints_ftheta(np.asarray(m, dtype=np.float64), rot, trans, K)
            if len(uv):
                ax.plot(uv[:, 0], uv[:, 1], "-", lw=1.7, color=RAMP[s], alpha=0.85,
                        label=f"{s} step{'s' if s > 1 else ''}" if i == 0 else None)
    uv = project_waypoints_ftheta(np.asarray(gt_xy, dtype=np.float64), rot, trans, K)
    if len(uv):
        ax.plot(uv[:, 0], uv[:, 1], "-", lw=2.0, color=GT, label="ground truth")
    ax.set_xlim(0, K[0]); ax.set_ylim(K[1], 0); ax.axis("off")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.set_title("front-wide @ t0 — 1 vs 10 steps, all 6 modes", fontsize=9, color=MUTED)
    ax.set_anchor("N")

    # Shared BEV limits across every panel: fan width is the comparison.
    allm = np.concatenate([per[s][clip][..., :2].reshape(-1, 2) for s in STEPS]
                          + [gt_xy[:, :2]])
    xs, ys = -allm[:, 1], allm[:, 0]
    pad_x = max((xs.max() - xs.min()) * 0.14, 1.5)
    pad_y = max((ys.max() - ys.min()) * 0.10, 2.0)
    xlim = (xs.min() - pad_x, xs.max() + pad_x)
    ylim = (min(ys.min(), 0) - pad_y, ys.max() + pad_y)

    for j, s in enumerate(STEPS):
        ax = fig.add_subplot(gs[0, j + 1])
        modes = per[s][clip]
        for m in modes:
            ax.plot(-m[:, 1], m[:, 0], "-", lw=1.9, color=RAMP[s], alpha=0.9, zorder=3)
        ax.plot(-gt_xy[:, 1], gt_xy[:, 0], "-", lw=1.7, color=GT, alpha=0.95, zorder=5)
        ax.plot(0, 0, marker="^", ms=9, color=INK, zorder=6)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_title(f"{s} step{'s' if s > 1 else ''}\nspread {spread(modes):.2f} m",
                     fontsize=10, color=INK)
        ax.tick_params(labelsize=7, colors=MUTED)
        if j:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("forward (m)", fontsize=8, color=MUTED, labelpad=1)
        ax.set_xlabel("left (m)", fontsize=8, color=MUTED)

    fig.text(0.5, 0.015, "BEV panels share axes so fan widths are comparable; x and y are "
             "independently scaled, so lateral is stretched and curvature exaggerated — the "
             "camera panel is the undistorted reference.", ha="center", fontsize=8, color=MUTED)
    path = os.path.join(out_dir, f"stepdist_{clip[:8]}.png")
    fig.savefig(path, dpi=125, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return path


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    viz = cfg.get("viz", {})
    out_dir = str(viz.get("out_dir", f"{SWEEP}/figs"))
    os.makedirs(out_dir, exist_ok=True)
    per, gt, common = load_sweep()
    clips = pick(per, gt, common, int(viz.get("n_clips", 3)))
    print(f"[viz] {len(common)} clips available; drawing {clips}", flush=True)

    # Dataset WITHOUT preprocessing: we only need pixels + calibration, so no model_config,
    # no tokenizer, no GPU.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("\n".join(clips) + "\n")
        uuid_file = fh.name
    ds_cfg = OmegaConf.to_container(cfg.data.val_dataset, resolve=True)
    ds_cfg.pop("vla_preprocess_args", None)
    ds_cfg["clip_uuid_filter"] = uuid_file
    ds = hyu.instantiate(ds_cfg, _convert_="partial", model_config=None,
                         vla_preprocess_args=None)
    print(f"[viz] dataset has {len(ds)} clips", flush=True)

    for i in range(len(ds)):
        s = ds[i]
        clip = s["clip_id"]
        if clip not in per[10]:
            print(f"[viz] skip {clip[:8]}: not in the sweep", flush=True)
            continue
        img = np.asarray(s["image_frames"][CAM_FW, FRAME_T0]).transpose(1, 2, 0)
        if img.max() <= 1.01:
            img = (img * 255).astype(np.uint8)
        p = draw(clip, img.astype(np.uint8), gt[clip], per,
                 ds.avdi.get_clip_feature(clip, "camera_intrinsics"),
                 ds.avdi.get_clip_feature(clip, "sensor_extrinsics"), out_dir)
        print(f"[viz] {clip[:8]} -> {p}", flush=True)
    print("DONE_VIZ", flush=True)


if __name__ == "__main__":
    main()
