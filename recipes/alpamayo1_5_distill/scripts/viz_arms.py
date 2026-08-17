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

"""Qualitative comparison: teacher vs blockonly vs kvonly, one clip per LCDrive scenario.

Renders, for each scenario category, a two-panel figure:

  left   front-wide camera at t0 with every arm's trajectory projected onto it
  right  bird's-eye view of the same trajectories

Every arm is driven through the TEACHER'S FROZEN ACTION EXPERT (``StitchedAlpamayoR1``) --
the same path the quantitative numbers come from -- so the pictures and the table are
measuring the same thing.  Scoring the students' own token heads instead would show a
completely different (and, for ``blockonly``, a degenerate) picture.

⚠️ Which of the 6 sampled modes is drawn: the one CLOSEST TO GROUND TRUTH, i.e. the mode
``min_ade`` reports.  That is the honest match to the headline metric, but it is an oracle
choice -- the model does not know which of its 6 modes is best at drive time.  Labelled as
such in the legend so a reader does not mistake it for a single deterministic prediction.

⚠️ Models are loaded and freed ONE AT A TIME, and each arm's trajectories for every clip are
computed before moving on.  Holding teacher (10 B) + two students (6.4 B each) resident at
once is ~46 GB and would leave no room for the rollout.

Usage::

    CUDA_VISIBLE_DEVICES=1 python -m alpamayo1_5_distill.scripts.viz_arms \\
        --config-path pkg://alpamayo1_5_distill/configs \\
        --config-name sft_eval_stitched_4b_lcdrive \\
        ++model.attn_implementation=sdpa ++viz.out_dir=/path/figs
"""

from __future__ import annotations

import csv
import os
import pickle

import hydra
import hydra.utils as hyu
import matplotlib
import numpy as np
import scipy.spatial.transform as spt
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from accelerate.utils import send_to_device  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from alpamayo.visualization.viz import project_waypoints_ftheta  # noqa: E402
from alpamayo1_5_distill.models.stitched_model import StitchedAlpamayoR1  # noqa: E402

TRAIN = "/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training"
TEACHER_CKPT = "/temp/achahe/hf_cache/hub/models--nvidia--Alpamayo-1.5-10B-A1-format"
COSMOS = (
    "/temp/achahe/hf_cache/hub/models--nvidia--Cosmos-Reason2-8B/"
    "snapshots/a9fae2cf89dc64db96b12860417f0eb403013bb9"
)
SCENARIO_CSV = (
    "/temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests/"
    "lcdrive_val_primary_scenario_mysubset.csv"
)

#: (label, colour). Order controls draw order and legend order; GT last so it sits on top.
ARMS = [
    ("teacher", "#e8b100"),
    ("blockonly", "#00b8d4"),
    ("kvonly_e3", "#e0479e"),
]
GT_COLOUR = "#22c55e"

#: Same seed for every arm on a given clip, so the diffusion noise is not what differs
#: between the pictures. Unseeded sampling is what invalidated an earlier finding here.
ROLLOUT_SEED = 1234


def pick_per_category(n: int) -> dict[str, str]:
    """First `n` clips of each scenario category, in file order (deterministic).

    Deterministic order matters: raising `n` must keep every previously chosen clip, so the
    caches from an earlier run stay valid and only the new clips cost a rollout.

    Returns clip_uuid -> "Category #k".
    """
    seen: dict[str, list[str]] = {}
    with open(SCENARIO_CSV) as fh:
        for row in csv.DictReader(fh):
            c = row["scenario_category"]
            if len(seen.setdefault(c, [])) < n:
                seen[c].append(row["clip_uuid"])
    return {u: f"{c} #{i + 1}" for c, us in seen.items() for i, u in enumerate(us)}


def build(kind: str, cfg):
    """One arm's stitched model. `kind` is 'teacher' or a student checkpoint directory."""
    if kind == "teacher":
        return StitchedAlpamayoR1.from_teacher(
            checkpoint_path=TEACHER_CKPT, vlm_name_or_path=COSMOS, attn_implementation="sdpa"
        )
    return hyu.instantiate({**cfg.model, "checkpoint_path": kind}, _convert_="partial")


def all_modes(pred_xyz: torch.Tensor) -> np.ndarray:
    """Every sampled mode, (n_modes, T, 3). Storing all of them is what lets the figure show
    the model's actual spread rather than only its luckiest draw."""
    return pred_xyz.detach().float().cpu().numpy().reshape(-1, *pred_xyz.shape[-2:])


def medoid_of(modes: np.ndarray) -> int:
    """Index of the mode closest to the mean of the modes -- a NON-oracle selection.

    Preferred over both "closest to GT" (an oracle the model cannot evaluate at drive time,
    which flatters whichever arm has the widest spread) and the mean trajectory itself (which
    can average a left-turn and a straight into a path through the divider that no mode
    proposed). The medoid always returns a trajectory the model actually generated.

    Measured on 15 clips, the arm ordering is identical under oracle, medoid, mean and a
    single draw -- so this choice costs nothing in conclusions and removes the oracle.
    """
    mu = modes.mean(axis=0)
    d = np.linalg.norm(modes[..., :2] - mu[None, ..., :2], axis=-1).mean(axis=-1)
    return int(d.argmin())


def draw(clip, category, image, gt_xy, preds, intr, extr, out_dir):
    fig, (ax_img, ax_bev) = plt.subplots(1, 2, figsize=(19, 8))
    fig.suptitle(f"{category}   —   {clip[:8]}", fontsize=13)

    cam_i, cam_e = intr.loc["camera_front_wide_120fov"], extr.loc["camera_front_wide_120fov"]
    K = [cam_i[k] for k in ("width", "height", "cx", "cy",
                            "fw_poly_0", "fw_poly_1", "fw_poly_2", "fw_poly_3", "fw_poly_4")]
    # Extrinsics are a quaternion + translation, NOT a 3x3 matrix -- same unpacking as
    # `viz_waypoints_pai`, whose f-theta projection this reuses.
    rot = spt.Rotation.from_quat(
        [cam_e["qx"], cam_e["qy"], cam_e["qz"], cam_e["qw"]]
    ).as_matrix()
    trans = np.array([cam_e["x"], cam_e["y"], cam_e["z"]], dtype=np.float64)

    # Camera panel: ONE line per arm -- its MEDOID, the mode closest to the mean of its 6.
    # 18 curves on a photo is unreadable, so one line per arm is required; the medoid is the
    # selection a deployed system could actually make. The BEV beside it shows every mode
    # unweighted; read the two together.
    ax_img.imshow(image)
    for name, modes in preds.items():
        colour = dict(ARMS)[name]
        pick = modes[medoid_of(modes)]
        uv = project_waypoints_ftheta(np.asarray(pick, dtype=np.float64), rot, trans, K)
        if len(uv):
            ax_img.plot(uv[:, 0], uv[:, 1], "-o", ms=2.5, lw=1.8, color=colour, alpha=0.9,
                        label=name)
    uv = project_waypoints_ftheta(np.asarray(gt_xy, dtype=np.float64), rot, trans, K)
    if len(uv):
        ax_img.plot(uv[:, 0], uv[:, 1], "-o", ms=2.5, lw=1.6, color=GT_COLOUR,
                    label="ground truth")
    ax_img.legend(loc="upper right", fontsize=8, framealpha=0.75)
    ax_img.set_xlim(0, K[0]); ax_img.set_ylim(K[1], 0); ax_img.axis("off")
    ax_img.set_title("front wide @ t0 — medoid of 6 per arm (non-oracle)", fontsize=10)

    # BEV in the ego frame: +x forward, +y left, so plot (-y, x) to put forward "up".
    # All 6 modes faint, the GT-closest one bold: the fan is what the model actually
    # proposes, the bold line is the single mode `min_ade` scores.
    xs, ys = [0.0], [0.0]
    for name, modes in preds.items():
        colour = dict(ARMS)[name]
        for i, m in enumerate(modes):
            xs += [-m[:, 1].min(), -m[:, 1].max()]; ys += [m[:, 0].min(), m[:, 0].max()]
            ax_bev.plot(-m[:, 1], m[:, 0], "-", lw=1.4, color=colour, alpha=0.55,
                        label=name if i == 0 else None, zorder=3)
    ax_bev.plot(-gt_xy[:, 1], gt_xy[:, 0], "-o", ms=2.5, lw=1.6, color=GT_COLOUR,
                label="ground truth", zorder=5)
    xs += [-gt_xy[:, 1].min(), -gt_xy[:, 1].max()]; ys += [gt_xy[:, 0].min(), gt_xy[:, 0].max()]
    ax_bev.plot(0, 0, marker="^", ms=13, color="k", zorder=6)

    # ⚠️ AXES ARE INDEPENDENTLY SCALED -- deliberately NOT equal aspect. A lane-keeping clip
    # runs ~100 m forward and ~2 m lateral; at equal aspect that is a hairline and the lateral
    # error between arms -- the entire point of the comparison -- is invisible. The cost is
    # that curvature is exaggerated, so these panels show WHERE the arms differ, not the true
    # geometric shape of the path. The camera panel is the undistorted reference.
    def _lim(vals, floor):
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.12, floor)
        return lo - pad, hi + pad

    ax_bev.set_xlim(*_lim(xs, 1.5))
    ax_bev.set_ylim(*_lim(ys, 2.0))
    ax_bev.grid(alpha=0.3)
    ax_bev.set_xlabel("left  →  (m)"); ax_bev.set_ylabel("forward  →  (m)")
    ax_bev.set_title("BEV (ego frame) — all 6 sampled modes, none highlighted\n"
                     "⚠ axes independently scaled: lateral is stretched, curvature exaggerated",
                     fontsize=9)
    ax_bev.legend(loc="best", fontsize=9)

    path = os.path.join(out_dir, f"{category.replace('/', '-').replace(' ', '_')}_{clip[:8]}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    viz = cfg.get("viz", {})
    out_dir = viz.get("out_dir", f"{TRAIN}/figs")
    os.makedirs(out_dir, exist_ok=True)

    per_cat = int(viz.get("per_category", 1))
    by_uuid = pick_per_category(per_cat)
    clips = list(by_uuid)
    print(f"[viz] {per_cat} per category -> {len(clips)} clips", flush=True)

    arms = {
        "teacher": "teacher",
        "blockonly": f"{TRAIN}/output_kd_4b_blockonly_lcdrive/checkpoint-1598",
        "kvonly_e3": f"{TRAIN}/output_kd_4b_kvonly_e3_lcdrive/checkpoint-4794",
    }

    device = torch.device("cuda")
    results: dict[str, dict[str, np.ndarray]] = {}
    frames: dict[str, dict] = {}

    # ⚠️ Cache per arm. A previous run was killed mid-`blockonly` and lost 15 completed
    # teacher rollouts, because results lived only in memory. Each arm is written as soon
    # as it finishes, so a kill costs at most the arm in flight.
    frames_cache = os.path.join(out_dir, "_frames.pkl")
    if os.path.exists(frames_cache):
        with open(frames_cache, "rb") as fh:
            frames = pickle.load(fh)
        print(f"[viz] reusing frames for {len(frames)} clips", flush=True)

    for arm, kind in arms.items():
        cache = os.path.join(out_dir, f"_modes_{arm}.npz")
        results[arm] = {k: v for k, v in np.load(cache).items()} if os.path.exists(cache) else {}
        # A clip still needs a rollout if this arm lacks it, or if we lack its image/calibration.
        missing = [c for c in clips if c not in results[arm] or c not in frames]
        print(f"[viz] {arm}: {len(results[arm])} cached, {len(missing)} to run", flush=True)
        if not missing:
            continue
        uuid_file = os.path.join(out_dir, f"_todo_{arm}.txt")
        with open(uuid_file, "w") as fh:
            fh.write("\n".join(missing) + "\n")
        model = build(kind, cfg).to(device=device).eval()
        # ⚠️ NOT include_extr_intr=True. That flag fetches three features and only two are
        # needed: it also pulls `vehicle_dimensions`, which exists for just 16 of 1818 chunks
        # locally (missing for these clips) and populates only ego_lwh / ego_length_offset --
        # neither of which the f-theta projection touches. Fetching extrinsics and intrinsics
        # directly skips the missing file entirely.
        ds = hyu.instantiate(
            {**cfg.data.val_dataset, "clip_uuid_filter": uuid_file},
            _convert_="partial", model_config=model.config,
        )
        collate = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=model.config)
        loader = DataLoader(ds, batch_size=1, collate_fn=collate, num_workers=2, shuffle=False)
        # NOTE: do NOT reset results[arm] here -- it already holds the clips loaded from the
        # cache above, and clearing it silently drops them from the npz on save.
        for batch in loader:
            clip = batch["clip_id"][0] if isinstance(batch["clip_id"], list) else batch["clip_id"]
            gpu = send_to_device(dict(batch), device)
            torch.manual_seed(ROLLOUT_SEED)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                xyz, _ = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=gpu, num_traj_samples=6, num_traj_sets=1,
                    top_p=0.98, temperature=0.6, max_generation_length=256,
                )
            gt = batch["ego_future_xyz"][0, 0].float().numpy()
            results[arm][clip] = all_modes(xyz)
            if clip not in frames:  # image + calibration are arm-independent; keep one copy
                frames[clip] = {
                    "gt": gt,
                    "image": batch["image_frames"][0].flatten(0, 1)[7].permute(1, 2, 0).numpy(),
                    "intr": ds.avdi.get_clip_feature(clip, "camera_intrinsics"),
                    "extr": ds.avdi.get_clip_feature(clip, "sensor_extrinsics"),
                }
            print(f"[viz] {arm} {by_uuid.get(clip, '?')} {clip[:8]}", flush=True)
        np.savez(cache, **results[arm])
        with open(frames_cache, "wb") as fh:
            pickle.dump(frames, fh)
        print(f"[viz] {arm}: cached {len(results[arm])} rollouts -> {cache}", flush=True)
        del model
        torch.cuda.empty_cache()

    for clip in clips:
        f = frames[clip]
        preds = {a: results[a][clip] for a in arms if clip in results.get(a, {})}
        img = f["image"]
        if img.max() <= 1.01:
            img = (img * 255).astype(np.uint8)
        path = draw(clip, by_uuid.get(clip, "?"), img.astype(np.uint8),
                    f["gt"], preds, f["intr"], f["extr"], out_dir)
        print(f"[viz] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
