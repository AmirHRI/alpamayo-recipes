#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
r"""Plot predicted vs ground-truth trajectories from eval `.npz` archives, front camera + BEV.

One arm:

    python scripts/plot_traj_npz.py \
        --arm "CD ep2=training/cd_eos4b_gt05_eos_checkpoint-3126_nfe1.npz" \
        --n 5 --out training/viz_cd_nfe1

Two arms on the same axes (the comparison this exists for):

    python scripts/plot_traj_npz.py \
        --arm "CD ep2=training/cd_eos4b_gt05_eos_checkpoint-3126_nfe1.npz" \
        --arm "EoS control=training/eos_4b_2cam_nav_eos_checkpoint-3126_nfe1.npz" \
        --n 5 --out training/viz_cd_vs_eos_nfe1

⚠️ NO MODEL IS LOADED AND NO GPU IS USED. The predictions are already in the archives written by
`evaluate_hf.py`; the dataset is instantiated only to fetch the camera frame and the calibration.
Building the model would cost 22 GB and ~4 minutes to reproduce numbers that are already on disk,
and -- worse -- a fresh rollout would draw different diffusion noise, so the picture would not be
the run that was scored. Everything here is read-only w.r.t. the experiment.

⚠️ The `.npz` is authoritative for BOTH the prediction and the ground truth. `gt_xyz` was written
as `ego_future_xyz[:, -1]`, which is the trajectory group the metrics actually score; re-deriving
GT from the dataset here would risk picking a different group and silently drawing a curve the
reported `ade` never saw.

⚠️ The FIRST `--arm` drives clip selection and the percentile labels; every later arm is looked
up by `clip_id` on those same clips. Selecting per-arm would put each arm's own p90 on the sheet
-- different clips per arm -- which looks like a comparison and is not one.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import hydra
import hydra.utils as hyu
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.transform as spt
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from alpamayo.visualization.viz import project_waypoints_ftheta

#: The loader's camera order (see `data/camera_subset.py`): 0 cross_left, 1 front_wide,
#: 2 cross_right, 3 front_tele. Only front_wide has a calibration entry we project into.
FRONT_WIDE_LOADER_IDX = 1
CAM_NAME = "camera_front_wide_120fov"

GT_COLOUR = "#00c853"
#: Arm colours in `--arm` order. The first arm is the subject, the second the control.
ARM_COLOURS = ["#d81b60", "#1e88e5", "#ff8f00", "#6a1b9a"]


def modes_of(npz, i):
    """All sampled modes for clip row `i` as [K, T, 3]."""
    p = npz["pred_xyz"][i].astype(np.float32)
    return p.reshape(-1, p.shape[-2], p.shape[-1])


def medoid(modes):
    """Index of the mode closest to the mean of the modes.

    ⚠️ The camera panel draws ONE line per arm, not six. With two arms, 12 curves over a photo
    is unreadable, and picking the GT-closest mode would be an oracle choice that flatters
    whichever arm samples more widely -- precisely the effect under test here. The medoid is
    GT-independent and is a selection a deployed system could actually make.
    """
    c = modes.mean(0, keepdims=True)
    return int(np.argmin(np.linalg.norm(modes[..., :2] - c[..., :2], axis=-1).mean(-1)))


def ade_of(modes, gt):
    """Mean-over-modes ade, matching the logged `ade` (`only_xy`)."""
    return float(np.linalg.norm(modes[..., :2] - gt[None, :, :2], axis=-1).mean(-1).mean())


def pick_clips(npz, n, mode="spread"):
    """Choose `n` clip ids. `spread` walks the per-clip error quantiles.

    ⚠️ Deliberately NOT random and NOT the first n. The first n are dataset order, which
    correlates with chunk and therefore with geography/time-of-day; a random draw is
    unreproducible. Quantile spread guarantees the sheet contains a typical case AND a tail
    case, which is the whole reason to look at pictures rather than at the mean.
    """
    pred, gt = npz["pred_xyz"].astype(np.float32), npz["gt_xyz"].astype(np.float32)
    pred = pred.reshape(pred.shape[0], -1, pred.shape[-2], pred.shape[-1])
    # only_xy, matching `compute_minade(only_xy=True)`. Scoring in 3D disagrees with the logs.
    ade = np.linalg.norm(pred[..., :2] - gt[:, None, :, :2], axis=-1).mean(-1).mean(-1)
    ids = np.asarray([str(c) for c in npz["clip_ids"]])
    if mode == "best":
        order = np.argsort(ade)[:n]
    elif mode == "worst":
        order = np.argsort(ade)[::-1][:n]
    else:
        q = np.linspace(0.1, 0.9, n)
        order = np.argsort(ade)[(q * (len(ade) - 1)).astype(int)]
    return [(ids[i], float(ade[i]), float(np.quantile(ade, 0).item())) for i in order], ade


def draw(clip, image, gt, arms, intr, extr, pct, out_dir):
    """`arms` is an ordered {label: [K, T, 3]} mapping; colours follow `--arm` order."""
    fig, (ax_img, ax_bev) = plt.subplots(1, 2, figsize=(19, 8))
    scores = "   ".join(f"{lab} ade {ade_of(m, gt):.3f}" for lab, m in arms.items())
    fig.suptitle(f"{clip[:8]}   —   p{pct:.0f} of the first arm   —   {scores}", fontsize=13)

    cam_i, cam_e = intr.loc[CAM_NAME], extr.loc[CAM_NAME]
    K = [
        cam_i[k]
        for k in ("width", "height", "cx", "cy",
                  "fw_poly_0", "fw_poly_1", "fw_poly_2", "fw_poly_3", "fw_poly_4")
    ]
    # Extrinsics are quaternion + translation, not a matrix -- same unpacking as
    # `viz_waypoints_pai`, whose f-theta projection this reuses.
    rot = spt.Rotation.from_quat(
        [cam_e["qx"], cam_e["qy"], cam_e["qz"], cam_e["qw"]]
    ).as_matrix()
    trans = np.array([cam_e["x"], cam_e["y"], cam_e["z"]], dtype=np.float64)

    ax_img.imshow(image)
    for (lab, modes), colour in zip(arms.items(), ARM_COLOURS):
        pick = modes[medoid(modes)]
        uv = project_waypoints_ftheta(np.asarray(pick, dtype=np.float64), rot, trans, K)
        if len(uv):
            ax_img.plot(uv[:, 0], uv[:, 1], "-", lw=2.0, color=colour, alpha=0.9,
                        label=f"{lab} (medoid)")
    uv = project_waypoints_ftheta(np.asarray(gt, dtype=np.float64), rot, trans, K)
    if len(uv):
        ax_img.plot(uv[:, 0], uv[:, 1], "-o", ms=2.5, lw=1.8, color=GT_COLOUR,
                    label="ground truth")
    ax_img.set_xlim(0, K[0])
    ax_img.set_ylim(K[1], 0)
    ax_img.axis("off")
    ax_img.legend(loc="upper right", fontsize=9, framealpha=0.8)
    # ⚠️ `project_waypoints_ftheta` DROPS points that fall outside the image or behind the
    # camera, so a short curve here means the path left the frame -- not that the model stopped.
    # The BEV panel is the complete trajectory; read them together.
    ax_img.set_title(f"{CAM_NAME} @ t0 — medoid per arm, non-oracle   "
                     f"(points outside the frame are clipped away)", fontsize=10)

    # BEV, ego frame: +x forward, +y left -> plot (-y, x) so forward points up.
    xs, ys = [0.0], [0.0]
    for (lab, modes), colour in zip(arms.items(), ARM_COLOURS):
        for i, m in enumerate(modes):
            ax_bev.plot(-m[:, 1], m[:, 0], "-", lw=1.4, color=colour, alpha=0.55,
                        label=f"{lab} ({len(modes)} modes)" if i == 0 else None, zorder=3)
            xs += [-m[:, 1].min(), -m[:, 1].max()]
            ys += [m[:, 0].min(), m[:, 0].max()]
    ax_bev.plot(-gt[:, 1], gt[:, 0], "-o", ms=2.5, lw=1.8, color=GT_COLOUR,
                label="ground truth", zorder=5)
    xs += [-gt[:, 1].min(), -gt[:, 1].max()]
    ys += [gt[:, 0].min(), gt[:, 0].max()]
    ax_bev.plot(0, 0, marker="^", ms=13, color="k", zorder=6)

    # ⚠️ AXES INDEPENDENTLY SCALED, as in `viz_arms.py`. A lane-keep clip runs ~100 m forward
    # and ~2 m lateral; at equal aspect the lateral spread -- the thing worth seeing -- is a
    # hairline. Cost: curvature is exaggerated and these panels show WHERE the modes differ,
    # not true path geometry. The camera panel is the undistorted reference.
    def _lim(v, floor):
        lo, hi = min(v), max(v)
        pad = max((hi - lo) * 0.12, floor)
        return lo - pad, hi + pad

    ax_bev.set_xlim(*_lim(xs, 1.5))
    ax_bev.set_ylim(*_lim(ys, 2.0))
    ax_bev.grid(alpha=0.3)
    ax_bev.set_xlabel("left  →  (m)")
    ax_bev.set_ylabel("forward  →  (m)")
    ax_bev.set_title("BEV (ego frame), all modes per arm\n"
                     "⚠ axes independently scaled: lateral stretched, curvature exaggerated",
                     fontsize=9)
    ax_bev.legend(loc="best", fontsize=9)

    path = os.path.join(out_dir, f"p{pct:02.0f}_{clip[:8]}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="LABEL=PATH",
                    help="repeatable; the FIRST arm drives clip selection")
    ap.add_argument("--config", default="sft_eval_eos_4b_2cam_nav_lcdrive")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--select", default="spread", choices=["spread", "best", "worst"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"--arm wants LABEL=PATH, got {spec!r}")
        # ⚠️ rsplit, not split: labels legitimately contain '=' ("CD ep2 @NFE=1"), and
        # splitting on the FIRST '=' silently turns "1=/path/x.npz" into the filename.
        lab, path = spec.rsplit("=", 1)
        arms[lab.strip()] = np.load(path.strip(), allow_pickle=True)
    labels = list(arms)

    os.makedirs(args.out, exist_ok=True)
    primary = arms[labels[0]]
    chosen, ade_all = pick_clips(primary, args.n, args.select)
    index = {lab: {str(c): i for i, c in enumerate(z["clip_ids"])} for lab, z in arms.items()}
    print(f"[viz] arms: {', '.join(f'{l} ({len(index[l])} clips)' for l in labels)}")
    print(f"[viz] selection driven by {labels[0]!r}, {args.select}, n={args.n}")

    # ⚠️ Every arm must contain every chosen clip, or the sheet would silently show a
    # different subset per arm.
    for lab in labels[1:]:
        gap = {c for c, _, _ in chosen} - set(index[lab])
        if gap:
            raise SystemExit(f"arm {lab!r} is missing chosen clips: {sorted(gap)}")

    # ⚠️ GT must be identical across arms; they were scored on the same val set, so a mismatch
    # means the archives came from different data and no comparison here is valid.
    for lab in labels[1:]:
        for c, _, _ in chosen:
            a = primary["gt_xyz"][index[labels[0]][c]]
            b = arms[lab]["gt_xyz"][index[lab][c]]
            if not np.allclose(a, b, atol=1e-3):
                raise SystemExit(f"GT differs between {labels[0]!r} and {lab!r} on {c[:8]}; "
                                 "these archives are not comparable")

    uuid_file = os.path.join(args.out, "_clips.txt")
    with open(uuid_file, "w") as fh:
        fh.write("\n".join(c for c, _, _ in chosen) + "\n")

    with hydra.initialize_config_dir(
        config_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs"),
        version_base=None,
    ):
        cfg = hydra.compose(config_name=args.config)

    # Config only -- never the weights. `AlpamayoR1Config.from_pretrained` eagerly builds a
    # processor from the config's STORED `vlm_name_or_path` (a bare hub repo id), so the local
    # snapshot must be forced in and the hub kept offline, exactly as the slurm launchers do.
    from alpamayo_r1.config import AlpamayoR1Config

    model_cfg = AlpamayoR1Config.from_pretrained(cfg.model.alpamayo_config_path)
    model_cfg.vlm_name_or_path = cfg.model.vlm_name_or_path

    ds = hyu.instantiate(
        {**OmegaConf.to_container(cfg.data.val_dataset, resolve=True),
         "clip_uuid_filter": uuid_file},
        _convert_="partial", model_config=model_cfg,
    )
    collate = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=model_cfg)
    # ⚠️ `clip_uuid_filter` does NOT subset a nav dataset. PAIDatasetWithNav builds one sample
    # per entry of `annotations_path`, so the filter trims the base clip list while the sample
    # list stays at all 1000 -- the loader would decode 1000 videos to reach 5 clips. Select by
    # sample index instead; `ds.clip_ids` is parallel to `ds._samples`.
    want = {c for c, _, _ in chosen}
    # ⚠️ Two levels of indirection: CameraSubsetPAIDataset wraps the nav dataset in `.base` and
    # maps its own index through `._indices`, so enumerate the SUBSET's positions and look the
    # clip up through both. Indexing `ds.base.clip_ids` positionally would be wrong the moment
    # `teacher_trajectory_cached_only` prunes `_indices`.
    base_ids = ds.base.clip_ids
    keep = [i for i, bi in enumerate(ds._indices) if base_ids[bi] in want]
    missing = want - {base_ids[ds._indices[i]] for i in keep}
    if missing:
        raise SystemExit(f"clips in the npz but not in {cfg.data.val_dataset.annotations_path}: "
                         f"{sorted(missing)}")
    print(f"[viz] {len(ds)} nav samples -> {len(keep)} for the {len(want)} chosen clips")
    loader = DataLoader(torch.utils.data.Subset(ds, keep), batch_size=1, collate_fn=collate,
                        num_workers=0, shuffle=False)

    cams = list(cfg.data.val_dataset.cameras)
    if FRONT_WIDE_LOADER_IDX not in cams:
        raise SystemExit(f"cameras={cams} has no front_wide (loader index "
                         f"{FRONT_WIDE_LOADER_IDX}); nothing to project onto")
    # ⚠️ Compute the flat index; do NOT copy the literal 7 from `viz_arms.py`. That constant is
    # for the 4-camera loader (4 cams x 4 frames, front_wide=cam 1 -> 1*4+3). After
    # CameraSubsetPAIDataset slices to [1,3], front_wide is LOCAL index 0 and the answer is 3.
    # Reusing 7 silently plots the front-telephoto frame under front-wide calibration.
    cam_local = cams.index(FRONT_WIDE_LOADER_IDX)
    pct_by_id = {c: 100.0 * (ade_all < a).mean() for c, a, _ in chosen}

    made = []
    for batch in loader:
        clip = batch["clip_id"][0] if isinstance(batch["clip_id"], list) else batch["clip_id"]
        frames = batch["image_frames"][0]
        n_frames = frames.shape[1]
        flat = cam_local * n_frames + (n_frames - 1)  # last frame = t0
        img = frames.flatten(0, 1)[flat].permute(1, 2, 0).float().numpy()
        if img.max() > 1.5:
            img = img / 255.0
        img = np.clip(img, 0, 1)

        per_arm = {lab: modes_of(arms[lab], index[lab][clip]) for lab in labels}
        gt = primary["gt_xyz"][index[labels[0]][clip]].astype(np.float32)
        pct = pct_by_id[clip]
        p = draw(clip, img, gt, per_arm,
                 ds.base.avdi.get_clip_feature(clip, "camera_intrinsics"),
                 ds.base.avdi.get_clip_feature(clip, "sensor_extrinsics"),
                 pct, args.out)
        made.append(p)
        scores = "  ".join(f"{lab} {ade_of(m, gt):.3f}" for lab, m in per_arm.items())
        print(f"[viz] {clip[:8]}  p{pct:.0f}  {scores}  -> {p}", flush=True)

    print(f"[viz] wrote {len(made)} figures to {args.out}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
