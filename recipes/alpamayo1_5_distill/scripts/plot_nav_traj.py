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

r"""Teacher vs student vs ground truth, at the ANCHORED t0, with the navigation instruction.

Front-wide camera at t0 beside a BEV showing all three: the GT the model is asked to predict,
the teacher's six proposals, and the student's six. Titled with the nav text the model was
actually given and each model's ADE on that clip.

**The teacher is not re-run.** ``stepsweep/teachernav_k10.npz`` already holds its predictions,
GT and per-clip metrics for all 1000 nav anchors at 10 denoising steps -- the same rows the
COMPARE_EVAL §7 table is computed from. Re-running a 10B to draw a picture would also risk the
picture disagreeing with the table. Only the student is executed here, on the handful of clips
being drawn.

⚠️ MATCHED CONDITIONS. Both models are read/run at the SAME anchored t0, the SAME 2 cameras and
the SAME route in the prompt -- the npz was produced with `components_order` containing "route"
(without it nav_text is silently discarded, which is why the eval that first tried this returned
BIT-IDENTICAL numbers with and without the instruction). The student's dataset here is built the
same way, and CameraSubsetPAIDataset raises if "route" is missing.

⚠️ The teacher runs its UNPRUNED 36-layer expert; the student runs the 28-layer set-C one. The
curves are directly comparable as trajectories -- same GT, same clip, same t0 -- but the ADE gap
is not attributable to the VLM alone.

Usage::

    CUDA_VISIBLE_DEVICES=3 python -m alpamayo1_5_distill.scripts.plot_nav_traj \
        --config-path pkg://alpamayo1_5_distill/configs \
        --config-name sft_eval_stitched_2b_prunedexpert_lcdrive \
        ++model.attn_implementation=sdpa ++evaluate.eval_ckpt=<student ckpt> \
        ++sweep.out=<png> ++sweep.n_turn=6 ++sweep.n_straight=2
"""

from __future__ import annotations

import json
import os

import hydra
import hydra.utils as hyu
import matplotlib
import numpy as np
import torch
from accelerate.utils import send_to_device
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ade(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean L2 over waypoints, XY only -- the convention every table in this tree uses."""
    return float(np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1).mean())


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    sw = cfg.get("sweep", {})
    T = "/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training"
    M = "/data/datasets/physical_ai_av/lcdrive_physicalai_av_manifests"
    npz_path = str(sw.get("teacher_npz", f"{T}/stepsweep/teachernav_k10.npz"))
    ann_path = str(sw.get("annotations", f"{M}/nav_lcdrive_val_mysubset_1k.json"))
    out = str(sw.get("out", f"{T}/figures/nav_traj.png"))
    n_turn = int(sw.get("n_turn", 6))
    n_straight = int(sw.get("n_straight", 2))
    os.makedirs(os.path.dirname(out), exist_ok=True)

    z = np.load(npz_path, allow_pickle=True)
    t_pred = {c: z["pred_xyz"][i] for i, c in enumerate(z["clip_ids"])}     # [ns, nj, T, 3]
    t_gt = {c: z["gt_xyz"][i] for i, c in enumerate(z["clip_ids"])}
    t_ade = {c: float(z["ade"][i]) for i, c in enumerate(z["clip_ids"])}
    ann = {e["clip_id"]: e for e in json.load(open(ann_path))}

    # pick clips: turns first (the ones the instruction is about), then a few straights
    turns = [c for c in z["clip_ids"] if ann.get(c, {}).get("nav_text", "").startswith("Turn")]
    straights = [c for c in z["clip_ids"] if ann.get(c, {}).get("nav_text") == "Continue straight"]
    turns.sort(key=lambda c: t_ade[c])           # teacher-easiest first, so the GT is legible
    chosen = turns[:n_turn] + straights[:n_straight]
    print(f"[traj] {len(turns)} turn clips available; drawing {len(chosen)} "
          f"({n_turn} turn + {n_straight} straight)", flush=True)

    ckpt = cfg.get("evaluate", {}).get("eval_ckpt", None)
    if ckpt is None:
        raise RuntimeError("set ++evaluate.eval_ckpt=<student checkpoint>")
    OmegaConf.update(cfg, "model.checkpoint_path", str(ckpt), merge=False)
    dev = torch.device("cuda")
    model = hyu.instantiate(cfg.model, _convert_="partial").to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # the student's dataset must match how the npz was produced: nav annotations, 2 cameras,
    # route in components_order (CameraSubsetPAIDataset raises without it)
    ds_cfg = OmegaConf.to_container(cfg.data.val_dataset, resolve=True)
    ds_cfg["_target_"] = "alpamayo1_5_distill.data.camera_subset.CameraSubsetPAIDataset"
    ds_cfg["cameras"] = list(sw.get("cameras", [1, 3]))
    ds_cfg["annotations_path"] = ann_path
    ds_cfg["vla_preprocess_args"]["components_order"] = [
        "image", "traj_history", "route", "prompt", "traj_future"]
    base = hyu.instantiate(ds_cfg, _convert_="partial", model_config=model.config)
    coll = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=model.config)

    want = {c: i for i, c in enumerate(base.base.clip_ids)}
    rows = []
    for c in chosen:
        if c not in want:
            print(f"[traj] {c[:8]} not in the student's dataset -- skipped", flush=True)
            continue
        s = base[want[c]]
        if s is None:
            continue
        batch = send_to_device(coll([s]), dev)
        with torch.no_grad():
            pxyz, _ = model.sample_trajectories_prefill_only(
                data=batch, num_traj_samples=6, num_traj_sets=1)
        sp = pxyz[0].float().cpu().numpy()                      # [ns, nj, T, 3]
        img = s["image_frames"][0, -1].permute(1, 2, 0).numpy()  # front-wide at t0
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 1) if img.max() <= 1.0 else np.clip(img / 255.0, 0, 1)
        gt = t_gt[c]
        rows.append({
            "clip": c, "img": img, "gt": gt,
            "t": t_pred[c].reshape(-1, *t_pred[c].shape[-2:]).astype(np.float32),
            "s": sp.reshape(-1, *sp.shape[-2:]),
            "nav": ann[c]["nav_text"], "t0": ann[c]["t0_relative"] / 1e6,
            "t_ade": t_ade[c],
            "s_ade": min(_ade(x, gt) for x in sp.reshape(-1, *sp.shape[-2:])),
        })
        print(f"[traj] {c[:8]}  {ann[c]['nav_text']:<20} teacher {t_ade[c]:5.2f}  "
              f"student {rows[-1]['s_ade']:5.2f}", flush=True)

    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(11.0, 3.1 * n),
                             gridspec_kw={"width_ratios": [1.7, 1]})
    axes = np.atleast_2d(axes)
    for i, r in enumerate(rows):
        ax_i, ax_b = axes[i, 0], axes[i, 1]
        ax_i.imshow(r["img"]); ax_i.axis("off")
        turn = r["nav"].startswith("Turn")
        ax_i.set_title(f'{r["clip"][:8]}   t0={r["t0"]:.1f}s   "{r["nav"]}"\n'
                       f'teacher min_ade {r["t_ade"]:.2f}   student {r["s_ade"]:.2f}',
                       fontsize=9, loc="left",
                       color="crimson" if turn else "black",
                       fontweight="bold" if turn else "normal")
        # ⚠️ GT FIRST, as a thick underlay. Drawn last it OCCLUDES the teacher exactly where the
        # teacher is good -- at min_ade 0.28 its six proposals sit on top of the GT, so a heavy
        # black line on top erased the one model that was working.
        ax_b.plot(r["gt"][:, 1], r["gt"][:, 0], "-", color="k", lw=4.0, alpha=0.30,
                  solid_capstyle="round", label="ground truth")
        # +y is LEFT in the rig frame -> invert x so a left turn bends left on the page
        for k, p in enumerate(r["t"]):
            ax_b.plot(p[:, 1], p[:, 0], "-", color="tab:blue", lw=1.3, alpha=0.75,
                      label="teacher (6)" if k == 0 else None)
        for k, p in enumerate(r["s"]):
            ax_b.plot(p[:, 1], p[:, 0], "-", color="tab:red", lw=1.3, alpha=0.70,
                      label="student (6)" if k == 0 else None)
        ax_b.plot(r["gt"][:, 1], r["gt"][:, 0], "--", color="k", lw=1.1, dashes=(4, 3))
        ax_b.plot(0, 0, "k^", ms=7)
        # where the instruction says the manoeuvre is: "Turn left in 14m" -> a ring at 14 m.
        # ⚠️ the GT is 6.4 s of travel, so at low speed it can END before that ring -- which is
        # the horizon mismatch NAVTEXT_SAMPLING §7 documents, made visible here.
        if turn:
            try:
                d = float(r["nav"].split(" in ")[1].rstrip("m"))
                ax_b.add_patch(plt.Circle((0, 0), d, fill=False, ec="crimson", ls=":", lw=1.0))
                ax_b.text(0.03, 0.03, f'instruction at {d:.0f} m', transform=ax_b.transAxes,
                          fontsize=6.5, color="crimson")
            except (IndexError, ValueError):
                pass
        nav_d = 0.0
        if turn and " in " in r["nav"]:
            try:
                nav_d = float(r["nav"].split(" in ")[1].rstrip("m"))
            except ValueError:
                nav_d = 0.0
        span = max(12.0, np.abs(r["gt"][:, 0]).max(), np.abs(r["gt"][:, 1]).max() * 1.5,
                   nav_d * 1.05) * 1.2
        ax_b.set_xlim(span, -span); ax_b.set_ylim(-span * 0.25, span * 1.25)
        ax_b.set_aspect("equal"); ax_b.grid(alpha=0.25, lw=0.4)
        ax_b.tick_params(labelsize=6.5)
        ax_b.set_xlabel("y (m)   ←left   right→", fontsize=7)
        if i == 0:
            ax_b.legend(fontsize=6.5, loc="upper left", framealpha=0.85)
    fig.suptitle("Teacher vs student vs ground truth — event-anchored t0, navigation "
                 "instruction in the prompt, front-wide camera at t0\n"
                 "teacher: unpruned 36-layer expert   |   student: 2B + 28-layer set-C expert   "
                 "|   both at 10 denoising steps, 6 proposals", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=125, bbox_inches="tight")
    print(f"[traj] wrote {out}", flush=True)
    print("DONE_TRAJ", flush=True)


if __name__ == "__main__":
    main()
