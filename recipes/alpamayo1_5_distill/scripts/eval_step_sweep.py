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

r"""Evaluate one model across DENOISING STEP COUNTS, saving the trajectories, not just metrics.

**Gate C.**  ``LATENCY_PROFILE.md`` measures denoising at **51-77% of end-to-end latency** and
its Consequence 1 names cutting steps as "by far the cheapest win ... This should be tested
before any further distillation work."  It never was: ``num_inference_steps`` is a python
default (``flow_matching.py:36``), absent from every config in this tree, and no eval in either
repo has run at a value other than 10.  This is that test.

Sibling of ``eval_camera_sweep.py`` -- same harness, same metrics, same output format -- with
the CAMERA axis pinned and the STEP-COUNT axis swept.  The two scripts' ``.npz``/``.json``
files are interchangeable, so the notebook tooling and the paired-comparison helpers work on
both.

**Why not ``slurm_eval_stitched.sh``.**  ``diffusion_kwargs={"inference_step": K}`` is already
plumbed end to end (``ReasoningSampler(**kwargs)`` -> ``sample_trajectories_from_data`` ->
``sample_trajectories_prefill_only`` -> ``diffusion.sample``), so the step override needs no
code.  What needs code is the CAMERA SUBSET: cameras must be sliced BEFORE preprocessing so the
prompt is rebuilt, and ``evaluate_hf`` drives the unsliced val set.  Hence this script.

**Two cameras by default** (front-wide + front-tele).  ``LATENCY_PROFILE.md`` recommends it
(-26% latency for +18% min_ade, strictly dominating 3cam) and it is where the step saving
matters most: of the teacher's 279.4 ms, **180.7 ms is denoising**.

⚠️ Camera indices are the loader's ``camera_features`` order -- 0 cross_left, **1 front_wide**,
2 cross_right, **3 front_tele** -- NOT the global camera-index table where front_tele is 6.

⚠️ Reduced-camera prompts are OUT OF DISTRIBUTION: the model was trained with all four cameras
present and named in a fixed order.  These are "starved" numbers.  That is fine for this
purpose -- every row here is at the same camera count, so the K comparison is paired and fair.

**Harness self-check.**  The ``k10`` row must reproduce the established 2cam teacher figures,
**min_ade 0.6981 / ade 1.6822** (``LATENCY_PROFILE.md:136``, n=1000, prefill-only).  If it does
not, fix that before reading any other row.

**What is reported per K**, beyond ``min_ade``/``ade``/``max_ade``:

* ``identical`` -- fraction of clips where all 6 samples coincide (``ade == min_ade``).  The
  mode-collapse tripwire ``slurm_eval_stitched.sh:128`` already uses; baseline band 15.6-17.8%.
* ``diversity`` -- mean pairwise L2 between the 6 sampled trajectories (XY, mean over time).
  At eval the six modes come ONLY from the diffusion noise (the prefill-only path skips VLM
  generation entirely), so this measures exactly what the sampler contributes.
* ``oob`` -- fraction of samples whose round-tripped (accel, curvature) leaves the action
  space's own bounds, via ``ActionSpace.is_within_bounds``.

⚠️ Prediction worth stating in advance: at **K=1** diversity should be near zero.  For the
optimal velocity field ``x_0 = eps`` is independent of the data, so ``v(x_0, 0) = m - x_0`` and
one Euler step of size 1 lands exactly on ``m``, the conditional mean.  A one-step flow sampler
is a mean predictor by construction, which is the argument that few-step *distillation* is
needed rather than a grid retune.

Usage::

    CUDA_VISIBLE_DEVICES=1 python -m alpamayo1_5_distill.scripts.eval_step_sweep \
      --config-path pkg://alpamayo1_5_distill/configs \
      --config-name sft_eval_stitched_4b_lcdrive \
      ++model._target_=alpamayo1_5_distill.models.stitched_model.StitchedAlpamayoR1.from_teacher \
      ++model.vlm_name_or_path=<cosmos 8b> ++model.checkpoint_path=<10b> \
      ++model.attn_implementation=sdpa ++sweep.tag=teacher

``++sweep.only=k2`` runs one row; ``++sweep.steps=[1,2,10]`` overrides the grid;
``++sweep.cameras=[0,1,2,3]`` overrides the camera subset; ``++sweep.limit=40`` for a smoke.
"""

from __future__ import annotations

import json
import os

import einops
import hydra
import hydra.utils as hyu
import numpy as np
import torch
from accelerate.utils import send_to_device
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

#: Denoising step counts to sweep. 10 is the deployed default and doubles as the harness
#: self-check; 1 and 2 are the targets; 3 and 5 locate the knee.
DEFAULT_STEPS = [1, 2, 3, 5, 10]

#: front-wide + front-tele, in the loader's camera_features order.
DEFAULT_CAMERAS = [1, 3]

#: Keys whose LEADING dimension is the camera axis.
_CAMERA_AXIS_KEYS = ("image_frames", "camera_indices", "absolute_timestamps",
                     "relative_timestamps")


class _CameraSubset(torch.utils.data.Dataset):
    """PAIDataset with cameras sliced before the prompt is built.

    Composition rather than a subclass: the base class runs its preprocess inside
    ``__getitem__``, so the only way to slice first is to build it WITHOUT a preprocess and
    apply ours afterwards.  Byte-identical to ``eval_camera_sweep._CameraSubset`` -- kept
    duplicated rather than imported so neither script can silently change the other's
    preprocessing.
    """

    def __init__(self, base, pre, cams: list[int]):
        self.base, self.pre, self.cams = base, pre, cams

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        s = dict(self.base[i])
        for k in _CAMERA_AXIS_KEYS:
            if k in s and torch.is_tensor(s[k]):
                s[k] = s[k][self.cams]
        s["tokenized_data"] = self.pre(data=s)
        return s


def _pairwise_diversity(pred_xyz: torch.Tensor) -> torch.Tensor:
    """Mean pairwise L2 between the K sampled trajectories, per clip.

    ``pred_xyz``: [B, ns, nj, Tf, 3].  XY only and mean over time, matching how
    ``distance_metrics.compute_ade`` defines distance, so the number is on the ADE scale and
    can be read against it directly.

    Returns [B].  Zero means every sample is identical; the ADE-scale reading is "how far
    apart, in metres, two proposals typically are".
    """
    p = pred_xyz[..., :2].flatten(1, 2)                      # [B, ns*nj, Tf, 2]
    n = p.shape[1]
    if n < 2:
        return torch.zeros(p.shape[0], device=p.device)
    # [B, n, n, Tf] -> mean over time -> mean over the n*(n-1)/2 unordered pairs
    d = torch.linalg.norm(p[:, :, None] - p[:, None, :], dim=-1).mean(-1)
    iu = torch.triu_indices(n, n, offset=1, device=p.device)
    return d[:, iu[0], iu[1]].mean(-1)


def _max_ade(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> torch.Tensor:
    """Worst-of-K ADE for every clip, averaged over trajectory sets.

    ``pred_xyz`` is ``[B, N, K, T, 3]`` and ``gt_xyz`` is ``[B, T, 3]``. This is
    the mirror of the evaluator's oracle ``min_ade``: first take the worst candidate
    along K, then average over N. XY-only distance and the time average exactly match
    ``DistanceMetrics``' ADE convention.
    """
    if pred_xyz.ndim != 5 or gt_xyz.ndim != 3:
        raise ValueError(
            f"expected pred [B,N,K,T,3] and gt [B,T,3], got "
            f"{tuple(pred_xyz.shape)} and {tuple(gt_xyz.shape)}"
        )
    sample_ade = torch.linalg.norm(
        pred_xyz[..., :2] - gt_xyz[:, None, None, :, :2], dim=-1
    ).mean(dim=-1)
    return sample_ade.amax(dim=2).mean(dim=1)


def _out_of_bounds(model, action: torch.Tensor) -> torch.Tensor:
    """Fraction of a clip's samples whose SAMPLED action leaves the action space's bounds.

    ``action`` is [B, ns, nj, N, 2] straight from the sampler -- ``return_action=True`` on
    ``sample_trajectories_prefill_only``.

    ⚠️ Do NOT recover this by inverting the trajectory with ``traj_to_action``.  That path
    runs ``theta_smooth`` and three ridge-regularised solves
    (``unicycle_accel_curvature.py:269-283``), so it returns a smoothed fit: the bounds check
    can then pass on a trajectory that is not feasible.  An earlier revision of this script
    did exactly that; ``return_action`` exists so this one does not have to.

    Returns [B] in [0, 1].
    """
    b = action.shape[0]
    ok = model.action_space.is_within_bounds(action.flatten(1, 2).flatten(0, 1).float())
    return (~ok).float().view(b, -1).mean(-1)


def _geometric_infeasible(pred_xyz: torch.Tensor, dt: float,
                          kappa_max: float, accel_max: float) -> torch.Tensor:
    """Fraction of a clip's samples whose TRAJECTORY GEOMETRY breaks the kinematic bounds.

    Independent of the action parameterisation and of any smoothing: finite-difference
    curvature ``|x'y'' - y'x''| / |v|^3`` and longitudinal acceleration, straight off the
    waypoints.  Kept alongside :func:`_out_of_bounds` because the two can disagree -- the
    action can be in-bounds while the integrated path is not, and that disagreement is
    itself worth seeing.

    Returns [B] in [0, 1].
    """
    xy = pred_xyz[..., :2].flatten(1, 2)                     # [B, n, T, 2]
    v = torch.gradient(xy, spacing=dt, dim=2)[0]
    a = torch.gradient(v, spacing=dt, dim=2)[0]
    sp = torch.linalg.norm(v, dim=-1)
    kap = (v[..., 0] * a[..., 1] - v[..., 1] * a[..., 0]).abs() / sp.pow(3).clamp_min(1e-6)
    lon = torch.gradient(sp, spacing=dt, dim=2)[0].abs()
    bad = (kap > kappa_max).any(-1) | (lon > accel_max).any(-1)
    return bad.float().mean(-1)


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    sw = cfg.get("sweep", {})
    tag = str(sw.get("tag", "model"))
    bs = int(sw.get("batch_size", 4))
    n_samples = int(sw.get("num_traj_samples", 6))
    limit = int(sw.get("limit", 0))
    only = str(sw.get("only", ""))
    steps = [int(k) for k in sw.get("steps", DEFAULT_STEPS)]
    cams = [int(c) for c in sw.get("cameras", DEFAULT_CAMERAS)]
    # ⚠️ Each worker decodes 16 video frames per clip, so this competes for CPU with whatever
    # else holds the node. Running this beside a training job at 8 workers moved that job from
    # 6.7 to 10.0 s/it. Default low; raise it only on an idle node.
    workers = int(sw.get("num_workers", 4))
    # Base for the per-batch diffusion seed. Identical across rows on purpose -- that is what
    # pairs them on noise. Change it only to measure the noise floor by re-running.
    seed = int(sw.get("seed", 1234))
    out_dir = str(sw.get("out_dir",
                         "/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/stepsweep"))
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device("cuda")

    # ⚠️ A stray PRUNE_EXPERT_LAYERS bypasses 8 of 36 layers inside `from_teacher` and would
    # profile the ablation under this run's name -- the trap LATENCY_PROFILE.md:47-50 records.
    pin = os.environ.get("PRUNE_EXPERT_LAYERS", "").strip()

    model = hyu.instantiate(cfg.model, _convert_="partial").to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[step] {tag}: expert {len(model.expert.layers)} layers"
          f"{f'  ⚠️ PRUNE_EXPERT_LAYERS={pin}' if pin else ''}, "
          f"cameras {cams}, bs {bs}, {n_samples} traj samples, K {steps}, "
          f"{workers} workers", flush=True)

    ds_cfg = OmegaConf.to_container(cfg.data.val_dataset, resolve=True)
    pre_args = ds_cfg.pop("vla_preprocess_args")
    base = hyu.instantiate(ds_cfg, _convert_="partial", model_config=model.config)
    pre = hyu.instantiate(pre_args, _convert_="partial", model_config=model.config)
    coll = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=model.config)
    metric = hyu.instantiate(cfg.evaluate.metric_runner.metrics[-1], _convert_="partial")
    print(f"[step] {len(base)} clips; metric {type(metric).__name__}", flush=True)

    summary = []
    for k in steps:
        name = f"k{k}"
        if only and only != name:
            continue
        loader = DataLoader(_CameraSubset(base, pre, cams), batch_size=bs, collate_fn=coll,
                            num_workers=workers, shuffle=False)
        ids, preds, gts, per = [], [], [], []
        seen = 0
        for bi, batch in enumerate(loader):
            if limit and seen >= limit:
                break
            clip_ids = batch.get("clip_id")
            gpu = send_to_device({kk: v for kk, v in batch.items() if kk != "clip_id"}, dev)
            # ⚠️ LOAD-BEARING, not tidiness. `flow_matching._euler` starts from
            # `torch.randn(...)` off the global RNG, and every step-count row is a separate
            # pass -- so without this each row draws INDEPENDENT initial noise and the rows
            # are paired on clip but not on noise. That is exactly what retracted the
            # R-KV-vs-random result (commit 12702e0): two runs of an identical arm differed
            # by 1.89 sigma on an effect that is zero by construction. Seeding per BATCH
            # (not per row) keeps the pairing even if a row consumes RNG unevenly, and the
            # noise shape [B*n_samples, 64, 2] does not depend on k, so the same seed gives
            # every row byte-identical x_0 for the same clip.
            torch.manual_seed(seed + bi)
            torch.cuda.manual_seed_all(seed + bi)
            with torch.no_grad():
                pred_xyz, pred_rot, action = model.sample_trajectories_prefill_only(
                    data=gpu, num_traj_samples=n_samples, num_traj_sets=1,
                    diffusion_kwargs={"inference_step": k}, return_action=True)
            out = {"pred_xyz": pred_xyz, "pred_rot": pred_rot}
            m = metric.evaluate(model, gpu, out)
            with torch.no_grad():
                max_ade = _max_ade(pred_xyz, gpu["ego_future_xyz"][:, -1])
                div = _pairwise_diversity(pred_xyz)
                oob = _out_of_bounds(model, action)
                geo = _geometric_infeasible(pred_xyz, model.action_space.dt,
                                            model.action_space.curvature_bounds[1],
                                            model.action_space.accel_bounds[1])
            b = pred_xyz.shape[0]
            for i in range(b):
                ids.append(str(clip_ids[i]))
                # [ns, nj, Tf, 3] -> keep every sample; float16 keeps the sweep small
                preds.append(pred_xyz[i].float().cpu().numpy().astype(np.float16))
                gts.append(gpu["ego_future_xyz"][i, -1].float().cpu().numpy().astype(np.float32))
                per.append({"clip_id": str(clip_ids[i]),
                            "diversity": float(div[i]), "oob": float(oob[i]),
                            "geo": float(geo[i]),
                            **{kk: float(v[i]) for kk, v in m.items()
                               if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == b},
                            "max_ade": float(max_ade[i])})
            seen += b
            if seen % 100 < bs:
                done = [p for p in per if "min_ade" in p]
                avg = np.mean([p["min_ade"] for p in done]) if done else float("nan")
                print(f"[step] {name} {seen}/{len(base)}  running min_ade {avg:.4f}", flush=True)

        npz = os.path.join(out_dir, f"{tag}_{name}.npz")
        np.savez_compressed(
            npz, clip_ids=np.array(ids), pred_xyz=np.stack(preds), gt_xyz=np.stack(gts),
            cameras=np.array(cams), description=np.array(f"{k} Euler steps"),
            inference_step=np.array(k),
            min_ade=np.array([p.get("min_ade", np.nan) for p in per], dtype=np.float32),
            ade=np.array([p.get("ade", np.nan) for p in per], dtype=np.float32),
            max_ade=np.array([p.get("max_ade", np.nan) for p in per], dtype=np.float32))
        with open(os.path.join(out_dir, f"{tag}_{name}.json"), "w") as fh:
            json.dump(per, fh)

        ma = float(np.nanmean([p.get("min_ade", np.nan) for p in per]))
        ad = float(np.nanmean([p.get("ade", np.nan) for p in per]))
        mx = float(np.nanmean([p.get("max_ade", np.nan) for p in per]))
        # ade is sample 0 and min_ade is oracle best-of-6, so equality means the 6 coincided.
        eq = float(np.mean([abs(p.get("ade", 0.0) - p.get("min_ade", 1.0)) < 1e-9 for p in per]))
        dv = float(np.nanmean([p["diversity"] for p in per]))
        ob = float(np.nanmean([p["oob"] for p in per]))
        ge = float(np.nanmean([p["geo"] for p in per]))
        summary.append((name, k, len(per), ma, ad, mx, eq, dv, ob, ge))
        print(f"[step] === {tag} {name}: n={len(per)}  min_ade {ma:.4f}  "
              f"ade {ad:.4f}  max_ade {mx:.4f}  "
              f"identical {100 * eq:.1f}%  diversity {dv:.4f}  oob {100 * ob:.2f}%  "
              f"geo_infeas {100 * ge:.2f}%  -> {npz}", flush=True)

    print(f"\n[step] {tag}, cameras {cams}, n_samples {n_samples}")
    # `steps`, not `K`: this repo already uses K for the CANDIDATE axis -- distance_metrics
    # documents pred_xyz as [B, N, K, T, 3] "N groups of K candidates" -- and K is pinned at
    # `num_traj_samples` here. The swept axis is the Euler step count.
    print(f"[step] {'row':>5} {'steps':>5} {'n':>5} {'min_ade':>9} {'ade':>9} "
          f"{'max_ade':>9} {'identical':>10} {'diversity':>10} {'oob':>8} {'geo':>8}")
    for name, k, n, ma, ad, mx, eq, dv, ob, ge in summary:
        print(f"[step] {name:>5} {k:>5} {n:>5} {ma:>9.4f} {ad:>9.4f} {mx:>9.4f} "
              f"{100 * eq:>9.1f}% {dv:>10.4f} {100 * ob:>7.2f}% {100 * ge:>7.2f}%")
    ref = [s for s in summary if s[1] == 10]
    if ref and cams == DEFAULT_CAMERAS:
        _, _, _, ma, ad, *_ = ref[0]
        print(f"[step] self-check vs LATENCY_PROFILE.md 2cam teacher (min_ade 0.6981 / "
              f"ade 1.6822): min_ade {ma - 0.6981:+.4f}, ade {ad - 1.6822:+.4f}")
    print("DONE_STEP_SWEEP", flush=True)


if __name__ == "__main__":
    main()
