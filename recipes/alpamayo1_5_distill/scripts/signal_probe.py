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

r"""WHICH student-vs-teacher signal actually PREDICTS trajectory error? Rank-correlated.

**The gap this fills.** Every objective in this line was chosen because it was conceptually the
right thing to match, then judged by whether training on it helped:

    L_block   matches the expert's block outputs      -- BEFORE the action head
    L_field   matches the velocity                    -- AFTER the action head
    L_roll    matches a 2-step rollout state
    L_span(m) same as L_block, chained over m layers

None was ever checked for whether it *tracks the metric*. That is why "loss down 730x, ade
flat" keeps recurring: the span curriculum drove block_loss 0.679 -> 0.00093 and moved min_ade
by 1.19x. A signal that cannot separate a clip the student nails (ade 0.30) from one it butchers
(ade 16.6) cannot be a useful training target, however cleanly it descends.

**Method.** Run the REAL 10-step Euler sampler twice per clip on the SAME noise -- once on the
student's cache, once on the teacher's -- and record, at every step, on both sides of the action
head. The only difference between the two runs is the cache, so the divergence is attributable.
Then rank-correlate each candidate against the outcome it claims to predict.

⚠️ THE OUTCOME IS ``ade``, NOT ``min_ade``. min_ade is best-of-6 over noise draws, so a signal
computed on one draw is handicapped against it by construction. Correlation is computed per
(clip, draw) against that draw's own ADE; per-clip min_ade is reported as a secondary.

⚠️ SAME NOISE IS THE WHOLE DESIGN. The two runs are seeded identically before each call, so
step k of run A and step k of run B see the same x_t. Without that, the "divergence" would be
dominated by noise, not by the cache.

⚠️ Spearman, not Pearson. ade is heavy-tailed (0.3 to 16.6 across clips in this eval set) and a
single catastrophic clip would otherwise set the correlation on its own.

Usage (teacher and student on separate cards, like cache_ladder)::

    CUDA_VISIBLE_DEVICES=0,3 python -m alpamayo1_5_distill.scripts.signal_probe \
        --config-path pkg://alpamayo1_5_distill/configs \
        --config-name sft_eval_stitched_2b_prunedexpert_lcdrive \
        ++model.attn_implementation=sdpa ++evaluate.eval_ckpt=<student ckpt> \
        ++sweep.teacher_checkpoint=<alpamayo 10B> ++sweep.teacher_vlm=<cosmos snapshot> \
        ++sweep.limit=200 ++sweep.tag=m14
"""

from __future__ import annotations

import json
import os

import hydra
import hydra.utils as hyu
import numpy as np
import torch
from accelerate.utils import send_to_device
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers.cache_utils import DynamicCache

from alpamayo1_5_distill.scripts.cache_ladder import _kv, _n_layers, _pi_map, _prefill


def _rel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-row relative squared error ||a-b||^2 / ||b||^2, flattened over everything else."""
    a = a.float().flatten(1)
    b = b.float().flatten(1)
    return (a - b).pow(2).mean(1) / b.pow(2).mean(1).clamp_min(1e-12)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(np.float64)
    ry = np.argsort(np.argsort(y[ok])).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    sw = cfg.get("sweep", {})
    tag = str(sw.get("tag", "student"))
    bs = int(sw.get("batch_size", 2))
    n_samples = int(sw.get("num_traj_samples", 6))
    limit = int(sw.get("limit", 200))
    seed = int(sw.get("seed", 1234))
    out_dir = str(sw.get("out_dir", "/temp/achahe/alpamayo-recipes/recipes/"
                                   "alpamayo1_5_distill/training/signalprobe"))
    os.makedirs(out_dir, exist_ok=True)
    if torch.cuda.device_count() < 2:
        raise RuntimeError("needs 2 visible GPUs: CUDA_VISIBLE_DEVICES=<teacher>,<student>")
    t_dev, s_dev = torch.device("cuda:0"), torch.device("cuda:1")

    ckpt = cfg.get("evaluate", {}).get("eval_ckpt", None)
    if ckpt is None:
        raise RuntimeError("set ++evaluate.eval_ckpt=<student checkpoint>")
    OmegaConf.update(cfg, "model.checkpoint_path", str(ckpt), merge=False)
    student = hyu.instantiate(cfg.model, _convert_="partial").to(s_dev).eval()
    t_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    t_cfg["_target_"] = ("alpamayo1_5_distill.models.stitched_model."
                         "StitchedAlpamayoR1.from_teacher")
    t_cfg["checkpoint_path"] = str(sw["teacher_checkpoint"])
    t_cfg["vlm_name_or_path"] = str(sw["teacher_vlm"])
    teacher = hyu.instantiate(t_cfg, _convert_="partial").to(t_dev).eval()
    for m in (student, teacher):
        for p in m.parameters():
            p.requires_grad_(False)
    print(f"[sig] student {ckpt}", flush=True)

    ds_cfg = OmegaConf.to_container(cfg.data.val_dataset, resolve=True)
    base = hyu.instantiate(ds_cfg, _convert_="partial", model_config=student.config)
    coll = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=student.config)
    loader = DataLoader(base, batch_size=bs, collate_fn=coll, num_workers=6, shuffle=False)

    rows: list[dict] = []
    pi = None
    seen = 0
    for batch in loader:
        if limit and seen >= limit:
            break
        clip_ids = batch.get("clip_id")
        s_batch = send_to_device({k: v for k, v in batch.items() if k != "clip_id"}, s_dev)
        t_batch = send_to_device({k: v for k, v in batch.items() if k != "clip_id"}, t_dev)

        with torch.no_grad():
            t_ids, t_cache = _prefill(teacher, t_batch)
            s_ids, s_cache = _prefill(student, s_batch)
            if not torch.equal(t_ids.cpu(), s_ids.cpu()):
                raise RuntimeError("teacher/student input_ids differ; positions do not align")
            n_slots = _n_layers(s_cache)
            if pi is None:
                pi = _pi_map(student, n_slots)
            t_kv = [(k.detach().to(s_dev), v.detach().to(s_dev))
                    for k, v in (_kv(t_cache, pi[j]) for j in range(n_slots))]
            del t_cache, s_cache

        def teacher_cache_hook(cache, input_ids, tokenized_data):
            mixed = DynamicCache()
            for j in range(n_slots):
                tk, tv = t_kv[j]
                mixed.update(tk.clone(), tv.clone(), j, {})
            return mixed

        # ---- two runs, identical noise, differing ONLY in the cache -------------------
        caught: dict[str, dict[int, tuple]] = {"student": {}, "teacher": {}}

        def make_probe(which):
            def probe(i, t, x, last_hidden, vel):
                caught[which][i] = (x.detach(), last_hidden.detach(), vel.detach(),
                                    float(t.reshape(-1)[0]))
            return probe

        preds = {}
        for which, hook in (("student", None), ("teacher", teacher_cache_hook)):
            torch.manual_seed(seed)           # ⚠️ same noise for both runs -- see docstring
            torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                pxyz, _ = student.sample_trajectories_prefill_only(
                    data=s_batch, num_traj_samples=n_samples, num_traj_sets=1,
                    cache_hook=hook, step_probe=make_probe(which))
            preds[which] = pxyz

        # ---- per (clip, draw) signals and outcome -------------------------------------
        steps = sorted(set(caught["student"]) & set(caught["teacher"]))
        sig = {k: [] for k in ("x_div", "h_err", "v_err")}
        for i in steps:
            xs, hs, vs, _ = caught["student"][i]
            xt, ht, vt, _ = caught["teacher"][i]
            sig["x_div"].append(_rel(xs, xt).cpu().numpy())
            sig["h_err"].append(_rel(hs, ht).cpu().numpy())
            sig["v_err"].append(_rel(vs, vt).cpu().numpy())
        sig = {k: np.stack(v) for k, v in sig.items()}      # [n_steps, b_star]

        gt = s_batch["ego_future_xyz"][:, -1]               # [B, T, 3]
        B = gt.shape[0]
        for which in ("student", "teacher"):
            p = preds[which].reshape(B, n_samples, *preds[which].shape[-2:])
            d = (p[..., :2] - gt[:, None, :, :2]).pow(2).sum(-1).sqrt().mean(-1)   # [B, ns]
            preds[which + "_ade"] = d.float().cpu().numpy()

        for b in range(B):
            for j in range(n_samples):
                r = b * n_samples + j       # repeat_interleave layout: clip-major
                rows.append({
                    "clip_id": str(clip_ids[b]), "draw": j,
                    "ade": float(preds["student_ade"][b, j]),
                    "ade_teacher": float(preds["teacher_ade"][b, j]),
                    **{f"{k}_mean": float(sig[k][:, r].mean()) for k in sig},
                    **{f"{k}_first": float(sig[k][0, r]) for k in sig},
                    **{f"{k}_last": float(sig[k][-1, r]) for k in sig},
                })
        seen += B
        if seen % 20 < bs:
            print(f"[sig] {seen}/{limit} clips, {len(rows)} (clip,draw) rows", flush=True)

    ade = np.array([r["ade"] for r in rows])
    print(f"\n[sig] === {tag}: does the signal predict ade?  n={len(rows)} (clip,draw) ===")
    print(f"{'signal':<14}{'spearman vs ade':>18}{'median':>12}   (higher |rho| = better proxy)")
    print("-" * 70)
    scored = []
    dropped = []
    for key in [k for k in rows[0] if k.endswith(("_mean", "_first", "_last"))]:
        v = np.array([r[key] for r in rows])
        # ⚠️ x_div_first is IDENTICALLY ZERO: both runs are seeded with the same noise, so x_0
        # is shared by construction. A rank correlation over an all-ties column is pure
        # tie-breaking noise (it read -0.5475 on the smoke), so degenerate columns are
        # reported as dropped rather than ranked.
        if np.nanstd(v) < 1e-12 or len(np.unique(v[np.isfinite(v)])) < 8:
            dropped.append(key)
            continue
        rho = _spearman(v, ade)
        scored.append((abs(rho), key, rho, float(np.nanmedian(v))))
    for _, key, rho, med in sorted(scored, reverse=True):
        print(f"{key:<14}{rho:>18.4f}{med:>12.4g}")
    print("-" * 70)
    if dropped:
        print(f"[sig] dropped as degenerate (constant / <8 distinct values): {dropped}")
    t_ade = np.array([r["ade_teacher"] for r in rows])
    print(f"[sig] student ade {ade.mean():.4f}   teacher-cache ade {t_ade.mean():.4f}   "
          f"(same noise, same expert; the gap is the cache)")
    print(f"[sig] spearman(teacher-cache ade, student ade) = "
          f"{_spearman(t_ade, ade):.4f}   <- clip difficulty, the floor any signal must beat")
    with open(os.path.join(out_dir, f"{tag}_signals.json"), "w") as fh:
        json.dump(rows, fh)
    print(f"[sig] wrote {out_dir}/{tag}_signals.json")
    print("DONE_SIGNAL", flush=True)


if __name__ == "__main__":
    main()
