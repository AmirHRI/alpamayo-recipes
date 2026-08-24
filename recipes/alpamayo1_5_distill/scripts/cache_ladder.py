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

r"""WHICH PART of the VLM cache does the action expert actually need? Measured in min_ade.

**Why this exists.** The expert's inputs are exactly two things: the noisy action embedding and
the VLM cache. The action embedding is the SHARED ``action_in_proj`` applied to the same
``traj_data``, so it is bit-identical for teacher and student. Therefore 100% of a student's
trajectory gap is attributable to cache differences -- there is nowhere else for it to come
from. That makes "substitute part of the teacher's cache and re-read min_ade" an exact
decomposition rather than an approximation.

**Why the block loss cannot answer this.** L_block grades the cache THROUGH the expert, which
sounds like the right signal, but it is an unweighted relative MSE on hidden states: every
direction is declared equally important. Two measurements say that geometry is wrong:

  * the loss is m-INVARIANT -- chaining 28 blocks on the student's own cache scores no worse
    than teacher-forcing every layer (0.001025 at m=28 vs 0.001112 at m=14, same weights). The
    expert's layer map DAMPS whatever the loss is measuring, so the loss lives in a subspace
    the output does not read.
  * the student has closed ~90% of the distance from a ZERO cache (0.0104) to perfect, and is
    still 3x off on min_ade (2.42 vs the 0.7893 ceiling). The residual 10% carries essentially
    all of the trajectory error.

So the cache matters enormously in aggregate (CE-only scores min_ade 6.99, KD-only 11.38) while
most DIRECTIONS in it evidently do not. This script finds which ones do, in the metric that
matters, without training anything.

**Cost model.** The 10B teacher prefill is the expensive part, so it runs ONCE per batch and is
reused across every substitution spec; each spec then pays one 2B student prefill plus the
expert rollout. Specs are cheap, clips are not -- start with a few hundred clips.

⚠️ POSITION ALIGNMENT is a precondition, not a detail. Substituting teacher K/V at position i
for student K/V at position i is meaningless unless both towers saw the same token at i. The
training objective already assumes this (one ``input_ids``, one ``traj_mask``, one
``first_traj`` for both towers), but assumed is not checked -- so this asserts it per batch and
refuses to produce a number otherwise.

⚠️ DEPTH MAP. The student is 28 layers and the teacher's VLM is 36. Expert slot j reads student
cache layer j, and the teacher layer that corresponds to slot j is pi(j) -- the same map
``kd_model`` applies when it subsets the teacher's 36-entry cache (``t_kv_b = {j: t_kv[i] for
j, i in enumerate(pi)}``). pi is derived from PRUNE_EXPERT_LAYERS, NOT hard-coded, so a
different pruning set cannot silently produce a misaligned ladder.

Usage (teacher and student on SEPARATE cards so neither process needs ~35 GB alone)::

    CUDA_VISIBLE_DEVICES=0,3 python -m alpamayo1_5_distill.scripts.cache_ladder \
        --config-path pkg://alpamayo1_5_distill/configs \
        --config-name sft_eval_stitched_2b_prunedexpert_lcdrive \
        ++model.attn_implementation=sdpa \
        ++evaluate.eval_ckpt=<student ckpt> \
        ++sweep.teacher_checkpoint=<alpamayo 10B> ++sweep.teacher_vlm=<cosmos snapshot> \
        ++sweep.limit=300 ++sweep.tag=m28
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

# (name, layer selector, region selector, which of K/V) -- None means "all"
#   layers: None = every slot, or a (lo, hi) inclusive band
#   region: None = every position, or one of vision/text/traj
#   part:   None = both K and V, or "k" / "v"
SPECS = [
    ("student",        "none",  None,         None,     "baseline: the student's own cache"),
    # ⚠️ THE NOISE FLOOR. min_ade is best-of-6 over UNSEEDED diffusion noise, so it is not a
    # deterministic number: the validated no-op run reproduced the production eval to 1.8%, not
    # exactly. Re-running the identical no-op gives the scale a delta must clear to mean
    # anything -- without it, a -0.3 row is uninterpretable.
    ("student_rerun",  "none",  None,         None,     "baseline again: sampling noise floor"),
    ("teacher_all",    "all",   None,         None,     "sanity: must recover the ceiling"),
    ("layers_0_9",     (0, 9),  None,         None,     "teacher cache in the first third"),
    ("layers_10_18",   (10, 18), None,        None,     "teacher cache in the middle third"),
    ("layers_19_27",   (19, 27), None,        None,     "teacher cache in the last third"),
    ("layers_last4",   (24, 27), None,        None,     "teacher cache in the last 4 layers"),
    ("region_traj",    "all",   "traj",       None,     "teacher cache at traj-history tokens"),
    ("region_text",    "all",   "text",       None,     "teacher cache at text tokens"),
    ("region_vision",  "all",   "vision",     None,     "teacher cache at vision tokens (~93%)"),
    ("k_only",         "all",   None,         "k",      "teacher K, student V"),
    ("v_only",         "all",   None,         "v",      "student K, teacher V"),
]


def _kv(cache, i):
    """(keys, values) for layer i, across both transformers cache APIs."""
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return layers[i].keys, layers[i].values
    return cache.key_cache[i], cache.value_cache[i]


def _n_layers(cache) -> int:
    layers = getattr(cache, "layers", None)
    return len(layers) if layers is not None else len(cache.key_cache)


def _prefill(model, data):
    """The prefix of ``sample_trajectories_prefill_only``: fused ids + one prompt prefill."""
    tok = dict(data["tokenized_data"])
    ids = tok.pop("input_ids")
    ids = model.fuse_traj_tokens(
        ids,
        {"ego_history_xyz": data["ego_history_xyz"],
         "ego_history_rot": data["ego_history_rot"]},
    )
    cache, _, _ = model._prefill_prompt_cache(ids, tok)
    return ids, cache


def _pi_map(model, n_slots: int) -> list[int]:
    """Teacher VLM layer for each expert slot, from PRUNE_EXPERT_LAYERS -- never hard-coded."""
    env = os.environ.get("PRUNE_EXPERT_LAYERS", "").strip()
    n_teacher = int(os.environ.get("TEACHER_TEXT_LAYERS", "36"))
    if not env:
        return list(range(n_slots))
    pruned = {int(x) for x in env.split(",") if x != ""}
    pi = [l for l in range(n_teacher) if l not in pruned]
    if len(pi) != n_slots:
        raise RuntimeError(
            f"PRUNE_EXPERT_LAYERS={env} leaves {len(pi)} teacher layers but the student cache "
            f"has {n_slots} -- the ladder would compare mismatched depths."
        )
    return pi


def _regions(ids: torch.Tensor, model) -> dict[str, torch.Tensor]:
    """vision / text / traj position masks, [B, T] -- same split kd_model reports per region."""
    tok = getattr(model, "tokenizer", None)
    img = tok.convert_tokens_to_ids("<|image_pad|>") if tok is not None else -1
    vision = ids == img
    start = getattr(model, "future_token_start_idx", None)
    if start is None:
        traj = torch.zeros_like(vision)
    else:
        traj = (ids >= start) & (ids < start + int(getattr(model.config, "traj_vocab_size", 0)))
    return {"vision": vision, "traj": traj, "text": ~vision & ~traj}


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    sw = cfg.get("sweep", {})
    tag = str(sw.get("tag", "student"))
    bs = int(sw.get("batch_size", 2))
    n_samples = int(sw.get("num_traj_samples", 6))
    limit = int(sw.get("limit", 300))
    only = str(sw.get("only", ""))
    out_dir = str(sw.get("out_dir", "/data/achahe/alpamayo-recipes/recipes/"
                                   "alpamayo1_5_distill/training/cacheladder"))
    os.makedirs(out_dir, exist_ok=True)

    n_vis = torch.cuda.device_count()
    if n_vis < 2:
        raise RuntimeError(
            f"needs 2 visible GPUs (teacher + student on separate cards); saw {n_vis}. "
            "Launch with CUDA_VISIBLE_DEVICES=<teacher>,<student>."
        )
    t_dev, s_dev = torch.device("cuda:0"), torch.device("cuda:1")

    # ⚠️ THE CHECKPOINT. The eval config ships `checkpoint_path: null` because
    # evaluate_hf.py:66-71 assigns `evaluate.eval_ckpt` onto it after composition. A script
    # that instantiates cfg.model directly must do the same, or it silently evaluates the
    # UNTRAINED base VLM -- which is what the first ladder run did (student min_ade 22.44 on
    # clips the trained checkpoint scores 3.89 on).
    ckpt = cfg.get("evaluate", {}).get("eval_ckpt", None)
    if ckpt is None:
        raise RuntimeError("set ++evaluate.eval_ckpt=<student checkpoint>; without it this "
                           "would grade the untrained base VLM's cache.")
    OmegaConf.update(cfg, "model.checkpoint_path", str(ckpt), merge=False)
    print(f"[ladder] student checkpoint: {ckpt}", flush=True)
    student = hyu.instantiate(cfg.model, _convert_="partial").to(s_dev).eval()
    for p in student.parameters():
        p.requires_grad_(False)

    t_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    t_cfg["_target_"] = ("alpamayo1_5_distill.models.stitched_model."
                         "StitchedAlpamayoR1.from_teacher")
    # from_teacher reads its trajectory settings and vlm.* weights from THIS path, so the
    # student's checkpoint must be overwritten, not merged onto.
    t_cfg["checkpoint_path"] = str(sw["teacher_checkpoint"])
    t_cfg["vlm_name_or_path"] = str(sw["teacher_vlm"])
    teacher = hyu.instantiate(t_cfg, _convert_="partial").to(t_dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"[ladder] student on {t_dev.type}:1, teacher on {t_dev.type}:0; "
          f"expert {len(student.expert.layers)} layers", flush=True)
    for d in (0, 1):
        free, total = torch.cuda.mem_get_info(d)
        print(f"[ladder]   cuda:{d} free {free/2**30:.1f} / {total/2**30:.1f} GiB", flush=True)

    ds_cfg = OmegaConf.to_container(cfg.data.val_dataset, resolve=True)
    pre_args = ds_cfg.pop("vla_preprocess_args", None)
    if pre_args is not None:
        ds_cfg["vla_preprocess_args"] = pre_args
    base = hyu.instantiate(ds_cfg, _convert_="partial", model_config=student.config)
    coll = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=student.config)
    metric = hyu.instantiate(cfg.evaluate.metric_runner.metrics[-1], _convert_="partial")
    loader = DataLoader(base, batch_size=bs, collate_fn=coll, num_workers=6, shuffle=False)
    print(f"[ladder] {len(base)} clips, limit {limit}, bs {bs}", flush=True)

    specs = [s for s in SPECS if not only or s[0] == only]
    acc = {s[0]: [] for s in specs}
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

            # ⚠️ position-for-position substitution is only meaningful on identical sequences
            if not torch.equal(t_ids.cpu(), s_ids.cpu()):
                raise RuntimeError(
                    "teacher and student input_ids DIFFER; the towers do not agree "
                    "position-for-position, so cache substitution is meaningless."
                )
            n_slots = _n_layers(s_cache)
            if pi is None:
                pi = _pi_map(student, n_slots)
                print(f"[ladder] depth map: slot j -> teacher layer pi(j), "
                      f"pi[:6]={pi[:6]} ... pi[-3:]={pi[-3:]}", flush=True)
            ts, ss = _kv(t_cache, pi[0])[0].shape[-2], _kv(s_cache, 0)[0].shape[-2]
            if ts != ss:
                raise RuntimeError(f"cache lengths differ: teacher {ts} vs student {ss}")

            # teacher K/V for each slot, moved once and reused by every spec
            t_kv = [(k.detach().to(s_dev), v.detach().to(s_dev))
                    for k, v in (_kv(t_cache, pi[j]) for j in range(n_slots))]
            s_kv = [(k.detach().clone(), v.detach().clone())
                    for k, v in (_kv(s_cache, j) for j in range(n_slots))]
            del t_cache, s_cache
            reg = _regions(s_ids, student)

        for name, layer_sel, region, part, _desc in specs:
            def hook(cache, input_ids, tokenized_data, _l=layer_sel, _r=region, _p=part):
                if _l == "none":
                    return cache
                mixed = DynamicCache()
                for j in range(n_slots):
                    sk, sv = s_kv[j]
                    k, v = sk.clone(), sv.clone()
                    take = _l == "all" or (_l[0] <= j <= _l[1])
                    if take:
                        tk, tv = t_kv[j]
                        if _r is None:
                            if _p != "v":
                                k = tk.clone()
                            if _p != "k":
                                v = tv.clone()
                        else:
                            # [B, T] -> [B, 1, T, 1] so it broadcasts over heads and head_dim
                            m = reg[_r][:, None, :, None].to(k.device)
                            if _p != "v":
                                k = torch.where(m, tk, k)
                            if _p != "k":
                                v = torch.where(m, tv, v)
                    mixed.update(k, v, j, {})
                return mixed

            with torch.no_grad():
                pred_xyz, pred_rot = student.sample_trajectories_prefill_only(
                    data=s_batch, num_traj_samples=n_samples, num_traj_sets=1,
                    cache_hook=hook)
            m = metric.evaluate(student, s_batch, {"pred_xyz": pred_xyz, "pred_rot": pred_rot})
            b = pred_xyz.shape[0]
            for i in range(b):
                acc[name].append({
                    "clip_id": str(clip_ids[i]),
                    **{k: float(v[i]) for k, v in m.items()
                       if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == b},
                })

        seen += bs
        if seen % 20 < bs:
            line = "  ".join(
                f"{n}:{np.nanmean([r.get('min_ade', np.nan) for r in acc[n]]):.3f}"
                for n, *_ in specs if acc[n])
            print(f"[ladder] {seen}/{limit}  {line}", flush=True)

    print(f"\n[ladder] === {tag}: min_ade by substitution (n={seen} clips) ===")
    print(f"{'spec':<16}{'min_ade':>9}{'ade':>9}{'vs student':>12}   description")
    print("-" * 86)
    base_ma = float(np.nanmean([r.get("min_ade", np.nan) for r in acc["student"]])) \
        if acc.get("student") else float("nan")
    rows = []
    for name, _l, _r, _p, desc in specs:
        if not acc[name]:
            continue
        ma = float(np.nanmean([r.get("min_ade", np.nan) for r in acc[name]]))
        ad = float(np.nanmean([r.get("ade", np.nan) for r in acc[name]]))
        rows.append({"spec": name, "min_ade": ma, "ade": ad, "n": len(acc[name]),
                     "delta_vs_student": ma - base_ma, "description": desc})
        print(f"{name:<16}{ma:>9.4f}{ad:>9.4f}{ma - base_ma:>+12.4f}   {desc}")
    with open(os.path.join(out_dir, f"{tag}_ladder.json"), "w") as fh:
        json.dump({"summary": rows, "per_clip": acc}, fh)
    print(f"\n[ladder] wrote {out_dir}/{tag}_ladder.json")
    print("DONE_LADDER", flush=True)


if __name__ == "__main__":
    main()
