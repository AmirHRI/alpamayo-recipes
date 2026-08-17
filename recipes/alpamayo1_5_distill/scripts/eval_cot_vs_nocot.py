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

"""Does the CoT change the action expert's trajectory, under NORMAL inference?

The companion to ``eval_evicted_expert.py``, and a cleaner instrument.  That script
answered "what if the CoT's K/V are surgically removed from the cache", which needed
per-head gathers, sequence edits and a rope compensation — three places to be subtly
wrong, and it perturbs 14 of ~3019 entries, four to seven times below the expert's
measured detection threshold.

This asks the same question with **no surgery at all**: run the model the way it is
actually run, twice, changing only whether the prompt has a CoT component.

    cot     vla_processor=distill_teacher_generate
            components_order [image, traj_history, prompt, cot]  -> the model reasons,
            then the expert reads a cache containing that reasoning
    nocot   vla_processor=default
            components_order [image, traj_history, prompt, traj_future]  -> no `cot`
            component exists, so the expert reads a cache that never held one

Both arms use the stock rollout: stochastic sampling, ``num_traj_samples`` trajectories,
10 Euler steps.  Nothing is patched.  The difference between them is the entire value
the chain-of-thought adds to the action expert.

Pairing.  The two arms need different processors and therefore different datasets, so
they are built separately and walked in lockstep by clip id (the same UUID filter yields
the same order).  ``torch.manual_seed`` is set identically per (clip, rep) before each
rollout, so the diffusion noise is shared; without that, the noise alone swamps the
signal — measured in the sibling script as a +-0.248 floor between two runs of an
*identical* arm.

Usage::

    python -m alpamayo1_5_distill.scripts.eval_cot_vs_nocot \\
        --limit 500 --shard 0 --num-shards 2
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path
from typing import Any

import torch

SUBSET_1K = (
    "/temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests/"
    "lcdrive_val_mysubset_1k_clip_uuids.txt"
)


def _batched_loader(dataset: Any, indices: list[int], processor: Any, batch: int, workers: int):
    """``(idx_list, collated)`` batches of ``batch`` clips.

    ``cache_common.sample_loader`` is fixed at one clip per batch, which leaves the GPU
    latency-bound: the CoT generation (~13 tokens) and the 10 Euler steps are
    bandwidth-bound, so they barely use the SMs at b_star=6. Batching multiplies the
    work per kernel launch. The rollout already supports B>1 -- it loops
    ``for i in range(b_star)`` when building the per-row attention mask and offsets.
    """
    from torch.utils.data import DataLoader

    class _IdxDataset:
        def __len__(self) -> int:
            return len(indices)

        def __getitem__(self, j: int):
            i = indices[j]
            try:
                return i, dataset[i]
            except Exception as ex:
                print(f"[cot-eval] idx={i} load error: {ex}", flush=True)
                return i, None

    def _collate(items):
        good = [(i, s) for i, s in items if s is not None]
        if not good:
            return [], None
        return [i for i, _ in good], processor.collate_fn([s for _, s in good])

    return iter(
        DataLoader(
            _IdxDataset(),
            batch_size=batch,
            num_workers=workers,
            collate_fn=_collate,
            prefetch_factor=2 if workers else None,
        )
    )


def _full_metrics(
    pred_xyz: torch.Tensor,
    pred_rot: torch.Tensor,
    gt_xyz: torch.Tensor,
    gt_rot: torch.Tensor,
    time_step: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Per-clip metrics, mirroring ``MetricAPI`` exactly so results are joinable.

    Reproduces the same set the standard eval writes to
    ``lcdrive_val_per_clip_metrics.json`` -- min_ade with the 0.5/1/3/5 s horizons, the
    single-sample ade and its 3 s variant, and corner distance -- rather than min_ade
    alone.  That matters: on the KAVA T=2 arm ``ade`` and ``min_ade`` told opposite
    stories (min_ade flat, ade 2x worse on a 10% tail), so recording one and not the
    other can hide the whole effect.

    ⚠️ ``ade`` is NOT the mean over modes and NOT the best mode.  metric_api.py:225 sets
    ``logprob = torch.zeros_like(...)`` with a "dummy logprob for now" comment, so its
    argmax is always 0 and ``ade`` is sample 0's error.  Since modes differ only by noise
    seed that makes it an unbiased estimate of a TYPICAL draw, which is arguably the more
    deployment-relevant number -- you get one trajectory at runtime, not the best of six.

    Returns:
        name -> ``[B]`` tensor, one value per clip.
    """
    from alpamayo.metrics import distance_metrics
    from alpamayo.metrics.metric_api import EGO_VEHICLE_LWH

    horizons = [int(t / time_step) for t in (0.5, 1.0, 3.0, 5.0)]
    out = distance_metrics.compute_minade(
        pred_xyz, gt_xyz, disable_summary=True, timestep_horizons=horizons, time_step=time_step
    )
    logprob = torch.zeros_like(pred_xyz[..., 0])
    idx = logprob.sum(dim=-1).argmax(dim=-1)
    top_xyz = torch.take_along_dim(pred_xyz, idx[..., None, None, None], dim=2)
    out["ade"] = distance_metrics.compute_ade(top_xyz, gt_xyz).squeeze(2).mean(-1)
    h3 = int(3.0 / time_step)
    if h3 <= pred_xyz.shape[3]:
        out["ade/by_t=3.0"] = (
            distance_metrics.compute_ade(top_xyz, gt_xyz, timestep_horizon=h3).squeeze(2).mean(-1)
        )
    out.update(
        distance_metrics.compute_grouped_corner_distance(
            pred_xyz,
            pred_rot,
            gt_xyz,
            gt_rot,
            torch.tensor(EGO_VEHICLE_LWH, dtype=torch.float32, device=gt_xyz.device),
            disable_summary=True,
        )
    )
    # every entry -> [B], one number per clip
    return {k: v.reshape(pred_xyz.shape[0], -1).mean(dim=-1) for k, v in out.items()}


def main() -> None:
    import hydra.utils as hyu

    from alpamayo.metrics import distance_metrics
    from alpamayo1_5_distill.scripts import cache_common
    from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA

    global _VLM_ONLY
    _VLM_ONLY = TrainableReasoningVLA.sample_trajectories_from_data

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--uuid-filter", default=SUBSET_1K)
    ap.add_argument("--teacher", default="teacher_ar1_5_10b_expert")
    ap.add_argument("--config-name", default="cache_teacher_kv_lcdrive")
    ap.add_argument("--num-traj-samples", type=int, default=6)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1,
                    help="clips per rollout; the CoT-generation and Euler phases are\n"
                         "bandwidth-bound, so batching is where the speedup is")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    # Validation aid only. The VLM's own sampling consumes RNG BEFORE the diffusion noise
    # and draws for all rows at once, so at batch>1 even row 0 gets different tokens and
    # leaves the generator in a different state. Forcing greedy removes that channel, so a
    # batch=1 vs batch=N comparison isolates padding. NOT for real runs -- greedy is not
    # normal usage and caps num_traj_samples at 1.
    ap.add_argument("--greedy", action="store_true")
    # Score the VLM's OWN trajectory tokens instead of the action expert's output.
    # TrainableAlpamayoR1 overrides sample_trajectories_from_data with the expert
    # rollout; the base TrainableReasoningVLA implementation generates the VLM's 128
    # discrete traj tokens and decodes them with traj_tokenizer.decode(). Calling the
    # base method unbound bypasses the expert entirely -- same weights, different head.
    ap.add_argument("--vlm-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    print(f"[cot-eval] subset={Path(args.uuid_filter).name} limit={args.limit}", flush=True)

    # One model, two datasets. `vla_processor@...` selects the prompt composition, which
    # is the ONLY thing differing between the arms.
    base = {"teacher": args.teacher, "data.cache_dataset.clip_uuid_filter": args.uuid_filter}
    cfg_cot = cache_common.compose_config(
        args.config_name,
        {**base, "vla_processor@data.cache_dataset.vla_preprocess_args": "distill_teacher_generate"},
    )
    cfg_no = cache_common.compose_config(
        args.config_name,
        {**base, "vla_processor@data.cache_dataset.vla_preprocess_args": "default"},
    )
    model = cache_common.build_teacher(cfg_cot, device)
    ds = {
        "cot": hyu.instantiate(cfg_cot.data.cache_dataset, _convert_="partial", model_config=model.config),
        "nocot": hyu.instantiate(cfg_no.data.cache_dataset, _convert_="partial", model_config=model.config),
    }
    if len(ds["cot"]) != len(ds["nocot"]):
        raise RuntimeError(f"datasets disagree: {len(ds['cot'])} vs {len(ds['nocot'])}")
    proc = {k: cache_common.build_processor(model) for k in ds}
    n = len(ds["cot"])
    idxs = list(range(n))[args.shard :: args.num_shards][: args.limit]
    print(f"[cot-eval] {n} clips in subset, this shard takes {len(idxs)}", flush=True)

    loaders = {
        k: _batched_loader(ds[k], idxs, proc[k], args.batch, workers=4) for k in ds
    }
    rows: list[dict[str, Any]] = []
    n_batch = 0

    for (ix_c, b_c), (ix_n, b_n) in zip(loaders["cot"], loaders["nocot"]):
        # Both arms must hold the SAME clips in the SAME order, or the pairing -- and the
        # shared diffusion seed -- silently compares different scenes.
        if ix_c != ix_n:
            raise RuntimeError(f"loaders desynced: {ix_c} != {ix_n}")
        if b_c is None or b_n is None:
            continue
        per_arm: dict[str, dict[str, list[list[float]]]] = {}
        for arm, batch in (("cot", b_c), ("nocot", b_n)):
            batch = cache_common.to_device(batch, device)
            gt_xyz = batch["ego_future_xyz"][:, -1].float()
            gt_rot = batch["ego_future_rot"][:, -1].float()
            tok0 = dict(batch["tokenized_data"])
            acc: dict[str, list[list[float]]] = {}
            for rep in range(args.reps):
                # Seed keyed on the BATCH, identical across arms. Both arms draw the same
                # noise because total_batch = len(batch) * num_traj_samples matches.
                seed = (args.seed * 1_000_003 + n_batch * 97 + rep) & 0x7FFFFFFF
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                arm_batch = {**batch, "tokenized_data": dict(tok0)}
                # ⚠️ Seeding before the rollout is NOT enough to pair the arms. The
                # rollout calls vlm.generate first, which consumes RNG while sampling
                # tokens -- and `cot` generates ~13 tokens where `nocot` generates 1-2.
                # The two arms therefore reach diffusion.sample with different generator
                # states and draw DIFFERENT initial noise, which is the +-0.248 floor this
                # pairing exists to remove. Re-seed at the diffusion call itself so both
                # arms provably start from the same noise.
                real_sample = model.diffusion.sample
                def _seeded_sample(*a, __s=real_sample, __seed=seed, **kw):
                    torch.manual_seed(__seed)
                    torch.cuda.manual_seed_all(__seed)
                    return __s(*a, **kw)
                model.diffusion.sample = _seeded_sample
                real_gen = model.vlm.generate
                if args.greedy:
                    def _greedy(*a, __r=real_gen, **kw):
                        gc = kw.get("generation_config")
                        if gc is not None:
                            gc.do_sample, gc.temperature = False, None
                            gc.top_p = gc.top_k = None
                            gc.num_return_sequences = 1
                        return __r(*a, **kw)
                    model.vlm.generate = _greedy
                try:
                  with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    if args.vlm_only:
                        # max_generation_length must cover the CoT AND the 128 traj
                        # tokens; the default is exactly 128, which would truncate the
                        # trajectory on the `cot` arm and silently score a short one.
                        pred_xyz, pred_rot = _VLM_ONLY(
                            model,
                            arm_batch,
                            num_traj_samples=args.num_traj_samples,
                            num_traj_sets=1,
                            max_generation_length=256,
                        )
                    else:
                        pred_xyz, pred_rot = model.sample_trajectories_from_data_with_vlm_rollout(
                            arm_batch,
                            num_traj_samples=args.num_traj_samples,
                            num_traj_sets=1,
                        )
                finally:
                    model.vlm.generate = real_gen
                    model.diffusion.sample = real_sample
                for name, vals in _full_metrics(
                    pred_xyz.float(), pred_rot.float(), gt_xyz, gt_rot
                ).items():
                    acc.setdefault(name, []).append([float(x) for x in vals])
            per_arm[arm] = acc

        for j, idx in enumerate(ix_c):
            # BARE uuid, not `_sample_key`'s "<uuid>::<t0>" composite -- the standard eval's
            # per-clip file keys on the plain uuid, and these results are meant to join
            # against it.
            row: dict[str, Any] = {
                "clip_id": ds["cot"].clip_ids[idx],
                "sample_key": ds["cot"]._sample_key(idx),
            }
            for arm in ("cot", "nocot"):
                row[arm] = {
                    name: st.mean(rep[j] for rep in reps)
                    for name, reps in per_arm[arm].items()
                }
            rows.append(row)
        n_batch += 1
        if n_batch % 5 == 1:
            d = [r["nocot"]["min_ade"] - r["cot"]["min_ade"] for r in rows]
            print(
                f"[cot-eval] {len(rows)}/{len(idxs)}  "
                f"cot={st.mean(r['cot']['min_ade'] for r in rows):.4f}  "
                f"nocot={st.mean(r['nocot']['min_ade'] for r in rows):.4f}  diff={st.mean(d):+.4f}",
                flush=True,
            )

    print(f"\n[cot-eval] n={len(rows)} clips, paired, normal inference\n")
    names = sorted(rows[0]["cot"]) if rows else []
    print(f"  {'metric':22s}{'cot':>10s}{'nocot':>10s}{'nocot - cot':>24s}")
    for name in names:
        c = [r["cot"][name] for r in rows]
        nc = [r["nocot"][name] for r in rows]
        d = [b - a for a, b in zip(c, nc)]
        m = st.mean(d)
        se = st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
        z = m / se if se and se == se else 0.0
        tag = "SIG " if abs(z) > 2 else "n.s."
        print(f"  {name:22s}{st.mean(c):10.4f}{st.mean(nc):10.4f}   {m:+8.4f} +- {se:.4f} ({z:+5.2f}s) {tag}")
    print("\n  positive `nocot - cot` => the CoT HELPS (removing it costs that much)")

    out = args.out or f"/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cot_vs_nocot_sh{args.shard}.json"
    Path(out).write_text(json.dumps(rows, indent=1))
    print(f"\n  per-clip -> {out}")


if __name__ == "__main__":
    main()
