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
    "/data/datasets/physical_ai_av/lcdrive_physicalai_av_manifests/"
    "lcdrive_val_mysubset_1k_clip_uuids.txt"
)


def main() -> None:
    import hydra.utils as hyu

    from alpamayo.metrics import distance_metrics
    from alpamayo1_5_distill.scripts import cache_common

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--uuid-filter", default=SUBSET_1K)
    ap.add_argument("--teacher", default="teacher_ar1_5_10b_expert")
    ap.add_argument("--config-name", default="cache_teacher_kv_lcdrive")
    ap.add_argument("--num-traj-samples", type=int, default=6)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
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

    loaders = {k: cache_common.sample_loader(ds[k], idxs, proc[k], num_workers=4) for k in ds}
    rows: list[dict[str, Any]] = []
    tfs = model.tokenizer.convert_tokens_to_ids("<|traj_future_start|>")

    for (i_c, b_c), (i_n, b_n) in zip(loaders["cot"], loaders["nocot"]):
        if i_c != i_n:
            raise RuntimeError(f"loaders desynced: {i_c} != {i_n}")
        if b_c is None or b_n is None:
            continue
        key = ds["cot"]._sample_key(i_c)
        row: dict[str, Any] = {"clip_id": key}
        acc: dict[str, list[float]] = {"cot": [], "nocot": []}
        for arm, batch in (("cot", b_c), ("nocot", b_n)):
            batch = cache_common.to_device(batch, device)
            gt = batch["ego_future_xyz"][:, -1].float()
            tok0 = dict(batch["tokenized_data"])
            for rep in range(args.reps):
                seed = (args.seed * 1_000_003 + len(rows) * 97 + rep) & 0x7FFFFFFF
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                arm_batch = {**batch, "tokenized_data": dict(tok0)}
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz, _ = model.sample_trajectories_from_data_with_vlm_rollout(
                        arm_batch,
                        num_traj_samples=args.num_traj_samples,
                        num_traj_sets=1,
                    )
                m = distance_metrics.compute_minade(
                    pred_xyz.float(), gt, disable_summary=True, timestep_horizons=[]
                )
                acc[arm].append(float(m["min_ade"].mean()))
                if rep == 0 and arm == "cot":
                    row["seq_len"] = int(arm_batch["tokenized_data"].get("input_ids", torch.zeros(1, 0)).shape[-1])
        row["cot"] = st.mean(acc["cot"])
        row["nocot"] = st.mean(acc["nocot"])
        rows.append(row)
        if len(rows) % 10 == 0 or len(rows) == 1:
            d = [r["nocot"] - r["cot"] for r in rows]
            print(
                f"[cot-eval] {len(rows)}/{len(idxs)}  cot={st.mean(r['cot'] for r in rows):.4f}  "
                f"nocot={st.mean(r['nocot'] for r in rows):.4f}  diff={st.mean(d):+.4f}",
                flush=True,
            )

    print(f"\n[cot-eval] n={len(rows)} clips, paired, normal inference\n")
    for a in ("cot", "nocot"):
        v = [r[a] for r in rows]
        print(f"  {a:6s} min_ade {st.mean(v):8.4f}   median {st.median(v):8.4f}")
    d = [r["nocot"] - r["cot"] for r in rows]
    m, se = st.mean(d), (st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan"))
    print(
        f"\n  nocot - cot  {m:+.4f} +- {se:.4f} ({m/se:+.2f}s) "
        f"{'SIG' if abs(m/se) > 2 else 'n.s.'}   median {st.median(d):+.4f}"
    )
    print(f"  positive => the CoT HELPS (removing it costs {m:+.4f})")
    out = args.out or f"/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cot_vs_nocot_sh{args.shard}.json"
    Path(out).write_text(json.dumps(rows, indent=1))
    print(f"\n  per-clip -> {out}")


if __name__ == "__main__":
    main()
