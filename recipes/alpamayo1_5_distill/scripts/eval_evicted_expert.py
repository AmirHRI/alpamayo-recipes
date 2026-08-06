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

"""Does R-KV eviction actually preserve what the action expert needs?

**The ceiling test.** Every KAVA number so far measures a *student* reading a
compressed cache, which confounds two things: how good the compression is, and how
well a 2B can imitate a 10B.  This separates them.  Run the **teacher's own** expert
on the **teacher's own** cache, evicted from ``N_C`` CoT entries down to ``M`` — the
exact tier we distil against.  Whatever min_ade that produces is the ceiling no
student can beat, because the student is trying to reproduce precisely this object.

    full     the teacher's expert on the uncompressed cache        <- reference
    rkv      evicted with the cached sel_idx (S = 0.1*I + 0.9*R)   <- what we distil
    crop     the first M CoT entries (the paper's naive baseline)
    random   M at random, same count                               <- floor

If ``rkv`` sits near ``full``, compression is cheap and the student's gap is a
capacity problem.  If ``rkv`` collapses toward ``random``, M=8 is simply too small
and no amount of student training can help — which would reframe the whole recipe.

Method.  Rather than reimplement the flow-matching sampler, this patches
``vlm.generate`` to hand the real rollout an already-evicted cache.  Three things
have to stay consistent, and two of them are easy to get wrong:

* ``sequences`` loses ``N_C - M`` columns so the downstream ``<traj_future_start>``
  search returns the *new* array index (the attention mask uses it as an index).
  Which columns are dropped from ``sequences`` does not matter — only the count —
  because R-KV selects per (layer, head) and a token id column cannot represent that.
* ``past_key_values`` is gathered **per (layer, head)**, which is what R-KV actually
  produces.  Different heads keep different tokens; that is fine because attention is
  computed per head and the stored keys are post-RoPE, so each surviving key carries
  its own rotation and the gaps are harmless.
* ``rope_deltas`` is incremented by ``N_C - M``.  The action tokens' positions are
  ``rope_deltas + offset``, and ``offset`` shrinks with the array.  Without this
  compensation every action token would sit ``N_C - M`` positions too early relative
  to cache keys whose rotations are already baked in — a silent, uniform corruption
  of every relative offset.

Usage::

    python -m alpamayo1_5_distill.scripts.eval_evicted_expert \\
        --limit 100 --arms full,rkv,crop,random
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path
from typing import Any

import torch

CACHE_ROOT = "/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/teacher_kv_lcdrive"
TIER = "compressed_M8_rkv0.1"


def _evict_cache_per_head(
    cache: Any, lo: int, hi: int, keep: torch.Tensor
) -> int:
    """Gather ``[lo, hi)`` down to ``keep`` per (layer, head), in place.

    Args:
        keep: ``[L, H, M]`` indices RELATIVE to ``lo``, ascending.

    Returns:
        the number of columns removed, ``(hi - lo) - M``.
    """
    n_layers, n_heads, m = keep.shape
    if len(cache.layers) != n_layers:
        raise ValueError(
            f"cache has {len(cache.layers)} layers but sel_idx has {n_layers}; "
            "the cache was built from a different teacher"
        )
    for layer_idx, layer in enumerate(cache.layers):
        for name in ("keys", "values"):
            t = getattr(layer, name)  # [B*, H, S, D]
            b, h, _, d = t.shape
            if h != n_heads:
                raise ValueError(f"{h} kv-heads in the cache, {n_heads} in sel_idx")
            idx = keep[layer_idx].to(t.device) + lo  # [H, M]
            # [B*, H, M, D]; the same per-head choice for every row, since all rows of
            # this batch are num_return_sequences copies of ONE clip.
            gather_idx = idx[None, :, :, None].expand(b, h, m, d)
            mid = t[:, :, lo:hi, :].gather(2, gather_idx - lo)
            setattr(layer, name, torch.cat([t[:, :, :lo, :], mid, t[:, :, hi:, :]], dim=2))
    return (hi - lo) - m


def _selection(mode: str, sel_idx: torch.Tensor, n_cot: int, m: int, seed: int) -> torch.Tensor:
    """``[L, H, M]`` indices relative to the CoT start, for one arm."""
    n_layers, n_heads, _ = sel_idx.shape
    if mode == "rkv":
        return sel_idx
    if mode == "crop":
        base = torch.arange(m)
        return base[None, None, :].expand(n_layers, n_heads, m).contiguous()
    if mode == "random":
        g = torch.Generator().manual_seed(seed)
        out = torch.stack(
            [
                torch.stack(
                    [torch.randperm(n_cot, generator=g)[:m].sort().values for _ in range(n_heads)]
                )
                for _ in range(n_layers)
            ]
        )
        return out
    raise ValueError(f"unknown arm {mode!r}")


def _patched_generate(model: Any, arm: str, state: dict[str, Any]) -> Any:
    """Wrap ``vlm.generate`` so the rollout receives an already-evicted cache."""
    real = model.vlm.generate
    tfs_id = model.tokenizer.convert_tokens_to_ids("<|traj_future_start|>")

    def wrapper(*a: Any, **kw: Any) -> Any:
        out = real(*a, **kw)
        out.rope_deltas = model.vlm.model.rope_deltas
        if arm == "full":
            state["n_cot"] = None
            return out

        seq = out.sequences
        keep_rel, lo, hi = state["keep"], state["lo"], state["hi"]
        dropped = _evict_cache_per_head(out.past_key_values, lo, hi, keep_rel)
        state["dropped"] = dropped
        if dropped == 0:
            return out

        # sequences: drop `dropped` columns from the CoT span. WHICH ones is irrelevant
        # (a token-id row cannot express a per-head choice); only the count matters, and
        # it is what shifts the <traj_future_start> array index the mask is built from.
        out.sequences = torch.cat([seq[:, : lo], seq[:, lo + dropped :]], dim=1)
        # Compensate the position shift: action-token positions are rope_deltas + offset,
        # and offset just shrank by `dropped`. The surviving cache keys keep their original
        # baked-in rotations, so without this every relative offset moves by `dropped`.
        out.rope_deltas = out.rope_deltas + dropped
        n_tfs = int((out.sequences == tfs_id).sum())
        if n_tfs == 0:
            raise RuntimeError("eviction removed the <traj_future_start> column")
        return out

    return wrapper


def main() -> None:
    import hydra.utils as hyu

    from alpamayo1_5_distill.data import kv_cache_io
    from alpamayo1_5_distill.scripts import cache_common

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--arms", default="full,rkv,crop,random")
    ap.add_argument("--cache-root", default=CACHE_ROOT)
    ap.add_argument("--tier", default=TIER)
    ap.add_argument("--config-name", default="cache_teacher_kv_lcdrive")
    # MUST be the expert variant: the plain teacher loads as DistillReasoningVLA, which
    # can generate a CoT but has no action expert, so there is no trajectory to score.
    # KaVaExpertTeacher extends TrainableAlpamayoR1 -> AlpamayoR1, which carries
    # sample_trajectories_from_data_with_vlm_rollout.
    ap.add_argument("--teacher", default="teacher_ar1_5_10b_expert")
    ap.add_argument("--out", default=None)
    ap.add_argument("--num-traj-samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    device = torch.device("cuda")
    cfg = cache_common.compose_config(args.config_name, {"teacher": args.teacher})
    print(f"[evict-eval] arms={arms} limit={args.limit} tier={args.tier}", flush=True)
    model = cache_common.build_teacher(cfg, device)
    dataset = hyu.instantiate(
        cfg.data.cache_dataset, _convert_="partial", model_config=model.config
    )
    print(f"[evict-eval] teacher up, dataset={len(dataset)}", flush=True)

    from alpamayo.metrics import distance_metrics
    from alpamayo1_5_distill.models.teacher_kv import find_cot_span

    cot_start = model.tokenizer.convert_tokens_to_ids("<|cot_start|>")
    cot_end = model.tokenizer.convert_tokens_to_ids("<|cot_end|>")
    tfs_id = model.tokenizer.convert_tokens_to_ids("<|traj_future_start|>")

    # Same loader the cache builder uses: workers decode frames on CPU while the GPU
    # runs, and processor.collate_fn produces the `tokenized_data` the model expects.
    processor = cache_common.build_processor(model)
    candidates = [
        i
        for i in range(len(dataset))
        if kv_cache_io.has_entry(args.cache_root, args.tier, dataset._sample_key(i))
    ]
    print(f"[evict-eval] {len(candidates)} clips have a cached tier entry", flush=True)
    loader = cache_common.sample_loader(dataset, candidates, processor, num_workers=6)

    rows: list[dict[str, Any]] = []
    skipped = {"no_sel_idx": 0, "cot_mismatch": 0, "m_gt_ncot": 0}
    done = 0
    for idx, batch in loader:
        if done >= args.limit:
            break
        if batch is None:
            continue
        key = dataset._sample_key(idx)
        try:
            entry = kv_cache_io.load_entry(args.cache_root, args.tier, key, names=("sel_idx",))
        except KeyError:
            skipped["no_sel_idx"] += 1
            continue
        sel_idx = entry.get("sel_idx")
        if sel_idx is None:
            skipped["no_sel_idx"] += 1
            continue

        batch = cache_common.to_device(batch, device)
        gt_xyz = batch["ego_future_xyz"][:, -1].float()
        # BOTH generate_cot_prefix and the rollout do `tokenized_data.pop("input_ids")`,
        # mutating the caller's dict. Keep a pristine copy and hand each arm a fresh one,
        # or arm 2 dies with a bare KeyError on a batch arm 1 emptied.
        tok0 = dict(batch["tokenized_data"])
        if "input_ids" not in tok0:
            raise KeyError(
                f"tokenized_data has no input_ids; keys={sorted(tok0)} "
                f"batch keys={sorted(batch)}"
            )

        # Locate the CoT span by regenerating the prefix greedily. The cache was built
        # the same way, so the spans should agree; where they do not, sel_idx points at
        # different tokens and the arm would be silently meaningless, so skip.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            trunc, _ = model.generate_cot_prefix(
                dict(tok0),
                ego_history_xyz=batch.get("ego_history_xyz"),
                ego_history_rot=batch.get("ego_history_rot"),
            )
        lo, hi = find_cot_span(trunc, cot_start, cot_end, tfs_id)
        n_cot, m = hi - lo, int(sel_idx.shape[-1])
        if m > n_cot:
            # The cached selection has more entries than this run's CoT -- the teacher
            # regenerated something shorter, so the indices do not apply.
            skipped["m_gt_ncot"] += 1
            continue
        if int(sel_idx.max()) >= n_cot:
            skipped["cot_mismatch"] += 1
            continue

        row: dict[str, Any] = {"clip_id": key, "n_cot": n_cot, "m": m}
        for arm in arms:
            state: dict[str, Any] = {"lo": lo, "hi": hi}
            if arm != "full":
                state["keep"] = _selection(arm, sel_idx.long(), n_cot, m, args.seed + done)
            arm_batch = {**batch, "tokenized_data": dict(tok0)}
            real_gen = model.vlm.generate
            model.vlm.generate = _patched_generate(model, arm, state)
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz, _ = model.sample_trajectories_from_data_with_vlm_rollout(
                        arm_batch, num_traj_samples=args.num_traj_samples, num_traj_sets=1
                    )
            finally:
                model.vlm.generate = real_gen
            mt = distance_metrics.compute_minade(
                pred_xyz.float(), gt_xyz, disable_summary=True, timestep_horizons=[]
            )
            row[arm] = float(mt["min_ade"].mean())
        rows.append(row)
        done += 1
        if done % 5 == 0 or done == 1:
            msg = "  ".join(f"{a}={st.mean(r[a] for r in rows):.4f}" for a in arms)
            print(f"[evict-eval] {done}/{args.limit}  running mean  {msg}", flush=True)

    print(f"\n[evict-eval] n={len(rows)} clips, paired   skipped={skipped}\n")
    print(f"  {'arm':10s}{'min_ade':>10s}{'vs full':>12s}")
    base = [r["full"] for r in rows] if "full" in arms else None
    for a in arms:
        v = [r[a] for r in rows]
        line = f"  {a:10s}{st.mean(v):10.4f}"
        if base is not None and a != "full":
            d = [x - y for x, y in zip(v, base)]
            se = st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
            line += f"{st.mean(d):+8.4f} +- {se:.4f}"
        print(line)
    out = args.out or f"/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/evict_expert_{args.tier}.json"
    Path(out).write_text(json.dumps(rows, indent=1))
    print(f"\n  per-clip -> {out}")


if __name__ == "__main__":
    main()
