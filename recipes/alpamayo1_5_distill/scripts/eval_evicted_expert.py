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

class _SkipClip(RuntimeError):
    """This clip cannot be scored coherently; skip it rather than record a number."""


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


def _selection(
    mode: str,
    sel_idx: torch.Tensor | None,
    n_cot: int,
    m: int,
    seed: int,
    geom: tuple[int, int],
) -> torch.Tensor:
    """``[L, H, M]`` indices relative to the CoT start, for one arm.

    ``sel_idx`` is needed ONLY by the ``rkv`` arm -- it is the cached R-KV choice.
    Every other arm is derivable from ``n_cot`` and the (layers, kv-heads) geometry,
    which is what lets this run on clips with no cache entry at all: the decisive
    ``full`` vs ``none`` comparison never touches the cache.
    """
    n_layers, n_heads = geom
    if mode == "rkv":
        if sel_idx is None:
            raise ValueError("the rkv arm needs a cached sel_idx for this clip")
        return sel_idx
    if mode == "crop":
        base = torch.arange(m)
        return base[None, None, :].expand(n_layers, n_heads, m).contiguous()
    if mode == "identity":
        # CONTROL for the surgery itself: run the per-head gather over the CoT span but
        # keep every entry, so nothing is removed and the sequence is not shortened. If
        # this differs from `full`, the gather path is biased and every removal arm's
        # ~-0.05 offset is an artifact of the machinery rather than a property of the CoT.
        return torch.arange(n_cot)[None, None, :].expand(n_layers, n_heads, n_cot).contiguous()
    if mode == "none" or mode.startswith("pre"):
        # `pre` is the shortening control: the wrapper has already pointed the span
        # at the tokens immediately BEFORE the CoT, so returning an empty selection
        # removes the same COUNT of entries as `none` does, from prompt/vision
        # content instead of from the reasoning.
        # Drop the CoT ENTIRELY. The expert reads the whole prefix cache (~3142 entries,
        # 91.7% of it vision) of which the CoT is ~13 -- 0.41%. If deleting all of it
        # does not move min_ade, the teacher's own reasoning contributes almost nothing
        # to the teacher's own trajectory, which bounds the whole recipe in a way that
        # evicting 13 -> 8 (0.16% of the expert's input) cannot.
        return torch.zeros(n_layers, n_heads, 0, dtype=torch.long)
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
    """Wrap ``vlm.generate``: force greedy, then hand back an evicted cache.

    ⚠️ The rollout's own generation is STOCHASTIC (``temperature=0.6, top_p=0.98``) and
    asks for ``num_return_sequences = num_traj_samples``, so by default it produces six
    DIFFERENT CoTs of different lengths. The cached ``sel_idx`` was computed against the
    greedy CoT, so under sampling it indexes a span that does not exist -- which shows up
    as an out-of-bounds gather, reported asynchronously at whatever CUDA call comes next.
    Forcing greedy makes the rollout's CoT identical to the cached one, which is the only
    condition under which this comparison means anything.

    The span is then read from the sequence that was actually generated, not from a
    separate pass, so the two can never drift apart silently.
    """
    from alpamayo1_5_distill.models.teacher_kv import find_cot_span

    real = model.vlm.generate
    tok = model.tokenizer
    cot_start = tok.convert_tokens_to_ids("<|cot_start|>")
    cot_end = tok.convert_tokens_to_ids("<|cot_end|>")
    tfs_id = tok.convert_tokens_to_ids("<|traj_future_start|>")

    def wrapper(*a: Any, **kw: Any) -> Any:
        gc = kw.get("generation_config")
        if gc is not None:
            gc.do_sample = False
            gc.temperature = None
            gc.top_p = None
            gc.top_k = None
            gc.num_return_sequences = 1
        out = real(*a, **kw)
        out.rope_deltas = model.vlm.model.rope_deltas

        lo, hi = find_cot_span(out.sequences, cot_start, cot_end, tfs_id)
        state["lo"], state["hi"], state["n_cot"] = lo, hi, hi - lo
        if arm.startswith("pre"):
            # CONTROL for cache SHORTENING, as opposed to CoT removal. Evict the same
            # number of entries from the span immediately BEFORE the CoT -- prompt/vision
            # content -- leaving the CoT fully intact. `identity` shows the gather is
            # neutral but never shortens; this shortens by the same amount without
            # touching the reasoning. If `pre` also improves min_ade, the gain is about
            # cache length, not about the CoT.
            # `preN` removes N entries; bare `pre` matches the CoT length exactly.
            span = int(arm[3:]) if len(arm) > 3 else (hi - lo)
            if lo - span < 1:
                raise _SkipClip(f"no room for a {span}-token pre-CoT span at lo={lo}")
            lo, hi = lo - span, lo
        if arm == "full":
            return out

        keep_rel = state["make_keep"](hi - lo)
        if keep_rel is None:
            raise _SkipClip(f"selection does not fit N_C={hi - lo}")
        dropped = _evict_cache_per_head(out.past_key_values, lo, hi, keep_rel)
        state["dropped"] = dropped
        if dropped == 0:
            return out

        seq = out.sequences
        # Drop `dropped` columns from the CoT span so the downstream
        # <traj_future_start> search returns the NEW array index (the attention mask
        # uses it as an index). WHICH columns is irrelevant -- only the count -- since a
        # token-id row cannot represent a per-head choice.
        out.sequences = torch.cat([seq[:, :lo], seq[:, lo + dropped :]], dim=1)
        # Action-token positions are rope_deltas + offset, and offset just shrank by
        # `dropped`, while surviving cache keys keep their baked-in rotations. Without
        # this every relative offset would silently move by `dropped`.
        out.rope_deltas = out.rope_deltas + dropped
        if int((out.sequences == tfs_id).sum()) == 0:
            raise _SkipClip("eviction removed the <traj_future_start> column")
        return out

    return wrapper


def main() -> None:
    import hydra.utils as hyu

    from alpamayo1_5_distill.data import kv_cache_io
    from alpamayo1_5_distill.scripts import cache_common

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--arms", default="full,rkv,crop,random,none")
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
    # Independent noise draws per clip, each shared across all arms. The
    # per-arm noise is the dominant variance at best-of-1, so reps buy far more
    # than extra clips do.
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--m", type=int, default=8, help="budget for arms that need no cache")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--uuid-filter", default=None, help="override the dataset clip filter")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    device = torch.device("cuda")
    cfg = cache_common.compose_config(args.config_name, {"teacher": args.teacher})
    print(f"[evict-eval] arms={arms} limit={args.limit} tier={args.tier}", flush=True)
    model = cache_common.build_teacher(cfg, device)
    if args.uuid_filter:
        cfg.data.cache_dataset.clip_uuid_filter = args.uuid_filter
    dataset = hyu.instantiate(
        cfg.data.cache_dataset, _convert_="partial", model_config=model.config
    )
    print(f"[evict-eval] teacher up, dataset={len(dataset)}", flush=True)

    from alpamayo.metrics import distance_metrics

    # Same loader the cache builder uses: workers decode frames on CPU while the GPU
    # runs, and processor.collate_fn produces the `tokenized_data` the model expects.
    processor = cache_common.build_processor(model)
    geom = (
        len(model.vlm.model.language_model.layers),
        int(getattr(model.vlm.config, "text_config", model.vlm.config).num_key_value_heads),
    )
    needs_cache = "rkv" in arms
    candidates = list(range(len(dataset)))
    if needs_cache:
        # Only the rkv arm needs the cached R-KV choice. Every other arm is derivable
        # from n_cot alone, which is what lets this run on clips with no cache entry --
        # e.g. the OOD-reasoning set, which was never cached.
        candidates = [
            i for i in candidates
            if kv_cache_io.has_entry(args.cache_root, args.tier, dataset._sample_key(i))
        ]
    candidates = candidates[args.shard :: args.num_shards]
    print(
        f"[evict-eval] {len(candidates)} candidate clips "
        f"(shard {args.shard}/{args.num_shards}, cache required={needs_cache})",
        flush=True,
    )
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
        sel_idx = None
        try:
            entry = kv_cache_io.load_entry(args.cache_root, args.tier, key, names=("sel_idx",))
            sel_idx = entry.get("sel_idx")
        except KeyError:
            pass
        if sel_idx is None and needs_cache:
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

        # No pre-pass: the wrapper reads the span from the sequence the rollout itself
        # generates, so the two cannot drift. `full` runs first and records n_cot.
        #
        # ⚠️ Every arm is re-seeded to the SAME value before its rollout, and each clip
        # is repeated over `--reps` independent seeds. This is load-bearing, not tidiness.
        # diffusion.sample() draws random initial noise; without seeding, each arm gets
        # its own draw and at best-of-1 that noise swamps the signal. Measured: two runs
        # of the IDENTICAL `full` arm over the same 150 clips differed by
        # -0.2475 +- 0.1308 -- 1.89 sigma on an effect that is exactly zero by
        # construction, and the same magnitude as every difference this script exists to
        # detect. Pairing the clips is not enough; the noise draw has to be paired too.
        acc: dict[str, list[float]] = {a: [] for a in arms}
        sel = None if sel_idx is None else sel_idx.long()
        m = int(sel.shape[-1]) if sel is not None else args.m
        n_cot_seen: int | None = None
        ok = True
        for rep in range(args.reps):
            seed_base = (args.seed * 1_000_003 + done * 97 + rep) & 0x7FFFFFFF
            for arm in arms:
                state: dict[str, Any] = {}

                def make_keep(n_cot: int, _arm: str = arm, _sel: Any = sel) -> Any:
                    if _arm not in ("none", "identity") and not _arm.startswith("pre"):
                        if m > n_cot:
                            return None
                        if _sel is not None and int(_sel.max()) >= n_cot:
                            return None
                    return _selection(_arm, _sel, n_cot, m, seed_base, geom)

                state["make_keep"] = make_keep
                arm_batch = {**batch, "tokenized_data": dict(tok0)}
                torch.manual_seed(seed_base)
                torch.cuda.manual_seed_all(seed_base)
                real_gen = model.vlm.generate
                model.vlm.generate = _patched_generate(model, arm, state)
                try:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        pred_xyz, _ = model.sample_trajectories_from_data_with_vlm_rollout(
                            arm_batch, num_traj_samples=1, num_traj_sets=1
                        )
                except _SkipClip as ex:
                    skipped["cot_mismatch"] += 1
                    print(f"[evict-eval] skip {key[:12]} ({arm}): {ex}", flush=True)
                    ok = False
                    break
                finally:
                    model.vlm.generate = real_gen
                mt = distance_metrics.compute_minade(
                    pred_xyz.float(), gt_xyz, disable_summary=True, timestep_horizons=[]
                )
                acc[arm].append(float(mt["min_ade"].mean()))
                n_cot_seen = state.get("n_cot", n_cot_seen)
            if not ok:
                break
        if not ok:
            continue
        row: dict[str, Any] = {"clip_id": key, "n_cot": n_cot_seen}
        row.update({a: st.mean(v) for a, v in acc.items()})
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
