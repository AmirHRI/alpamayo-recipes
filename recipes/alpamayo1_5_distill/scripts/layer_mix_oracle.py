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

r"""Can 28 cache layers drive a 36-layer action expert?  The ceiling, before any training.

``models/layer_mix.py`` proposes keeping the teacher's full 36-layer expert on a 28-layer
student by synthesising its 36 cache slots from the student's 28 with learned block-convex
matrices.  Training that costs GPU-weeks.  This costs one eval pass, and it answers the
question the training arm cannot separate: **is the 28 -> 36 reconstruction good enough, or
is the student's capacity the binding constraint?**

The trick is to take the student out of it entirely.  Run the TEACHER, whose cache is by
definition the right answer, throw away 8 of its 36 layers, reconstruct all 36 from the
surviving 28 by CLOSED-FORM least squares, and drive the teacher's own expert with the
result.  Whatever min_ade that produces is an upper bound on what any learned P can reach --
the fit is optimal for the objective, and a trained student's cache is strictly worse input
than the teacher's own.

**Decision rule.**  Pruning to 28 costs the teacher 0.5776 -> 0.7893 (PRUNING.md).  If the
fitted reconstruction does not land clearly below 0.7893, a learned mix cannot beat pruning
and ``ARM=mix2bnav`` should not be launched.

Four variants, each on the same clips with the same diffusion noise:

    full         the teacher's own 36-layer cache                  -- the harness control
    fit          36 rebuilt from 28 by the fitted convex P         -- THE CEILING
    tent         36 rebuilt by the untrained depth-matched init    -- where training starts
    gather       36 rebuilt by nearest-source one-hot (no mixing)  -- does MIXING buy anything?

It also reports how far P must TRAVEL in logit space from the tent init to the fitted
optimum, which calibrates ``lr_multiplier`` instead of guessing it: under Adam a logit moves
~``lr`` per step, so the travel divided by ``base_lr x steps`` is the multiplier that just
lets P arrive.

``gather`` is the one that is easy to leave out and shouldn't be.  Without it a good ``fit``
number cannot distinguish "a convex mix reconstructs the missing layers" from "28 layers were
always enough and the expert never needed the other 8".

⚠️ **The fit is RoPE-invariant, so it does not matter that the prefill cache is post-RoPE.**
RoPE is orthogonal per position, so ``|| sum_i a_i R k_i - R k_j || == || sum_i a_i k_i - k_j ||``
for any weights -- the least-squares problem, and its solution, are identical in either space.
The same linearity is what lets one P serve training (pre-RoPE) and deployment (post-RoPE).

⚠️ **Seeded per batch**, like ``eval_step_sweep.py``: ``flow_matching._euler`` draws its
initial noise off the global RNG, and unpaired noise across variants is precisely what
retracted the R-KV-vs-random result (commit 12702e0) -- two runs of an identical arm differed
by 1.89 sigma on an effect that is zero by construction.

Usage::

    CUDA_VISIBLE_DEVICES=2 python -m alpamayo1_5_distill.scripts.layer_mix_oracle \
        --config-path pkg://alpamayo1_5_distill/configs \
        --config-name sft_eval_stitched_2b_layermix_lcdrive \
        ++model.attn_implementation=sdpa \
        ++probe.n_fit=32 ++probe.n_clips=100 \
        ++probe.out=/temp/achahe/layer_mix_oracle.json
"""

from __future__ import annotations

import json

import hydra
import hydra.utils as hyu
import numpy as np
import torch
from accelerate.utils import send_to_device
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from alpamayo1_5_distill.models.kv_distill import build_layer_map
from alpamayo1_5_distill.models.layer_mix import (
    LayerMixer,
    gather_weights,
    tent_weights,
)
from alpamayo1_5_distill.models.stitched_model import StitchedAlpamayoR1
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA

TEACHER_CKPT = "/temp/achahe/hf_cache/hub/models--nvidia--Alpamayo-1.5-10B-A1-format"
COSMOS = (
    "/temp/achahe/hf_cache/hub/models--nvidia--Cosmos-Reason2-8B/"
    "snapshots/a9fae2cf89dc64db96b12860417f0eb403013bb9"
)
#: Same for every variant of a clip. The value is arbitrary; using the SAME one is not.
SEED = 1234

#: The set-C survivors -- the OTHER principled 36 -> 28 subset, and the one the pruned arm
#: actually uses for the expert. Offered as `++probe.subset=pi` so the choice is measured
#: rather than assumed, in the spirit of BLOCK_PI_PROBE.
PI = [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 14, 16, 17, 18, 20,
      21, 22, 23, 24, 26, 28, 29, 30, 31, 32, 33, 35]


def paired_stats(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Mean paired difference ``a - b``, its standard error, and z.

    ⚠️ PAIRED, not two independent means. Every variant sees the same clips in the same
    order with the same diffusion noise, so the per-clip difference removes clip difficulty
    entirely -- which is most of the variance. Comparing the two means and their separate
    SEMs would understate the significance by a large factor, and is not what the design
    supports.
    """
    d = np.array([x - y for x, y in zip(a, b)], dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return float("nan"), float("nan"), float("nan")
    sem = float(d.std(ddof=1) / np.sqrt(len(d)))
    return float(d.mean()), sem, float(d.mean() / sem) if sem > 0 else float("inf")


def source_layers(kind: str, n_src: int, n_tgt: int) -> list[int]:
    """Which ``n_src`` of the teacher's ``n_tgt`` cache layers stand in for the student's.

    ``stride`` is ``build_layer_map``'s own depth-proportional convention, i.e. the layers a
    28-deep tower is taken to correspond to. ``pi`` is the set-C survivor list. They differ,
    and which is the fairer proxy is exactly the kind of thing this tree scores rather than
    argues about.
    """
    if kind == "stride":
        return build_layer_map(n_src, n_tgt)
    if kind == "pi":
        if len(PI) != n_src:
            raise ValueError(f"pi has {len(PI)} entries, need {n_src}")
        return list(PI)
    raise ValueError(f"probe.subset must be stride|pi, got {kind!r}")


# --------------------------------------------------------------------- the fit


class BlockFit:
    """Streaming normal equations for one block's ``g_in -> g_out`` convex reconstruction.

    Per block we need one Gram ``G[i,i'] = <k_i, k_i'>`` over the block's ``g_in`` source
    layers (shared by all ``g_out`` targets) and one cross term ``c_j[i] = <k_i, k_j>`` per
    target.  Both accumulate in **float64** across clips -- these are sums over ~1.6k
    positions x 8 heads x 128 dims per clip, and fp32 loses the tail.
    """

    def __init__(self, g_in: int, g_out: int) -> None:
        self.g = torch.zeros(g_in, g_in, dtype=torch.float64)
        self.c = torch.zeros(g_in, g_out, dtype=torch.float64)
        self.t = torch.zeros(g_out, dtype=torch.float64)      # ||target||^2, for rel-L2
        self.n = 0

    def add(self, src: torch.Tensor, tgt: torch.Tensor) -> None:
        """``src``: ``[g_in, N]``, ``tgt``: ``[g_out, N]``, both flattened over B/H/T/D."""
        s, t = src.double(), tgt.double()
        self.g += s @ s.T
        self.c += s @ t.T
        self.t += (t * t).sum(dim=1)
        self.n += s.shape[1]

    def solve(self, iters: int = 4000) -> tuple[torch.Tensor, torch.Tensor]:
        """Convex weights ``[g_in, g_out]`` and the residual rel-L2 per target.

        Projected gradient on ``a^T G a - 2 a^T c``, restricted to the simplex.  With 7
        variables this converges in well under a second and, unlike an unconstrained solve
        followed by a projection, the answer is actually the constrained optimum -- projecting
        a least-squares solution onto the simplex is NOT the same as minimising over it, and
        the difference is exactly the "does the convexity constraint cost anything" question
        this probe exists to answer.
        """
        g_in, g_out = self.c.shape
        a = torch.full((g_in, g_out), 1.0 / g_in, dtype=torch.float64)
        # 1/L step, L = largest eigenvalue of 2G: the standard guarantee, no tuning.
        lr = 1.0 / (2.0 * torch.linalg.eigvalsh(self.g).max().clamp_min(1e-12))
        for _ in range(iters):
            a = _project_simplex(a - lr * 2.0 * (self.g @ a - self.c))
        resid = ((a * (self.g @ a)).sum(0) - 2.0 * (a * self.c).sum(0) + self.t)
        return a.float(), (resid.clamp_min(0) / self.t.clamp_min(1e-12)).sqrt().float()


def _project_simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of each COLUMN onto the probability simplex (Duchi et al.)."""
    n = v.shape[0]
    u = torch.sort(v, dim=0, descending=True).values
    css = u.cumsum(dim=0) - 1.0
    idx = torch.arange(1, n + 1, dtype=v.dtype, device=v.device).unsqueeze(1)
    rho = ((u - css / idx) > 0).to(v.dtype).cumsum(dim=0).argmax(dim=0)
    theta = css.gather(0, rho.unsqueeze(0)) / (rho + 1).to(v.dtype).unsqueeze(0)
    return (v - theta).clamp_min(0.0)


def logit_travel(fit: torch.Tensor, tent: torch.Tensor) -> float:
    """How far P must move in logit space to get from ``tent`` to ``fit``.

    ⚠️ **CENTRED per column, and that is not cosmetic.**  Softmax is shift-invariant, so
    ``log(fit) - log(tent)`` is only defined up to an arbitrary constant per column: adding
    0.9 to every logit leaves P bit-identical yet reports 0.9 of "travel", and the raw max is
    therefore not a distance at all.  Subtracting the column mean picks the minimum-norm
    representative of that equivalence class -- the displacement an optimiser actually has to
    produce.  Verified: a pure constant shift returns ~0.

    Under Adam a logit moves ~``lr`` per step regardless of gradient magnitude, so this
    divided by ``base_lr x steps`` is the ``lr_multiplier`` that just lets P arrive.
    """
    d = torch.log(fit.clamp_min(1e-3)) - torch.log(tent.clamp_min(1e-3))
    return float((d - d.mean(dim=-2, keepdim=True)).abs().max())


def subset_cache(cache, keep: list[int]):
    """A fresh cache holding only ``keep``, in order -- the 28 a shallow tower would have."""
    from transformers.cache_utils import DynamicCache

    out = DynamicCache()
    for j, i in enumerate(keep):
        out.update(cache.layers[i].keys, cache.layers[i].values, j, {})
    return out


def make_hook(keep: list[int], mixer: LayerMixer | None):
    """``cache_hook`` for ``sample_trajectories_prefill_only``: 36 -> 28 -> 36."""

    def hook(cache, _input_ids, _tokenized):
        if mixer is None:
            return cache                       # the `full` control: untouched
        sub = subset_cache(cache, keep)
        return mixer.to(device=sub.layers[0].keys.device, dtype=sub.layers[0].keys.dtype).mix_cache(sub)

    return hook


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    probe = cfg.get("probe", {})
    n_fit = int(probe.get("n_fit", 32))
    n_clips = int(probe.get("n_clips", 100))
    n_blocks = int(probe.get("blocks", 4))
    n_src = int(probe.get("n_src", 28))
    subset = str(probe.get("subset", "stride"))
    source = str(probe.get("source", "teacher"))
    out_path = probe.get("out", "layer_mix_oracle.json")
    n_samples = int(probe.get("num_traj_samples", 6))
    baseline = probe.get("baseline_min_ade", None)
    if source not in ("teacher", "student"):
        raise ValueError(f"probe.source must be teacher|student, got {source!r}")

    dev = torch.device("cuda")

    # ---- models -------------------------------------------------------------
    # `teacher`: one model. Sources are a SUBSET of the teacher's own cache layers, so 28 of
    #     the 36 targets are available verbatim -- the reconstruction is degenerate and what
    #     this measures is how much the expert minds 8 bad slots, not whether mixing works.
    # `student`: two towers. Sources are a REAL 2B student's 28 layers, targets the teacher's
    #     36. No pass-through exists, so this is the honest test of the mix -- and the one
    #     whose answer decides ARM=mix2bnav.
    if cfg.evaluate.get("eval_ckpt"):
        cfg.model.checkpoint_path = cfg.evaluate.eval_ckpt
    if source == "teacher":
        driver = StitchedAlpamayoR1.from_teacher(
            checkpoint_path=TEACHER_CKPT, vlm_name_or_path=COSMOS,
            attn_implementation=cfg.model.get("attn_implementation", "sdpa"),
        ).to(dev).eval()
        target_vlm = None
    else:
        if not cfg.model.get("checkpoint_path"):
            raise ValueError(
                "probe.source=student needs the student checkpoint: pass "
                "++evaluate.eval_ckpt=<checkpoint-N dir>"
            )
        # layer_mix_init=tent: this checkpoint predates the mix and has no `layer_mixer.*`.
        # The probe overwrites the logits per variant, so nothing trained is being ignored.
        driver = hyu.instantiate(
            cfg.model, _convert_="partial", layer_mix_init="tent"
        ).to(dev).eval()
        target_vlm = TrainableReasoningVLA.from_alpamayo_checkpoint(
            checkpoint_path=TEACHER_CKPT, vlm_name_or_path=COSMOS,
        ).to(dev).eval()
        for q in target_vlm.parameters():
            q.requires_grad_(False)
    for q in driver.parameters():
        q.requires_grad_(False)

    n_tgt = len(driver.expert.layers)
    g_in, g_out = n_src // n_blocks, n_tgt // n_blocks
    if source == "teacher":
        keep = source_layers(subset, n_src, n_tgt)
    else:
        # The student HAS 28 layers; there is nothing to pick.
        keep = list(range(n_src))
        n_have = len(driver.vlm.model.language_model.layers)
        if n_have != n_src:
            raise ValueError(f"student VLM has {n_have} layers, probe.n_src={n_src}")
    print(f"[oracle] source={source}; expert {n_tgt} layers; reconstructing from {n_src} "
          f"in {n_blocks} blocks of {g_in} -> {g_out}", flush=True)
    print(f"[oracle] source layers: {keep}", flush=True)

    ds = hyu.instantiate(cfg.data.val_dataset, _convert_="partial", model_config=driver.config)
    coll = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=driver.config)
    metric = hyu.instantiate(cfg.evaluate.metric_runner.metrics[-1], _convert_="partial")
    loader = DataLoader(ds, batch_size=1, collate_fn=coll, num_workers=4, shuffle=False)

    def _prefill(model, batch):
        """The prompt cache this clip produces, from whichever tower is passed."""
        td = dict(batch["tokenized_data"])
        ids = driver.fuse_traj_tokens(
            td.pop("input_ids"),
            {"ego_history_xyz": batch["ego_history_xyz"],
             "ego_history_rot": batch["ego_history_rot"]},
        )
        with torch.no_grad():
            out = model.vlm(input_ids=ids, use_cache=True, logits_to_keep=1, **td)
        return out.past_key_values, model.vlm.model.rope_deltas

    # ---- phase A: accumulate the normal equations --------------------------
    fits = {w: [BlockFit(g_in, g_out) for _ in range(n_blocks)] for w in ("k", "v")}
    seen = 0
    for batch in loader:
        if seen >= n_fit:
            break
        gpu = send_to_device({k: v for k, v in batch.items() if k != "clip_id"}, dev)
        src_cache, src_rd = _prefill(driver, gpu)
        if target_vlm is None:
            tgt_cache = src_cache
        else:
            tgt_cache, tgt_rd = _prefill(target_vlm, gpu)
            # ⚠️ Both towers must have read the SAME prompt, or every position-wise target is
            # misaligned and the fit is meaningless while still converging happily. This is
            # the check kd_model.forward makes before L_block for the same reason.
            if src_cache.get_seq_length() != tgt_cache.get_seq_length():
                raise RuntimeError(
                    f"student prefill is {src_cache.get_seq_length()} positions, teacher "
                    f"{tgt_cache.get_seq_length()}; their caches are not comparable"
                )
            if not torch.equal(src_rd.to(tgt_rd.device), tgt_rd):
                raise RuntimeError("student and teacher rope_deltas differ for one prompt")
        for which, sel in (("k", "keys"), ("v", "values")):
            src_all = torch.stack([getattr(l, sel) for l in src_cache.layers], dim=1)
            tgt_all = torch.stack([getattr(l, sel) for l in tgt_cache.layers], dim=1)
            for b in range(n_blocks):
                src = torch.stack(
                    [src_all[:, keep[b * g_in + i]] for i in range(g_in)], dim=0
                ).flatten(1)
                tgt = tgt_all[:, b * g_out : (b + 1) * g_out].transpose(0, 1).flatten(1)
                fits[which][b].add(src.cpu(), tgt.cpu())
        del src_cache, tgt_cache
        seen += 1
        if seen % 8 == 0:
            print(f"[oracle] fit {seen}/{n_fit} clips", flush=True)

    solved, resid = {}, {}
    for which in ("k", "v"):
        ws, rs = zip(*(f.solve() for f in fits[which]))
        solved[which] = torch.stack(ws)                        # [n_blocks, g_in, g_out]
        resid[which] = torch.stack(rs)                         # [n_blocks, g_out]
        print(f"[oracle] P_{which.upper()} reconstruction rel-L2 per slot: "
              + " ".join(f"{float(x):.3f}" for x in resid[which].flatten()), flush=True)

    tent = tent_weights(g_in, g_out).expand(n_blocks, -1, -1).clone()
    gather = gather_weights(g_in, g_out).expand(n_blocks, -1, -1).clone()

    def _logits(wk, wv):
        """Fitted weights as logits. ``softmax(log a) == a`` on the simplex, so this is exact."""
        return torch.log(wk.clamp_min(1e-12)), torch.log(wv.clamp_min(1e-12))

    variants = {"fit": _logits(solved["k"], solved["v"]),
                "tent": _logits(tent, tent),
                "gather": _logits(gather, gather)}

    # ---- phase B: drive the expert -----------------------------------------
    # ⚠️ EVAL CLIPS ARE DISJOINT FROM FIT CLIPS. The loader is not shuffled, so without this
    # the first n_fit clips would be scored by a P fitted on them. With 7 parameters and ~50M
    # samples the in-sample bias is tiny, but "tiny" is not a number this run can report.
    ref = "full" if source == "teacher" else "teacher"
    per: dict[str, list[float]] = {ref: []}
    per.update({name: [] for name in variants})
    mixer = LayerMixer(n_student=n_src, n_expert=n_tgt, n_blocks=n_blocks).to(dev)
    scored = 0
    for bi, batch in enumerate(loader):
        if bi < n_fit:
            continue
        if scored >= n_clips:
            break
        gpu = send_to_device({k: v for k, v in batch.items() if k != "clip_id"}, dev)

        run = {ref: None}                    # the ceiling, unmixed; None => no mixer
        run.update(variants)
        for name, logits in run.items():
            if logits is not None:
                with torch.no_grad():
                    mixer.logit_k.copy_(logits[0].to(dev))
                    mixer.logit_v.copy_(logits[1].to(dev))
            if source == "teacher":
                driver.layer_mixer = None
                hook = make_hook(keep, None if name == "full" else mixer)
            else:
                # from_stitch's own layer_mix path applies mix_cache inside
                # sample_trajectories_prefill_only -- the PRODUCTION path, exercised as-is.
                driver.layer_mixer = None if logits is None else mixer
                hook = None
                if logits is None:
                    tgt_cache, _ = _prefill(target_vlm, gpu)

                    def hook(c, _i, _t, _tc=tgt_cache):
                        if c.get_seq_length() != _tc.get_seq_length():
                            raise RuntimeError(
                                f"student prefill {c.get_seq_length()} != teacher "
                                f"{_tc.get_seq_length()}; prefix_len is read off the "
                                "student's cache, so substituting would misalign it")
                        return _tc
            # ⚠️ Per BATCH, before every variant: identical initial diffusion noise is what
            # makes these a paired comparison rather than independent draws.
            torch.manual_seed(SEED + bi)
            torch.cuda.manual_seed_all(SEED + bi)
            with torch.no_grad():
                pred_xyz, pred_rot = driver.sample_trajectories_prefill_only(
                    data=gpu, num_traj_samples=n_samples, num_traj_sets=1, cache_hook=hook,
                )
            m = metric.evaluate(driver, gpu, {"pred_xyz": pred_xyz, "pred_rot": pred_rot})
            per[name].append(float(m["min_ade"][0]))
        scored += 1
        if scored % 25 == 0:
            print(f"[oracle] {scored}/{n_clips}  " + "  ".join(
                f"{n} {np.nanmean(v):.4f}" for n, v in per.items()), flush=True)
    driver.layer_mixer = None

    # ---- report -------------------------------------------------------------
    print(f"\n[oracle] n={scored} clips (disjoint from the {n_fit} used for the fit), "
          f"paired on clip AND on diffusion noise", flush=True)
    print(f"[oracle] {'variant':10s}{'min_ade':>10s}{'vs ' + ref:>12s}{'SEM':>9s}{'z':>8s}",
          flush=True)
    stats = {}
    for name, vals in per.items():
        mean = float(np.nanmean(vals))
        if name == ref:
            print(f"[oracle] {name:10s}{mean:>10.4f}{'--':>12s}{'':>9s}{'':>8s}", flush=True)
            stats[name] = {"min_ade": mean}
            continue
        d, sem, z = paired_stats(vals, per[ref])
        stats[name] = {"min_ade": mean, "delta": d, "sem": sem, "z": z}
        print(f"[oracle] {name:10s}{mean:>10.4f}{d:>+12.4f}{sem:>9.4f}{z:>+8.2f}", flush=True)

    d_mg, sem_mg, z_mg = paired_stats(per["gather"], per["fit"])
    print(f"\n[oracle] mixing vs one-hot gathering: {d_mg:+.4f} +- {sem_mg:.4f} (z {z_mg:+.2f})",
          flush=True)
    if baseline is not None:
        print(f"[oracle] against the supplied 28-layer pruned-expert baseline "
              f"{float(baseline):.4f}: fit is {stats['fit']['min_ade'] - float(baseline):+.4f}",
              flush=True)
    else:
        print("[oracle] ⚠️ no probe.baseline_min_ade given. For source=student the number that "
              "matters is THIS student through the 28-layer PRUNED expert on these clips; "
              "pass it so the verdict is same-harness.", flush=True)

    travel = max(logit_travel(solved[w], tent) for w in ("k", "v"))
    base_lr = float(probe.get("base_lr", 1e-5))
    steps = int(probe.get("train_steps", 1563 * 5))
    print(f"\n[oracle] P must travel {travel:.3f} in logit space from the tent init to the "
          f"fitted optimum; at lr={base_lr:g} over {steps} steps Adam moves a logit "
          f"~{base_lr * steps:.3f}, so lr_multiplier ~{travel / max(base_lr * steps, 1e-30):.1f} "
          f"is what lets P arrive from the tent. Seeding P AT the fit removes that need.",
          flush=True)

    with open(out_path, "w") as fh:
        json.dump({
            "config": {"source": source, "n_fit": n_fit, "n_clips": scored, "subset": subset,
                       "keep": keep, "n_blocks": n_blocks, "g_in": g_in, "g_out": g_out,
                       "seed": SEED, "eval_ckpt": cfg.model.get("checkpoint_path")},
            "summary": stats,
            "per_clip_min_ade": per,
            "recon_rel_l2": {w: resid[w].tolist() for w in ("k", "v")},
            "logit_travel": travel,
            "P": {w: solved[w].tolist() for w in ("k", "v")},
        }, fh, indent=1)
    print(f"[oracle] wrote {out_path}", flush=True)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    main()
