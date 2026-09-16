# Ablation: reshaping `L_block`, the KD tower's only training loss

The 4B student tower every downstream arm loads is trained by **`L_block` and nothing else**.
This file is the ablation over that objective. It closes the line: six interventions, none of
which improved trajectory quality, against a bottleneck that `CENTER_GAP.md` shows is not in
this objective's reach.

## 0. ⚠️ Comparability

All rows: **stitched** eval (student VLM → the teacher's *untuned* 36-layer expert), n=1000,
`nav_lcdrive_val_mysubset_1k`, cameras `[1,3]`, event-anchored `t0`, `"route"` in
`components_order`, K=6, no `inference_step` override so the budget is the model default and
is **identical across every row**. Paired on `clip_id`. Distances are **XY**, recomputed from
the `.npz` archives.

⚠️ **Do NOT compare these numbers to `CENTER_GAP.md`.** That file scores the **EoS** model —
the expert *fine-tuned on the student's own cache* — at NFE=2. These score the teacher's
**untuned** expert at the default budget. The gap between e.g. centre 1.6513 here and 1.5352
there is the EoS stage, not a regression.

⚠️ **The parent has no epoch-1/2/3 checkpoint.** Job 20724 ran `save_steps=500` with
`save_total_limit=8`, which prunes to steps 10500–13752. Only `checkpoint-13752` (**epoch 4**)
survives, so no arm can be paired against the parent at a matched epoch. The new arms use
`save_steps=0.125`, which lands saves on the epoch boundaries; that change exists because of
this.

## 1. The objective under test

Resolved config of the tower every 4B arm loads
(`output_kd_4b_nav4bspan2camallfc_m1-9-18-36_framecache1080p_lcdrive/checkpoint-13752`, job 20724):

```
ce_weight 0.0   kd_weight 0.0   kv_weight 0.0     <- L_block is the ONLY loss
block_weight 1.0        block_timestep beta
block_norm 'teacher'    block_span 1 (scheduled 1->9->18->36)
block_span_mix 0        block_freerun_weight 0.0
```

`L_block` runs each frozen expert block twice on the **same teacher-forced** action state
`h^T_l` — once on the teacher's VLM cache, once on the student's — and matches the outputs
with normalised MSE + cosine distance. Teacher-forcing `h^T_l` is what makes each layer
independently well-posed, and is also what removes every gradient path to compounding.

## 2. The ablation

| # | intervention | what it changes | result |
|---|---|---|---|
| 1 | span curriculum m=1→7→14→28 (§3) | chain m blocks before scoring | **null** |
| 2 | `ladder` (0.25 floor) (§4) | cross-layer reweighting | **+0.149 min_ade** |
| 3 | `ladder_add` (1.0 floor) (§4) | cross-layer reweighting | **+0.179 min_ade** |
| 4 | `mix` m=1 + m=7 (§4) | teacher-forced + span together | **+0.164 min_ade** |
| 5 | `block_norm=cache` (job 20828) | cache-attributable normaliser | **no accuracy gain** (§3 below) |
| 6 | `block_freerun_weight=2e-4` (job 20831) | adds the free-running chain | **decisively worse** (§3) |

§1–§4 references are to `COMPARE_EVAL.md`. Rows 5–6 are new and are detailed below.

## 3. Rows 5 and 6 — `block_norm=cache` and `block_freerun`

Both arms are single-variable against job 20724: same 4B student, cameras `[1,3]`, nav route,
frame cache, span schedule 1→9→18→36, and **identical BS/ACCUM/LR/warmup/epochs**. Each writes
its own `output_dir`, so neither can overwrite the parent or each other.

| arm | E[1 draw] | medoid | centre | minADE@6 | spread |
|---|---|---|---|---|---|
| parent @ep4 | 2.0653 | 1.6729 | 1.6513 | 0.9016 | 1.2610 |
| cachenorm @ep1 | 2.3406 | 1.8631 | 1.8592 | 0.9816 | 1.4695 |
| cachenorm @ep2 | 2.1607 | 1.7428 | 1.7461 | 0.9270 | 1.3107 |
| freerun @ep1 | 2.6006 | 2.1162 | 2.1311 | 1.1609 | 1.5329 |
| freerun @ep2 | 2.3186 | 1.8666 | 1.8718 | 1.0015 | 1.4000 |

### 3a. `block_freerun` is a clear negative — epoch-matched

| freerun − cachenorm | E[1 draw] | centre | minADE@6 |
|---|---|---|---|
| @ep1 | +0.2600 (t=+10.8) | +0.2719 (t=+9.2) | +0.1793 (t=+8.7) |
| @ep2 | +0.1579 (t=+8.7) | +0.1257 (t=+5.8) | +0.0744 (t=+4.7) |

Worse on every axis at both epochs. It improves *faster* per epoch (centre −0.2593 vs
cachenorm's −0.1131) only because it starts far worse — catching up, not overtaking. On the
observed halving it would finish epoch 4 still behind.

⚠️ **This is `L_field`'s failure mode again (§4a), and it survived damping.** `fr_term` matches
`captured[n_layers-1]["y_out"]` — **only the final layer's output** after chaining all 36
blocks — so it is an ENDPOINT-ONLY objective, structurally what §4a measured at 7.8× worse
than `L_block` and worse than the CE-only and KD-only floors. At the shipped
`block_freerun_weight=1.0` it was **~1,600×** `block_loss` (freerun ~89–190 vs block ~0.083,
i.e. 99.94% of the total), and a 20-step smoke showed `freerun_loss` flat
(89,143,106,130,114,143,190,130) while `block_loss` ROSE 0.0789 → 0.0893 with `grad_norm`
swinging 2.5k–9.8k. The run above uses **2e-4**, which keeps L_block at ~75% — and it is still
decisively negative.

⚠️ The docstring's "L_block keeps ~4% of the gradient at weight 1.0" is **not true on this
stack**; it is ~0.07%. Anyone reusing that term must re-derive the weight from the live loss
magnitudes, not trust the comment.

⚠️ **`BLOCK_FR_FP32=1` IS BROKEN — do not set it.** `kd_model.py` casts the activations to fp32
while DISABLING autocast, but the frozen expert's weights stay bf16, so the first `q_proj`
raises `expected mat1 and mat2 to have the same dtype, float != BFloat16` at
`modeling_qwen3_vl.py:428`, on every rank, at step 0 (job 20829 died on it four times through
the restart loop). The flag appears in no other script or doc — the fp32 opt-in has never run
end-to-end here. Casting the whole expert to fp32 is NOT the fix: `L_block` runs OUTSIDE
autocast against bf16 weights, so that would silently change the parent-comparable term too; a
correct fix needs a separate fp32 copy of the expert for that chain only.
⚠️ And the failure it was guarding against did not reproduce: **bf16 is fine** — 20 clean steps,
no nan, no `grad_norm: 2.0` sentinel.

### 3b. `block_norm=cache` — a better diagnostic, not a better model

The measured defect it fixes is real. Per `block_losses.py`, a **zero** cache scores 0.0104, so
**99% of `||y_teacher||²` is cache-INDEPENDENT residual stream** — a component the student
cannot get wrong. The informative band is 0…0.0104 and a trained student sits at 0.0012: 88%
already captured, the whole remaining gap in the last 12%. That is why *"500 steps moved this
loss within noise while `ade` moved −13% at z=−3.44."* `block_norm=cache` divides by
`||y_teacher − y_zero||²` instead, putting zero-cache at exactly 1.0.

It delivered exactly that and no more. Against the parent — ⚠️ **ep2 vs ep4, not epoch-matched**:

| cachenorm@ep2 − parent@ep4 | Δ | t |
|---|---|---|
| E[1 draw] | +0.0955 | +4.4 |
| centre | +0.0947 | +3.8 |
| minADE@6 | +0.0254 | +1.4 **ns** |

Within-arm ep1→ep2: E[1 draw] −0.1798 (t=−7.2), centre −0.1131 (t=−3.8), minADE@6 −0.0545
(t=−2.8). Real movement, decaying; extrapolated to ep4 it lands at roughly parity with the
parent, not past it. Consistent with §4's three negative reweightings — the docstring's own
caveat is that under Adam this changes **cross-layer weighting and curve readability, not
within-layer gradient direction**.

⚠️ **It also silently demotes the cosine term.** `block_output_loss` divides the MSE by the
normaliser but, by design, never divides the cosine. Measured live, same step range:

| arm | `block_norm` | mse | cosine | cosine share of `L_block` |
|---|---|---|---|---|
| freerun | `teacher` | 0.00228 | 0.00113 | **33%** |
| cachenorm | `cache` | 0.10221 | 0.00056 | **0.5%** |

So `block_norm=cache` is not a pure normaliser change — it is also "drop the direction term".
Any future use should rescale the cosine by the same factor, or the two effects stay confounded.

⚠️ Early in warmup the cache-attributable scale makes `block_loss_early` read ~67× `deep`
(early 67.1 while mid/deep sit at ~1.0), because layers 0–9 have near-zero cache-attributable
denominators. This is **transient**: by ~2.5 h it settles to early 0.031 / mid 0.147 / deep
0.114 — `early` smallest, and the mid > deep > early ordering reproduces §4's measured
difficulty profile. Do not judge the arm from the first few hundred steps.

## 4. Why the line is closed

`CENTER_GAP.md` measures the actual bottleneck: the student's distribution is **not**
under-dispersed (spread 0.578 teacher vs 0.480 student at NFE=2) — it is **centred 0.1965 m
wrong** (t=+5.4), and that centre is invariant to step budget **and** to a second epoch of
frozen-VLM EoS (+0.0005). The centre is set by the VLM cache, and `COMPARE_EVAL.md` §5 puts
100% of the gap there.

`L_block` scores each layer **independently** against the teacher. §5 proves the cache is not
separable that way: substituting the teacher's `region_vision` — 93% of the cache — while
leaving text/traj as the student's scores **+2.68 (harmful)**, worse than substituting nothing.
`k_only` is +0.84 and `v_only` +0.38, both harmful. K and V, and the regions, must be mutually
**consistent**; a per-layer objective can drive every layer's own error down while degrading
the joint configuration the expert actually reads. §4's per-band diagnostic is the fingerprint:
the deep band improved 15–19% while `min_ade` got **worse** by 0.16–0.18.

Six interventions, six non-improvements, against a bottleneck this objective's decomposition
cannot express. **Work on the cache directly instead** — the partial-VLM-cotrain arm closed 35%
of the centre gap in a single epoch (`CENTER_GAP.md`).

## 5. What was NOT tested

* **Multi-`t` probes per prefill.** `COMPARE_EVAL.md` §9's top open item, still unimplemented
  (`_sample_block_t` returns one `t` per batch row). The cache does not depend on `t` at all —
  the VLM prefill is computed once — so `t` selects only *which action state probes the cache*,
  and extra probes per prefill are nearly free. This is the one L_block change that reallocates
  across the **noise schedule** rather than across **layers**, so the six negatives above do not
  speak to it.
* **`block_timestep=uniform`.** One flag, no code: the cheap A/B for the above. `beta` is
  verbatim the teacher's own schedule (`Beta(1.5,1)` then `0.999 - t*0.999`, checked against
  `flow_matching.py:145-149`), and with `noisy_x = t*x + (1-t)*noise` it concentrates probes at
  **t≈0, pure noise** — action states carrying almost no trajectory content — while *centring*
  is a low-noise-end property. Copying the teacher's TRAINING schedule as a PROBE schedule is a
  category error: the teacher needed mass at high noise because that is where its velocity field
  is hardest to learn.
* **KL on attention maps.** Attractive for one reason: a normalised distribution needs no
  normaliser, which is the problem §3b keeps fumbling. But `A = softmax(QK^T)` contains no V, so
  it supervises **K only** — and §5's `k_only` +0.84 says K and V must move together. If tried,
  add it as a small auxiliary alongside `L_block`, restricted to the text+traj regions (vision
  is 93% of the cache and `region_vision` is the harmful one), never as a replacement.

## 6. Reproducing

```bash
cd recipes/alpamayo1_5_distill
# the two arms (4 GPUs each). Both are single-variable against ARM=nav4bspan2camallfc.
APPROVED=1 ARM=nav4bspan2camallfc_cachenorm sbatch --nodelist=amhrisvh200b slurm_train_kd.sh
APPROVED=1 ARM=nav4bspan2camallfc_freerun   sbatch --nodelist=amhrisvh200b slurm_train_kd.sh
# ⚠️ SMOKE=1 FIRST for anything touching block_freerun -- it is what caught the 1,600x weight.

# stitched eval. NAV2CAM auto-fires on "framecache" in the ARM -> the 2cam+nav config.
# ⚠️ CKPT is REQUIRED while a run is still training: the newest-checkpoint default would
# silently pick up a later epoch mid-sweep.
for A in cachenorm freerun; do for C in 3438 6876; do
  ARM=nav4bspan2camallfc_${A}_m1-9-18-36_framecache1080p CKPT=checkpoint-$C \
    sbatch --nodelist=amhrisvh100c --gpus=1 --cpus-per-task=12 --mem=96G ./slurm_eval_stitched.sh
done; done
```

⚠️ `--partition=gpu` and an explicit `--nodelist`: there is no `debug` partition on this
cluster and some in-script `#SBATCH` defaults are stale.

Archives, all per-clip and joinable on `clip_id`:
`training/stitch_4b_nav4bspan2camallfc{,_cachenorm,_freerun}_m1-9-18-36_framecache1080p_checkpoint-*.{json,npz}`

The decomposition is offline from the `.npz` — no GPU; see `CENTER_GAP.md` §7 for the snippet.
