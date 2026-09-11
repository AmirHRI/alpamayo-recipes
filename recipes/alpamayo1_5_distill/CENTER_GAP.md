# The center gap — where the 4B student actually loses to the teacher

The corrected-EoS / CD comparison twice pointed at the wrong conclusion because single-draw
`ade` conflates two independent quantities: **where** the predicted distribution sits and **how
wide** it is. Decomposing them says the student's dispersion is already right and its location
is not. This file records the decomposition, the measurement that settles it, and the two
retractions it forces.

## 0. ⚠️ Comparability

Every row: n=1000, `nav_lcdrive_val_mysubset_1k`, cameras `[1,3]`, event-anchored `t0`, `"route"`
in `components_order`, K=6 samples, same seed. Teacher and student archives were checked to
carry the **same 1000 clip_ids and bit-identical `gt_xyz`** — the `COMPARE_EVAL.md` §0 rules are
satisfied, not assumed.

⚠️ **All distances here are XY**, recomputed from the `.npz` archives, so `E[1 draw]` is
comparable to `min_ade` within this file. The eval harness's JSON `ade` field is **3D** and its
`min_ade` field is XY — that is why the same run reads 1.6283 (`ade`, 3D) and 1.6178
(`E[1 draw]`, XY). Never put those two in one column.

## 1. The four numbers, separated

`center` = ADE of the barycentre of the 6 samples (dispersion removed). `medoid` = the sample
minimising mean distance to the other five — a **realizable** single output, unlike the mean.
`spread` = mean distance from a sample to the barycentre.

| arm | E[1 draw] | medoid | **center** | minADE@6 | spread |
|---|---|---|---|---|---|
| Teacher @1 | 1.6756 | 1.5852 | 1.5856 | 1.2817 | 0.4180 |
| **Teacher @2** | 1.4788 | 1.3476 | **1.3387** | 0.9321 | 0.5778 |
| Teacher @10 | 1.7410 | 1.3531 | 1.3887 | 0.7442 | 1.1099 |
| EoS ep2 @1 | 1.8003 | 1.7537 | 1.7406 | 1.5075 | 0.3197 |
| **EoS ep2 @2** | 1.6178 | 1.5417 | **1.5352** | 1.0930 | 0.4801 |
| EoS ep2 @10 | 1.8428 | 1.5441 | 1.5360 | 0.8340 | 1.0232 |
| CD ep1 @2 | 1.8308 | 1.6223 | 1.6046 | 0.9920 | 0.8569 |
| CD ep2 @1 | 1.8259 | 1.6686 | 1.6599 | 1.0955 | 0.7320 |
| CD ep2 @2 | 1.8328 | 1.6197 | 1.6045 | 0.9891 | 0.8609 |
| CD ep2 @10 | 1.9317 | 1.6872 | 1.6751 | 1.0038 | 0.9525 |

**Center gap at the operating point: student − teacher = +0.1965, t = +5.4, n=1000 paired.**

## 2. It is location, not dispersion

At 2 steps the teacher's spread is **0.5778** and the student's is **0.4801** — the same regime.
The student is *not* under-dispersed. At essentially matched width its distribution simply sits
0.197 m further from the truth.

The minADE@6 gap at 2 steps is 0.1609, **smaller** than the center gap, because best-of-6
partially forgives a centering error. The center gap is therefore the larger and more
fundamental quantity: close it and both columns close.

⚠️ **Both centers saturate by step 2 and neither moves after.** Teacher 1.5856 → **1.3387** →
1.3847 → 1.3887 at k = 1/2/9/10; student 1.5352 @2 → 1.5360 @10. The center is a property of the
**conditioning**, not of the sampler. No step budget, no consistency objective and no damper
setting on either model touches it.

## 3. Two retractions

⚠️ **"The student already beats the teacher on `ade`" — WRONG, retracted.** It came from
comparing the student at its best step count (EoS @2, 1.6178) against the teacher at k=1 and
k=10 only. `COMPARE_EVAL.md` §7a states the `ade`-dip at k=2 is a property of the *sampler*, so
the teacher dips there too, and it does: 1.6362 → **1.4459** → … → 1.6976 (§7 table). The teacher
leads at every matched step count. Measuring one model across NFE and the other at two endpoints
is the same error class as comparing across camera counts.

⚠️ **"The teacher's center keeps improving with steps while the student's saturates" — WRONG,
retracted.** Inferred from k1/k9/k10 with k2 missing. The teacher's center is *best* at k=2 and
flat thereafter, exactly like the student's. The `teachernav_k2` archive (§5) is what settled it.

## 4. What this says about CD

CD moves the center the **wrong way**: 1.5352 → 1.6045 (+0.069, t=+4.3) while inflating spread
0.480 → 0.861 (+79%). Splitting its +0.215 single-draw regression: **+0.146 is variance
inflation, +0.069 is genuine center degradation.**

Contracting each arm's samples toward their own barycentre (α scales spread and nothing else)
traces each checkpoint's frontier. CD @2 sits **inside** the corrected EoS's frontier at both
ends:

| spread | arm | E[1 draw] | minADE@6 |
|---|---|---|---|
| 0.51 | EoS @10 (α=0.50) | **1.621** | **1.040** |
| 0.51 | CD @2 (α≈0.58) | 1.69 | 1.15 |
| 0.86 | EoS @10 (α≈0.84) | **1.76** | **0.88** |
| 0.86 | CD @2 (native) | 1.833 | 0.989 |

Everything CD bought was already available by raising the step count on the corrected EoS, at a
better exchange rate. CD's surviving win is **NFE=1 only** — minADE 1.5075 → 1.0955 at equal
`ade`, which is what `student4B_CD.md` always claimed.

⚠️ The `x0_gt_weight: 0.5` damper was swept (`COMPARE_EVAL.md` §7e) against the **pre-maskfix**
EoS, whose starting dispersion was much wider. That sweep does not transfer; the damper is now
far too weak. Re-sweeping recovers the +0.146 variance component, not the +0.069 center
component.

## 5. Free win: report the medoid

`medoid` needs no retraining and no extra prefill — the 6 draws already share it. EoS @2 medoid
**1.5417** vs 1.6178 for a random draw; EoS @10 medoid **1.5441** vs 1.8428. Unlike the
barycentre it is an actual sampled trajectory, so it is kinematically valid.

**Recommendation: report `medoid` + `minADE@6` and drop single-draw `ade` as a decision metric.**
It is what steered the recipe toward tighter sampling twice.

## 6. Why the center is stuck — the KD objective

The center is set by the VLM cache (`COMPARE_EVAL.md` §5: 100% of the gap is the cache, 95% in
layers 19–27). The cache is trained by `L_block` **and nothing else**. Resolved config of the run
that produced the tower every 4B arm loads
(`output_kd_4b_nav4bspan2camallfc_m1-9-18-36_framecache1080p_lcdrive/checkpoint-13752`, job 20724):

```
ce_weight 0.0   kd_weight 0.0   kv_weight 0.0     <- L_block is the only loss
block_weight 1.0        block_timestep beta
block_norm 'teacher'                              <- §6a
block_span 1 (scheduled 1->9->18->36)   block_span_mix 0
block_freerun_weight 0.0 (default, never set)     <- §6b
```

### 6a. `block_norm='teacher'` makes the loss ~99% inert

Measured in `block_losses.py`: a **zero** cache scores 0.0104, so 99% of `||y_teacher||²` is
cache-independent residual stream — a component the student cannot get wrong. The informative
band is 0…0.0104 and a trained student sits at 0.0012: 88% already captured, the whole remaining
gap in the last 12%. The docstring ties this to the symptom directly — *"500 steps moved this
loss within noise while `ade` moved −13% at z=−3.44."*

`block_norm='cache'` normalises by `||y_teacher − y_zero||²` instead, putting zero-cache at
exactly 1.0 and the model at ~0.115. **Never run.**

⚠️ Same docstring: under Adam this changes **cross-layer weighting and curve readability, not
within-layer gradient direction** — and §4 found three hand-designed cross-layer reweightings all
negative. Expect a readable diagnostic; treat the accuracy gain as unproven.

### 6b. Teacher forcing hides the error that actually occurs

`freerun_probe.py` (n=32): the teacher-forced error `L_block` trains on is flat at **~4.1e-4**
across all 36 layers; the free-running error grows to **1.3e-2 — 32×, and 72× at the deepest
layers**. `block_freerun_weight` was written for exactly this and is **0.0**.

The span schedule is not a substitute: §3 measured the objective as **m-invariant** (loss *falls*
0.00170 → 0.00093 as spans deepen, peak memory flat at 65.8 GiB from m=1 to m=28, and m=28 scores
better than m=14 on identical weights), because the expert's layer map is contractive.

### 6c. `t` is a probe distribution, and the teacher's schedule is the wrong one

`_sample_block_t` is verbatim correct against `flow_matching.py:145-149` — `Beta(1.5,1)` then
`0.999 - t*0.999`, and with `noisy_x = t*x + (1-t)*noise`, **t=0 is noise**. Mean t ≈ 0.40, mass
at the noise end. No bug.

But **the cache does not depend on `t` at all** — the VLM prefill is computed once and `t` enters
only through `action_embeds`. So `t` does not set a training distribution; it selects *which
action state probes the cache*. Concentrating probes at t≈0 tests the cache almost entirely
against near-pure-noise queries that carry no trajectory content, and under-samples the low-noise
end where the trajectory is positioned — which is what `center` measures.

Copying the teacher's schedule is a category error: the teacher needed mass at high noise because
that is where its *velocity field* is hardest to learn. The cache needs probes wherever its
contribution to the block output is most **discriminative**, which is a different distribution and
has never been measured. `block_timestep=uniform` is the cheap A/B.

### 6d. One `t` per sample; per-layer scoring assumes separability that §5 disproves

`COMPARE_EVAL.md` §9 already lists multi-`t` as the top open deficiency. Since the cache is shared
across all `t`, extra probes per prefill are nearly free — the same argument §9 makes for CD's
multi-rung item. **Not currently implemented** (`_sample_block_t` returns one `t` per batch row).

And §5 proves the cache is not separable: `region_vision` substitution scores **+2.68 (harmful)**
— swapping the teacher's 93% majority while leaving text/traj as the student's is worse than
swapping nothing. `L_block` scores each layer independently, so it can drive every layer's own
error down while degrading the joint consistency the expert reads. §4's fingerprint: the deep
band improved 15–19% while `min_ade` got **worse** by 0.16–0.18.

## 7. Reproducing

```bash
cd recipes/alpamayo1_5_distill
# the missing teacher row (job 20827). ⚠️ --partition=gpu: there is no debug partition here,
# and the in-script #SBATCH default is stale.
M=/temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests
TAG=teachernav STEPS='[2]' CAMERAS='[1,3]' \
  EXTRA_ARGS="++data.val_dataset._target_=alpamayo.data.pai_nav.PAIDatasetWithNav \
    ++data.val_dataset.annotations_path=$M/nav_lcdrive_val_mysubset_1k.json \
    ++data.val_dataset.vla_preprocess_args.components_order=[image,traj_history,route,prompt,traj_future]" \
  sbatch --partition=gpu --nodelist=amhrisvh200b ./slurm_eval_step_sweep.sh
# self-check: min_ade 0.9320 / ade 1.4470 against COMPARE_EVAL.md §7's 0.9318 / 1.4459.
```

The decomposition is offline from the archives — no GPU:

```python
p = np.load(f)['pred_xyz'][:, 0][..., :2]; g = np.load(f)['gt_xyz'][..., :2]
e      = np.linalg.norm(p - g[:, None], axis=-1).mean(-1)      # [N,6] per-sample ADE
center = np.linalg.norm(p.mean(1) - g, axis=-1).mean(-1)       # dispersion removed
spread = np.linalg.norm(p - p.mean(1)[:, None], axis=-1).mean((1, 2))
D      = np.linalg.norm(p[:, :, None] - p[:, None], axis=-1).mean(-1)
medoid = e[np.arange(len(e)), D.sum(-1).argmin(-1)]            # realizable single output
```

Archives: `training/stepsweep/teachernav_k2.npz`, `training/teacher_eval/teachernav_k{1,9,10}.npz`,
`training/{eos_4b_maskfix_20812,cd4b_maskfix_fp32_20820}_eos_checkpoint-{3438,6876}{,_nfe1,_nfe2}.npz`.
All per-clip and joinable on `clip_id`.

## 8. Open

* ~~**`block_norm=cache`**~~ and ~~**`block_freerun_weight > 0`**~~ — both RUN and both closed.
  See `KD_LOSS_ABLATION.md`: cachenorm buys a readable curve and no accuracy gain; freerun is
  decisively worse epoch-matched (centre +0.1257, t=+5.8 at ep2) even damped to 2e-4. That
  makes SIX non-improvements over `L_block`, and the line is closed.
* **`block_timestep=uniform`** — still the cheap A/B against the probe-distribution argument (§6c).
* **Multi-`t` probes per prefill** — nearly free, needs code (§6d). The one L_block change that
  reallocates across the NOISE SCHEDULE rather than across layers, so the six negatives do not
  speak to it.
* ~~**`cotrain_vlm: true` in EoS**~~ — RUN, and it is the first thing to move the centre.
  Full cotrain does not fit (~81 GB under DDP on 80 GB cards), so job 20835 trains VLM text
  layers **27-35** only (~0.91 B of VLM + 2.28 B expert = 3.19 B). Epoch 1, paired, NFE=2:

  | | teacher | baseline @ep1 | cotrain @ep1 | Δ vs baseline |
  |---|---|---|---|---|
  | centre | 1.3387 | 1.5371 | **1.4676** | **−0.0695 (t=−4.0)** |
  | E[1 draw] | 1.4788 | 1.6197 | **1.5377** | −0.0821 (t=−5.0) |
  | minADE@6 | 0.9321 | 1.0950 | **1.0487** | −0.0463 (t=−3.0) |

  **35% of the centre gap in one epoch.** At NFE=10 the same checkpoint gives centre −0.0348
  (t=−2.0), so the effect is larger at the operating point.

  ⚠️ **This is meaningful precisely because the quantity had never moved.** The frozen-VLM
  baseline's centre is 1.5355 @ep1/NFE=10, 1.5360 @ep2/NFE=10, 1.5371 @ep1/NFE=2, 1.5352
  @ep2/NFE=2 — flat across BOTH epochs and budgets. A second epoch of frozen EoS buys +0.0005.

  ⚠️ **The centre's budget-response changed character**, which the magnitudes alone hide:
  teacher 1.3387@2 vs 1.3887@10 (better at 2 by 0.050); frozen baseline 1.5371@2 vs 1.5355@10
  (FLAT); cotrain 1.4676@2 vs 1.5007@10 (better at 2 by 0.033). The student is acquiring the
  teacher's asymmetry — a structural change in the cache, not just a smaller number.

  ⚠️ **Verified loaded, not assumed.** The eval log reads `Loaded 714 VLM tensors from
  .../output_eos_cotrain_deep27_.../checkpoint-3438 (missing=416, unexpected=0)`, and against
  the KD tower both runs started from: baseline is bit-identical at every probed layer, while
  cotrain differs at 27 (4.01e-3) and 30 (4.14e-3) and is identical at 5/20/26.
  ⚠️ Layer 35 moves ONLY in `k_proj` (5.22e-3) and `v_proj` (3.68e-3); its q/o/mlp are exactly
  0. That is correct, not a defect: with `cotrain_vlm_ce=False` layer 35's hidden state feeds
  only the discarded `lm_head`, so its sole route to the loss is the K/V it puts in the cache.
  It is also why `ddp_find_unused_parameters: true` is required. The effective intervention is
  "layers 27-34 in full + layer 35's K/V" — exactly the parameters able to affect the cache.

* ~~**Cotrain dose-response**~~ — RUN. Two dials on top of the 27-35 arm: capacity (19-35,
  §5's full 95% band, 1.72 B of VLM vs 0.91 B) and step size (27-35 with
  `trainer.lr_multiplier: {"vlm.": 3.0}`). All arms share a BIT-IDENTICAL step-1 loss (0.2677)
  with the frozen baseline, so every cell is paired and single-variable.

  NFE=2, n=1000, paired against the MATCHED baseline epoch. Teacher: centre 1.3387,
  E[1 draw] 1.4788, minADE@6 0.9321, spread 0.5778.

  | arm | ep | E[1 draw] | centre | minADE@6 | spread | % of centre gap |
  |---|---|---|---|---|---|---|
  | baseline | 1 | 1.6197 | 1.5371 | 1.0950 | 0.4803 | 0% |
  | baseline | 2 | 1.6178 | 1.5352 | 1.0930 | 0.4801 | 0% |
  | 27-35 | 1 | 1.5377 | 1.4676 | 1.0487 | 0.4378 | 35% |
  | 27-35 | 2 | 1.5193 | 1.4550 | 1.0487 | 0.4172 | 41% |
  | 19-35 | 1 | 1.5255 | 1.4548 | 1.0303 | 0.4394 | 42% |
  | 19-35 | 2 | 1.5074 | 1.4453 | 1.0365 | 0.4099 | 46% |
  | 27-35 x3LR | 1 | 1.6152 | 1.5480 | 1.1134 | 0.4517 | **-5%** |
  | **27-35 x3LR** | **2** | **1.5010** | **1.4376** | **1.0235** | 0.4170 | **50%** |

  ⚠️ **DO NOT JUDGE AN LR ARM AT EPOCH 1.** The x3 arm is the WORST cell at epoch 1 (centre
  +0.0109, t=+0.5, ns -- indistinguishable from baseline) and the BEST at epoch 2 (-0.0976,
  t=-4.8). That is the ordinary signature of a larger step size: more disruption through
  warmup, better convergence after. An epoch-1 reading of this arm produced exactly the wrong
  conclusion once already.

  * **Capacity helps, sublinearly.** 19-35 trains 1.9x the VLM parameters of 27-35 and buys
    +5-6 points (35->42% at ep1, 41->46% at ep2). Epochs are similarly flat (~+5 points).
    Both dimensions diminish hard, so training the whole tower would not close the remaining
    half -- consistent with `COMPARE_EVAL.md` §7g's "the residual is conditioning error".
  * **Step size is the stronger single lever**, at equal capacity: x3 beats x1 on centre by
    0.0174 (t=+1.9) and on minADE@6 by 0.0252 (t=+3.0) at ep2.
  * **19-35 at x3 LR is untested** and is the obvious next arm.

  ### Budget profile of the two best arms (all @ep2, ckpt-6876, n=1000, paired)

  | metric | arm | NFE=1 | NFE=2 | NFE=10 |
  |---|---|---|---|---|
  | centre | teacher | 1.5856 | **1.3387** | 1.3887 |
  | | baseline | 1.7406 | 1.5352 | 1.5360 |
  | | 19-35 x1 | 1.6055 | 1.4453 | 1.4771 |
  | | **27-35 x3** | 1.5995 | **1.4376** | 1.4737 |
  | spread | teacher | 0.4180 | 0.5778 | 1.1099 |
  | | baseline | 0.3197 | 0.4801 | 1.0232 |
  | | **27-35 x3** | 0.3212 | 0.4170 | 0.9639 |

  Centre gap to the teacher, paired: NFE=1 baseline +0.1550 -> x3 **+0.0140 (91% closed)**;
  NFE=2 +0.1965 -> +0.0989 (50%); NFE=10 +0.1473 -> +0.0850 (42%).

  ⚠️ **The 91% at NFE=1 is flattering, not a better operating point.** The gap is small there
  largely because the TEACHER is weak at one step (centre 1.5856 vs its 1.3387 at two). In
  absolute terms NFE=1 is the worst budget for every model -- the student's own NFE=2 centre
  (1.4376) beats its NFE=1 centre (1.5995). **NFE=2 remains the operating point.**

  ⚠️ **At NFE=1 cotrain's `E[1 draw]` (1.6631) "beats" the teacher's (1.6756). DO NOT REPORT
  THAT AS BEATING THE TEACHER.** `E[1 draw]` rewards under-dispersion and this student is now
  under-dispersed; §3 of this file retracts an earlier claim built on exactly that artifact. On
  the honest comparator the student is still +0.0140 behind.

  * **27-35 x3 edges 19-35 x1 at EVERY budget on EVERY metric**, corroborating step size as the
    stronger dial.

  ⚠️ **THE DISPERSION SIGN HAS FLIPPED, and it changes what to try next.** Every cotrain arm
  TIGHTENS the distribution: spread 0.4801 (baseline) -> ~0.41, against the teacher's 0.5778.
  So E[1 draw] is now within **0.022** of the teacher (1.5010 vs 1.4788) while minADE@6 is
  still **0.091** behind (1.0235 vs 0.9321). The student is UNDER-dispersed for the first time.
  ⚠️ Confirmed at EVERY budget by the profile above, not just at NFE=2: cotrain is tighter than
  baseline at 1, 2 and 10, and far tighter than the teacher at 10 (0.9639 vs 1.1099). So the
  tightening is a property of the CACHE, not an artifact of the 2-step sampler.
  §4 of this file showed CD was strictly dominated *because the student already matched the
  teacher's spread*; that premise no longer holds, so **CD on a cotrained student is worth
  re-testing** -- same objective, materially different starting point. The `x0_gt_weight`
  damper must be re-swept from scratch there, for the same reason the pre-maskfix sweep did
  not transfer.
* **CD `x0_gt_weight` re-sweep** against the corrected EoS — recovers the variance component only.
* **Teacher `center` at k=3..8** — never dumped; would confirm the saturation shape at k=2 is not
  a single-point artefact.
