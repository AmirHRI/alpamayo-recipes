# Eval comparison: every number in this recipe, and what it may be compared against

Numbers in this tree come from **three different heads**, **two camera counts**, **two `t0`
choices** and **two metrics**, and mixing any of those has produced retracted findings before.
So the comparability rules come first; the tables are useless without them.

## 0. ⚠️ Comparability rules

| axis | variants | may they be compared? |
|---|---|---|
| head | trajectory-**token** head vs the **action expert** | **NO.** Different heads entirely. The surviving 10B token-head baselines (ade 1.2111 / min_ade 0.6413) are not comparable to anything the expert produces. |
| expert depth | 36-layer (full) vs 28-layer (set-C pruned) | **NO.** A pruned expert has its own ceiling (§1). Every 2B student runs on the pruned one because depth follows the VLM's text config. |
| cameras | 4-cam `[0,1,2,3]` vs 2-cam `[1,3]` | **NO.** Scoring a 2-camera-trained student through the shipped 4-camera `val_dataset` is the same off-distribution error class that moved min_ade 1.116 → 0.626. Filenames carry `_cam13`. |
| `t0` | default 5.1 s keyframe vs event-anchored | **NO.** Measured: event-anchored is **+0.0440 min_ade** harder at k=10 (§5), because the turn rate goes 4.0% → 7.9%. |
| metric | `ade` (single draw) vs `min_ade` (best-of-6) | Both, but they can **disagree in sign** — see §3 and §5. Report both. |
| prompt flags | `include_camera_ids` / `include_frame_nums` | Must both be **true** everywhere. Dropping them moved min_ade 1.116 → 0.626. |

⚠️ **`nav_text` does nothing unless `"route"` is in `components_order`.** `r1_5.py` emits the
route via `case "route":`, so with the shipped eval order
(`[image, traj_history, prompt, traj_future]`) the annotation is silently discarded — the model
runs, the metrics look reasonable, and nothing warns you. Caught only because a matched control
came back **bit-identical** (both 1.0275 / 1.6020). The canonical order is
`[image, traj_history, route, prompt, traj_future]` (asserted in
`tests/test_recipe_static_contracts.py:171`). **Keep the no-nav control permanently**: it is the
only thing that distinguishes "nav helped a little" from "nav was not connected".

---

## 1. Ceilings — teacher through the action expert

4 cameras, default keyframe, n=1000. These are the only valid references for expert-head numbers.

| model | `ade` | `min_ade` |
|---|---|---|
| teacher, full 36-layer expert | 1.3039 | **0.5776** |
| teacher, set-C 28-layer expert ← **the 2B students' ceiling** | 1.6112 | **0.7893** |
| teacher, prune-A 28-layer | 1.5398 | 0.9323 |
| teacher, prune-B 28-layer | 1.7016 | 0.9306 |

## 2. Student arms (stitched: student VLM → teacher's expert)

4B on the full expert; 2B on the set-C pruned expert. `_cam13` = 2 cameras.

| arm | `ade` | `min_ade` |
|---|---|---|
| **4B**, expert-on-student (`eos`, ckpt-1598) | **2.5840** | **1.4044** |
| 4B, `blockrandt` 3 epochs (sampled `t`) | 2.9885 | 1.6008 |
| 4B, `blockonly` 3 epochs (fixed `t`) | 3.0033 | 1.6576 |
| 4B, `kvonly` 3 epochs | 3.8176 | 2.4098 |
| 4B, `cekv` | 4.9100 | 2.7601 |
| 4B, `kv` | 5.9970 | 2.9554 |
| 4B, `kvband` | 5.5903 | 3.1763 |
| 4B, CE only *(floor: cache never trained toward the teacher)* | 12.5500 | 6.9948 |
| 4B, KD only *(floor)* | 17.4294 | 11.3750 |
| **2B**, expert-on-student (`eos`, ckpt-1598) | **3.8103** | **2.3668** |
| 2B, `block2b` ckpt-3000 *(4-cam — not comparable to the `_cam13` rows)* | 4.8203 | 2.6444 |

`blockrandt` vs `blockonly` is the **sampled-`t` result**: 1.6008 vs 1.6576. Random `t` beats
fixed `t`, which §4 explains and §6 says how to push further.

## 3. The span curriculum — a null result

2B, 2 cameras, set-C expert. Four stages, one epoch each, resumed in sequence from the previous
checkpoint. Paired on 1000 clips.

| stage | `block_loss` | `min_ade` | Δ | z | `ade` | Δ | z |
|---|---|---|---|---|---|---|---|
| m=1 (ckpt-1199) | 0.00170 | 2.8367 | — | — | 5.8744 | — | — |
| m=7 (ckpt-2398) | 0.00150 | 2.5360 | −0.3008 | −10.61 | 5.2581 | −0.6163 | −6.74 |
| m=14 (ckpt-3597) | 0.00110 | 2.4160 | −0.1200 | −5.87 | 5.1292 | −0.1288 | −2.26 |
| m=28 (ckpt-4796) | 0.00093 | 2.3760 | −0.0400 | **−1.90** | 5.1550 | **+0.0257** | **+0.63** |
| total | | | −0.4607 | −11.57 | | −0.7194 | −7.32 |

**The final stage is null and the two metrics disagree in sign.** The deltas decay
−0.301 / −0.120 / −0.040, tracking the 4B's epoch-over-epoch curve at *constant* m=1
(−0.329, −0.043) — i.e. the gains are the **epoch effect**, not the span change.

Three independent signs that span length was never the active variable:

* the loss **falls** as spans deepen (0.00170 → 0.00093), so longer spans are not a harder
  target. On identical weights, m=28 scores 0.001025 against m=14's 0.001112 — the objective is
  **m-invariant**, and compounding across layers is damped by construction.
* peak memory is **flat** at 65.8 GiB from m=1 to m=28 — the physical corollary: there is almost
  no extra state to chain.
* four epochs land where 2B expert-on-student sits after **one** (2.3760 vs 2.3668 min_ade), and
  on `ade` expert-on-student is decisively better (3.8103 vs 5.1550).

## 4. Three attempts to reshape `L_block` — three negatives

All from ckpt-1199, 1199 steps, paired on 1000 clips. Baseline is the uniform span-7 epoch.

| run | early-layer grad share | `min_ade` | `ade` | Δ min_ade | z |
|---|---|---|---|---|---|
| **uniform, span 7** ← best in this family | 1/28 | **2.5360** | **5.2581** | — | — |
| `ladder` (0.25 floor) | −72% | 2.6846 | 5.9495 | +0.1487 | +7.84 |
| `ladder_add` (1.0 floor) | −13% | 2.7146 | 5.8496 | +0.1787 | +8.62 |
| mix m=1 + m=7, ep 2 | 1/28 | 2.5615 | 5.4640 | +0.0256 (ade +0.2059, z +3.00) | +1.27 |
| mix m=1 + m=7, ep 3 | 1/28 | 2.5804 | 5.9268 | +0.1644 | +7.68 |

⚠️ **Not monotone in reallocation** — a −72% and a −13% cut to the early layers cost the *same*
(+0.149 vs +0.179), so there is no tunable optimum between uniform and `ladder`.

⚠️ **These are negatives, not tuning failures.** The per-band diagnostic
(`block_loss_early/mid/deep`) rules out "the intervention didn't take":

| run | early | mid | deep | outcome |
|---|---|---|---|---|
| `ladder_add` | −37.4% | −11.5% | **−14.8%** | min_ade **+0.179** |
| mix m1+m7 | −37.4% | −11.5% | −19.2% (tf −17%, span −28%) | min_ade **+0.164** |

The deep band — which §5 shows holds **95%** of the recoverable gap — improved by 15–19% while
the trajectories got worse. Reducing the deep-layer block loss does not improve trajectories.

Per-band loss on the trained ckpt-1199, for reference: early **0.000422**, mid **0.002445**,
deep **0.001934**. Difficulty is **not** monotone in depth — mid is the hardest band, and early
is 5.8× easier than mid while carrying zero importance.

### 4a. `L_field` alone — the objective is NECESSARY, not just weak

The counterpart question: drop the per-layer loss entirely and match only what the expert
**outputs**. `L_field` chains all 28 layers on the student's own cache (no teacher forcing
anywhere), applies `expert.norm` + `action_out_proj`, and MSEs the **velocity** against the
teacher's at a random `t`, expert frozen. It had never been run alone — every prior use was
alongside `L_block` (arm `blockfield`, which itself never passed checkpoint-500 and was never
evaluated).

Motivated by a real measurement: `BLOCK_ODE` puts the **velocity at 30% RMS error** while the
hidden states `L_block` matches are only **3.2%** off per layer (`PRUNING.md:353`) — the
quantity the trajectory integrates was ten times more wrong than the one four epochs optimised.

From scratch, 1 epoch, 2 cameras, matched to the block-only epoch in every other respect:

| objective | `min_ade` | `ade` |
|---|---|---|
| `L_block` (m=1, random `t`) | **2.8367** | **5.8744** |
| **`L_field` only** (random `t`) | **22.0637** | **27.3783** |

**7.8× worse — and worse than the CE-only (6.9948) and KD-only (11.3750) floors.** Wired
correctly, too: the field self-test gives `identity(teacher cache) 4.8e-06`, so this is not a
convention error.

⚠️ **Why, and it is the opposite of what was predicted.** `L_field` constrains only the
ENDPOINT, so intermediate representations are free to drift as long as they *compose* to the
right velocity. The optimiser duly reduced velocity error 5.2× by moving the cache **away** from
the teacher's (block diagnostic, taken under no_grad throughout):

| | step 1 | step 1199 |
|---|---|---|
| field loss (the objective) | 5.761 | **1.107** ↓ 5.2× |
| `block_loss` | 0.679 | **17.374** ↑ 25.6× |
| `block_loss_early` | 0.998 | 4.385 ↑ 4.4× |
| `block_loss_deep` | 0.826 | **46.836** ↑ **56.7×** |
| `kv_ratio_v` | — | **9.539** (every block run held 2.65–2.71) |

The arm's documented objection was that field-only would under-train the **shallow** layers; the
damage is concentrated in the **deep** band (56.7× vs early's 4.4×), because the layers nearest
the output have the most freedom to be wrong in hidden-state terms while still landing on the
right projection. A 20-step smoke showed early rising fastest and was misleading.

**Consequence, taken with §3–§4:** `L_block` is a **weak but NECESSARY** constraint. Driving it
down harder barely moves `ade` (four negatives), and letting it go destroys the model. The 30%
velocity error is real but is not a better training target — it can be reduced by moving the
cache away from the teacher's, which is exactly what happened.

⚠️ Also: `_field_layer` checkpointing was disabled here by analogy with `_span_sweep` (where
peak memory was FLAT at 65.8 GiB from m=1 to m=28). That inference was **wrong for this path** —
it runs at 71.9 GiB with ~8 GiB headroom, and `expandable_segments` converted the pressure into
throughput loss rather than an OOM: **38 s/it vs 9.3**, 5:44 instead of ~3.5 h. Leave
checkpointing ON for the field chain.

## 5. Cache ladder — which part of the cache the expert reads

Substitute the teacher's K/V into part of the student's prefill cache and re-read `min_ade`.
Exact decomposition: the expert's only inputs are the noisy action embedding (shared
`action_in_proj`, identical for both) and the cache, so **100% of the gap is the cache**.
n=300, noise floor **0.0335** (a duplicate baseline), total recoverable gap **−1.3452**.

| substitution | `min_ade` | Δ | share of gap |
|---|---|---|---|
| student (baseline) | 2.5584 | — | — |
| *student rerun (noise floor)* | 2.5249 | *−0.0335* | — |
| **teacher_all** | **1.2132** | −1.3452 | 100% |
| `layers_19_27` | 1.2739 | **−1.2846** | **95%** |
| `region_text` | 1.4114 | −1.1470 | 85% |
| `layers_10_18` | 1.6638 | −0.8946 | 66% |
| `layers_last4` | 1.8655 | −0.6929 | 52% |
| `region_traj` | 2.2140 | −0.3444 | 26% |
| `layers_0_9` | 2.5705 | **+0.0120** | **0%** (inside noise) |
| `region_vision` | 5.2399 | +2.6815 | harmful |
| `k_only` / `v_only` | 3.3980 / 2.9338 | +0.84 / +0.38 | harmful |

Sensitivity rises monotonically with depth (0.001, 0.099, 0.143, 0.173 per layer) because the
expert's layer map is **contractive** — the same fact that makes §3 a null result.

⚠️ **Rows are not additive.** `region_vision` at +2.68 proves it: substituting the 93% majority
while leaving text/traj as the student's is worse than not substituting at all. K and V, and the
regions, must be mutually consistent. Each row reads "how much does this piece help **given the
rest is the student's**"; the percentages do not sum.

⚠️ **The information being present does not mean this objective can extract it.** §4 weighted
training toward layers 19–27 and got worse.

## 6. Signal probe — what actually predicts `ade`

Run the real 10-step sampler twice on the **same noise** — student cache vs teacher cache — and
correlate each candidate against the outcome. n=1200 (200 clips × 6 draws), Spearman.
`excess = student_ade − teacher_ade` on the same clip and noise: the part distillation can fix.

| signal | side of the head | ρ vs `ade` | **ρ vs excess** |
|---|---|---|---|
| `v_err_mean` | post-`action_out_proj` | 0.3159 | **0.4228** |
| `h_err_mean` | pre-head (what `L_block` matches) | 0.2878 | 0.4128 |
| `v_err_last` / `h_err_last` | | 0.2248 / 0.2248 | 0.3837 / 0.3765 |
| `x_div_mean` | trajectory state | 0.1535 | 0.3312 |
| `v_err_first` / `h_err_first` | | 0.2939 / 0.2047 | 0.2833 / 0.2639 |
| `x_div_last` | final trajectory | **0.0211** | 0.2011 |
| *clip difficulty* | — | *0.5565* | *0.0506* |

* **Pre- and post-head are effectively tied** (0.4228 vs 0.4128). Matching the velocity instead
  of the hidden state is **not** the win.
* **`_mean` beats both endpoints for every signal**, while `L_block` samples **one** timestep.
  This is why sampled `t` beat fixed `t` in §2 — and says to cover the sampler's whole `t` grid.
* **`x_div_last` ρ = 0.021**: how close the student's trajectory is to the teacher's says almost
  nothing about whether it is *correct*, because `ade` is measured against ground truth.
* ⚠️ **Only 52% of the student's `ade` is addressable.** Same expert, same noise, the teacher's
  cache scores **2.7823** against the student's **5.7627**. The other **48% is the teacher's own
  error against GT** — irreducible by any amount of distillation, and every objective in this
  line optimises teacher agreement. (Specific to these 200 clips / 1 seeded draw / 2 cameras;
  the qualitative point is solid, the exact split is not a corpus constant.)

---

## 7. Denoising steps — the nav sweep, and few-step distillation

Teacher, **unpruned 36-layer** expert, 2 cameras, n=1000, event-anchored `t0`, paired per clip.
Both arms carry `"route"` in `components_order`; they differ only in whether `nav_text` is
present (`null` in the control ⇒ `construct_route` returns `[]`).

| steps | nav `min_ade` | ctl | Δ | z | nav `ade` | ctl | Δ | z | diversity |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.2817 | 1.3064 | −0.0247 | −1.93 | 1.6362 | 1.6609 | −0.0247 | −1.92 | 0.648 |
| **2** | 0.9318 | 0.9587 | −0.0269 | −2.11 | **1.4459** | **1.4822** | **−0.0363** | **−2.80** | 0.892 |
| 3 | 0.8412 | 0.8631 | −0.0219 | −1.87 | 1.4767 | 1.5137 | −0.0370 | −2.65 | 1.117 |
| 4 | 0.7985 | 0.8187 | −0.0202 | −1.83 | 1.5081 | 1.5446 | −0.0364 | −2.49 | 1.253 |
| 5 | 0.7812 | 0.8019 | −0.0207 | −2.04 | 1.5533 | 1.5850 | −0.0317 | −2.17 | 1.368 |
| 6 | 0.7659 | 0.7855 | −0.0196 | −1.83 | 1.5875 | 1.6162 | −0.0288 | −1.92 | 1.460 |
| 7 | 0.7555 | 0.7742 | −0.0187 | −1.74 | 1.6209 | 1.6452 | −0.0243 | −1.80 | 1.541 |
| 8 | 0.7509 | 0.7696 | −0.0187 | −1.76 | 1.6557 | 1.6756 | −0.0199 | −1.58 | 1.609 |
| 9 | 0.7473 | 0.7657 | −0.0183 | −1.73 | 1.6796 | 1.6991 | −0.0195 | −1.55 | 1.661 |
| **10** | **0.7441** | **0.7625** | −0.0184 | −1.75 | 1.6976 | 1.7188 | −0.0212 | −1.63 | 1.701 |

### 7a. The step trade-off dominates, and the metrics disagree

`min_ade` improves **monotonically** 1.2817 → 0.7441 (−42%, k=1→10) while `ade` is **best at
k=2** (1.4459) and degrades **17%** by k=10. Diversity rises monotonically 0.648 → 1.701, which
is the mechanism: more denoising spreads the six proposals, so best-of-6 improves while any
single draw gets worse.

**Consequence.** A deployed single-draw planner wants **k=2** — better `ade` *and* **5× cheaper**,
against denoising being 51–77% of end-to-end latency (`LATENCY_PROFILE.md`). An oracle-metric
benchmark wants k=10. There is no single best step count; it is a choice of metric.

### 7b. Navigation helps consistently but modestly

Δ`min_ade` ≈ **−0.02** at every step count. Individually marginal (z −1.7 to −2.1), but **pooled
over all ten counts: −0.0208, se 0.0035, z −5.88** (n=10000) — a real ~2.4% effect. The `ade`
gain is **largest at low step counts** (−0.036 at k=2–4) and shrinks by k=10, i.e. the
instruction helps most exactly where the single-draw metric is best.

### 7c. Event-anchored `t0` is harder, and nav recovers ~40% of it

Control (event `t0`, no nav) vs the earlier Gate C (default keyframe, no nav), same script, same
cameras, k=10, 1000 shared clips: **+0.0440 min_ade**. Consistent with the turn rate rising
4.0% → 7.9% (`NAVTEXT_SAMPLING.md`). Nav recovers −0.0184 of that, about **42%**.

For reference, the default-keyframe no-nav sweep (`teacher_k*`):

| steps | `ade` | `min_ade` |
|---|---|---|
| 1 | 1.6116 | 1.2436 |
| 2 | **1.4236** | 0.8876 |
| 3 | 1.4767 | 0.8078 |
| 5 | 1.5623 | 0.7547 |
| 10 | 1.6736 | **0.7185** |

The `ade`-best-at-k=2 / `min_ade`-best-at-k=10 shape is identical with and without nav and at
both `t0` choices — it is a property of the sampler, not of the conditioning.

### 7d. ⚠️ Caveats on the nav number

* **`--horizon-start` is 0**, so the route is derived from the GT future the model is being asked
  to predict. This measures *what a correct instruction is worth* — an **upper bound** on a
  planner-supplied route.
* **92% of the instructions are "Continue straight"** (7.9% turns), which caps the achievable
  effect. A per-category split over the 79 turn clips would show whether the gain concentrates
  there; not yet computed.
* **The 2B student has never seen nav text.** Every KD run used the plain 2-camera config with no
  annotations file, so route tokens carry no training signal for it. Nav is in-distribution for
  the teacher and **out-of-distribution for the student** — adding it at eval time to the 2B is
  not a valid experiment. Train annotations exist (109,997 anchors, or the 50k subsets).

---

### 7e. Few-step distillation of the EoS-2B expert — 2 NFE beats 10

2 cameras, nav, event-anchored `t0`, n=1000, `sft_eval_eos_2b_nav_lcdrive`. Every row is the
SAME student VLM; only the expert's weights and the denoising step count change, so these are
paired end to end. `diversity` is mean pairwise L2 between the 6 draws (§7a's definition);
`coverage` is, per clip, the distance from each *teacher* draw to the nearest *student* draw —
i.e. can the student produce what the teacher produces; `floor` is the ADE of the barycentre of
the 6 draws, an estimate of what a conditional-mean predictor of this model reaches.

| arm | NFE | `ade` | `min_ade` | diversity | coverage | floor |
|---|---|---|---|---|---|---|
| teacher 10B, 36-layer expert *(ceiling)* | 2 | 1.4460 | 0.9319 | 0.892 | **0.425** | 1.3382 |
| **EoS-2B baseline ← the reference** | **10** | **2.4859** | **1.3974** | 1.609 | 1.2628 | 2.3015 |
| EoS-2B baseline | 2 | 2.4661 | 1.8148 | 0.882 | 1.4980 | 2.4000 |
| CD bootstrap + 0.1·gt (job 591) | 2 | 2.5304 | 1.5837 | 1.344 | 1.2946 | 2.3796 |
| endpoint, `x0_gt_weight` 0.0 | 2 | 2.8433 | 1.3429 | 2.511 | 1.1678 | 2.3851 |
| endpoint + 0.1·gt | 2 | 2.6996 | **1.3214** | 2.256 | 1.1366 | 2.3148 |
| **endpoint + 0.3·gt** | **2** | 2.5141 | 1.3344 | 1.864 | **1.1173** | 2.2397 |
| **endpoint + 0.5·gt** | **2** | **2.3726** | 1.4276 | 1.451 | 1.1699 | 2.1952 |
| endpoint + 1.0·gt *(past the turn)* | 2 | 2.3107 | 1.5345 | 1.148 | 1.2825 | 2.1949 |
| endpoint, 0.0 | 1 | 2.8598 | 1.5928 | 2.036 | 1.3224 | 2.5779 |
| endpoint + 0.3·gt | 1 | 2.6968 | 1.6629 | 1.598 | 1.3524 | 2.5047 |
| endpoint + 0.5·gt | 1 | 2.5720 | 1.7620 | 1.235 | 1.4136 | 2.4132 |
| endpoint + 1.0·gt | 1 | 2.5182 | 1.8731 | 1.007 | 1.5211 | 2.3916 |

Paired per-clip against the 10-step reference:

| arm (2 NFE) | Δ`min_ade` | z | Δ`ade` | z |
|---|---|---|---|---|
| endpoint + 0.0·gt | −0.0545 | −2.12 | +0.3574 | +9.34 |
| endpoint + 0.1·gt | **−0.0760** | **−3.14** | +0.2137 | +6.04 |
| **endpoint + 0.3·gt** | −0.0630 | **−2.63** | +0.0282 | **+0.84 (n.s.)** |
| **endpoint + 0.5·gt** | +0.0302 | **+1.11 (n.s.)** | **−0.1133** | **−3.23** |
| endpoint + 1.0·gt | +0.1371 | +4.53 | −0.1752 | −4.62 |

**Two operating points, each a PARETO IMPROVEMENT on the 10-step sampler at 1/5 the denoising
cost** — and 2 epochs is converged; §7e-i has the third-epoch check.
`0.3·gt` beats the reference on `min_ade` and ties `ade`; `0.5·gt` beats it on `ade` and
ties `min_ade`. Neither wins both with significance, and that is the shape of the problem, not a
shortfall — `ade` and `min_ade` trade through diversity, the same tension §7a documents across
step count. Choose by deliverable: a single-draw planner takes 0.5, an oracle benchmark 0.3.

⚠️ **1 NFE never gets there.** Best rows are `min_ade` 1.5928 (gt 0.0) and `ade` 2.5182
(gt 1.0), both worse than the 10-step reference on the other metric. 2 NFE is the operating
point; do not quote a 1-NFE row as the result.

### 7e-i. The `x0_gt` damper sweep, one variable at a time

All rows: `x0_teacher_weight` 1.0, `cd_weight` 0.0, 2 epochs (`checkpoint-3126`), same student
init, same schedule. Only `model.cd.x0_gt_weight` changes. Reference is this student's own
10-step sampler: `ade` **2.4859** / `min_ade` **1.3974**.

| `x0_gt_weight` | 1 NFE `ade` | 1 NFE `min_ade` | 2 NFE `ade` | 2 NFE `min_ade` |
|---|---|---|---|---|
| 0.0 *(endpoint only)* | 2.8598 | 1.5928 | 2.8433 | **1.3429** |
| 0.1 | 2.7896 | 1.6094 | 2.6996 | **1.3214** |
| 0.3 | 2.6968 | 1.6629 | 2.5141 | **1.3344** |
| 0.5 | 2.5720 | 1.7620 | **2.3726** | 1.4276 |
| 1.0 | 2.5182 | 1.8731 | **2.3107** | 1.5345 |
| *reference: 10 NFE, no CD* | — | — | 2.4859 | 1.3974 |

Read down the two 2-NFE columns: `ade` falls monotonically with the damper while `min_ade`
turns between 0.3 and 0.5. **At 1 NFE both columns are monotone in the same direction and
nothing beats the reference** — do not quote a 1-NFE row as the result.

Two diagnostics locate the turn more sharply than either headline metric:

* **Coverage is U-SHAPED, with its minimum at 0.3** (1.1678 → 1.1366 → **1.1173** → 1.1699 →
  1.2825). At gt 1.0 it has reverted to the UNTRAINED baseline's 1.2628 — the GT term is now
  pulling the student AWAY from the teacher's trajectory set toward the conditional mean, i.e.
  job 591's failure mode arriving from the other direction, and it shows up here while `ade` is
  still improving.
* **Diversity tracks it monotonically** (2.511 → 2.256 → 1.864 → 1.451 → 1.148 against the
  teacher's 0.892), which is the mechanism: `ade` and `min_ade` trade through spread.

⚠️ **THIS SWEEP IS CONVERGED IN WEIGHT, NOT IN TRAINING TIME.** An earlier revision of this
section claimed the barycentre floor "bottoms at ~2.195" and that the sweep was complete. That
was wrong, and the error is worth recording because it is easy to repeat: 2.1952 (gt 0.5) and
2.1949 (gt 1.0) are two weights at the SAME epoch, which says nothing about convergence in
epochs. Measured ep1 → ep2, paired, n=1000:

| run | Δ`ade` | z | Δ`min_ade` | z | floor ep1 → ep2 |
|---|---|---|---|---|---|
| gt=0.5 | −0.0240 | −3.64 | −0.0314 | −5.88 | 2.2268 → 2.1952 |
| gt=1.0 | −0.0686 | −7.53 | −0.0688 | −9.34 | 2.2729 → 2.1949 |

Both metrics were still falling together at 3126 steps, and so was the floor.

**A third epoch on gt=0.5 then settled it: the curve is flat by epoch 2.** True resume
(optimizer + scheduler restored, LR continuing mid-cosine at 6.1e-6 and re-annealing, world
size held at 2 because `rng_state_*.pth` is per-rank), so it even got a mild LR bump:

| gt=0.5 @ 2 NFE | `ade` | `min_ade` | diversity | coverage | floor |
|---|---|---|---|---|---|
| ep1 (1563) | 2.3967 | 1.4590 | 1.441 | 1.1977 | 2.2268 |
| ep2 (3126) | 2.3726 | 1.4276 | 1.451 | 1.1699 | 2.1952 |
| ep3 (4689) | 2.3739 | 1.4299 | 1.448 | 1.1681 | 2.1970 |

| paired delta | Δ`ade` | z | Δ`min_ade` | z |
|---|---|---|---|---|
| ep2 − ep1 | −0.0240 | −3.64 | −0.0314 | −5.88 |
| **ep3 − ep2** | **+0.0013** | **+0.86** | **+0.0023** | **+1.60** |

So the ep1 → ep2 gain was the TAIL of the curve, not a rate to extrapolate — an extrapolation
predicting `ade` ≈ 2.35 / `min_ade` ≈ 1.40 for ep3 was wrong; it delivered 2.3739 / 1.4299.
Diversity, coverage and the floor are unchanged to three decimals. **2 epochs is the recipe;
do not spend a third.** gt=1.0's ep1 → ep2 delta was larger (−0.069) so it may have had a
little more left, untested — but it is the worse operating point on `min_ade`.

`min_ade` never closes the 1.4276 → 1.3974 gap, so the both-metrics win does not exist at any
epoch count here: 0.3 and 0.5 remain two separate operating points. Final gt=0.5 @ 2 NFE vs the
10-step reference: `ade` **−0.1120 (z −3.22)**, `min_ade` +0.0325 (z +1.21, n.s.).

⚠️ **The action-space training loss is NOT a convergence signal for these metrics, and this
tree has now been fooled by it twice.** `x0_gt_loss` is statistically flat across the whole of
epoch 2 at every weight (Spearman rho −0.002 / −0.004 / −0.009, p = 0.97 / 0.95 / 0.87;
Theil-Sen CIs straddling zero) while `ade` and `min_ade` improved by up to 0.069 with z beyond
7. The loss is a mean squared residual on (accel, curvature) dominated by irreducible
conditional variance (~0.45 at gt 1.0), so the reducible part is buried; the metric depends on
those actions INTEGRATED over 6.4 s, where small systematic gains compound. Judge convergence
by evaluating successive checkpoints, never by the loss curve.

⚠️ Corollary for the two traps above: the action-space loss said "converged" at a point where
the metrics still had 0.03-0.07 left (ep1 → ep2), and it said the same thing at a point where
they genuinely were converged (ep2 → ep3). It carries no information about either. Only
successive-checkpoint evals do.

### 7f. The CD bootstrap does not work on this budget — the endpoint pair does

Job 591 ran the textbook consistency objective (one teacher solver step, EMA target, M=10) and
it went the WRONG WAY on the band that matters. Per-band `x0` loss, first vs last 10% of the run:

| band | job 591 (cached 10B teacher) | job 580 (online teacher) |
|---|---|---|
| anchor (τ≈0.1) | 3.7e-4 → 4.4e-4 *(flat)* | 1.3e-5 → 7.7e-5 |
| mid | 3.0e-3 → 1.8e-3 | 7.2e-4 → 4.5e-4 |
| **noise (τ≥0.8, what 1 NFE evaluates)** | **4.0e-2 → 5.3e-2 (+32%)** | 1.8e-2 → 1.3e-2 (−25%) |

Three compounding causes, all measurable:

* **The anchor is unreachable.** With an *online* teacher the frozen reference IS the student's
  initialisation, so the last rung is exact (1e-5) and the chain propagates from it. With the
  10B cache the anchor asks a 28-layer expert on a 2B cache to reproduce the 36-layer expert's
  transition velocity: 28x larger, and flat for 3126 steps. A bootstrap anchored on a rung the
  student cannot fit propagates error instead of signal.
* **The anchor gets ~0.3% of the gradient.** `cd_loss` weights the residual by `tau**2`
  (correct for a uniform f-space objective), so the anchor rung carries 1/100 the per-sample
  gradient of the noise rung and is 1/10 of samples. The only teacher-anchored rung in the
  objective is ~300x underweighted relative to the purely bootstrapped one.
* **The EMA cannot cross the grid in time.** `decay: 0.999` is a 1000-step horizon; information
  travels roughly one rung per horizon, so M=10 rungs needs ~10k optimizer steps. The run had
  3126.

The fix is not to tune the bootstrap. The cache already stores, for every clip, `K` pairs of
(initial noise `states[k,0]`, the teacher's answer for that noise `states[k,M]`) — the exact
input/output signature of a 1-NFE sampler. Regressing on it directly,

    L = || x + tau*v_theta(x, tau) - states[k, M] ||^2

removes the EMA, the solver step and the chain at once, converges on a 2-epoch budget, and runs
ONE expert forward per step instead of two. `model.cd.cd_weight: 0.0` + `x0_teacher_weight: 1.0`.

⚠️ **The two `x0` targets are not interchangeable, and the difference is diversity.**
`x0_source: gt` is the dataset action, SHARED by all `K` draws, so its minimiser at `tau=1` is
the conditional mean — alone it collapses the spread. `x0_teacher_weight` uses the cached
endpoint of *this* noise draw, so `K` draws keep `K` distinct targets — alone it OVERSHOOTS,
to diversity 2.511 against the teacher's 0.892. Blending them is what lands the answer, and the
two turn out complementary rather than antagonistic: 0 → 0.1 improved `ade` AND `min_ade`
together; only 0.1 → 0.3 traded, and cheaply (−0.0130 `min_ade` for −0.1855 `ade`).

⚠️ The GT residual means something DIFFERENT in the two teacher modes and is not a controlled
variable across them. Online: `x_hi = interpolate(x0_gt, eps, tau)` is built from the GT action,
so the term is near self-fulfilling (0.148). Cached: `x_hi` sits on the 10B's path toward the
10B's endpoint while the term demands the GT one (0.617, and RISING). Job 591's config comment
claiming it "changes only the teacher reference" was wrong.

### 7g. What few-step distillation cannot fix

`ade` **< 1.5 is unreachable for this student on this axis**, and the floor column is why: strip
all sampling variance and the EoS-2B still scores ~2.30. The residual is conditioning error, not
sampler error. Two independent measurements agree:

* **Coverage.** The teacher's own mode scale (leave-one-out NN among its 6 draws) is **0.425 m**.
  The student's *closest* proposal to a teacher trajectory is 1.12–1.50 m — ~3x that, and nearly
  the size of the teacher's entire `ade`. Distillation moves it monotonically
  (1.2946 → 1.1678 → 1.1173) and closes maybe 10% of a gap that must reach 0.425.
* **Diversity is not the explanation.** The baseline at k=2 matches the teacher's spread almost
  exactly (0.882 vs 0.892) and still has the WORST coverage in the table, 1.4980.

⚠️ **A capacity-matched intermediate teacher does not help — already tested.** `nav_cdp28_e2_s3126`
is the 8B VLM with 28 active expert layers (36 slots, 8 identity-bypassed so the cache pairing
survives), consistency-trained: **min_ade 1.4703 / `ade` 2.4730** at 1 NFE. That is no better
than the 2B student it would teach (1.3974 / 2.4859). CD recovered it from the raw ablation
(2.1363 → 1.4703 min_ade) and it still lands below where we already are, so there is nothing to
distil. Two independent 28-active-expert configurations — one on an 8B tower, one on a 2B tower
with a natively trained expert — converge to `ade` 2.47 vs 2.49 and `min_ade` 1.47 vs 1.40.

⚠️ **Expert depth is NOT a free parameter.** The expert self-attends over the VLM's cache and
expert layer `l` reads cache layer `l` (`_SkippedExpertLayer`'s docstring), so its depth is
welded to the tower's: Cosmos-Reason2-2B has 28 text layers, Qwen3-VL-4B has 36. "Give the 2B a
36-layer expert" is not an available intervention. The only sub-1.5 rows in this tree run a
36-layer expert — including consistency-distilled ones: `nav_cd1563` reaches **`ade` 1.4983 at
1 NFE** and **1.4886 at 2 NFE** (vs the undistilled teacher's best of 1.4459 at k=2). The method
delivers sub-1.5 `ade` at 1–2 steps; the depth cap is what stops the 2B.

## 8. Reproducing

```bash
# stitched student/teacher eval (2 cameras -- REQUIRED for the 2-cam arms)
ARM=<arm> CKPT=checkpoint-N CAMERAS='[1,3]' PIN_GPU=1 sbatch --gpus=4 slurm_eval_stitched.sh

# denoising-step sweep, with nav; the ORDER override is what makes nav_text take effect
M=/data/datasets/physical_ai_av/lcdrive_physicalai_av_manifests
TAG=teachernav STEPS='[1,2,3,4,5,6,7,8,9,10]' CAMERAS='[1,3]' \
  EXTRA_ARGS="++data.val_dataset._target_=alpamayo.data.pai_nav.PAIDatasetWithNav \
    ++data.val_dataset.annotations_path=$M/nav_lcdrive_val_mysubset_1k.json \
    ++data.val_dataset.vla_preprocess_args.components_order=[image,traj_history,route,prompt,traj_future]" \
  sbatch --gpus=4 slurm_eval_step_sweep.sh
# ...and the matched control: same command with nav_lcdrive_val_mysubset_1k_nonav.json

# few-step distillation of the EoS-2B expert (§7e-7g). The rollout cache is built ONCE;
# every arm below reuses it. K=6 noises x M=10 Euler states per clip, 50k clips.
python -m alpamayo1_5_distill.scripts.generate_teacher_trajectories \
  config=cache_full_teacher_trajectories_2cam_nav_lcdrive cache_root=<dir on /data>
GT_W=0.3 GPUS=0,1 ./run_cd_eos2b_endpoint_gt.sh          # the winning arm
GT_W=0.0 ./run_cd_eos2b_endpoint.sh                      # endpoint only (no damper)
# eval at 1 and 2 NFE; ARM=eos points BOTH model paths at the CD checkpoint
EOS_DIR=<run dir> TAG_BASE=<tag> CKPT=checkpoint-3126 NFE=2 ./slurm_eval_eos.sh

# cache ladder / signal probe -- eval only, teacher and student on separate cards
CUDA_VISIBLE_DEVICES=0,3 python -m alpamayo1_5_distill.scripts.cache_ladder   ...  # §5
CUDA_VISIBLE_DEVICES=0,3 python -m alpamayo1_5_distill.scripts.signal_probe   ...  # §6
```

Artefacts: `training/stitch_*.json` (§1–4), `training/stepsweep/` (§7),
`training/cacheladder/` (§5), `training/signalprobe/` (§6). All per-clip, joinable on `clip_id`.

## 9. Open

* **Multi-`t` block loss** — the one deficiency §6 identified and §2 corroborates. Untried.
* ~~**Ground-truth trajectory error in the objective.**~~ Addressed in §7e: `x0_gt_weight`
  puts the dataset action in the objective alongside teacher agreement, and the blend is what
  lands 2 NFE at the 10-step model's `ade`. Note it is a DIVERSITY control, not just an
  accuracy term — see the ⚠️ in §7f.
* ~~**The `x0_gt` weight is not swept out.**~~ Swept 0 / 0.1 / 0.3 / 0.5 / 1.0 in §7e; the
  frontier turns between 0.3 and 0.5 and the floor saturates at 0.5. Closed.
* **Multi-rung sampling per prefill.** τ=1.0 is 1 of 10 rungs, so only ~10% of samples train
  the point 1 NFE actually evaluates. The VLM prefill dominates cost and is shared, so several
  (noise, τ) draws per prefill are nearly free — untried, and the obvious 1-NFE lever.
* **Nav-conditioned student** — data ready, never trained; needs `"route"` in the *training*
  config's `components_order` too, or it will train on nav data with the instruction invisible.
* **`--horizon-start` variant** of the val annotations, for a leak-free nav number.
* Whether the nav gain concentrates on the 79 turn clips (§7d).
* **`blockfield` (block + field)** — the one arm that tests whether the velocity term adds
  anything ON TOP of a constraint that keeps the cache anchored. Built for exactly this, never
  ran past checkpoint-500, never evaluated. Run it with field checkpointing ON (§4a).
