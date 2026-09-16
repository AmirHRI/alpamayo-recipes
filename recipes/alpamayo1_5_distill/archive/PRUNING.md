# Pruning the action expert — what we tried, and what it cost

Goal: cut the 36-layer action expert to **28 layers** so it can pair 1:1 with a 28-layer
2B VLM (`Cosmos-Reason2-2B`), then train it with the teacher-forced block loss.

**Headline: no similarity metric survived a causal test.** Selection driven by
*structural* facts about how the model is wired beat metric-driven selection by
−0.143 min_ade (z = −5.97), and pruning is still not free (+37%).

All numbers are `min_ade` on the LCDrive val 1k subset, n = 1000, paired per clip, scored
through the stitched harness. The teacher's own pairing is **0.5776**.

---

## 1. The ablation results

8 of 36 expert layers replaced by identity stand-ins. **Indices preserved**, so surviving
layer *l* still reads VLM cache layer *l* — shortening the `ModuleList` would re-pair every
layer above the cut with the wrong cache and measure something else entirely.

| set | layers removed | chosen by | min_ade | vs teacher |
|---|---|---|---|---|
| — | none | — | **0.5776** | — |
| **C** | `{4,10,13,15,19,25,27,34}` | **BI (cosine family) as the objective, structure as the constraint** — lowest-BI block per depth bin | **0.7893** | **+37%** (z = +10.85) |
| B | `{18,19,24,25,27,32,33,34}` | CKA, with L35 swapped for L27 | 0.9306 | +61% (z = +12.78) |
| A | `{18,19,24,25,32,33,34,35}` | CKA, contiguous runs | 0.9323 | +61% (z = +13.65) |

Paired comparisons:

* **C beats A** by **−0.1430** (z = −5.97), better on 53% of clips — it recovers **40%** of
  A's damage.
* **C beats B** by −0.1413 (z = −6.63).
* **B vs A: −0.0017, z = −0.07 — NOT significant.** The last layer is *not* special:
  dropping L35 (A) costs the same as keeping it and dropping L27 instead (B). The concern
  that `action_out_proj` would be unusually sensitive to its input layer was unfounded.
* A and B are worse than the teacher on **72%** of clips; C on fewer.

⚠️ These ablations **bypass layers without retraining**, and keep the original cache
indices. +0.212 for C is the *pre-training bar*, not a ceiling —
`configs/sft_prunedexpert_10b_lcdrive.yaml` trains the pruned expert against the teacher's
own cache to find out how much comes back. It also does **not** test the re-indexing that
deployment requires (surviving layer *j* reading cache *j* of a 28-layer VLM), which is a
separate perturbation.

Per-clip results: `training/stitch_4b_teacher_prune{A,B,C}.json`.
Mechanism: `PRUNE_EXPERT_LAYERS=4,10,13,15,19,25,27,34`, handled by
`_apply_expert_pruning` in `models/stitched_model.py`. Every run prints
`[stitch] PRUNED expert layers [...] -- 8 of 36 bypassed`; treat its absence as a failed run.

---

## 2. Metrics we computed, and their verdicts

`scripts/expert_cka.py`. Representations are saved (`reps_t*.npy`, `grams_tsweep.npz`), so
every number below is re-derivable with **no GPU**.

| metric | most-prunable 8 | verdict |
|---|---|---|
| **CKA** (Kornblith, linear) | `{18,19,24,25,32,33,34,35}` (adjacent CKA ≥ 0.998) | **REFUTED** — sets A/B, **+61%** |
| **Block Influence** `1 − cos` (ShortGPT) | `{2,3,4,5,6,15,33,34}`, Σ BI 0.0221 | **untested unconstrained**; constrained by structure it gave set C, **+37%** |
| **cosine** (token-wise) | same ranking as BI | same |
| **angular distance** `arccos(cos)/π` (Gromov) | same ranking; best spans L4 / L4-5 / L2-6 / L1-8 | same |

The cosine family's raw ranking, lowest BI first:
`L4 0.00141 · L2 0.00153 · L34 0.00217 · L6 0.00258 · L33 0.00320 · L15 0.00363 · L5 0.00373 · L3 0.00382 · L35 0.00392 · L10 0.00455`

⚠️ **Set C is BI-selected, not metric-free.** Every member is the lowest-BI block in its depth
bin: `L4` rank 1/35, `L34` rank 3, `L15` rank 6, `L10` rank 10, `L25` rank 12, `L27` rank 13,
`L13` rank 14, `L19` rank 20. The structural constraints supplied the BINS; BI chose within
them. That is why C's Σ BI is 0.0342 against BI's unconstrained 0.0221 — the constraints
forced the budget to spread across depth instead of concentrating in the early stack, and
members like `L19` are there because their bin held nothing cheaper.

So the causal scoreboard is: **CKA's set failed (+61%); BI's set, constrained, did materially
better (+37%)**. BI has one supporting data point. What is still untested is BI
*unconstrained* — `{2,3,4,5,6,15,33,34}` — which would separate "BI is a better metric" from
"the structural constraints did the work".

**The last three are one metric in three forms.** `1 − cos` and `arccos(cos)/π` are both
monotone decreasing in cosine, so on identical inputs they induce *identical rankings*.
Their agreement is one vote, not three. Final-token-only (as Gromov specifies) versus
token-averaged changes almost nothing here: rank correlation **0.9951**.

**Why CKA disagrees with the rest: it is invariant to orthogonal transformations.** A layer
that rotates the representation scores CKA ≈ 1 while completely changing the vectors the
next layer reads. For *deletion* that invariance is a liability — removing layer *l* means
layer *l+1* receives `h_{l−1}` in the same basis, so literal alignment is what matters.
Rank correlation between CKA and BI is only **0.38**.

The clearest case is **block 22**: CKA rates it near-identity (1 − CKA ≈ 0.005) while BI
calls it the single most influential block in the network (0.0161).

Both families agree on one thing: **L7–L14 is the working band.** CKA's sharpest adjacent
drop is L8 (0.9751), and BI's most influential blocks are 22, 21, 14, 9, 12, 17.

### Rankings

Most prunable by CKA (adjacent CKA, mean over 3 studies × all timesteps, sd ≤ 0.0007):
`L33 0.99951, L34 0.99949, L35 0.99923, L1 0.99922, L3 0.99887, L24 0.99882, L25 0.99856, L32 0.99856`

Least prunable: `L8 0.97502, L7 0.98153, L11 0.98177, L9 0.98748, L13, L6`

Lowest Block Influence: `block 4 (0.00141), 2, 34, 6, 33, 15, 5, 3` — i.e. the **early** stack.

---

## 3. The three structural facts that actually mattered

None of these is visible to a similarity metric computed on the expert alone.

**Depth alignment.** Expert layer *j* reads VLM cache layer *j*. Removing 8 layers
*uniformly* gives `k ≈ i·(8/36)`, so `(i−k)/28 = i/36` **exactly** — every survivor reads a
cache at its original relative depth. A contiguous cut shifts everything above it by the
full 8.

| set | Σ BI | max depth misalignment |
|---|---|---|
| depth-aligned, deepstack-protected `{4,10,13,15,19,25,27,34}` | 0.0342 | **3.9%** |
| depth-aligned, unprotected `{2,4,10,15,18,25,27,34}` | 0.0295 | 3.2% |
| two groups of 4 `{2,3,4,5,32,33,34,35}` | 0.0251 | 11.4% |
| one span of 8 `{28..35}` | 0.0425 | 22.9% |
| CKA set A | 0.0417 | 14.4% |

**Deepstack.** `modeling_qwen3_vl.py` injects the 3 multi-level ViT features into LLM layers
**0, 1, 2** (`if layer_idx in range(len(deepstack_visual_embeds))`, comment: *"add visual
features to the hidden states of first several layers"*). `deepstack_visual_indexes` are the
*ViT* layers features are taken from — `[8,16,24]` for Cosmos-8B, `[5,11,17]` for the 2B —
and there are always 3, so the injection target is always LLM 0–2. Expert layers 0–2 read
those caches; set C leaves them alone. Costs only Σ BI 0.0295 → 0.0342 and misalignment
3.2% → 3.9%.

**Spans are super-additive.** Removing a contiguous run costs *more* than the sum of its
per-block influences, so scattered removals beat runs:

| span | span BI | Σ per-block BI | ratio |
|---|---|---|---|
| L2–L6 | 0.0254 | 0.0131 | **1.94×** |
| L24–L27 | 0.0397 | 0.0252 | 1.57× |
| L32–L34 | 0.0129 | 0.0106 | 1.21× |
| L18–L19 | 0.0148 | 0.0129 | 1.14× |

Cheapest contiguous span of each length: `1: L4 (0.0014) · 2: L4-5 (0.0062) · 3: L2-4
(0.0109) · 5: L2-6 (0.0254) · 8: L28-35 (0.0717)`. The cheapest 8-run costs **51×** the
cheapest single block.

⚠️ Aggregation matters and the two natural choices disagree about whether to split. For
small angles `d ≈ √(2·BI)/π`, so summing angular distances sums square roots — and √ is
concave, which penalises splitting *by construction*. If separate removals perturb roughly
independently, their squared magnitudes add, which is what Σ BI approximates. Under Σ BI,
two groups of 4 costs about **half** a single span of 8.

---

## 4. Robustness of the measurements

* **Train vs val** (n = 64 each, zero overlap, seeded): C-matrix correlation **0.9870**,
  mean |diff| 0.0005, **same five most-active transitions in the same order**. The structure
  is architectural, not memorised.
* **On-policy vs GT-interpolation** `x_t` (n = 64, same clips): correlation **0.9865**, max
  |diff| 0.0064. The sampler's own path and the closed-form
  `x_t = t·x_GT + (1−t)·ε` give the same layer structure — so the cheap interpolation is
  justified and no 10-step integration is needed.
* **Timestep dependence**: transformation peaks at **t ≈ 0.2–0.4**, not at either endpoint
  (corner CKA 0.181 at t = 0.2 vs 0.572 at t = 1). L7→L8 is the most-active transition at
  every step from t = 0.2 to 0.8. `t = 1` is the least stable row across runs.
* **Reproducibility**: two independent runs gave t=0 corner CKA 0.2513 vs 0.2579, so ±0.007
  is the noise floor at n = 64.

---

## 5. Figures

Expert (`training/cka/`, `training/cka_fine/`):

| file | what |
|---|---|
| `cka_fine/block_influence.png` | BI per block with the CKA view beneath — the two metrics rank blocks differently (rank corr 0.38) |
| `cka_fine/span_BI_matrix.png` | cost of removing any contiguous span `a…b`; cyan diagonal = single block, super-additivity visible |
| `cka_fine/cka_expert_teacher_tsweep.png` | 36×36 CKA at 11 points along the flow |
| `cka_fine/C_adjacent_cka.png` | timestep × layer-transition; the L7–L13 stripe |
| `cka/cka_expert_teacher_step0.png` | the original t=0 matrix (superseded — t=0 is not representative) |
| `cka_val64/`, `cka_train64/`, `cka_train64_onpolicy/` | the robustness checks above |

VLM (`training/cka_vlm/`, `training/cka_vlm_front4/`) — see the retraction below:

| file | what |
|---|---|
| `cka_vlm_front4/cka_vlm_rowL2_transitions.png` | **the valid one**: per-token L2 normalised, `l−1→l` cells outlined, value strip |
| `cka_vlm_front4/cka_vlm_rowL2_labeled.png` | same matrix, all 36 layers labelled |
| `cka_vlm/cka_vlm_corrected.png` | raw vs two channel-based fixes, side by side |
| `cka_vlm/cka_vlm_teacher.png`, `cka_vlm_front4/cka_vlm_teacher{,_labeled}.png` | **RAW — invalid from L6 on** |

---

## 6. ⚠️ Retracted: the raw VLM CKA

Reported, then withdrawn. From L6 onward **one channel** (index 2276, |a| = 28288 against a
median channel of 213) makes the centred Gram **rank-1** — top-1 eigenvalue 99.9%, effective
rank 1.0 — so CKA ≈ 1 *by construction*, giving `1.000` for 28 consecutive layers.

Channel-based fixes (dropping the top-20 channels; per-channel standardisation) were **not
enough** — the dominance also comes from ~61 of 828 outlier **token** positions (attention
sinks). Only **per-token L2 normalisation** fixes it: top-1 eigenvalue 99.9% → 12.8%,
effective rank 1 → 91.

What changed once corrected:

| transition | raw | per-token L2 |
|---|---|---|
| L5→L6 | 0.008 | **0.990** — the "hard boundary" was pure artifact |
| L15→L16 | 0.079 | 0.855 — real, mild |
| L34→L35 | 0.993 | **0.680** — the one real boundary, which the raw version HID |

So the raw matrix simultaneously invented two boundaries and concealed the only genuine one.
The expert CKA is **unaffected** (ratio 21–46×, top-1 eigenvalue 19–34%, effective rank
7.3–13.7).

Also retired: reducing to 4 front-wide images (prefill 3073 → 828 tokens) does **not** fix
this — the problem is channels/tokens, not sequence length. It was still useful as a
robustness check: identical boundary set at both operating points, full-matrix correlation
0.9557.

---

## 7. If you build the 28-layer expert

Recommended set: **`{4, 10, 13, 15, 19, 25, 27, 34}`** — depth-aligned, no adjacent pairs
(so no super-additive penalty), deepstack reads untouched, and only 2 of 8 drawn from the
L24–35 band that the causal cache-swap sweep credited with 105% of the recoverable gap.

Survivors give `π(j)`: new expert layer *j* was teacher layer `π(j)`, reads student cache
*j*, and must absorb the teacher span `[π(j−1)+1 … π(j)]`. **20 of 28 layers are pure
identity mappings** (no span to absorb — their objective is exactly today's `L_block`); only
8 absorb one removed layer each. Two boundaries come out clean by construction:
`π(0..2) = 0,1,2` preserves the deepstack correspondence, and `π(27) = 35` means
`action_out_proj` still receives the layer it was trained on.

```
L_prune = 1/28 · Σ_j ‖ B^S_j(h^T_{a_j}; K^S_j, V^S_j) − sg h^T_{π(j)+1} ‖² / ‖h^T_{π(j)+1}‖²
          a_j = π(j−1)+1      (a_0 = 0)
```

Teacher-force the span **entry**, target the teacher's state at the span **exit**. When
`a_j = π(j)` this is bit-identical to today's `L_block`, so the existing implementation
covers 20 of the 28 terms and only the target index changes for the other 8. The teacher
side stays free: one forward with `output_hidden_states=True` yields every `h^T`.

**What breaks:** today's well-posedness rests on the expert being *identical and frozen* on
both sides, so the cache is the only difference. A pruned expert must be trainable, which
introduces a second error source and also reverses a deliberate memory decision — the frozen
expert is held outside `nn.Module` registration precisely so its 2.28 B never enters the
optimizer, the ZeRO shard, or the 68 GB checkpoints. Stage it: pruned expert against the
*teacher's* cache first (isolates block error), then the VLM against the frozen pruned
expert (which restores today's setup), then jointly only if needed. Initialise surviving
layer *j* from teacher layer `π(j)`.

**The trap:** 20 of 28 terms have targets that are already near-identity, so the loss starts
tiny and looks excellent while teaching almost nothing — the same failure as the first
`L_block`. Assert that the identity mapping reproduces current `L_block` exactly, that the
teacher's own cache gives 0, and that a **span-shuffled** mapping scores clearly worse.

**Compatibility check that makes this possible at all**: `kv_heads × head_dim = 8 × 128 =
1024` for the teacher VLM *and* Cosmos-Reason2-2B, so the cache shape works. The 2B is 28
layers × 2048 hidden, and `expert_cfg.hidden_size` is also 2048.

---

## 8. Building it: the 2B student on the pruned expert (job 457)

`ARM=block2b` — Cosmos-Reason2-2B student (28 text layers) + set-C expert (28 layers),
`L_block` with `t` sampled from the teacher's own Beta schedule, 3 epochs, step-based saves
every 500 with `save_total_limit: 10`. Configs: `configs/sft_kd_cosmos2b_prunedexpert_lcdrive.yaml`,
`configs/models/cosmos2b_prunedexpert_kd.yaml`.

**The expert is genuinely 28 layers, not 36 with stand-ins.** Its depth is *derived from the
VLM's text config*, so a 28-layer student builds a 28-layer expert automatically — and
loading the teacher's 36-layer expert state_dict into it dies on
`unexpected=['expert.layers.28...']`. Pruning at this stage is therefore a **load-time
remap** in `expert_holder._load`: teacher expert layer `π(j)` → slot `j`, dropped layers
never loaded. The identity stand-ins of §1 belong only to the *ablation* path, where the
whole 36-layer teacher runs with 8 layers bypassed.

That inverts the cache mapping relative to the ablation. The student now pairs 1:1 (its
layer *j* is expert slot *j*), and it is the **teacher's** 36-entry cache that must be
subset: slot *j* reads `t_kv[π(j)]`, the layer those weights were trained against.

**The mapping was verified causally, not assumed** (`BLOCK_PI_PROBE=1`, one real batch —
the span-shuffle self-test §7 asked for). Every candidate has identical shapes and none
would have crashed:

| teacher-cache pairing | `block_loss` |
|---|---|
| **π (set-C survivors)** | **0.4144** |
| teacher's first 28 | 2.0650 |
| teacher's last 28 | 1.9468 |
| π reversed | 2.2679 |

5× separation, so π is load-bearing and not one of many equivalent orderings.

**Operational trap, measured the hard way.** `PIN_GPUS` did not work under `srun` in either
of its two forms: a plain `export` is overwritten, and `srun --export=ALL,CUDA_VISIBLE_DEVICES=…`
is *also* discarded because slurm re-derives the device list from the step's GPU binding
afterwards. Job 456 therefore ran on physical GPUs 0–1 while its log claimed 1–2, putting a
co-tenant's card at 80.0/80 GB. Pinning now launches `torchrun` directly from the batch
script (allocation and slurm bookkeeping kept, nothing to overwrite the list), sets
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, and prints each rank's device **UUID** beside `nvidia-smi -L`.
Read that, not the `[slurm] gpus=` line.

Baselines this arm must be read against (n=1000, stitched, LCDrive val):

| | min_ade | ade |
|---|---|---|
| teacher VLM + full 36-layer expert | 0.5776 | 1.3039 |
| teacher VLM + set-C expert bypassed | 0.7893 | — |
| 4B student + teacher expert (`blockrandt` e3) | 1.6008 | 2.9885 |
| 4B student + expert tuned on it, epoch 1 | 1.4044 | 2.5840 |

---

## 9. The 2B plateau: five hypotheses, one survivor

`L_block` on the 2B converges to `block_loss` ~8.8e-4 and stops. Stitched eval at
checkpoint-3000: **min_ade 2.6444 / ade 4.8203** (n=1000, tripwires clean, mode-collapse 16.7%
which is inside the 15.6-17.8% band of every other arm). Against the teacher's 0.5776 and the
pruned expert's 0.7893 ceiling, that is a large gap, and the obvious explanations were tested
one at a time. Four failed.

| hypothesis | probe | verdict |
|---|---|---|
| compounding along LAYERS hides error from `L_block` | span n = 1/2/4/7/14 | **refuted**: amplification 1.08 -> 1.19 to n=7, then **0.88** at n=14 |
| the 10-step ODE integrates a biased field error | rollout vs per-step sum | **refuted**: endpoint error *below* the sum (0.69), though bias cos = 0.93 |
| MSE is the wrong readout (direction vs scale) | cosine decomposition | **refuted**: `mse = 2(1-cos)` to 3 digits, scale term 1000x smaller |
| the velocity head is under-weighted | `L_field` arm, lambda 2e-4 and 1e-3 | **no effect**: block and field both flat over 715 steps |
| the LR is too low to escape | 5e-5 and 1e-4, 300 steps | **refuted**: both DAMAGED it (1.25e-3 / 1.44e-3 vs 8.8e-4) |
| aggregate capacity | single-clip overfit | **survives**: 1.85e-4, **5x below** the training floor, still falling |

**The span curve is the interesting one.** Amplification rises with span length only to n = 7 and
then falls below 1, because the two halves differ: layers 0-13 amplify (1.23) and layers 14-27
**contract** (0.79). So the deep half damps what the shallow half amplifies.

**And it corrects a conclusion recorded earlier in this file's history.** The "32x" free-running
ratio was read as evidence that compounding dominates. It is `fr_last / tf_MEAN`, so it is
~L x 1: measured `fr_last / (L x tf_mean)` is **0.74** (4B) and **0.92** (2B). Errors accumulate
ADDITIVELY along depth, and `L_block` already minimises that sum. Only the ~14% amplification
was ever invisible to it.

**Why "small loss, bad model" is not a paradox.** Every stage on one scale (2B, RMS):

| stage | RMS | factor |
|---|---|---|
| per-layer block output (what `L_block` trains) | 3.2% | - |
| accumulated over 28 layers | 18.5% | x5.8 |
| after `action_out_proj` -> velocity | 30% | x1.6 |
| after 10 Euler steps -> action | 50% | x1.7 |

3.2% per layer becomes a 50% endpoint error. Calibrating min_ade against final-layer RMS
(teacher 0%/0.5776, 4B 11.4%/1.6008, 2B 18.5%/2.6444) gives ~0.09-0.11 min_ade per 1% RMS, so
reaching min_ade ~1.0 needs the per-layer error **~13x lower** -- which no reweighting delivers.

**The cache is not "mis-rotated" either.** A closed-form ridge fit from the student's pre-RoPE
K/V to the teacher's at pi(j), scored on held-out clips, cuts the raw distance 20x (K) and 13.5x
(V) but leaves 0.37 / 0.73 where predicting zero scores 1.0. Meanwhile the same checkpoint
matches the expert's *block outputs* to 1.14e-3. The student found a cache that is 744%
different from the teacher's in raw terms and functionally equivalent to 0.1% -- so raw-cache
equality is neither necessary nor achievable, and an adapter fitted to K/V distance would
optimise something the expert ignores. (This is also a mechanism for `L_KV` losing to `L_block`
by -0.752, z = -11.53, though that was measured on the 4B, not here.)

**What did work: train the expert on the student's cache** (`sft_expert_on_student_2b_lcdrive`),
1 epoch, VLM frozen, 1.77 B trainable:

| | min_ade | ade |
|---|---|---|
| 2B student + frozen teacher expert | 2.6444 | 4.8203 |
| **2B student + expert trained on it** | **2.3668** | **3.8103** |

Paired n=1000: min_ade **-0.2776, z = -7.51**, better on 63.0% of clips; ade **-1.0099,
z = -13.42**, better on 72.7%. That closes 13.4% of the gap to the teacher (the 4B's version
closed 19%, but this is larger in absolute terms: -0.278 vs -0.196). Note the `ade` gain (-21%)
is twice the `min_ade` gain (-10.5%) -- the typical draw improved more than the best-of-6,
which is the metric deployment actually gets.

### Deployment latency (H100, bs=1, sdpa, bf16, model only)

| cameras x 4 frames | tokens | 2B prefill | 2B step | 2B total | 4B prefill | 4B step | 4B total |
|---|---|---|---|---|---|---|---|
| 4 | 3073 | 90.7 | 14.9 | **240.2** | 131.6 | 21.6 | **347.7** |
| 3 (L/wide/R) | 2324 | 70.2 | 16.1 | 231.2 | 101.5 | 18.1 | 282.5 |
| 2 (wide+tele) | 1577 | 50.0 | 14.4 | 194.1 | 69.5 | 18.1 | 250.9 |
| 1 (wide) | 828 | 30.6 | 14.6 | 176.9 | 40.6 | 18.1 | 221.3 |

Prefill is linear in tokens (2B 0.0295 ms/tok, 4B 0.0428). The expert step is **flat** in camera
count -- it is 28 (or 36) blocks over 64 action tokens, not a function of cache length -- so the
10 denoising steps are a 150-215 ms floor that no camera reduction touches, and are 60-65% of
total latency everywhere. At 4 cameras the 2B buys 107 ms (-31%) for +0.96 min_ade. Halving
`num_inference_steps` would save ~108 ms on the 4B alone, i.e. the same latency the entire 2B
distillation delivers.

---

## 9. Lessons

1. **Similarity metrics did not predict causal damage.** Three refutations: `kvband`
   (reweighting on marginal cache-swap importance, +0.5451, z = +11.22), prune A, prune B.
2. **Structural constraints did.** Depth alignment, deepstack, and super-additivity are
   properties of the wiring, not of any statistic, and they cost almost nothing in Σ BI.
3. **Check the Gram's rank before trusting CKA on an LLM.** A `1.000` plateau is the
   signature of rank-1 domination, not similarity.
4. **A falling loss is not evidence.** Verified via a self-test against the published
   reference (agreement 1e-15, `CKA(X,X) = 1`) after an earlier layer analysis in this
   project measured nothing while looking healthy.
5. **Check the diagonal.** The first span-BI matrix had a diagonal of 0.9972 instead of 1.0 —
   float32 accumulating 8.4M terms in one register, losing ~0.3%, the same magnitude as the
   values being measured. Contract over the feature dimension only, average in float64.
