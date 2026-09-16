# Deployment latency: teacher vs 4B vs 2B student, by camera count

Closed-loop planning latency of the Alpamayo-1.5-10B teacher and the two distilled students.
Each student is paired with **its own action expert trained on its cache** (the
`expert_on_student` arm — the deployable configuration, not the frozen-teacher-expert one).

Produced by `scripts/latency_profile.py`. All three tables were measured on the **same idle
H100** back to back, so they are directly comparable.

```
CUDA_VISIBLE_DEVICES=1 python -m alpamayo1_5_distill.scripts.latency_profile \
  --config-path pkg://alpamayo1_5_distill/configs \
  --config-name sft_eval_stitched_2b_prunedexpert_lcdrive \
  ++model.attn_implementation=sdpa \
  ++model.checkpoint_path=<eos ckpt> ++model.teacher_checkpoint_path=<eos ckpt> \
  ++lat.n_warmup=3 ++lat.n_timed=10
```

## What is measured

| phase | what it is | scales with |
|---|---|---|
| **prefill** | one VLM forward over the whole prompt → the KV cache | **tokens** (~93% of them are vision) |
| **1 step** | `action_in_proj` → 28/36 expert blocks over 64 action tokens attending the cached prefix → `action_out_proj` | expert **depth**, not cache length |
| **10 steps** | one trajectory = `num_inference_steps` = 10 Euler steps | — |

Batch 1, bf16, `sdpa`, 3 warmup + 10 timed iterations, **CUDA-synced**. ⚠️ Without the sync the
first phase reports near-zero while its kernels are still queued behind the launch.

**Model only.** No image decode, no preprocessing, no postprocessing. The 16-frame video decode
is excluded and is not negligible on a real deployment path — it is what starved the training
dataloader (2 workers → 21.7 s/it, 16 workers → 10.3 s/it) and it scales directly with camera
count in a way these numbers do not capture.

⚠️ Cameras are sliced **before** preprocessing and the prompt is rebuilt. The text carries one
placeholder run per image, so dropping images without re-running the processor would leave the
token count disagreeing with the pixel values.

⚠️ Camera indices are the **loader's `camera_features` order** — `0 cross_left, 1 front_wide,
2 cross_right, 3 front_tele` — *not* the global camera-index table where front_tele is 6.

Input images are 1080×1920, resized into the teacher's pixel budget (`min_pixels` 163,840 /
`max_pixels` 196,608, patch 16 / merge 2) → **192 tokens per image**. That budget lives only in
the teacher's config and is inherited by both students, which is what keeps their token counts
identical position-for-position.

---

## Teacher — Alpamayo-1.5-10B (Cosmos-Reason2-8B tower + full 36-layer expert)

`models--nvidia--Alpamayo-1.5-10B-A1-format` — min_ade **0.5776**, ade **1.3039**. The ceiling
every student number in this tree is measured against.

⚠️ Profiled with `PRUNE_EXPERT_LAYERS` **unset**. `from_teacher` calls `_apply_expert_pruning`
(unlike `from_stitch`), so a stray pin would bypass 8 of the 36 layers and profile the ablation
instead of the teacher. The log confirms `expert 36 layers` and prints no `PRUNED` line.

| cameras × 4 frames | images | tokens | prefill | 1 step | 10 steps | **total** |
|---|---|---|---|---|---|---|
| 4 (front-left, front-wide, front-right, front-tele) | 16 | 3073 | 187.7 ms | 19.2 ms | 191.8 ms | **379.5 ms** |
| 3 (front-left, front-wide, front-right) | 12 | 2324 | 144.0 ms | 18.0 ms | 180.4 ms | **324.5 ms** |
| 2 (front-wide, front-tele) | 8 | 1577 | 98.7 ms | 18.1 ms | 180.7 ms | **279.4 ms** |
| 1 (front-wide) | 4 | 828 | 54.6 ms | 18.1 ms | 180.9 ms | **235.5 ms** |

---

## 2B student + tuned 28-layer expert

`output_eos_2b_lcdrive/checkpoint-1598` — min_ade **2.3668**, ade **3.8103**.

| cameras × 4 frames | images | tokens | prefill | 1 step | 10 steps | **total** |
|---|---|---|---|---|---|---|
| 4 (front-left, front-wide, front-right, front-tele) | 16 | 3073 | 90.7 ms | 14.9 ms | 149.5 ms | **240.2 ms** |
| 3 (front-left, front-wide, front-right) | 12 | 2324 | 70.2 ms | 16.1 ms | 161.0 ms | **231.2 ms** |
| 2 (front-wide, front-tele) | 8 | 1577 | 50.0 ms | 14.4 ms | 144.1 ms | **194.1 ms** |
| 1 (front-wide) | 4 | 828 | 30.6 ms | 14.6 ms | 146.2 ms | **176.9 ms** |

## 4B student + tuned 36-layer expert

`output_expert_on_student_lcdrive/checkpoint-1598` — min_ade **1.4044**, ade **2.5840**.

| cameras × 4 frames | images | tokens | prefill | 1 step | 10 steps | **total** |
|---|---|---|---|---|---|---|
| 4 (front-left, front-wide, front-right, front-tele) | 16 | 3073 | 131.6 ms | 21.6 ms | 216.1 ms | **347.7 ms** |
| 3 (front-left, front-wide, front-right) | 12 | 2324 | 101.5 ms | 18.1 ms | 181.0 ms | **282.5 ms** |
| 2 (front-wide, front-tele) | 8 | 1577 | 69.5 ms | 18.1 ms | 181.4 ms | **250.9 ms** |
| 1 (front-wide) | 4 | 828 | 40.6 ms | 18.1 ms | 180.7 ms | **221.3 ms** |

---

## Reading the three together

| cameras | teacher | 4B student | 2B student |
|---|---|---|---|
| 4 | 379.5 ms | 347.7 ms | 240.2 ms |
| 3 | 324.5 ms | 282.5 ms | 231.2 ms |
| 2 | 279.4 ms | 250.9 ms | 194.1 ms |
| 1 | 235.5 ms | 221.3 ms | 176.9 ms |
| **min_ade** | **0.5776** | **1.4044** | **2.3668** |

**The 4B saves only 9% over the teacher.** 347.7 vs 379.5 ms at 4 cameras, for +0.83 min_ade
(a 143% regression). That is the uncomfortable headline: distilling 10B → 4B costs most of the
accuracy and buys almost none of the latency, because the two models share an identical
36-layer expert whose 10 denoising steps are ~55% of the teacher's total and ~62% of the 4B's.
The 2B is the only configuration that moves latency materially (−37% vs teacher), and it gives
up 1.79 min_ade to do it.

**Prefill is linear in tokens, and only prefill differs between the towers:**

| model | ms/token | 4-cam prefill |
|---|---|---|
| teacher (8B) | 0.0611 | 187.7 ms |
| 4B student | 0.0428 | 131.6 ms |
| 2B student | 0.0295 | 90.7 ms |

**The expert step is flat in camera count and identical across the 36-layer models** — teacher
18.0–19.2 ms, 4B student 18.1–21.6 ms, 2B (28 layers) 14.4–16.1 ms. It is depth-bound, not
cache-bound: the block runs 64 action-token queries whatever the prefix length. The 28/36 ratio
predicts 1.29×, measured 1.24×.

**Denoising dominates every configuration**: 51–77% of total latency. At 1 camera it is *77%*
of the teacher's 235 ms. No camera reduction touches it.

---

## Accuracy vs camera count — teacher, n=1000

The latency tables above cannot say what dropping cameras *costs*. This measures it:
Alpamayo-1.5 on the same LCDrive val 1k subset, **prefill-only**, 6 samples per clip, metrics
from the repo's own `DistanceMetrics`.

| setting | cameras | min_ade | ade | mean_ade | max_ade | vs 4cam (min_ade) | z | worse on |
|---|---|---|---|---|---|---|---|---|
| **4cam** | L, wide, R, tele | **0.5918** | 1.3143 | 1.3141 | 2.2993 | — | — | — |
| **3cam** | L, wide, R | **1.3647** | 2.3399 | 2.3018 | 3.6485 | +0.7729 | 14.2 | 75.0% |
| **2cam** | wide, tele | **0.6981** | 1.6822 | 1.6286 | 3.0040 | +0.1063 | 4.4 | 65.6% |
| **1cam** | wide | **1.6798** | 3.2087 | 3.1938 | 5.3636 | +1.0879 | 18.9 | 88.0% |

`min_ade` = best of 6 (oracle), `ade` = the selected draw, `mean_ade` / `max_ade` = mean and
worst of the 6, recomputed from the saved trajectories with `compute_ade` (XY only, as the repo
defines it). Self-check: recomputing best-of-6 from the saved trajectories gives 0.5919 against
the harness's 0.5918.

**Harness validated**: 4cam lands at 0.5918 against the established 0.5776 from the standard
eval — +0.014 on 1000 clips, ~1.3σ of the per-clip diffusion-noise spread. Same measurement.

**2 cameras beat 3 — the telephoto is what matters.** Dropping *both* side cameras costs
+0.106; dropping *only* the telephoto costs +0.773, seven times more. The horizon breakdown
shows why:

| setting | 0.5 s | 1 s | 3 s | 5 s |
|---|---|---|---|---|
| 4cam | 0.0089 | 0.0270 | 0.1778 | 0.3864 |
| 2cam (wide+tele) | 0.0112 | 0.0336 | 0.2054 | 0.4453 |
| 3cam (no tele) | 0.0124 | 0.0433 | 0.3372 | **0.8550** |
| 1cam (wide only) | 0.0184 | 0.0612 | 0.4290 | **1.0531** |

Losing the telephoto more than doubles the 5 s error (0.386 → 0.855) and barely moves 0.5 s —
long-horizon prediction needs the distant detail only the 30° FOV camera resolves.

**Latency and accuracy together, teacher:**

| setting | latency | min_ade | trade |
|---|---|---|---|
| 4cam | 379.5 ms | 0.5918 | baseline |
| **2cam** | **279.4 ms (−26%)** | **0.6981 (+18%)** | **best exchange rate measured in this tree** |
| 3cam | 324.5 ms (−14%) | 1.3647 (+131%) | dominated by 2cam on both axes |
| 1cam | 235.5 ms (−38%) | 1.6798 (+184%) | — |

For comparison, distilling 10B → 4B buys 8% latency for +143% min_ade. **Dropping to
front-wide + telephoto is a far better trade than anything the distillation line produced**, and
3cam is strictly worse than 2cam on *both* axes — it is never the right choice.

⚠️ These are **starved** numbers, not achievable ones. The model was trained with all four
cameras present and named in fixed order, so reduced-camera prompts are out of distribution
(verified: the prompt text is adaptive, so a dropped camera loses its label, its four `frame N`
tags and its four vision blocks). A fine-tune at 2 cameras would likely beat 0.6981. The side
cameras' low value also reflects this val subset; they plausibly matter in tight turns, cut-ins
and intersections that it underrepresents.

## Accuracy vs camera count — teacher with the set-C PRUNED expert, n=1000

Same sweep, `PRUNE_EXPERT_LAYERS=4,10,13,15,19,25,27,34` (8 of 36 layers replaced by identity
stand-ins, indices preserved).

| setting | cameras | min_ade | ade | mean_ade | max_ade | vs full expert (min_ade) | z | worse on |
|---|---|---|---|---|---|---|---|---|
| **4cam** | L, wide, R, tele | **0.7950** | 1.7025 | 1.6236 | 2.7506 | +0.2032 | 9.9 | 69.1% |
| **3cam** | L, wide, R | **1.4527** | 2.6970 | 2.6565 | 4.4272 | +0.0880 | 2.9 | 62.3% |
| **2cam** | wide, tele | **1.1871** | 2.4241 | 2.4268 | 4.3508 | +0.4890 | 15.7 | 75.0% |
| **1cam** | wide | **3.9363** | 7.8732 | 7.9155 | 12.8343 | +2.2566 | 17.2 | 80.9% |

Harness check: 4cam lands at 0.7950 against the established 0.7893 for this ablation.

## Accuracy vs camera count — 4B student + its tuned 36-layer expert, n=1000

`output_expert_on_student_lcdrive/checkpoint-1598` — harness check 1.4066 vs established 1.4044.

| setting | min_ade | ade | mean_ade | max_ade | vs 4cam | z |
|---|---|---|---|---|---|---|
| 4cam | 1.4066 | 2.5667 | 2.5337 | 4.0813 | — | — |
| 3cam | 2.0997 | 3.5659 | 3.5263 | 5.5996 | +0.6930 | 11.5 |
| 2cam | 1.6512 | 3.2990 | 3.3933 | 5.9904 | +0.2445 | 5.2 |
| 1cam | 2.5844 | 4.7588 | 4.7977 | 8.4451 | +1.1778 | 17.2 |

## Accuracy vs camera count — 2B student + its tuned 28-layer expert, n=1000

`output_eos_2b_lcdrive/checkpoint-1598` — harness check 2.3611 vs established 2.3668.

| setting | min_ade | ade | mean_ade | max_ade | vs 4cam | z |
|---|---|---|---|---|---|---|
| 4cam | 2.3611 | 3.7980 | 3.8578 | 6.1085 | — | — |
| 3cam | 2.4470 | 4.1863 | 4.2215 | 7.1395 | +0.0859 | 2.2 |
| 2cam | 2.7861 | 4.8798 | 4.7743 | 8.2385 | +0.4250 | 7.6 |
| 1cam | 4.9886 | 9.6678 | 9.7865 | 16.0566 | +2.6275 | 17.4 |

## Where the results are saved

`/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/camsweep/`

| file | contents |
|---|---|
| `teacher_<setting>.npz` | `clip_ids` [1000], `pred_xyz` [1000, 1, 6, 64, 3] fp16 (**all 6 samples**), `gt_xyz` [1000, 64, 3], `cameras`, `description`, `min_ade`, `ade` |
| `teacher_<setting>.json` | per-clip metrics keyed by clip UUID, same format as the other evals in `training/` so it joins the notebook tooling |
| `teacherPruneC_<setting>.npz` | same fields, set-C pruned expert |
| `teacherPruneC_<setting>.json` | same fields, set-C pruned expert |
| `eos4b_<setting>.npz` / `.json` | same fields, 4B student + its tuned 36-layer expert |
| `eos2b_<setting>.npz` / `.json` | same fields, 2B student + its tuned 28-layer expert |

`<setting>` is `4cam` / `3cam` / `2cam` / `1cam`. Trajectories are saved so any further metric
(corner distance, per-horizon, per-category, mode diversity) or plot needs no model re-run —
`max_ade` and `mean_ade` above were computed this way. Regenerate with
`scripts/eval_camera_sweep.py`.

## Consequences

1. **Cutting denoising steps is by far the cheapest win, and it applies to the TEACHER too.**
   10 → 5 steps saves ~96 ms on the teacher, ~108 ms on the 4B — more than distillation to 4B
   delivers (32 ms), at zero accuracy cost to the VLM. One eval determines what the sampler
   tolerates. This should be tested before any further distillation work.
2. **Neither student's exchange rate is good as it stands.** vs the teacher: the 4B buys 32 ms
   (−8%) for +0.83 min_ade; the 2B buys 139 ms (−37%) for +1.79. See `PRUNING.md` §9 for why the
   2B's cache is stuck and which five fixes were refuted.
3. **Drop to 2 cameras (front-wide + telephoto) before anything else.** Measured on the
   teacher: −26% latency for +18% min_ade, and it strictly dominates 3cam. See the accuracy
   table above. Untested on the students -- `scripts/eval_camera_sweep.py` takes any of them.
