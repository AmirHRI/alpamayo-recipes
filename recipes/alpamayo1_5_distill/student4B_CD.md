# 4B student — endpoint consistency distillation (CD) on the EoS expert

Does endpoint distillation let the 4B 2-camera student run at **1 denoising step** without
losing trajectory quality? Answer: **yes at NFE=1, nothing at NFE=2, and it costs a little at
NFE=10.** The effect is monotonic in the step budget and changes sign.

- **Student**: 4B Qwen3-VL, 2 cameras `[1,3]`, nav-conditioned, event-anchored `t0`.
- **Baseline (EoS)**: `output_eos_4b_2cam_nav_lcdrive/checkpoint-3126` — expert trained on the
  frozen student's own cache, no CD.
- **CD**: `output_cd_eos4b_endpoint_gt05_2cam_nav_e2_bs32/checkpoint-{1563,3126}` — the EoS
  checkpoint above, further trained with
  `configs/sft_cd_eos_4b_endpoint_gt05_2cam_nav_lcdrive.yaml`
  (`cd_weight: 0`, `x0_teacher_weight: 1.0`, `x0_gt_weight: 0.5`, cached 10B rollouts).
- Val: `nav_lcdrive_val_mysubset_1k`, n=1000, 6 samples/clip, 0 malformed on every run.

## 0. ⚠️ Comparability

Everything below is one student, one val set, one sampler config; **only the checkpoint and
`inference_step` vary**, and all rows are paired on `clip_id`. The `COMPARE_EVAL.md` rules still
apply — these are action-expert, 2-camera, event-anchored numbers and are not comparable to the
token-head, 4-camera or default-keyframe tables there.

⚠️ **NFE is a confound, not a constant.** Lowering the step budget improves `ade` on its own,
with no training at all: the EoS baseline goes 1.9588 → 1.8380 from NFE=10 → 2 purely because
fewer Euler steps contract samples toward the conditional mean. An earlier reading of this
experiment credited endpoint distillation with a −0.11 `ade` win that was **entirely** this
effect. Every CD row must be compared against the EoS row **at the same NFE**, which is why the
NFE=1 and NFE=2 baselines were run.

⚠️ Two different directories both contain a `checkpoint-3126` (both runs were 2 × 1563 steps).
They are structurally identical — 1130 tensors, 36-layer expert — so pointing at the wrong one
does **not** error, it silently gives you the control. Always carry the full path.

## 1. `ade` and `min_ade` @ NFE 1 / 2 / 10

n=1000, paired. `ade` = single draw; `min_ade` = best-of-6 (oracle).

| NFE | EoS `ade` | CD ep1 `ade` | CD ep2 `ade` | EoS `min_ade` | CD ep1 `min_ade` | CD ep2 `min_ade` |
|---|---|---|---|---|---|---|
| **1** | 2.0500 | 1.8193 | **1.8141** | 1.7640 | 1.4256 | **1.4083** |
| **2** | 1.8380 | 1.8442 | 1.8358 | 1.3460 | 1.2089 | **1.1912** |
| **10** | 1.9588 | 2.0218 | 2.0077 | **1.0510** | 1.0931 | 1.0608 |

### CD ep2 − EoS, paired, same NFE

| NFE | Δ`ade` | t | Δ`min_ade` | t |
|---|---|---|---|---|
| **1** | **−0.2358** | −8.6 | **−0.3557** | −16.9 |
| 2 | −0.0022 | −0.1 **ns** | −0.1548 | −9.0 |
| 10 | +0.0489 | +2.1 | +0.0099 | +0.6 **ns** |

**The sign flips with the step budget.** Large win at 1 step, nothing at 2, small regression at
10. This is what the objective was built to do: the loss is
`|| x_hi + tau_hi * v(x_hi, tau_hi) − teacher_endpoint ||²`, and a 1-step Euler solve from τ=1 is
`x + 1.0 · v(x, 1.0)` — **the same computation**. Training and 1-step inference coincide; at
NFE=10 the sampler does something the objective never optimized. The training loss predicted
this: `x0_loss_noise` fell 28% (0.1897 → 0.1369) while `x0_loss_mid` stayed flat.

## 2. Deployment view

`E[1 draw]` is the expected error of a single sampled trajectory — the number that matters if
production draws once. `min_ade` and `corner_distance` are **oracle** metrics (both take
`.min()` over samples) and flatter any model that merely samples more widely.

| arm | `E[1 draw]` | worst-of-6 | diversity |
|---|---|---|---|
| EoS @1 | 2.0351 | 2.3557 | 0.2617 |
| CD ep1 @1 | 1.8182 | **2.2666** | 0.3934 |
| **CD ep2 @1** | **1.8131** | 2.2745 | 0.4045 |
| EoS @2 | 1.8387 | 2.3723 | 0.4417 |
| CD ep2 @2 | 1.8544 | 2.6104 | 0.6725 |
| EoS @10 | 1.9728 | 3.1348 | 0.8703 |
| CD ep2 @10 | 2.0211 | 3.1922 | 0.9513 |

**CD ep2 @ NFE=1 is the best operating point measured and simultaneously the cheapest.**

| vs | Δ`E[1 draw]` | t | Δworst-of-6 | t |
|---|---|---|---|---|
| EoS @1 | −0.2221 | −9.5 | −0.0813 | −2.8 |
| EoS @2 | −0.0256 | −1.3 **ns** | −0.0978 | −3.9 |
| EoS @10 | −0.1597 | −6.6 | −0.8603 | −20.4 |

It ties the EoS NFE=2 baseline on `E[1 draw]` while being significantly better in the tail, at
**half** the compute; and beats the NFE=10 baseline on both at **1/10** the compute.

⚠️ **Mechanism is diversity, and the mode-collapse tripwire cannot see it.** CD raises sample
spread at every budget (+0.13 at NFE=1, t=+29.5), which is what drives the `min_ade` gains. The
`all-6-samples-identical` counter moved only 16.0% → 15.6% and would have reported "no change".
Absolute diversity at NFE=1 (0.405) is still less than half the NFE=10 baseline's (0.870), so
**any consumer that genuinely needs multi-modality is degraded by the cheap configs regardless
of arm.**

## 3. Epoch 2 vs epoch 1 — converged

| NFE | Δ`ade` | t | Δ`min_ade` | t |
|---|---|---|---|---|
| 1 | −0.0051 | −2.7 | −0.0173 | −10.1 |
| 2 | −0.0084 | −2.2 | −0.0177 | −5.8 |
| 10 | −0.0141 | −3.8 | −0.0323 | −10.6 |

Every delta is significant and every delta is negligible — significance comes from n=1000 and
tight pairing, not magnitude. Epoch 2 does retire the NFE=10 `min_ade` regression (+0.0421,
t=+2.5 at ep1 → +0.0099, **ns** at ep2), so it is a strict improvement. A third epoch is not
worth 6.5 h.

## 4. Reproduce

Training (~6.5 h, 4 GPUs). `RESUME=auto` continues an interrupted run; job 20661 died at 66% to
an unexplained NCCL rank-0 watchdog hang and was finished by 20663.

```bash
sbatch --nodelist=amhrisvh200b recipes/alpamayo1_5_distill/slurm_train_cd_eos_4b.sh
RESUME=auto WORKERS=8 sbatch --nodelist=amhrisvh100b recipes/alpamayo1_5_distill/slurm_train_cd_eos_4b.sh
```

Eval. **`EOS_DIR_OVERRIDE` is what keeps this honest** — the CD checkpoint is scored through the
*same* config, sampler and code path as the EoS baseline, so the only variable is the weights.

```bash
cd recipes/alpamayo1_5_distill
CD=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/output_cd_eos4b_endpoint_gt05_2cam_nav_e2_bs32
for n in 1 2 10; do
  MODEL=4b EOS_DIR_OVERRIDE=$CD TAG_BASE=cd_eos4b_gt05 CKPT=checkpoint-3126 NFE=$n \
    sbatch --nodelist=amhrisvh200b ./slurm_eval_eos.sh          # CD
  MODEL=4b CKPT=checkpoint-3126 NFE=$n \
    sbatch --nodelist=amhrisvh200b ./slurm_eval_eos.sh          # EoS control
done
```

## 5. Serving

CD changes **weights only**. `ConsistencyExpertVLA` subclasses `KaVaExpertTeacher` and defines
no sampling method — all CD machinery lives in `forward()`, the training path — so inference
loads the checkpoint into the ordinary `StitchedAlpamayoR1.from_stitch` exactly like any EoS
checkpoint. `consistency_expert.py`, the teacher rollout cache and the endpoint loss are
build-time only.

```
/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/
    output_cd_eos4b_endpoint_gt05_2cam_nav_e2_bs32/checkpoint-3126     # inference_step: 1
```

22 GB on disk, of which `optimizer.pt` is 9.1 GB; the servable model is the 3 safetensors shards
+ index + `config.json` (~14.3 GB). The checkpoint feeds **both** loader paths —
`evaluate.eval_ckpt` (714 VLM tensors) and `model.teacher_checkpoint_path` (397 expert tensors).

Raw per-clip JSON and `pred_xyz`/`gt_xyz` `.npz` archives for all nine cells are in `training/`
as `{eos_4b_2cam_nav,cd_eos4b_gt05}_eos_checkpoint-*{,_nfe1,_nfe2}.{json,npz}`, so any metric
here can be recomputed offline. ⚠️ `compute_minade` uses `only_xy=True`; scoring the archives in
3D gives 1.2392 where the logs say 1.1714.
