# Two-Step EoS to One-Step Consistency

This is a new experiment, not a measured performance claim. Its target is the
**two-step sampler of the best cotrained 4B EoS model**, not the full 10B teacher,
ground truth, or the previous ten-step rollout cache.

Initialization for the online student, EMA target, and frozen solver expert:

```text
/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/
  output_eos_cotrain_4b_all36_lr1x_2cam_nav_framecache/checkpoint-6876
```

The checkpoint's cotrained VLM is loaded and then frozen. All expert branches read
the same cache: cameras `[1,3]`, event-anchored training manifest, stripped navigation
text, camera IDs and frame numbers. Only the expert and action projections train.

## Objective

Use repository time $s$, with noise at $s=0$. For fixed EoS weights $\phi$ and fresh
noise $\epsilon$, compute the actual two-step Euler path online:

$$
x_{1/2}=\epsilon+\tfrac12 v_\phi(\epsilon,0),\qquad
x_1=x_{1/2}+\tfrac12 v_\phi(x_{1/2},1/2).
$$

Define the consistency output as
$f_\theta(x,s)=x+(1-s)v_\theta(x,s)$, so $f_\theta(x,1)=x$ exactly.
Sample either path edge with probability 1/2 per row:

$$
L_\mathrm{upper}=\|f_\theta(\epsilon,0)-
\operatorname{sg}[f_{\bar\theta}(x_{1/2},1/2)]\|^2,
\qquad
L_\mathrm{lower}=\|f_\theta(x_{1/2},1/2)-\operatorname{sg}[x_1]\|^2.
$$

The lower edge is the consistency boundary anchor, not an extra GT or cached
endpoint loss. The upper edge bootstraps through the EMA target. Internally the
existing loss uses reversed time `tau=1-s` and an algebraically equivalent fp32
velocity residual to avoid cancellation.

At a zero-loss fixed point with EMA equal to the student, both consistency outputs
equal this frozen EoS model's two-step endpoint for the **same noise draw**. This is
a representable target condition, not a guarantee that optimization will reach it.
Unlike setting the old `m_rungs=2`, the midpoint comes from a real teacher rollout,
not interpolation between GT and noise.

Keep the solver expert **fixed**. An EMA teacher that also generates the solver
path would continually redefine the two-step reference as the student changes.
EMA belongs to the consistency target here: fp32 shadow, decay 0.99, updated once
per optimizer step, initialized from checkpoint-6876. The decay is a starting
hyperparameter, not a measured optimum. No EMA warmup is used.

No step-count conditioning is added. This experiment specializes the existing
head for one-step generation, whose Euler output is already `f(noise, s=0)`.
Preserving the original two-step velocity field as well would be a separate
multi-budget objective; running this CM for two steps is not guaranteed to retain
the original EoS NFE=2 performance.

## Training

Config: [configs/sft_cd_eos_4b_consistency2to1_2cam_nav_lcdrive.yaml](configs/sft_cd_eos_4b_consistency2to1_2cam_nav_lcdrive.yaml).
The existing endpoint configs and launcher are unchanged.

```bash
cd /home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
SMOKE=1 sbatch --nodelist=amhrisvh200b slurm_train_cd_eos_4b_consistency2to1.sh
```

The smoke run uses a separate output directory, two optimizer steps, the same
BS=4/accumulation=2 as training, no W&B, and a checkpoint save at step 2. Inspect
finite loss, nonzero EMA liveness, restored live weights after saving, and saved
weights matching EMA to serialization precision. It still loads the full models
and writes a full checkpoint. H200 is the conservative first smoke target: this
path adds a frozen expert and an fp32 EMA shadow, so the old endpoint run's memory
measurements do not establish that it fits on an 80 GB GPU.

After the smoke passes:

```bash
sbatch --nodelist=amhrisvh200b slurm_train_cd_eos_4b_consistency2to1.sh
```

Defaults: four GPUs, effective batch 32, LR 2e-5, 430 warmup steps, two epochs,
epoch checkpoint saves. On the full 109,997-anchor dataset expect checkpoints
3438 and 6876; verify the actual dataset count and trainer step count. Each batch
uses one VLM prefill, two frozen teacher forwards, an EMA forward when needed,
and one gradient-bearing student forward.

`WORKERS`, `EOS_CKPT`, `OUTPUT_DIR`, and whitespace-separated `EXTRA_ARGS` are
supported. A nonempty output directory is refused. Exact resume is deliberately
unsupported: the existing trainer saves deployment EMA weights but does not
persist both online weights and EMA state for an exact continuation. Do not inject
`trainer.resume_from_checkpoint` through `EXTRA_ARGS` as a workaround.

Evaluate `checkpoint-*`, not the root output directory: the existing final
`save_model` path is not the EMA-swapped checkpoint-save path.

## Evaluation

Run the same 1,000-clip validation set with K=6 and identical RNG settings. The
reference is **this EoS checkpoint at NFE=2**, not the 10B teacher at NFE=1.

```bash
OUT=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
MODEL=4b NFE=2 CKPT=checkpoint-6876 TAG_BASE=eos4b_cm_reference \
  EOS_DIR_OVERRIDE="$OUT/output_eos_cotrain_4b_all36_lr1x_2cam_nav_framecache" \
  EXTRA_ARGS='++data.val_dataset.strip_nav_turn_distance=true' \
  sbatch slurm_eval_eos.sh

MODEL=4b NFE=1 CKPT=checkpoint-6876 TAG_BASE=cm4b_2to1 \
  EOS_DIR_OVERRIDE="$OUT/output_cd_eos4b_consistency2to1_2cam_nav" \
  EXTRA_ARGS='++data.val_dataset.strip_nav_turn_distance=true' \
  sbatch slurm_eval_eos.sh
```

Also evaluate CM epoch 1 by changing `CKPT` to `checkpoint-3438`. First run with
`MAX_EVAL_STEPS=5` to check loading and shapes. Both the VLM and expert must load
from the selected checkpoint, with 36 expert layers and no pruning.

Join archives by clip ID and verify identical GT before scoring. Use the XY
decomposition in [CENTER_GAP.md](CENTER_GAP.md): centre, medoid, minADE@6 and spread.
The recorded EoS NFE=2 reference is centre **1.4035**, medoid **1.4061**,
minADE@6 **1.0012**, spread **0.4072**; a fresh matched evaluation takes precedence.
Report paired differences and uncertainty, and choose an acceptable non-inferiority
margin before declaring that one step matches two. An insignificant difference is
not proof of equivalence. Lower minADE with inflated spread, or lower single-draw
ADE with collapsed spread, is not sufficient. NFE halves; total inference latency
does not necessarily halve because VLM prefill is unchanged.

## Precision Probes

The original run used bf16 trainable parameters and bf16 Adam moments. Its fp32
EMA shadow is not an optimizer master copy. To investigate rounding-limited
updates, `model.expert_precision` accepts two opt-in modes:

| Mode | Online/EMA expert computation | Trainable weights and Adam moments | Frozen VLM/solver |
|---|---|---|---|
| `fp32` | fp32, autocast disabled | fp32 | original bf16 |
| `fp16` | fp16 autocast with Trainer GradScaler | fp32 | original bf16 |

The default `checkpoint` mode preserves the previous behavior. The shared cache
and additive attention mask are converted for the online/EMA expert and cache
references are restored afterward. The fixed solver continues using the original
cache precision. EMA skips updates when GradScaler skips an optimizer step.

```bash
PRECISION=fp32 sbatch slurm_probe_cd_precision.sh
PRECISION=fp16 sbatch slurm_probe_cd_precision.sh
```

Submitted on 2026-09-14 as jobs 21053 and 21054, respectively. Each starts from the
original cotrained EoS checkpoint-6876, not a CM checkpoint. These diagnostics use
one GPU, batch 1, no accumulation, 16 steps, fixed LR 2e-5, no warmup, and the
0-120 chunk subset. TF32 and W&B are disabled. No node is pinned. They are not
epoch-matched experiments or evidence of ADE improvement.

The callback checks fp32 trainable weights and Adam moments, requires a scaler
for fp16, and reports successful/skipped steps, scale, a representative weight's
changed fraction and update RMS, and peak allocated GPU memory. A checkpoint is
saved at step 16 using the existing EMA-save path. Inspect these diagnostics and
EMA liveness before launching longer matched experiments. Batch-1 memory success
does not establish that the original four-GPU, batch-4 recipe fits.

### Initial Probe Results

Both jobs exited 0 on H100c, but only the fp32 run is a valid initial diagnostic.

| Job | Mode | Result |
|---|---|---|
| 21053 | fp32 | 16 updates, zero skips, peak allocated 57.11 GiB; saved probe tensor equals fp32 EMA exactly |
| 21054 | fp16 AMP | INVALID comparison: expert updates absent on 9/16 steps despite zero scaler skips |

The fp32 probe changed 99.947%-99.985% of the monitored layer-0 q-projection
elements per step, with update RMS 2.71e-6 to 8.01e-6. Its mean training loss was
0.02062. These are 16 different batch-1 examples with no warmup, not evidence that
the earlier full-run plateau or validation gap is solved.

The new fp16 path exposed an autocast weight-cache bug: a no-grad EMA forward
cached half-precision casts of the live parameter objects, then the online forward
reused them after the fp32 weights were restored. A CPU autocast reproduction showed
both stale values and absent gradients. Disabling the weight cache around the
online/EMA velocity calls fixes the reproduction; a regression test pins it. The
diagnostic now also raises on a missing monitored-expert gradient before stepping.
Corrected fp16 probe submitted as job 21056; its GPU outcome is not yet recorded.

## 2B All-Layer Cotrained EoS

The 2B run uses the two-camera all-layer cotrained EoS checkpoint, not the
four-camera sibling:

```text
output_eos_cotrain_2bmix_all28_lr1x_nav_framecache/checkpoint-6876
```

Config: [configs/sft_cd_eos_2b_consistency2to1_2cam_nav_lcdrive.yaml](configs/sft_cd_eos_2b_consistency2to1_2cam_nav_lcdrive.yaml).
The 28-layer Cosmos VLM and trained 28->36 layer mixer are loaded and frozen.
Mixer geometry is four blocks, gain disabled, sharpen 0.75. The fixed solver and
online/EMA experts all retain 36 layers and read the same mixed conditioning.
The frozen solver constructor now explicitly receives the configured expert depth
instead of inheriting 28 from the VLM.

```bash
SMOKE=1 sbatch slurm_train_cd_eos_2b_consistency2to1.sh
sbatch --dependency=afterok:<smoke-job-id> --kill-on-invalid-dep=yes \
  slurm_train_cd_eos_2b_consistency2to1.sh
```

Submitted 2026-09-14: smoke 21081, full training 21082 with success-only dependency.
No node is pinned. Settings: corrected fp16 AMP, fp32 trainable weights/Adam/EMA,
EMA decay 0.99, four GPUs x batch 2 x accumulation 4 = 32, LR 2e-5, warmup 430,
two epochs on the full training manifest. No GT or cached-10B endpoint loss.
The target is this 2B checkpoint's own two-step output, not the 4B model's output.

Full output directory under `training/`:
`output_cd_eos2bmix_all28_consistency2to1_fp16_master32_2cam_nav`.
Evaluate epoch checkpoints at NFE=1 with `MODEL=2b MIX=1` in the EoS eval launcher,
overriding the run directory and enabling stripped navigation. Compare against
the original 2B all-layer checkpoint at NFE=2 with the same mixer and cameras.