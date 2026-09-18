# Resources and Training

This document describes the resources and training used for the final two-camera
2B and 4B models. Both follow **knowledge distillation (KD) -> VLM/action-expert
cotraining (EoS) -> consistency training (CM)** and retain the full 36-layer
Alpamayo action expert. The size labels refer to the VLM, not the combined model.

## Final Model Identification

The final model identifiers used in the saved trajectory comparison are:

| VLM | Final identifier / NPZ archive stem | Training directory under `training/` |
|---|---|---|
| Cosmos-Reason2-2B | `cm2b_all28_fp16_master32_h100c_eos_checkpoint-6876_nfe1` | `output_cd_eos2bmix_all28_consistency2to1_fp16_master32_2cam_nav` |
| Qwen3-VL-4B-Instruct | `cm4b_fp16_master32_eos_checkpoint-6876_nfe1` | `output_cd_eos4b_consistency2to1_fp16_master32_2cam_nav` |

Both use the **epoch-2 EMA checkpoint `checkpoint-6876`**, evaluated with one
action-expert function evaluation (NFE=1). Use the checkpoint subdirectory, not
the output-directory root. The NPZ stems identify evaluation artifacts rather
than separate pretrained model repositories. See the
[evaluation guide](EVALUATION.md) and
[metrics notebook](../notebooks/npz_metrics_comparison.ipynb).

## Data and Annotation Provenance

- **PhysicalAI Autonomous Vehicles (PAI-AV), NVIDIA:** the source of camera data,
  ego-motion history, and ground-truth future trajectories.
- **LCDrive manifest:** a list of PAI-AV clip identifiers supplied by the LCDrive
  authors, used to select the training and validation subsets. It is not an
  additional private driving dataset.
- **Availability limitation:** 1,159 clips in the LCDrive manifest are absent
  from the public NVIDIA release used here: **732 training and 427 validation
  clips**. We report that NVIDIA withdrew a data chunk from the
  public release; these clips are not part of the available data used here.
  Counts therefore describe the available subset, not an intact original
  LCDrive release.
- **Navigation annotations:** we generated these from ground-truth
  trajectories using an algorithm similar to Alpamayo-Labeler. This is a
  description of the method, not a claim that Alpamayo-Labeler was run unchanged.
  Turn-distance text is stripped for the final training pipeline.
- **Derived local artifacts:** navigation JSONs, clip filters, decoded-frame
  caches, trained checkpoints, and evaluation trajectories are experiment
  outputs, not additional independently sourced datasets or annotations.

**Permission disclosure.** We confirm that NVIDIA authorizes our
research, training, and evaluation use of the PAI-AV data described here. We
used no other private input datasets, annotations, or pretrained weights for
these final models. Author-provided access to the LCDrive clip list and permission
to use data do not by themselves establish permission to redistribute the list,
data, or derived artifacts; applicable upstream terms still govern each resource.

### Resource Inventory and Licenses

| Resource | Role in this work | License / access terms |
|---|---|---|
| [NVIDIA PAI-AV](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) | Publicly listed, access-gated source data and ego-motion labels for the LCDrive subset | [NVIDIA Autonomous Vehicle Dataset License Agreement](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/blob/main/LICENSE.pdf); access requires acceptance. Not Apache-2.0 and not unrestricted redistribution. |
| Author-supplied LCDrive clip manifest | Split membership and clip selection within PAI-AV | No separate redistribution license was supplied for this disclosure. The underlying clips remain governed by NVIDIA's dataset terms. |
| Locally generated event anchors and navigation annotations | Select training keyframes and provide route-intent text derived from GT motion | Derived from PAI-AV; no independent public-data license is asserted. The [navigation generator](../scripts/gen_nav_annotations.py) is repository code; licensing that code does not relicense its inputs or outputs. |
| [Alpamayo-1.5-10B](https://huggingface.co/nvidia/Alpamayo-1.5-10B) | Online KD teacher; initial full 36-layer action expert and action projections for both students | Current official card lists **OpenMDW-1.1** for weights and Apache-2.0 for source. The card also contains non-commercial/commercial-license wording; retain and check the license accompanying the downloaded revision before release or commercial use. |
| [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) | Pretrained 28-text-layer VLM for the 2B student | **NVIDIA Open Model License**, including applicable attribution and use conditions; separate component notices also apply. |
| [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | Pretrained 36-text-layer VLM for the 4B student | **Apache-2.0**. |
| [Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B) | Teacher backbone architecture/tokenizer resource referenced by the KD configs; teacher parameters come from Alpamayo | Consult its NVIDIA Open Model License and separate component notices; this is not an additional driving dataset. |
| Local frame caches, intermediate KD/EoS weights, final CM weights, and NPZ predictions | Derived experiment artifacts | Not granted a new public redistribution license by this README. Upstream model/data terms and local release authorization must be respected. |
| This repository | Training, annotation, and evaluation implementation | [Apache-2.0](../../../LICENSE), distinct from data and model-weight licenses. |

License labels above reflect the official cards checked on 2026-09-18; they do
not establish the historical license of an unpinned download. The local
Alpamayo `A1-format` directory is a layout conversion of the released teacher,
not another independently pretrained model. Its converted directory does not
contain a license file, so the original download's license must be retained
separately. Upstream pretraining data, including proprietary data disclosed by
NVIDIA for Alpamayo, were not directly accessed for these experiments; their
influence is inherited through the public pretrained weights.

**Access is not publication permission.** NVIDIA's dataset agreement restricts
use and distribution and includes provisions concerning benchmarking/results.
The research-use confirmation above does not independently establish permission
to publish data-derived artifacts or results. Check the agreement accepted for
the actual release and any additional written permissions before distribution.

### Annotations and Evaluation Scope

The [annotation notes](../archive/NAVTEXT_SAMPLING.md) record 109,997 distinct
training `(clip_id, t0)` anchors over 32,022 clips. These event-based anchors are
not the same as using one fixed timestamp for every available training clip.
Event timing and route text are algorithmic annotations, not human-written
reasoning supervision. The navigation generator records that its heading-change
classifier follows AlpaSim; we describe our overall GT-derived
labeling approach as similar to Alpamayo-Labeler.

| Split / artifact | Size and sampling | Use |
|---|---|---|
| `nav_lcdrive_train_anchors_all.json` | 109,997 event anchors; 32,022 distinct clips | All three training stages |
| `nav_lcdrive_val_mysubset_1k.json` | 1,000 validation clips | Development comparisons and the final NPZ archive stems listed above |
| Available full validation | 23,331 clips at fixed `t0 = 5.1 s`; 427 original-manifest clips unavailable | Separate full-validation evaluation, not an additional training set |
| `lcdrive_val_primary_scenario_for_table2.csv` | Primary scenario categories aligned by clip UUID | Optional category-wise validation reporting in the notebook, not a training loss target |

The training/validation split is clip-disjoint according to the annotation
inventory. The 732 unavailable training clips and 427 unavailable validation
clips must not be counted as used data. The original LCDrive clip lists, the
available release, and the event-anchor manifest are different inventories.
Full-validation loading is recorded in
[slurm_eval_cm_fullval_fixedt0.sh](../slurm_eval_cm_fullval_fixedt0.sh). Saved
trajectory comparisons use six draws per clip and 64 future waypoints at 10 Hz
(6.4 s); XY metrics are recomputed consistently in the notebook.

**Evaluation conditioning caveat:** route instructions are generated from GT
future trajectories, not an independently supplied map/planner route. Removing
turn distances removes a numeric cue, but not the GT-derived directional cue.
Results are therefore **GT-derived-navigation-conditioned** trajectory metrics,
not navigation-free prediction or evidence of independent route planning.
No generated CoT or reasoning-label loss is used in the final pipeline.

## Training Overview

All stages use two front cameras `[1, 3]` (front-wide and front-telephoto), four
frames per camera, navigation text without turn distances, and the same
109,997-event-anchor training manifest. Event anchors are training examples,
not counts of unique clips. The 2B model uses a learned 28-to-36 KV layer mixer;
the 4B model has 36 text layers and does not need that mixer.

| Stage | Epochs | Trainable components | Objective | Selected checkpoint |
|---|---:|---|---|---|
| KD | 4 | Student VLM and, for 2B, the layer mixer | Frozen-expert block/span-output matching; spans 1, 9, 18, 36, one per epoch | `checkpoint-13752` |
| EoS cotraining | 2 | All VLM text layers, full action expert, action projections | Ground-truth flow matching; trajectory-token CE disabled | `checkpoint-6876` |
| CM | 2 | Full action expert and action projections; VLM and mixer frozen | Online two-step EoS teacher with an EMA consistency target; no GT or cached-10B endpoint loss | EMA `checkpoint-6876` |

The [training guide](TRAINING.md) contains launch commands and environment
setup. Historical pruning, partial-layer cotraining, cached-teacher experiments,
and smoke tests are not the final pipeline described here.

### Ablations and Experiment Records

We also ablated and tested alternative training methods before selecting this
final pipeline. We recorded the training directories, datasets, completed
epochs, evaluation artifacts, and ablation comparisons in two experiment books:

- [trainng_exp_book_h100e.md](trainng_exp_book_h100e.md): the H100e
  inventory, including CE/logit-KD/KV/block-loss comparisons, KAVA controls,
  layer mixing, and endpoint/GT-weight/NFE sweeps.
- [trainng_exp_book.md](trainng_exp_book.md): the local inventory,
  including block/span curricula, normalization and free-running variants,
  camera counts, expert adaptation, VLM cotraining depth and learning rates,
  and endpoint/consistency training with full-validation comparisons.

These books document the broader experimental history, not additional stages
in the final models. We keep their inclusion rules and protocol caveats:
differences in datasets, timestamps, navigation, cameras, action heads, or NFE
must not be treated as controlled loss-only ablations.

## Training Schedule and Hyperparameters

The following values were checked against the **saved configs and selected
checkpoint trainer states of all six runs**, not just today's launcher defaults.
Each run used four GPUs and effective batch 32. There are 3,438 optimizer
updates per epoch, giving 27,504 updates across the three stages for each final
model. A new stage starts its own optimizer/LR schedule; the KD span changes
occur within one continuous four-epoch run.

| Stage | VLM | Completed epochs | Updates | Peak LR | Warmup updates | Batch/GPU | Gradient accumulation | Effective batch | Training precision | Parallelism |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| KD | 2B | 4 | 13,752 | 1e-5 | 733 | 8 | 1 | 32 | bf16 | DeepSpeed ZeRO-2 |
| KD | 4B | 4 | 13,752 | 1e-5 | 733 | 8 | 1 | 32 | bf16 | DeepSpeed ZeRO-2 |
| EoS | 2B | 2 | 6,876 | 2e-5 | 430 | 4 | 2 | 32 | bf16 AMP | DDP; no DeepSpeed |
| EoS | 4B | 2 | 6,876 | 2e-5 | 430 | 4 | 2 | 32 | bf16 AMP | DDP; no DeepSpeed |
| CM | 2B | 2 | 6,876 | 2e-5 | 430 | 2 | 4 | 32 | fp16 AMP; fp32 trainable/master weights and EMA | DDP; no DeepSpeed |
| CM | 4B | 2 | 6,876 | 2e-5 | 430 | 2 | 4 | 32 | fp16 AMP; fp32 trainable/master weights and EMA | DDP; no DeepSpeed |

**Recorded runs versus launch examples:** the current KD 2B launcher defaults
to batch 2 / accumulation 4, and the training guide's EoS examples use 2 / 4.
The completed runs above recorded 8 / 1 for KD and 4 / 2 for EoS in both sizes.
These alternatives preserve effective batch 32 but are not identical memory
configurations. The all-layer 4B EoS run requires H200-class memory in the
documented DDP setup; use the smaller launch-example microbatch where needed.

| Setting | Final pipeline value |
|---|---|
| LR schedule, all stages | `cosine_warmup_with_min_lr`; linear warmup followed by cosine decay to `1e-6` |
| Optimizer | AdamW; installed Transformers default `adamw_torch_fused`, with ZeRO-2 handling KD state partitioning |
| Adam settings | $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$ |
| Weight decay / gradient clipping | 0.0 / max gradient norm 1.0 |
| Trainer seed | 42 |
| KD curriculum | One span per epoch: 1, 9, 18, 36; `block_span_mix=0`; teacher-output normalization |
| KD coefficients | Block/span 1.0; token CE 0; logit KD 0; direct KV 0 |
| KD checkpointing | Trainer gradient checkpointing enabled; expert-span checkpointing controlled separately by `SPAN_CKPT_MIN` |
| EoS VLM LR multiplier | 1x: text layers and expert use the same base LR |
| EoS/CM gradient checkpointing | Disabled |
| CM teacher / grid | Frozen matching EoS expert; two Euler steps; uniform edge sampling on $\tau\in\{0,0.5,1\}$ |
| CM coefficients | Consistency 1.0; GT endpoint 0; cached-teacher endpoint 0; MSE normalizer 1.0 |
| CM EMA | Decay 0.99; warmup 0; update each optimizer step; save EMA weights at epoch checkpoints |
| Input / output | Two front cameras, four frames each; 16 ego-history waypoints; 64 future waypoints; 10 Hz trajectory sampling |
| Text preprocessing | Camera and frame labels included; turn distances removed; no generated CoT |

Adam settings, weight decay, clipping, optimizer name, and seed are inherited
`TrainingArguments` defaults verified in the installed Transformers 4.57.1
environment; they are not explicit overrides in the saved configs. The table
does not claim bitwise reproducibility across hardware or package builds.

### Checkpoint Lineage

All directories below are relative to the experiment artifact store's
`training/` directory. They are local artifacts, not bundled public downloads.

| VLM | Stage | Run directory | Checkpoint used by the next stage / evaluation |
|---|---|---|---|
| 2B | KD | `output_kd_2b_mixspan2bnavfc_m1-9-18-36_framecache1080p_lcdrive` | `checkpoint-13752` |
| 2B | EoS | `output_eos_cotrain_2bmix_all28_lr1x_nav_framecache` | `checkpoint-6876` |
| 2B | CM | `output_cd_eos2bmix_all28_consistency2to1_fp16_master32_2cam_nav` | EMA `checkpoint-6876` |
| 4B | KD | `output_kd_4b_nav4bspan2camallfc_m1-9-18-36_framecache1080p_lcdrive` | `checkpoint-13752` |
| 4B | EoS | `output_eos_cotrain_4b_all36_lr1x_2cam_nav_framecache` | `checkpoint-6876` |
| 4B | CM | `output_cd_eos4b_consistency2to1_fp16_master32_2cam_nav` | EMA `checkpoint-6876` |

The checkpoint trainer states confirm epoch 4.0 / step 13,752 for KD and epoch
2.0 / step 6,876 for EoS and CM. The two student sizes have separate lineages;
neither CM model uses the other size's EoS teacher.

## Training Objectives

Let $c$ denote images, ego-motion history, navigation text, and the associated
camera/frame labels. Let $C_\psi(c)$ be the student VLM's prompt KV cache,
$C_T(c)$ the frozen Alpamayo teacher cache, and $\operatorname{sg}$ stop-gradient.
Action-space variables below are normalized acceleration/curvature sequences,
not the XY positions used to report ADE. MSE means an elementwise mean, not a
sum over the action horizon or hidden width.

### 1. Knowledge Distillation: Expert Block/Span Matching

The teacher VLM and action expert are frozen. The student is trained to produce
a cache that drives that same expert similarly to the teacher cache. Teacher
and student receive matching prompt inputs; supervision is through expert
activations, not generated reasoning or token logits.

For a sampled noisy GT action and time, let $h_l^T$ be the teacher-conditioned
expert state entering block $B_l$. For span length $m$, define disjoint starts
$\mathcal A_m=\{0,m,\ldots,36-m\}$ and teacher-force only the span entry:

$$
y_{a,m}^{S} = B_{a+m-1}^{C_\psi}\circ\cdots\circ B_a^{C_\psi}
  \bigl(\operatorname{sg}(h_a^T)\bigr),
\qquad y_{a,m}^{T}=\operatorname{sg}(h_{a+m}^T).
$$

The normalized output discrepancy is

$$
D_{\mathrm{MSE}}(y,z)=
\frac{\operatorname{mean}[(y-z)^2]}
     {\max(\operatorname{mean}[z^2],10^{-6})},
\qquad
D_{\cos}(y,z)=1-\operatorname{mean}_{b,t}
\frac{\langle y_{b,t},z_{b,t}\rangle}
{\max(\|y_{b,t}\|_2,10^{-8})\max(\|z_{b,t}\|_2,10^{-8})}.
$$

The **implemented curriculum is piecewise**:

$$
\mathcal L_{\mathrm{KD}}^{(m)}=
\begin{cases}
\displaystyle \frac{1}{36}\sum_{a=0}^{35}
\left[D_{\mathrm{MSE}}(y_{a,1}^{S},y_{a,1}^{T})
+D_{\cos}(y_{a,1}^{S},y_{a,1}^{T})\right], & m=1,\\[6pt]
\displaystyle \frac{1}{m|\mathcal A_m|}\sum_{a\in\mathcal A_m}
D_{\mathrm{MSE}}(y_{a,m}^{S},y_{a,m}^{T}), & m\in\{9,18,36\}.
\end{cases}
$$

Epochs 1, 2, 3, and 4 use $m=1,9,18,36$, respectively. **Only epoch 1 includes
cosine distance.** Later epochs match span exits with normalized MSE, divided by
span length. Normalization uses teacher-output energy (`block_norm=teacher`),
not the experimental zero-cache denominator. Teacher forcing at span entry
allows errors to propagate inside each span as its length grows.

The KD time sampler is $u\sim\operatorname{Beta}(1.5,1)$,
$s=0.999(1-u)$, with noisy actions $x_s=sx_0+(1-s)\epsilon$ and
$\epsilon\sim\mathcal N(0,I)$. Here $x_0$ denotes the clean GT action, not a
time-indexed noisy state. The block-loss coefficient is 1; CE, logit KD,
elementwise KV, extra free-running, velocity-field, and rollout losses are off.

For 2B, each group of seven VLM layers supplies nine expert cache slots using
separate learned convex combinations for K and V:

$$
\widetilde K_{g,j}=\sum_{i=1}^{7}\operatorname{softmax}_i(A^K_{g,:,j})K_{g,i},
\qquad
\widetilde V_{g,j}=\sum_{i=1}^{7}\operatorname{softmax}_i(A^V_{g,:,j})V_{g,i}.
$$

There are four groups, giving 28-to-36 alignment and 504 mixing logits. The
mixer trains during KD, with initialization sharpening 0.75 and no learned
gain. The expert is not pruned. See [KD implementation](../models/kd_model.py),
[block losses](../models/block_losses.py), and [layer mixer](../models/layer_mix.py).

### 2. VLM and Action-Expert Cotraining: GT Flow Matching

Initialize the student from its epoch-4 KD checkpoint and the expert from the
released Alpamayo weights. Train all VLM **text layers** (2B: 0-27; 4B: 0-35),
the expert, and action input/output projections jointly. Vision, embeddings,
the LM head, and the learned 2B mixer remain frozen; the frozen mixer still
passes gradients into the trainable text layers.

With $x_0$ the normalized GT action sequence and
$\epsilon\sim\mathcal N(0,I)$, native flow time runs from noise ($s=0$) to
data ($s=1$). The inherited beta timestep sampler uses
$u\sim\operatorname{Beta}(1.5,1)$ and $s=0.999(1-u)$:

$$
x_s=sx_0+(1-s)\epsilon,\qquad v^*=x_0-\epsilon,
$$

$$
\mathcal L_{\mathrm{EoS}}(\theta,\psi)=
\mathbb E_{c,x_0,\epsilon,s}
\left[\operatorname{MSE}\left(v_\theta(x_s,s;C_\psi(c)),x_0-\epsilon\right)\right].
$$

This is a velocity regression loss in action space. It is not trajectory-token
cross-entropy, direct XY ADE optimization, or another KD loss. With
`cotrain_vlm_ce=false`, no CE term is added. The cache-gradient path is open even
though the configs use `cotrain_vlm=false`: the explicit
`cotrain_vlm_layers` selection enables text-layer training in
[the stitched model](../models/stitched_model.py).

### 3. Consistency Training: Two-Step EoS to One Step

Initialize from each model's own epoch-2 EoS checkpoint. Freeze its VLM and 2B
mixer. Let $\phi$ be a fixed copy of that EoS expert, $\theta$ the trainable
expert/projections, and $\bar\theta$ their EMA target. All three use the same
frozen student conditioning. The teacher here is **the matching cotrained EoS
model**, not the original 10B teacher.

Use consistency time $\tau=1-s$, so noise is at $\tau=1$. Write
$v_\theta(x,\tau;c)$ as shorthand for the expert's native velocity evaluated
with timestep input $s=1-\tau$. A fresh Gaussian draw produces the actual
two-step frozen-teacher Euler path:

$$
x_1=\epsilon,\qquad
x_{1/2}=x_1+\tfrac12v_\phi(x_1,1;c),\qquad
x_{\mathrm{end}}=x_{1/2}+\tfrac12v_\phi(x_{1/2},\tfrac12;c).
$$

Sample either adjacent edge uniformly per example: noise-to-midpoint or
midpoint-to-endpoint. Denote its endpoints by $(x_h,\tau_h)$ and
$(x_l,\tau_l)$. Define the consistency prediction and loss as

$$
f_\theta(x,\tau;c)=x+\tau v_\theta(x,\tau;c),\qquad f_\theta(x,0;c)=x,
$$

$$
\mathcal L_{\mathrm{CM}}=
\mathbb E\left[\operatorname{MSE}\left(
f_\theta(x_h,\tau_h;c),
\operatorname{sg}\left[f_{\bar\theta}(x_l,\tau_l;c)\right]\right)\right].
$$

The implementation evaluates the algebraically equivalent residual in fp32:

$$
\lambda=\frac{\tau_h-\tau_l}{\tau_h},\qquad
r=\tau_h\left[v_\theta(x_h,\tau_h;c)-
\operatorname{sg}\left(\lambda v_\phi(x_h,\tau_h;c)
+(1-\lambda)v_{\bar\theta}(x_l,\tau_l;c)\right)\right],
\qquad \mathcal L_{\mathrm{CM}}=\mathbb E[\operatorname{mean}(r^2)].
$$

The $\tau_h$ factor is retained. At the lower edge $\lambda=1$, so the EMA
velocity term vanishes and the frozen teacher anchors the endpoint. At the
upper edge $\lambda=1/2$, the target blends teacher and EMA velocities. The
EMA updates after optimizer steps:

$$
\bar\theta\leftarrow0.99\bar\theta+0.01\theta.
$$

The final settings are `teacher_source=online_eos2`, `m_rungs=2`, `metric=mse`,
`normalizer=1`, and consistency weight 1. GT endpoint and cached-teacher
endpoint weights are both zero. No GT interpolation or cached 10B trajectory
enters this objective. Training uses fp16 AMP with fp32 trainable/master
weights, optimizer moments, EMA, and residual reduction. Deployment uses the
EMA model and one evaluation $\widehat x=f_{\bar\theta}(\epsilon,1;c)$.
See [consistency loss](../models/consistency_losses.py),
[model](../models/consistency_expert.py), and [EMA callback](../models/ema.py).

## External Software and Reproduction References

The recipe specifies Python 3.12, PyTorch 2.8.0, Transformers 4.57.1, and
DeepSpeed 0.19.1. The full dependency declarations are in
[pyproject.toml](../pyproject.toml); several other dependencies are lower-bounded
rather than pinned, so retain the resolved environment for exact reproduction.
W&B is used for experiment tracking and can be disabled for a new run; it is
not a source of training labels. Slurm launchers contain site-specific paths
and GPU settings that must be adapted.

| External resource | Purpose | Code license |
|---|---|---|
| [NVlabs/Alpamayo](https://github.com/NVlabs/alpamayo) and this repository's [SFT recipe](../../alpamayo1_5_sft/README.md) | Model/action-space implementation, flow matching, released-weight conversion, and shared trainer environment | Apache-2.0; model weights have separate terms |
| [PhysicalAI-AV developer kit](https://github.com/NVlabs/physical_ai_av) | Read PAI-AV metadata, sensors, and ego-motion | [MIT](https://github.com/NVlabs/physical_ai_av/blob/main/LICENSE); dataset has separate terms |
| [AlpaSim](https://github.com/NVlabs/alpasim) | Navigation-classifier reference identified by the local annotation generator; not an additional training dataset | [Apache-2.0](https://github.com/NVlabs/alpasim/blob/main/LICENSE) |
| [PyTorch](https://github.com/pytorch/pytorch) | Tensor computation and optimization | BSD-3-Clause with bundled third-party notices |
| [Transformers](https://github.com/huggingface/transformers), [Accelerate](https://github.com/huggingface/accelerate), [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) | Pretrained model loading and distributed training | Apache-2.0 |
| [Hydra](https://github.com/facebookresearch/hydra) | Compose configs and save resolved run settings | MIT |

We cite Alpamayo-Labeler as a methodological reference,
not as an installed dependency, a source of separately licensed
annotations, or an additional pretrained model used in these runs. Dependency
licenses and component notices remain separate from data/model licenses; this
table is not a replacement for the resolved environment's full license inventory.

### Configuration Sources

| Stage | Recorded recipe sources |
|---|---|
| KD | [Launcher](../slurm_train_kd.sh), arms `mixspan2bnavfc` and `nav4bspan2camallfc`; [2B base config](../configs/sft_kd_cosmos2b_2cam_nav_layermix_lcdrive.yaml); [4B base config](../configs/sft_kd_qwen3_4b_2cam_nav_lcdrive.yaml); [curriculum callback](../callbacks.py) |
| EoS | [2B all-text-layer config](../configs/sft_eos_cotrain_2bmix_all28_lr1x_nav_framecache_lcdrive.yaml); [4B all-text-layer config](../configs/sft_eos_cotrain_4b_all36_lr1x_2cam_nav_framecache_lcdrive.yaml) |
| CM | [2B config](../configs/sft_cd_eos_2b_consistency2to1_2cam_nav_lcdrive.yaml); [4B config](../configs/sft_cd_eos_4b_consistency2to1_2cam_nav_lcdrive.yaml); [2B launcher](../slurm_train_cd_eos_2b_consistency2to1.sh); [4B launcher](../slurm_train_cd_eos_4b_consistency2to1.sh) |
| Common | [Shared trainer defaults](../../alpamayo1_5_sft/configs/sft_base.yaml); [ZeRO-2 config](../../alpamayo1_5_sft/configs/deepspeed/zero2.json); each completed run's saved `config.yaml` and checkpoint `trainer_state.json` |

The base KD YAMLs alone do not select the final four-epoch curriculum: the
launcher overrides are essential. Likewise, the 4B CM base config predates the
final precision correction and requires the fp16/fp32-master overrides shown
in the training guide. Saved run configs, completed checkpoint states, and the
implemented active loss branches take precedence over stale historical comments.