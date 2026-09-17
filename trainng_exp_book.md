# Local Training Experiment Book

Inventory date: 2026-09-17. **39 locally retained training runs** with at least one saved epoch: 11 KD, 17 expert adaptation/cotraining, and 11 endpoint/consistency runs. **174 full evaluation artifacts** are catalogued below (160 root-level, 10 nested step-sweep, 4 legacy teacher); 13 short smoke evaluations are omitted. Four legacy teacher files have incomplete dataset provenance and are not used in matched comparisons. This is a local-artifact inventory, not a transcription of historical experiment notes.

Quick navigation: [training inventory](#training-inventory), [evaluation file index](#evaluation-file-index), [ablation comparisons](#ablation-comparisons), [full-validation comparison](#6-full-available-validation-eos-vs-consistency), [exclusions](#exclusions-and-limitations).

## Evidence and Inclusion Rules

- Include training runs only when local saved training state or logs demonstrate at least one completed epoch. Configured epoch counts alone do not establish completion.
- Exclude smoke tests, failed launches, and experiments documented elsewhere without a corresponding local artifact directory.
- The actual checkpoints/results are on this machine under the `/temp` root below, not in the home checkout's nearly empty training folder. Both training and evaluation directories were checked for existence. No remote-machine-only experiments are included.
- Report saved evaluation results, keeping datasets, camera/prompt settings, action heads, checkpoint epochs, and sampling budgets separate. A cross-setting comparison is not a controlled loss ablation.
- Missing evaluation evidence is not a zero score. Teacher and untrained baselines, where locally available, are reference evaluations rather than eligible training runs.

## Paths and Datasets

Directory aliases used throughout the tables:

```text
R = /temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
M = /temp/achahe/physical_ai_av/lcdrive_physicalai_av_manifests
D = /temp/achahe/physical_ai_av
H = /home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/outputs
```

Every training directory below is relative to `R`. Its saved configuration is `R/<directory>/config.yaml`; completion evidence is `R/<directory>/checkpoint-<step>/trainer_state.json`. These are the **training files**, not merely submission scripts. Epoch/step columns report the latest surviving state, not the requested training budget. Evaluation rows later identify the actual result/log files and source run.

All datasets are Physical AI AV clips selected by LCDrive split manifests. Training set size means anchors, not necessarily unique clips. All use chunk range `0-3146` unless stated otherwise. The train clip filter is `M/lcdrive_train_clip_uuids.txt`.

| Code | Dataset / selection | Anchors or evaluated clips | Navigation / timing |
|---|---|---:|---|
| T0 | Physical AI AV, LCDrive train clip filter, default keyframe; no navigation annotation manifest | Not recovered from surviving config | No route input; original K04 training |
| T50 | `M/nav_lcdrive_train_anchors_50k_turnpreserved.json` | 50,000 | Turn-preserved navigation anchors; distance stripping not explicitly enabled |
| T110 | `M/nav_lcdrive_train_anchors_all.json` | 109,997 | All navigation anchors; distance stripped; 1080p frame cache |
| V0 | LCDrive validation subset, `M/lcdrive_val_mysubset_1k_clip_uuids.txt`, no navigation annotation manifest | 1,000 | Default keyframe; no route input; K04 evaluation only |
| V1 | `M/nav_lcdrive_val_mysubset_1k.json` | 1,000 | Event/navigation anchors; stripping not enabled in logged config |
| V1S | Same manifest as V1, `strip_nav_turn_distance=true` | 1,000 | Event/navigation anchors; runtime distance stripping |
| V1P | `M/nav_lcdrive_val_mysubset_1k_stripped.json` | 1,000 | Pre-stripped navigation manifest; kept distinct from V1S |
| VF | `M/nav_lcdrive_val_available_fixedt0_23331_stripped.json`; filter `M/lcdrive_val_available_23331_clip_uuids.txt` | 23,331 | Available validation clips, fixed t0=5.1 s, distance stripped; 427 missing-index clips excluded |

Camera codes: `13` = cameras `[1,3]`; `0123` = `[0,1,2,3]`. T110 uses `D/framecache_nav2cam_1080p` or `D/framecache_nav4cam_1080p` accordingly. Cached-endpoint training additionally uses `R/teacher_action_rollouts_full10b_2cam_nav_k6_m10`, with `teacher_trajectory_cached_only=true`; 109,997/50,000 are manifest sizes, not an independent recount of cache-filtered examples.

## Training Inventory

### KD Tower Runs

All included KD runs have **CE=0, logit-KD=0, direct-KV=0, block-output weight=1**. Thus the retained full-run loss ablation is primarily over block supervision, not a local CE-vs-KL-vs-KV sweep. `span` is expert depth in one forward, not diffusion NFE. The 2B runs learn 28-to-36 blockwise KV mixing (4 mixer blocks, mixer LR multiplier 10); 4B has native 36-layer correspondence. Base LR is 1e-5.

| ID | Training directory under R | Training data / cameras | Verified epoch / step | Loss variant |
|---|---|---|---|---|
| K01 | `output_kd_2b_mixspan2b4camnavfc_m1-9-18-36_framecache4cam1080p_lcdrive` | T110 / 0123 | 4 / 13752 | 2B; span curriculum 1,9,18,36; teacher normalization |
| K02 | `output_kd_2b_mixspan2bnavfc_m1-9-18-36_framecache1080p_lcdrive` | T110 / 13 | 4 / 13752 | 2B; same curriculum |
| K03 | `output_kd_2b_mixspanmix2bnavfc_m1-9-18-36mix_w1.0_framecache1080p_lcdrive` | T110 / 13 | 4 / 13752 | 2B; fixed span 1 plus scheduled span-mix 1,9,18,36, mix weight 1 |
| K04 | `output_kd_4b_blockrandt_lcdrive` | T0 / default cameras | 3 / 2397 | 4B; block-output, beta-random timestep, no navigation; see overwritten-config warning |
| K05 | `output_kd_4b_nav4bmix2cam_m9w1.0_lcdrive` | T50 / 13 | 5 / 7815 | 4B; span 1 plus span 9, mix weight 1 |
| K06 | `output_kd_4b_nav4bmix_m9w1.0_lcdrive` | T50 / default four-camera dataset | 5 / 7815 | 4B; span 1 plus span 9, mix weight 1 |
| K07 | `output_kd_4b_nav4bspan2camallfc_cachenorm_m1-9-18-36_framecache1080p_lcdrive` | T110 / 13 | 2.5 / 8595 | 4B; span curriculum; cache-attributable normalization; evaluated epochs 1,2 |
| K08 | `output_kd_4b_nav4bspan2camallfc_freerun_m1-9-18-36_framecache1080p_lcdrive` | T110 / 13 | 2 / 6876 | 4B; span curriculum + free-running loss weight 0.0002 |
| K09 | `output_kd_4b_nav4bspan2camallfc_m1-9-18-36_framecache1080p_lcdrive` | T110 / 13 | 4 / 13752 | 4B parent; span curriculum 1,9,18,36; teacher normalization |
| K10 | `output_kd_4b_nav4bspan4camallfc_m1-9-18-36_framecache4cam1080p_lcdrive` | T110 / 0123 | 4 / 13752 | 4B; same curriculum, four cameras |
| K11 | `output_kd_4b_nav4bspanmix2camallfc_m1-9-18-36mix_w1.0_framecache1080p_lcdrive` | T110 / 13 | 4 / 13752 | 4B; fixed span 1 plus scheduled span-mix 1,9,18,36, mix weight 1 |

**K04 provenance correction:** its root configuration was overwritten by later navigation/span-mix probes. The completed checkpoint belongs to the earlier three-epoch, no-route run. Use the [original training config](recipes/alpamayo1_5_distill/outputs/2026-08-17/18-24-24/.hydra/config.yaml) and [original evaluation config](recipes/alpamayo1_5_distill/outputs/2026-08-19/10-17-48/.hydra/config.yaml), not the newer root config, to describe it.

### EoS and VLM Cotraining

These use **GT flow matching**, not token CE or block KD. The expert/action projections train; selected VLM text layers additionally train in cotraining arms. Layer intervals are inclusive and zero-based: `deep27` means layers 27-35 (9 layers), not 27 trainable layers. Base LR is 2e-5; `3x` and `5x` apply to VLM parameter groups. Parent is the initialization, not another epoch count to add to this stage.

| ID | Training directory under R | Data / cameras | Epoch / step | Trainable text layers; VLM LR | KD parent |
|---|---|---|---|---|---|
| E01 | `output_eos_2b_mix_nav_all_framecache_lcdrive` | T110 / 13 | 2 / 6876 | Frozen | K02 @13752 |
| E02 | `output_eos_2b_mix_nav_all_framecache_maskfix_20260910_174742` | T110 / 13 | 2 / 6876 | Frozen; maskfix-labeled rerun | K02 @13752 |
| E03 | `output_eos_4b_2cam_nav_all_framecache_lcdrive` | T110 / 13 | 2 / 6876 | Frozen | K09 @13752 |
| E04 | `output_eos_4b_2cam_nav_all_framecache_maskfix_20260909_204531` | T110 / 13 | 2 / 6876 | Frozen; maskfix-labeled rerun | K09 @13752 |
| E05 | `output_eos_4b_2cam_nav_lcdrive` | T50 / 13 | 4 / 6252 | Frozen | K05 @7815 |
| E06 | `output_eos_cotrain_2bmix_all28_lr1x_4cam_nav_framecache` | T110 / 0123 | 2 / 6876 | 0-27; 1x | K01 @13752 |
| E07 | `output_eos_cotrain_2bmix_all28_lr1x_nav_framecache` | T110 / 13 | 2 / 6876 | 0-27; 1x | K02 @13752 |
| E08 | `output_eos_cotrain_2bmix_deep21_lr3x_nav_framecache` | T110 / 13 | 2 / 6876 | 21-27; 3x | K02 @13752 |
| E09 | `output_eos_cotrain_2bmix_deep21_lr5x_nav_framecache` | T110 / 13 | 2 / 6876 | 21-27; 5x | K02 @13752 |
| E10 | `output_eos_cotrain_2bmix_wide14_lr1x_nav_framecache` | T110 / 13 | 2 / 6876 | 14-27; 1x | K02 @13752 |
| E11 | `output_eos_cotrain_4b_all36_lr1x_2cam_nav_framecache` | T110 / 13 | 2 / 6876 | 0-35; 1x | K09 @13752 |
| E12 | `output_eos_cotrain_4b_all36_lr1x_4cam_nav_framecache` | T110 / 0123 | 2 / 6876 | 0-35; 1x | K10 @13752 |
| E13 | `output_eos_cotrain_deep27_4b_2cam_nav_framecache` | T110 / 13 | 2 / 6876 | 27-35; 1x | K09 @13752 |
| E14 | `output_eos_cotrain_deep27_lr3x_4b_2cam_nav_framecache` | T110 / 13 | 2 / 6876 | 27-35; 3x | K09 @13752 |
| E15 | `output_eos_cotrain_deep27_lr5x_4b_2cam_nav_framecache` | T110 / 13 | 2 / 6876 | 27-35; 5x | K09 @13752 |
| E16 | `output_eos_cotrain_wide19_4b_2cam_nav_framecache` | T110 / 13 | 2 / 6876 | 19-35; 1x | K09 @13752 |
| E17 | `output_eos_cotrain_wide19_lr3x_4b_2cam_nav_framecache` | T110 / 13 | 2 / 6876 | 19-35; 3x | K09 @13752 |

### Endpoint and Consistency Distillation

Endpoint rows use `teacher_source=cached_full`, teacher-endpoint MSE weight 1 and the listed GT-endpoint MSE weight; `cd_weight=0`. These must not be called EMA consistency training. CM rows use the frozen online two-step EoS solver, `cd_weight=1`, both endpoint/GT weights zero, two rungs, and EMA decay 0.99. All freeze the VLM; none of these retained runs prunes expert layers.

| ID | Training directory under R | Data / cameras | Epoch / step | Objective / precision variant | Initialization |
|---|---|---|---|---|---|
| C01 | `output_cd_cotrain27lr3x_gt03_2cam_nav_e2` | T110 / 13 | 1 / 3438 | Endpoint + GT weight 0.3; only epoch 1 retained | E14 @6876 |
| C02 | `output_cd_cotrain27lr3x_gt10_2cam_nav_e2` | T110 / 13 | 2 / 6876 | Endpoint + GT weight 1.0 | E14 @6876 |
| C03 | `output_cd_cotrain27lr3x_gt20_2cam_nav_e2` | T110 / 13 | 2 / 6876 | Endpoint + GT weight 2.0 | E14 @6876 |
| C04 | `output_cd_eos2b_mix_endpoint_gt05_2cam_nav_allstudent_all110k_e2_bs32` | T110 / 13 | 2 / 6876 | 2B endpoint + GT weight 0.5; nested sweep results V161-V163 | E01 @6876 |
| C05 | `output_cd_eos2bmix_all28_consistency2to1_fp16_master32_2cam_nav` | T110 / 13 | 2 / 6876 | 2B CM; fp16 AMP / fp32 master-weight variant | E07 @6876 |
| C06 | `output_cd_eos4b_consistency2to1_2cam_nav` | T110 / 13 | 2 / 6876 | 4B CM; original bf16 variant | E11 @6876 |
| C07 | `output_cd_eos4b_consistency2to1_fp16_master32_2cam_nav` | T110 / 13 | 2 / 6876 | 4B CM; fp16 AMP / fp32 master-weight variant | E11 @6876 |
| C08 | `output_cd_eos4b_endpoint_gt05_2cam_nav_allstudent_all110k_e2_bs32` | T110 / 13 | 2 / 6876 | 4B endpoint + GT weight 0.5 | E03 @6876 |
| C09 | `output_cd_eos4b_endpoint_gt05_2cam_nav_e2_bs32` | T50 / 13 | 2 / 3126 | 4B endpoint + GT weight 0.5 | E05 @3126 |
| C10 | `output_cd_eos4b_maskfix_fp32_20260910_102924` | T110 / 13 | 2 / 6876 | 4B endpoint + GT weight 0.5; maskfix/fp32-labeled rerun | E04 @6876 |
| C11 | `output_cd_preeos4b_endpoint_gt05_2cam_nav_allstudent_all110k_e2_bs32` | T110 / 13 | 4 / 13752 | 4B endpoint + GT weight 0.5, without trained EoS initialization | Local frozen pre-EoS export, derived from K09 |

E02 has no full local result linked to its exact checkpoint; do not substitute E01's results. C11's one-step frozen pre-EoS export is initialization only, not an included training experiment. Names such as `maskfix`/`fp32` identify historical variants; they do not prove those code changes are isolated causal factors.

## Evaluation File Index

For each **stem** below, the evaluation directory is `R/<stem>/`, the per-clip result file is `R/<stem>.json`, and the log is `R/<stem>.log`. Trajectories, when retained, are `R/<stem>.npz`. VF runs use the job log `R/cm_fullval_<last numeric stem component>.out` when there is no stem log. Empty evaluation directories are normal: their artifacts are sibling files. Dataset/camera provenance comes from the logged `Dataset Configs`; checkpoint and NFE come from `Evaluate Configs`, with Hydra model config used for controls.

**Metrics:** arithmetic means of the saved per-clip JSON `ade` and `min_ade`, in meters, lower is better. ADE is first-sample XY trajectory error, **not** the average of all six samples and not 3D ADE. minADE selects the best trajectory among six. This explains differences from historical tables labeled E[1 draw], centre, or medoid. No new model inference was run. All non-VF rows have 1,000 records; VF rows have 23,331. `D` means no explicit inference-step override in the saved stitched-evaluation config; compare D rows with each other rather than silently assuming an override. `T` is the local pretrained-teacher reference, not a training run.

Rows containing `control` use the KD student's VLM with the untuned teacher expert, **not** the trained EoS expert; the source ID identifies the associated run/checkpoint container. V0 default-camera configuration is retained literally rather than retrospectively rewritten. The index is comprehensive, not a single matched-protocol ranking.

### Endpoint, Consistency, and Cotraining Results

| Eval ID | Source | Dataset/cameras | NFE | ADE | minADE@6 | Evaluation directory / file stem under R |
|---|---|---|---:|---:|---:|---|
| V001 | C10 | V1S/13 | 10 | 1.9373 | 1.0048 | `cd4b_maskfix_fp32_20820_eos_checkpoint-3438` |
| V002 | C10 | V1S/13 | 1 | 1.8285 | 1.0988 | `cd4b_maskfix_fp32_20820_eos_checkpoint-3438_nfe1` |
| V003 | C10 | V1S/13 | 2 | 1.8441 | 0.9920 | `cd4b_maskfix_fp32_20820_eos_checkpoint-3438_nfe2` |
| V004 | C10 | V1S/13 | 10 | 1.9308 | 1.0038 | `cd4b_maskfix_fp32_20820_eos_checkpoint-6876` |
| V005 | C10 | V1S/13 | 1 | 1.8303 | 1.0955 | `cd4b_maskfix_fp32_20820_eos_checkpoint-6876_nfe1` |
| V006 | C10 | V1S/13 | 2 | 1.8447 | 0.9891 | `cd4b_maskfix_fp32_20820_eos_checkpoint-6876_nfe2` |
| V007 | C08 | V1S/13 | 10 | 1.9243 | 0.9962 | `cd_4b_2cam_nav_all_eos_checkpoint-6876` |
| V008 | C08 | V1S/13 | 1 | 1.8061 | 1.1765 | `cd_4b_2cam_nav_all_eos_checkpoint-6876_nfe1` |
| V009 | C08 | V1S/13 | 2 | 1.8478 | 1.0609 | `cd_4b_2cam_nav_all_eos_checkpoint-6876_nfe2` |
| V010 | C01 | V1/13 | 1 | 2.5191 | 1.1341 | `cd_cotrain27_gt03_eos_checkpoint-3438_nfe1` |
| V011 | C01 | V1/13 | 2 | 2.0202 | 0.9354 | `cd_cotrain27_gt03_eos_checkpoint-3438_nfe2` |
| V012 | C09 | V1/13 | 10 | 2.0218 | 1.0931 | `cd_eos4b_gt05_eos_checkpoint-1563` |
| V013 | C09 | V1/13 | 1 | 1.8193 | 1.4256 | `cd_eos4b_gt05_eos_checkpoint-1563_nfe1` |
| V014 | C09 | V1/13 | 2 | 1.8442 | 1.2089 | `cd_eos4b_gt05_eos_checkpoint-1563_nfe2` |
| V015 | C09 | V1/13 | 10 | 2.0077 | 1.0608 | `cd_eos4b_gt05_eos_checkpoint-3126` |
| V016 | C09 | V1/13 | 1 | 1.8141 | 1.4083 | `cd_eos4b_gt05_eos_checkpoint-3126_nfe1` |
| V017 | C09 | V1/13 | 2 | 1.8358 | 1.1912 | `cd_eos4b_gt05_eos_checkpoint-3126_nfe2` |
| V018 | C02 | V1/13 | 1 | 1.6752 | 1.2554 | `cd_gt10_eos_checkpoint-6876_nfe1` |
| V019 | C02 | V1/13 | 2 | 1.6936 | 1.1263 | `cd_gt10_eos_checkpoint-6876_nfe2` |
| V020 | C03 | V1/13 | 1 | 1.6457 | 1.3495 | `cd_gt20_eos_checkpoint-6876_nfe1` |
| V021 | C03 | V1/13 | 2 | 1.6627 | 1.2340 | `cd_gt20_eos_checkpoint-6876_nfe2` |
| V022 | C11 | V1S/13 | 10 | 2.0287 | 0.9926 | `cdpre4ep_4b_2cam_nav_all_eos_checkpoint-13752` |
| V023 | C11 | V1S/13 | 1 | 1.8210 | 1.1120 | `cdpre4ep_4b_2cam_nav_all_eos_checkpoint-13752_nfe1` |
| V024 | C11 | V1S/13 | 2 | 1.8566 | 0.9931 | `cdpre4ep_4b_2cam_nav_all_eos_checkpoint-13752_nfe2` |
| V025 | C11 | V1S/13 | 10 | 2.0408 | 0.9957 | `cdpre_4b_2cam_nav_all_eos_checkpoint-6876` |
| V026 | C11 | V1S/13 | 1 | 1.8314 | 1.1101 | `cdpre_4b_2cam_nav_all_eos_checkpoint-6876_nfe1` |
| V027 | C11 | V1S/13 | 2 | 1.8695 | 0.9920 | `cdpre_4b_2cam_nav_all_eos_checkpoint-6876_nfe2` |
| V028 | C05 | V1S/13 | 1 | 1.7894 | 1.1874 | `cm2b_all28_fp16_master32_h100c_eos_checkpoint-3438_nfe1` |
| V029 | C05 | V1S/13 | 1 | 1.7815 | 1.1783 | `cm2b_all28_fp16_master32_h100c_eos_checkpoint-6876_nfe1` |
| V030 | C05 | VF/13 | 1 | 1.7116 | 1.1352 | `cm2b_availableval23331_fixedt0_stripped_ep2_nfe1_21167` |
| V031 | C05 | VF/13 | 2 | 1.7920 | 0.9699 | `cm2b_availableval23331_fixedt0_stripped_ep2_nfe2_21408` |
| V032 | C06 | V1S/13 | 1 | 1.5644 | 1.0646 | `cm4b_2to1_ema_verified_eos_checkpoint-3438_nfe1` |
| V033 | C06 | V1S/13 | 1 | 1.5684 | 1.0704 | `cm4b_2to1_ema_verified_eos_checkpoint-6876_nfe1` |
| V034 | C07 | VF/13 | 1 | 1.5142 | 1.0368 | `cm4b_availableval23331_fixedt0_stripped_ep2_nfe1_21166` |
| V035 | C07 | VF/13 | 2 | 1.5765 | 0.8870 | `cm4b_availableval23331_fixedt0_stripped_ep2_nfe2_21409` |
| V036 | C07 | V1S/13 | 1 | 1.5420 | 1.0443 | `cm4b_fp16_master32_eos_checkpoint-3438_nfe1` |
| V037 | C07 | V1S/13 | 1 | 1.5369 | 1.0376 | `cm4b_fp16_master32_eos_checkpoint-6876_nfe1` |
| V038 | E13 | V1/13 | 10 | 1.7688 | 0.8111 | `cotrain27_eos_checkpoint-6876` |
| V039 | E13 | V1/13 | 1 | 1.7941 | 1.4902 | `cotrain27_eos_checkpoint-6876_nfe1` |
| V040 | E13 | V1/13 | 2 | 1.5299 | 1.0487 | `cotrain27_eos_checkpoint-6876_nfe2` |
| V041 | E13 | V1/13 | 10 | 1.7935 | 0.8094 | `cotrain27_ep1_eos_checkpoint-3438` |
| V042 | E13 | V1/13 | 2 | 1.5537 | 1.0487 | `cotrain27_ep1_eos_checkpoint-3438_nfe2` |
| V043 | E07 | V1/13 | 10 | 2.0783 | 0.9204 | `cotrain2b_all28_eos_checkpoint-6876` |
| V044 | E07 | V1/13 | 1 | 1.8537 | 1.5253 | `cotrain2b_all28_eos_checkpoint-6876_nfe1` |
| V045 | E07 | V1/13 | 2 | 1.7368 | 1.1491 | `cotrain2b_all28_eos_checkpoint-6876_nfe2` |
| V046 | E07 | V1S/13 | 10 | 2.0681 | 0.9176 | `cotrain2b_all28_navstripped_h100c_eos_checkpoint-6876` |
| V047 | E07 | V1S/13 | 6 | 1.9717 | 0.9365 | `cotrain2b_all28_navstripped_h100c_eos_checkpoint-6876_nfe6` |
| V048 | E08 | V1/13 | 10 | 2.1166 | 0.9294 | `cotrain2b_d21lr3x_eos_checkpoint-6876` |
| V049 | E08 | V1/13 | 1 | 1.9995 | 1.5452 | `cotrain2b_d21lr3x_eos_checkpoint-6876_nfe1` |
| V050 | E08 | V1/13 | 2 | 1.7983 | 1.1482 | `cotrain2b_d21lr3x_eos_checkpoint-6876_nfe2` |
| V051 | E08 | V1/13 | 2 | 2.1950 | 1.4206 | `cotrain2b_d21lr3x_ep1_eos_checkpoint-3438_nfe2` |
| V052 | E09 | V1/13 | 10 | 2.1519 | 0.9448 | `cotrain2b_d21lr5x_eos_checkpoint-6876` |
| V053 | E09 | V1/13 | 1 | 2.2303 | 1.5480 | `cotrain2b_d21lr5x_eos_checkpoint-6876_nfe1` |
| V054 | E09 | V1/13 | 2 | 1.8348 | 1.1370 | `cotrain2b_d21lr5x_eos_checkpoint-6876_nfe2` |
| V055 | E10 | V1/13 | 10 | 2.0769 | 0.9211 | `cotrain2b_w14lr1x_eos_checkpoint-6876` |
| V056 | E10 | V1/13 | 1 | 1.8931 | 1.5842 | `cotrain2b_w14lr1x_eos_checkpoint-6876_nfe1` |
| V057 | E10 | V1/13 | 2 | 1.7396 | 1.1532 | `cotrain2b_w14lr1x_eos_checkpoint-6876_nfe2` |
| V058 | E11 | V1/13 | 10 | 1.7496 | 0.7977 | `cotrain4b_all36_eos_checkpoint-6876` |
| V059 | E11 | V1/13 | 1 | 1.5919 | 1.3441 | `cotrain4b_all36_eos_checkpoint-6876_nfe1` |
| V060 | E11 | V1/13 | 2 | 1.4791 | 1.0012 | `cotrain4b_all36_eos_checkpoint-6876_nfe2` |
| V061 | E11 | V1S/13 | 6 | 1.6710 | 0.8147 | `cotrain4b_all36_eos_checkpoint-6876_nfe6` |
| V062 | E15 | V1/13 | 10 | 1.7968 | 0.8094 | `cotrain_d27lr5x_eos_checkpoint-6876` |
| V063 | E15 | V1/13 | 1 | 1.7096 | 1.4034 | `cotrain_d27lr5x_eos_checkpoint-6876_nfe1` |
| V064 | E15 | V1/13 | 2 | 1.5371 | 1.0324 | `cotrain_d27lr5x_eos_checkpoint-6876_nfe2` |
| V065 | E14 | V1/13 | 2 | 1.6313 | 1.1134 | `cotrain_lr3x_eos_checkpoint-3438_nfe2` |
| V066 | E14 | V1/13 | 10 | 1.7741 | 0.8067 | `cotrain_lr3x_eos_checkpoint-6876` |
| V067 | E14 | V1/13 | 1 | 1.6696 | 1.3950 | `cotrain_lr3x_eos_checkpoint-6876_nfe1` |
| V068 | E14 | V1/13 | 2 | 1.5171 | 1.0235 | `cotrain_lr3x_eos_checkpoint-6876_nfe2` |
| V069 | E14 | V1/13 | 2 | 1.5171 | 1.0235 | `cotrain_lr3x_h200ctl_eos_checkpoint-6876_nfe2` |
| V070 | E17 | V1/13 | 10 | 1.7885 | 0.8189 | `cotrain_w19lr3x_eos_checkpoint-6876` |
| V071 | E17 | V1/13 | 1 | 1.6892 | 1.3871 | `cotrain_w19lr3x_eos_checkpoint-6876_nfe1` |
| V072 | E17 | V1/13 | 2 | 1.5271 | 1.0186 | `cotrain_w19lr3x_eos_checkpoint-6876_nfe2` |
| V073 | E16 | V1/13 | 2 | 1.5367 | 1.0303 | `cotrain_wide19_eos_checkpoint-3438_nfe2` |
| V074 | E16 | V1/13 | 10 | 1.7515 | 0.8163 | `cotrain_wide19_eos_checkpoint-6876` |
| V075 | E16 | V1/13 | 1 | 1.6653 | 1.4143 | `cotrain_wide19_eos_checkpoint-6876_nfe1` |
| V076 | E16 | V1/13 | 2 | 1.5153 | 1.0365 | `cotrain_wide19_eos_checkpoint-6876_nfe2` |
| V077 | E07 | VF/13 | 1 | 1.7947 | 1.4814 | `eos2b_availableval23331_fixedt0_stripped_ep2_nfe1_21333` |
| V078 | E07 | VF/13 | 2 | 1.6665 | 1.0935 | `eos2b_availableval23331_fixedt0_stripped_ep2_nfe2_21284` |
| V079 | E01 | V1/13 | 1 | 2.3378 | 2.0774 | `eos2b_mix_base_eos_checkpoint-6876_nfe1` |
| V080 | E01 | V1/13 | 2 | 1.9037 | 1.3849 | `eos2b_mix_base_eos_checkpoint-6876_nfe2` |
| V081 | E01 | V1/13 | 2 | 1.8886 | 1.3690 | `eos2b_mix_base_ep1_eos_checkpoint-3438_nfe2` |
| V082 | E11 | VF/13 | 1 | 1.5824 | 1.3600 | `eos4b_availableval23331_fixedt0_stripped_ep2_nfe1_21332` |
| V083 | E11 | VF/13 | 2 | 1.4683 | 1.0050 | `eos4b_availableval23331_fixedt0_stripped_ep2_nfe2_21285` |
| V084 | E11 | V1S/13 | 2 | 1.4788 | 0.9998 | `eos4b_cm_matched_reference_eos_checkpoint-6876_nfe2` |
| V085 | E04 | V1/13 | 2 | 1.6285 | 1.0950 | `eos4b_maskfix_ep1_eos_checkpoint-3438_nfe2` |

### Expert Adaptation and Camera Transfer

| Eval ID | Source | Dataset/cameras | NFE | ADE | minADE@6 | Evaluation directory / file stem under R |
|---|---|---|---:|---:|---:|---|
| V086 | E01 | V1P/13 | 10 | 2.3640 | 1.0208 | `eos_2b_mix_nav_framecache_control_checkpoint-6876` |
| V087 | E01 | V1P/13 | 10 | 2.0758 | 1.0966 | `eos_2b_mix_nav_framecache_eos_checkpoint-6876` |
| V088 | E03 | V1S/13 | 10 | 2.0352 | 0.9015 | `eos_4b_2cam_nav_all_control_checkpoint-6876` |
| V089 | E03 | V1S/13 | 10 | 1.7989 | 0.9521 | `eos_4b_2cam_nav_all_eos_checkpoint-3438` |
| V090 | E03 | V1S/13 | 10 | 1.7966 | 0.9516 | `eos_4b_2cam_nav_all_eos_checkpoint-6876` |
| V091 | E03 | V1S/13 | 1 | 2.0451 | 1.7830 | `eos_4b_2cam_nav_all_eos_checkpoint-6876_nfe1` |
| V092 | E03 | V1S/13 | 2 | 1.6797 | 1.2206 | `eos_4b_2cam_nav_all_eos_checkpoint-6876_nfe2` |
| V093 | E05 | V1/13 | 10 | 2.1536 | 0.9689 | `eos_4b_2cam_nav_control_checkpoint-3126` |
| V094 | E05 | V1/13 | 10 | 1.9826 | 1.0668 | `eos_4b_2cam_nav_eos_checkpoint-1563` |
| V095 | E05 | V1/13 | 10 | 1.9588 | 1.0510 | `eos_4b_2cam_nav_eos_checkpoint-3126` |
| V096 | E05 | V1/13 | 1 | 2.0500 | 1.7640 | `eos_4b_2cam_nav_eos_checkpoint-3126_nfe1` |
| V097 | E05 | V1/13 | 2 | 1.8380 | 1.3460 | `eos_4b_2cam_nav_eos_checkpoint-3126_nfe2` |
| V098 | E05 | V1/13 | 10 | 1.9548 | 1.0489 | `eos_4b_2cam_nav_eos_checkpoint-4689` |
| V099 | E05 | V1/13 | 10 | 1.9548 | 1.0494 | `eos_4b_2cam_nav_eos_checkpoint-6252` |
| V100 | E03 | V1P/13 | 10 | 2.0352 | 0.9016 | `eos_4b_2cam_nav_framecache_control_checkpoint-6876` |
| V101 | E03 | V1P/13 | 10 | 1.7966 | 0.9517 | `eos_4b_2cam_nav_framecache_eos_checkpoint-6876` |
| V102 | E04 | V1S/13 | 10 | 1.8272 | 0.8352 | `eos_4b_maskfix_20812_eos_checkpoint-3438` |
| V103 | E04 | V1S/13 | 10 | 1.8286 | 0.8340 | `eos_4b_maskfix_20812_eos_checkpoint-6876` |
| V104 | E04 | V1S/13 | 1 | 1.8095 | 1.5075 | `eos_4b_maskfix_20812_eos_checkpoint-6876_nfe1` |
| V105 | E04 | V1S/13 | 2 | 1.6283 | 1.0930 | `eos_4b_maskfix_20812_eos_checkpoint-6876_nfe2` |
| V106 | E06 | V1S/0123 | 10 | 1.9765 | 0.9148 | `eos_cotrain_2bmix_all28_4cam_21001_eos_checkpoint-6876` |
| V107 | E06 | V1S/0123 | 1 | 1.8868 | 1.6079 | `eos_cotrain_2bmix_all28_4cam_21001_eos_checkpoint-6876_nfe1` |
| V108 | E06 | V1S/0123 | 2 | 1.6447 | 1.0994 | `eos_cotrain_2bmix_all28_4cam_21001_eos_checkpoint-6876_nfe2` |
| V109 | E12 | V1S/0123 | 10 | 1.6855 | 0.8154 | `eos_cotrain_4b_all36_4cam_20993_eos_checkpoint-6876` |
| V110 | E12 | V1S/0123 | 1 | 1.6058 | 1.4283 | `eos_cotrain_4b_all36_4cam_20993_eos_checkpoint-6876_nfe1` |
| V111 | E12 | V1S/0123 | 2 | 1.4514 | 0.9968 | `eos_cotrain_4b_all36_4cam_20993_eos_checkpoint-6876_nfe2` |
| V112 | E07 | V1S/0123 | 10 | 2.7990 | 1.5497 | `eos_crosscam_2b_train2cam_eval4cam_eos_checkpoint-6876` |
| V113 | E07 | V1S/0123 | 1 | 3.0405 | 2.6399 | `eos_crosscam_2b_train2cam_eval4cam_eos_checkpoint-6876_nfe1` |
| V114 | E07 | V1S/0123 | 2 | 2.6132 | 1.8057 | `eos_crosscam_2b_train2cam_eval4cam_eos_checkpoint-6876_nfe2` |
| V115 | E06 | V1S/13 | 10 | 2.6044 | 1.1082 | `eos_crosscam_2b_train4cam_eval2cam_eos_checkpoint-6876` |
| V116 | E06 | V1S/13 | 1 | 2.4473 | 2.0025 | `eos_crosscam_2b_train4cam_eval2cam_eos_checkpoint-6876_nfe1` |
| V117 | E06 | V1S/13 | 2 | 2.1402 | 1.3767 | `eos_crosscam_2b_train4cam_eval2cam_eos_checkpoint-6876_nfe2` |
| V118 | E11 | V1S/0123 | 10 | 2.1570 | 1.1667 | `eos_crosscam_4b_train2cam_eval4cam_eos_checkpoint-6876` |
| V119 | E11 | V1S/0123 | 1 | 2.4736 | 2.1646 | `eos_crosscam_4b_train2cam_eval4cam_eos_checkpoint-6876_nfe1` |
| V120 | E11 | V1S/0123 | 2 | 2.1151 | 1.4064 | `eos_crosscam_4b_train2cam_eval4cam_eos_checkpoint-6876_nfe2` |
| V121 | E12 | V1S/13 | 10 | 2.3836 | 1.1463 | `eos_crosscam_4b_train4cam_eval2cam_eos_checkpoint-6876` |
| V122 | E12 | V1S/13 | 1 | 2.4936 | 2.1537 | `eos_crosscam_4b_train4cam_eval2cam_eos_checkpoint-6876_nfe1` |
| V123 | E12 | V1S/13 | 2 | 2.0480 | 1.4125 | `eos_crosscam_4b_train4cam_eval2cam_eos_checkpoint-6876_nfe2` |

### KD and Teacher References

| Eval ID | Source | Dataset/cameras | NFE | ADE | minADE@6 | Evaluation directory / file stem under R |
|---|---|---|---:|---:|---:|---|
| V124 | K02 | V1S/13 | 1 | 2.1482 | 1.8428 | `kd2b_navstripped_h100c_control_checkpoint-13752_nfe1` |
| V125 | K02 | V1S/13 | 2 | 1.9947 | 1.3281 | `kd2b_navstripped_h100c_control_checkpoint-13752_nfe2` |
| V126 | K09 | V1S/13 | 1 | 2.1305 | 1.7627 | `kd4b_navstripped_h100c_control_checkpoint-13752_nfe1` |
| V127 | K09 | V1S/13 | 2 | 1.7740 | 1.1732 | `kd4b_navstripped_h100c_control_checkpoint-13752_nfe2` |
| V128 | K01 | V1P/0123 | D | 2.1601 | 0.9823 | `stitch_2b_mixspan2b4camnavfc_m1-9-18-36_framecache4cam1080p_checkpoint-13752_cam0123_stripped` |
| V129 | K01 | V1P/13 | D | 2.9367 | 1.2457 | `stitch_2b_mixspan2b4camnavfc_m1-9-18-36_framecache4cam1080p_checkpoint-13752_cam13_stripped` |
| V130 | K02 | V1S/13 | D | 2.3640 | 1.0208 | `stitch_2b_mixspan2bnavfc_m1-9-18-36_framecache1080p_checkpoint-13752` |
| V131 | K02 | V1/13 | D | 2.3712 | 1.0280 | `stitch_2b_mixspan2bnavfc_m1-9-18-36_framecache1080p_checkpoint-13752_cam13_nav` |
| V132 | K03 | V1S/13 | D | 2.4157 | 1.0669 | `stitch_2b_mixspanmix2bnavfc_m1-9-18-36mix_w1.0_framecache1080p_checkpoint-13752` |
| V133 | K03 | V1/13 | D | 2.4326 | 1.0759 | `stitch_2b_mixspanmix2bnavfc_m1-9-18-36mix_w1.0_framecache1080p_checkpoint-13752_cam13_nav` |
| V134 | K04 | V0/default | D | 3.2965 | 1.7563 | `stitch_4b_blockrandt_checkpoint-2397` |
| V135 | K05 | V1/13 | D | 2.3847 | 1.0715 | `stitch_4b_nav4bmix2cam_m9w1.0_checkpoint-4500_cam13_nav` |
| V136 | K05 | V1/13 | D | 2.1536 | 0.9689 | `stitch_4b_nav4bmix2cam_m9w1.0_checkpoint-7815_cam13_nav` |
| V137 | K06 | V1/0123 | D | 1.8073 | 0.8613 | `stitch_4b_nav4bmix_m9w1.0_checkpoint-7815_cam0123_nav` |
| V138 | K06 | V1/13 | D | 2.3816 | 1.0780 | `stitch_4b_nav4bmix_m9w1.0_checkpoint-7815_cam13_nav` |
| V139 | K07 | V1S/13 | D | 2.3124 | 0.9816 | `stitch_4b_nav4bspan2camallfc_cachenorm_m1-9-18-36_framecache1080p_checkpoint-3438` |
| V140 | K07 | V1S/13 | D | 2.1398 | 0.9270 | `stitch_4b_nav4bspan2camallfc_cachenorm_m1-9-18-36_framecache1080p_checkpoint-6876` |
| V141 | K08 | V1S/13 | D | 2.5732 | 1.1609 | `stitch_4b_nav4bspan2camallfc_freerun_m1-9-18-36_framecache1080p_checkpoint-3438` |
| V142 | K08 | V1S/13 | D | 2.2874 | 1.0015 | `stitch_4b_nav4bspan2camallfc_freerun_m1-9-18-36_framecache1080p_checkpoint-6876` |
| V143 | K09 | V1S/13 | D | 2.0352 | 0.9016 | `stitch_4b_nav4bspan2camallfc_m1-9-18-36_framecache1080p_checkpoint-13752` |
| V144 | K09 | V1/13 | D | 2.0375 | 0.9012 | `stitch_4b_nav4bspan2camallfc_m1-9-18-36_framecache1080p_checkpoint-13752_cam13_nav` |
| V145 | K10 | V1P/0123 | D | 1.7825 | 0.8271 | `stitch_4b_nav4bspan4camallfc_m1-9-18-36_framecache4cam1080p_checkpoint-13752_cam0123_stripped` |
| V146 | K10 | V1P/13 | D | 2.5176 | 1.1594 | `stitch_4b_nav4bspan4camallfc_m1-9-18-36_framecache4cam1080p_checkpoint-13752_cam13_stripped` |
| V147 | K11 | V1S/13 | D | 1.9729 | 0.8678 | `stitch_4b_nav4bspanmix2camallfc_m1-9-18-36mix_w1.0_framecache1080p_checkpoint-13752` |
| V148 | K11 | V1/13 | D | 1.9744 | 0.8585 | `stitch_4b_nav4bspanmix2camallfc_m1-9-18-36mix_w1.0_framecache1080p_checkpoint-13752_cam13_nav` |
| V149 | T | V1S/13 | D | 1.6882 | 0.7260 | `stitch_4b_teacher_models--nvidia--Alpamayo-1.5-10B-A1-format` |
| V150 | T | V1P/0123 | D | 1.3587 | 0.6078 | `stitch_4b_teacher_models--nvidia--Alpamayo-1.5-10B-A1-format_cam0123_stripped` |
| V151 | T | V1/13 | D | 1.6800 | 0.7155 | `stitch_4b_teacher_models--nvidia--Alpamayo-1.5-10B-A1-format_cam13_nav` |
| V152 | T | V1P/13 | D | 1.6886 | 0.7265 | `stitch_4b_teacher_models--nvidia--Alpamayo-1.5-10B-A1-format_cam13_stripped` |
| V153 | T | V1S/13 | 6 | 1.6051 | 0.7495 | `teacher15_2cam_nav_stripped_eos_models--nvidia--Alpamayo-1.5-10B-A1-format_nfe6` |
| V154 | T | V1S/0123 | 1 | 1.2372 | 1.0751 | `teacher15_4cam_navstripped_1k_h100c_eos_models--nvidia--Alpamayo-1.5-10B-A1-format_nfe1` |
| V155 | T | V1S/0123 | 2 | 1.1502 | 0.7600 | `teacher15_4cam_navstripped_1k_h100c_eos_models--nvidia--Alpamayo-1.5-10B-A1-format_nfe2` |
| V156 | T | VF/13 | 1 | 1.6450 | 1.2327 | `teacher15_availableval23331_fixedt0_stripped_nfe1_21170` |
| V157 | T | VF/13 | 2 | 1.4006 | 0.8628 | `teacher15_availableval23331_fixedt0_stripped_nfe2_21283` |
| V158 | T | V1S/13 | 10 | 1.6886 | 0.7265 | `teacher15_nav_fp32archive_h100c_eos_models--nvidia--Alpamayo-1.5-10B-A1-format` |
| V159 | T | V1S/13 | 1 | 1.7146 | 1.3087 | `teacher15_nav_fp32archive_h100c_eos_models--nvidia--Alpamayo-1.5-10B-A1-format_nfe1` |
| V160 | T | V1S/13 | 2 | 1.4660 | 0.9463 | `teacher15_nav_fp32archive_h100c_eos_models--nvidia--Alpamayo-1.5-10B-A1-format_nfe2` |

Trajectory archives are absent for V134, V135, V137, V138, and V151; their JSON/log results exist. The other 155 index rows have NPZ companions. Repeated evaluations and control exports remain separate artifact rows, not independent training seeds.

### Nested Step Sweeps and Legacy Teacher Files

These use **shared existing directories** `R/stepsweep/` and `R/teacher_eval/`, not `R/<stem>/` directories. For each relative stem below, append `.json` for metrics or `.npz` for trajectories; all 14 have both. All have 1,000 clips. The sweep harness seeds each batch with `1234 + batch_index`; its conditioning/batching differs from the main evaluator, so do not merge its numbers into the tables above as reruns of the same protocol.

| Eval ID | Source / conditioning | Dataset/cameras | NFE | ADE | minADE@6 | Relative file stem under R |
|---|---|---|---:|---:|---:|---|
| V161 | C04 / student VLM | V1P/13 | 1 | 2.1175 | 1.3082 | `stepsweep/nav_cd_span2b_e2_studentvlm_k1` |
| V162 | C04 / student VLM | V1P/13 | 10 | 2.2592 | 1.0967 | `stepsweep/nav_cd_span2b_e2_studentvlm_k10` |
| V163 | C04 / student VLM | V1P/13 | 2 | 2.1659 | 1.1614 | `stepsweep/nav_cd_span2b_e2_studentvlm_k2` |
| V164 | C08 expert / teacher VLM | V1P/13 | 1 | 1.5872 | 1.0974 | `stepsweep/nav_cd_span4b_e2_k1` |
| V165 | C08 expert / teacher VLM | V1P/13 | 10 | 1.6350 | 0.8660 | `stepsweep/nav_cd_span4b_e2_k10` |
| V166 | C08 expert / teacher VLM | V1P/13 | 2 | 1.5545 | 0.9327 | `stepsweep/nav_cd_span4b_e2_k2` |
| V167 | C08 / student VLM | V1P/13 | 1 | 1.7908 | 1.1702 | `stepsweep/nav_cd_span4b_e2_studentvlm_k1` |
| V168 | C08 / student VLM | V1P/13 | 10 | 1.9365 | 0.9982 | `stepsweep/nav_cd_span4b_e2_studentvlm_k10` |
| V169 | C08 / student VLM | V1P/13 | 2 | 1.8432 | 1.0503 | `stepsweep/nav_cd_span4b_e2_studentvlm_k2` |
| V170 | T / teacher VLM | V1/13 | 2 | 1.4470 | 0.9320 | `stepsweep/teachernav_k2` |
| V171 | Legacy teacher-labeled reference | Unverified | 1 | 1.6116 | 1.2436 | `teacher_eval/teacher_k1` |
| V172 | Legacy teacher-nav-labeled reference | Unverified | 1 | 1.6362 | 1.2817 | `teacher_eval/teachernav_k1` |
| V173 | Legacy teacher-nav-labeled reference | Unverified | 10 | 1.6976 | 0.7441 | `teacher_eval/teachernav_k10` |
| V174 | Legacy teacher-nav-labeled reference | Unverified | 9 | 1.6796 | 0.7473 | `teacher_eval/teachernav_k9` |

Exact sweep provenance (all student/expert checkpoints are step 6876, epoch 2):

| Eval rows | Config | Existing log under R |
|---|---|---|
| V161-V163 | [2B student-conditioned sweep](recipes/alpamayo1_5_distill/outputs/2026-09-09/18-01-42/.hydra/config.yaml) | `stepsweep/nav_cd_span2b_e2_studentvlm_0909-1801.log` |
| V164-V166 | [4B expert with teacher VLM](recipes/alpamayo1_5_distill/outputs/2026-09-09/10-49-47/.hydra/config.yaml) | `stepsweep/nav_cd_span4b_e2_0909-1049.log` |
| V167-V169 | [4B student-conditioned sweep](recipes/alpamayo1_5_distill/outputs/2026-09-09/17-55-59/.hydra/config.yaml) | `stepsweep/nav_cd_span4b_e2_studentvlm_0909-1755.log` |
| V170 | [Teacher two-step sweep](recipes/alpamayo1_5_distill/outputs/2026-09-10/14-03-03/.hydra/config.yaml) | `stepsweep/teachernav_0910-1402.log` |
| V171-V174 | No corresponding original config/log recovered; exact dataset, cameras, and stripping unknown | JSON/NPZ only; Euler-step counts confirmed by embedded archive descriptions |

The earlier 4B sweep launch at 10:46 used a different manifest and failed; the successful 10:49 config and completed result files are the provenance used here. The 2B sweep's config specifies a historical Cosmos-Reason2-2B base path with the C04 VLM checkpoint and layer mixer; retain that config when reproducing the old harness, rather than silently replacing it with a current launcher.

## Ablation Comparisons

These are descriptive comparisons of retained runs, not confidence intervals or multi-seed estimates. Eval IDs resolve to exact files above. **Clip ordering and GT arrays were byte-verified identical** across the eight block-ablation archives, eleven cotraining-comparison archives, and ten VF archives used below. Matching GT does not erase differences in conditioning or optimization.

### 1. Block-Output KD Objective

T110 training; V1S/13 evaluation; frozen, untuned 36-layer teacher expert; default stitched NFE (`D`) throughout. Rows compare losses within a model size. The 4B cache-normalization/free-run rows are **not epoch-matched to the parent**; the parent has no surviving epoch-1/2 checkpoint. They do not establish what those arms would do after four epochs. Cache normalization also changes the MSE-to-cosine balance, not just the displayed loss scale.

| Model / objective | Train epoch | ADE | minADE@6 | Eval |
|---|---:|---:|---:|---|
| 4B span curriculum, K09 | 4 | 2.0352 | 0.9016 | V143 |
| 4B span 1 + span-mix curriculum, K11 | 4 | 1.9729 | 0.8678 | V147 |
| 4B cache normalization, K07 | 1 | 2.3124 | 0.9816 | V139 |
| 4B cache normalization, K07 | 2 | 2.1398 | 0.9270 | V140 |
| 4B free-run weight 0.0002, K08 | 1 | 2.5732 | 1.1609 | V141 |
| 4B free-run weight 0.0002, K08 | 2 | 2.2874 | 1.0015 | V142 |
| 2B span curriculum, K02 | 4 | 2.3640 | 1.0208 | V130 |
| 2B span 1 + span-mix curriculum, K03 | 4 | 2.4157 | 1.0669 | V132 |

For the retained epoch-4 runs, mixing improves 4B minADE by 0.0338 m (3.75%) but worsens 2B by 0.0461 m (4.52%). This is narrower than any claim about all historical mixing experiments. K04's no-route data and K05/K06's 50k training set belong in the inventory, not in this controlled T110 loss table.

### 2. VLM Cotraining Scope and LR

T110 training, epoch 2, V1/13, NFE=2. All use GT flow matching. The 2B frozen-VLM reference is E01; the 4B frozen-VLM epoch-2 evaluations use V1S, so they are deliberately not inserted into this V1 table.

| Model / text layers / LR | Train epoch | ADE | minADE@6 | Eval |
|---|---:|---:|---:|---|
| 4B layers 27-35, 1x, E13 | 2 | 1.5299 | 1.0487 | V040 |
| 4B layers 27-35, 3x, E14 | 2 | 1.5171 | 1.0235 | V068 |
| 4B layers 27-35, 5x, E15 | 2 | 1.5371 | 1.0324 | V064 |
| 4B layers 19-35, 1x, E16 | 2 | 1.5153 | 1.0365 | V076 |
| 4B layers 19-35, 3x, E17 | 2 | 1.5271 | 1.0186 | V072 |
| 4B all 36 layers, 1x, E11 | 2 | 1.4791 | 1.0012 | V060 |
| 2B frozen VLM, E01 | 2 | 1.9037 | 1.3849 | V080 |
| 2B layers 21-27, 3x, E08 | 2 | 1.7983 | 1.1482 | V050 |
| 2B layers 21-27, 5x, E09 | 2 | 1.8348 | 1.1370 | V054 |
| 2B layers 14-27, 1x, E10 | 2 | 1.7396 | 1.1532 | V057 |
| 2B all 28 layers, 1x, E07 | 2 | 1.7368 | 1.1491 | V045 |

All-layer 4B cotraining has the lowest ADE and minADE in its displayed scope/LR sweep. For 2B, all-layer cotraining has the lowest ADE, while the deep-21 5x arm has the lowest minADE: selecting solely by best-of-six would give a different answer.

### 3. Cached Endpoint GT-Weight Sweep

Same E14 initialization, T110, V1/13, NFE=2. Teacher-endpoint weight remains 1. **Only weights 1.0 and 2.0 are epoch-matched here.** Weight 0.3 has only an epoch-1 checkpoint and should not be ranked as a controlled two-epoch comparison.

| Objective | Train epoch | ADE | minADE@6 | Eval |
|---|---:|---:|---:|---|
| E14 before endpoint stage, reference | 2 (EoS) | 1.5171 | 1.0235 | V068 |
| Endpoint + GT 0.3, C01 | 1 (CD) | 2.0202 | 0.9354 | V011 |
| Endpoint + GT 1.0, C02 | 2 (CD) | 1.6936 | 1.1263 | V019 |
| Endpoint + GT 2.0, C03 | 2 (CD) | 1.6627 | 1.2340 | V021 |

Increasing GT weight from 1 to 2 improves ADE by 0.0309 m but worsens minADE by 0.1077 m. The weight-0.3 result is another example where better minADE accompanies much worse typical-draw ADE.

### 4. EoS Initialization and Endpoint Adaptation

T110, V1S/13, NFE=2; stage epochs explicitly shown. C08 and C11 share endpoint + GT 0.5 but differ in whether the expert had EoS adaptation before endpoint training. The maskfix-labeled pair is a historical implementation variant, not a pure loss-only ablation.

| Variant | Stage epoch | ADE | minADE@6 | Eval |
|---|---:|---:|---:|---|
| Frozen-VLM EoS, E03 | 2 | 1.6797 | 1.2206 | V092 |
| Endpoint after E03, C08 | 2 | 1.8478 | 1.0609 | V009 |
| Endpoint without trained EoS init, C11 | 2 | 1.8695 | 0.9920 | V027 |
| Same no-EoS-init endpoint run, C11 | 4 | 1.8566 | 0.9931 | V024 |
| Maskfix-labeled EoS, E04 | 2 | 1.6283 | 1.0930 | V105 |
| Endpoint after E04, C10 | 2 | 1.8447 | 0.9891 | V006 |

### 5. Consistency Precision Variant

4B, same E11 initialization, same online two-step teacher/EMA objective, T110, V1S/13, NFE=1. Treat this as a precision/implementation variant comparison, not a new KD loss. Both GT and cached-endpoint loss weights are zero.

| Precision variant | CM epoch | ADE | minADE@6 | Eval |
|---|---:|---:|---:|---|
| Original bf16, C06 | 1 | 1.5644 | 1.0646 | V032 |
| Original bf16, C06 | 2 | 1.5684 | 1.0704 | V033 |
| fp16 AMP / fp32 master weights, C07 | 1 | 1.5420 | 1.0443 | V036 |
| fp16 AMP / fp32 master weights, C07 | 2 | 1.5369 | 1.0376 | V037 |

At epoch 2 the corrected-precision variant is lower by 0.0315 m ADE and 0.0328 m minADE. The E11 matched two-step reference is V084 (ADE 1.4788, minADE 0.9998); moving to C07's one step costs about 3.9% ADE while halving expert NFE. This is an NFE saving, not a measured wall-clock speedup.

### 6. Full Available Validation: EoS vs Consistency

**Headline deployment comparison:** VF/13, 23,331 identical clips and GT arrays, fixed t0=5.1 s, navigation distance stripped, six samples. EoS and CM each use their epoch-2 checkpoint; CM adds a further training stage. Teacher training data are not inferred here. Model size and NFE are explicit, so this is a pipeline/budget comparison rather than a loss-only ablation.

| Model / stage | NFE | ADE | minADE@6 | Eval |
|---|---:|---:|---:|---|
| 10B pretrained teacher | 1 | 1.6450 | 1.2327 | V156 |
| 10B pretrained teacher | 2 | 1.4006 | 0.8628 | V157 |
| 4B EoS cotrain, E11 | 1 | 1.5824 | 1.3600 | V082 |
| 4B EoS cotrain, E11 | 2 | 1.4683 | 1.0050 | V083 |
| 4B CM, C07 | 1 | 1.5142 | 1.0368 | V034 |
| 4B CM, C07 | 2 | 1.5765 | 0.8870 | V035 |
| 2B EoS cotrain, E07 | 1 | 1.7947 | 1.4814 | V077 |
| 2B EoS cotrain, E07 | 2 | 1.6665 | 1.0935 | V078 |
| 2B CM, C05 | 1 | 1.7116 | 1.1352 | V030 |
| 2B CM, C05 | 2 | 1.7920 | 0.9699 | V031 |

At NFE=1, CM improves both metrics over its same-size one-step EoS baseline. Relative to two-step EoS, one-step CM increases ADE by about **3.1% for 4B** and **2.7% for 2B**, with half the expert NFE. Running CM at two steps lowers minADE further but increases ADE for both sizes; the best-of-six metric alone would conceal that tradeoff.

### 7. Camera-Set Transfer

T110, epoch-2 cotrained models, V1S, NFE=2. This tests camera conditioning, not KD loss. Compare models under the same evaluation-camera column.

| Model / training cameras -> evaluation cameras | Epoch | ADE | minADE@6 | Eval |
|---|---:|---:|---:|---|
| 4B E12, 0123 -> 0123 | 2 | 1.4514 | 0.9968 | V111 |
| 4B E11, 13 -> 0123 | 2 | 2.1151 | 1.4064 | V120 |
| 4B E12, 0123 -> 13 | 2 | 2.0480 | 1.4125 | V123 |
| 4B E11, 13 -> 13 | 2 | 1.4788 | 0.9998 | V084 |
| 2B E06, 0123 -> 0123 | 2 | 1.6447 | 1.0994 | V108 |
| 2B E07, 13 -> 0123 | 2 | 2.6132 | 1.8057 | V114 |
| 2B E06, 0123 -> 13 | 2 | 2.1402 | 1.3767 | V117 |

The two-camera-trained 2B same-camera NFE=2 evaluation is V045, but it uses V1 rather than V1S, so it is not silently inserted into this table.

### 8. Historical Sweep: Conditioning Source

V1P/13, NFE=2, same step-sweep harness, epoch-2 endpoint checkpoints. The first two rows change VLM conditioning while keeping the C08 expert fixed. The 2B row is a model-size/pipeline comparison, not a loss-only ablation. Teacher-VLM conditioning is a diagnostic control and must not be presented as the deployable 4B student's accuracy.

| Expert / conditioning | CD epoch | ADE | minADE@6 | Eval |
|---|---:|---:|---:|---|
| C08 expert + teacher VLM | 2 | 1.5545 | 0.9327 | V166 |
| C08 expert + its 4B student VLM | 2 | 1.8432 | 1.0503 | V169 |
| C04 expert + its 2B student VLM | 2 | 2.1659 | 1.1614 | V163 |

## Exclusions and Limitations

- No full local latent/KAVA, CE-only, logit-KD-only, direct-KV-only, depth-ladder, or pruned-expert training run was verified. Historical tables about such experiments are not reproduced as local experiments. The existing KV-labeled directory only has logs up to about epoch 0.21.
- Excluded all explicit smoke/precision probes and runs that did not establish one completed epoch. In particular, the locality IO run associated with `kdtrain_20700.out` stopped at step 3435/3438 of its first epoch; its printed epoch rounded to 1.0, which is not completion. The RAM-cache run reached about epoch 0.82 in logs, not a full epoch. These were not counted as ablation runs.
- Excluded all 13 short evaluation JSONs: each contains only 10 or 20 records. They are not substitutes for the 1,000-clip or 23,331-clip results.
- E02 passed the training-duration test but has no full result tied to its exact local checkpoint in this inventory. Missing results are deliberately not filled from neighboring models. C04 does have the full nested sweep results V161-V163.
- Earlier checkpoint directories may have been pruned even where their evaluation artifacts survive. Training inclusion is based on a surviving full-epoch state; an older evaluation remains a historical measurement, not a promise its checkpoint can still be loaded.
- Teacher rows refer to locally retained pretrained-teacher evaluations. This audit does not claim the teacher was trained locally or establish its pretraining dataset.
- Metrics are saved JSON values, not a retrospective uniform re-evaluation. Historical code revisions, precision, hardware, prompt handling, and stochastic sampling can still matter. V1, V1S, V1P, V0, and VF remain separate protocols. No significance claim or repeated-seed estimate is made.

## Audit Checks

- Verified all 39 training directories, their saved epoch/step states, and parent-checkpoint paths.
- Matched all 160 root-level evaluation JSONs to their local directories and logged dataset/checkpoint/NFE, plus 10 nested sweep results to configs/logs. Retained 4 additional local legacy teacher files with explicitly unverified dataset provenance. Recomputed both reported means for all 174 directly from per-clip JSON.
- Verified unique clip IDs in every result and matching clip sets within the eight dataset/camera/stripping groups.
- Verified matching clip order and GT archive bytes for the main block-loss, cotraining, and full-validation tables; five missing historical NPZ archives are identified above.
- No checkpoints, source code, existing experiment notes, or notebook outputs were changed to create this book.