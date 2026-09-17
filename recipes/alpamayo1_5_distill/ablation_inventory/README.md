# Distillation Ablation Inventory

Local artifact audit, 2026-09-17. Metrics recomputed from saved per-clip JSON, not copied from documentation.

**52 training records in 51 directories; 142 non-smoke trained-model evaluations; 42 teacher/baseline evaluations.**

## Scope And Rules

- Training and result paths below are relative to `/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training` (the artifact root).
- Scope: the distillation recipe artifact root, its top-level and camera/step-sweep per-clip results, and historical Hydra configs across neighboring recipes. No remote W&B/Hugging Face history was fetched.
- Keep training with observed epoch >= 1, including interrupted longer runs and resumed stages. Configured epochs alone do not qualify a run. Epoch values on resumes are cumulative, not additional epochs.
- Explicit tiny-data smoke jobs are excluded even when their counters exceed one epoch. Completed 2-30-step jobs were identified this way; see completed_log_audit.csv for the full list.
- Exclude evaluated checkpoints below one epoch, explicit smoke evaluations, and tiny diagnostic sets (<100 clips). Keep 500-clip KAVA evaluations and the 1,000-clip validation studies.
- A retained checkpoint supplies the epoch. If deleted, estimate it from a retained step/epoch ratio and flag the CSV row. This estimate assumes unchanged steps per epoch.
- ADE and minADE are metres, lower is better. Different heads, cameras, expert depth, annotation timing, prompt flags, or NFE are separate protocols, not clean loss-only comparisons.
- NFE is the number of denoising steps. Token-head rows have no NFE. Expert rows without an explicit override use the documented 10-step default, marked in the CSV.
- Navigation annotations use GT future direction (`--horizon-start 0` in the recorded study). These are annotation-conditioned, potentially label-leaking evaluations, not leak-free planner-route results. See [NAVTEXT_SAMPLING.md](../NAVTEXT_SAMPLING.md#12-comparability-and-the-label-leak).
- Repeated evaluation filenames/trajectory-export reruns are retained as separate artifacts, not independent seeds. Shared clip IDs do not imply shared timestamps or prompts.
- The KAVA directory was reused: checkpoint-7191 belongs to a historical three-epoch run; checkpoint-2397 belongs to the later one-epoch rerun. An `e3_` evaluation filename does not prove three epochs.
- Raw configs/logs can disagree. Completed training logs take precedence where recovered; CSV source columns identify the evidence. Five trained-model sweep protocols use a sibling config with checkpoint identity verified in the launch log. Two teacher camera references lack recoverable dataset configs and are explicitly marked unknown.

## Matched 4B Loss Ablation

One training epoch on the original LCDrive train split (38,340 clips); 1,000 default-keyframe validation clips, four cameras, full frozen teacher expert. This is the cleanest loss-only comparison. All are no-CoT student runs. Teacher is a reference, not a trained arm.

| Objective | ADE | minADE | Result JSON |
|---|---|---|---|
| Teacher reference | 1.3039 | 0.5776 | `stitch_4b_teacher.json` |
| CE | 12.5500 | 6.9948 | `stitch_4b_ce.json` |
| CE + logit KD | 17.4294 | 11.3750 | `stitch_4b_kd.json` |
| CE + logit KD + KV | 5.9970 | 2.9554 | `stitch_4b_kv.json` |
| CE + KV | 4.9100 | 2.7601 | `stitch_4b_cekv.json` |
| KV only | 4.1085 | 2.6313 | `stitch_4b_kvonly.json` |
| KV only, depth-banded | 5.5903 | 3.1763 | `stitch_4b_kvband_checkpoint-1598.json` |
| Block, t=0 | 3.5824 | 2.0293 | `stitch_4b_blockonly_checkpoint-1598.json` |
| Block, random t | 3.4005 | 1.9518 | `stitch_4b_blockrandt_checkpoint-1598.json` |

On this expert endpoint, block matching outperforms direct KV matching; logit KD does not improve the CE control. Do not transfer that ranking to the token head. The full tables retain the token-head reversal, later epochs, 2B field/span/layer-mix studies, KAVA controls, and endpoint/GT-weight/NFE sweeps.

## Datasets

All identified training/evaluation datasets are LCDrive subsets of PhysicalAI-AV. The original train manifest has 38,340 clips; the KAVA teacher cache covers 38,336 of them (cache coverage is not necessarily dataset length). The navigation training file contains 50,000 event anchors, not 50,000 unique clips. The usual eval subset is 1,000 clips; KAVA used 500 evaluated clips from the validation manifest.

Dataset IDs include cameras, timestamp selection and prompt flags. Full absolute manifest paths and dataset classes are in both CSVs.

| ID | Clip manifest | Annotation file | Chunks | Cameras | Timing | Route slot | Camera/frame IDs |
|---|---|---|---|---|---|---|---|
| D01 | lcdrive_train_clip_uuids.txt | nav_lcdrive_train_anchors_50k_turnpreserved.json | 0-3146 | [1, 3] | annotation anchored | True | True/True |
| D02 | lcdrive_train_clip_uuids.txt | none | 0-3146 | [0, 1, 2, 3] | default keyframe | False | True/True |
| D03 | lcdrive_train_clip_uuids.txt | none | 0-3146 | [0, 1, 2, 3] | default keyframe | False | False/False |
| D04 | lcdrive_train_clip_uuids.txt | none | 0-3146 | [1, 3] | default keyframe | False | True/True |
| D05 | lcdrive_val_mysubset_1k_clip_uuids.txt | none | 0-3146 | [1, 3] | default keyframe | False | True/True |
| D06 | lcdrive_val_mysubset_1k_clip_uuids.txt | none | 0-3146 | [1] | default keyframe | False | True/True |
| D07 | lcdrive_val_mysubset_1k_clip_uuids.txt | none | 0-3146 | [0, 1, 2] | default keyframe | False | True/True |
| D08 | lcdrive_val_mysubset_1k_clip_uuids.txt | none | 0-3146 | [0, 1, 2, 3] | default keyframe | False | True/True |
| D09 | unknown | none |  | [0, 1, 2] | unknown | False | unknown/unknown |
| D10 | unknown | none |  | [0, 1, 2, 3] | unknown | False | unknown/unknown |
| D11 | lcdrive_val_mysubset_1k_clip_uuids.txt | nav_lcdrive_val_mysubset_1k.json | 0-3146 | [1, 3] | annotation anchored | True | True/True |
| D12 | lcdrive_val_clip_uuids.txt | none | 0-3146 | [0, 1, 2, 3] | default keyframe | False | False/False |
| D13 | lcdrive_val_mysubset_1k_clip_uuids.txt | nav_lcdrive_val_mysubset_1k.json | 0-3146 | [1, 3] | annotation anchored | True | True/True |
| D14 | lcdrive_val_mysubset_1k_clip_uuids.txt | nav_lcdrive_val_mysubset_1k_nonav.json | 0-3146 | [1, 3] | annotation anchored | True | True/True |
| D15 | lcdrive_val_mysubset_1k_clip_uuids.txt | none | 0-3146 | [1, 3] | default keyframe | False | True/True |
| D16 | lcdrive_val_mysubset_1k_clip_uuids.txt | nav_lcdrive_val_mysubset_1k_nonav.json | 0-3146 | [1, 3] | annotation anchored | True | True/True |

Route slot means `route` occurs in the prompt component order; `_nonav` annotation files intentionally leave its content empty. Repeated-looking dataset rows may differ in dataset class; see the CSV.


## Training Directory Index

Loss weights below are the recorded active objective settings. The CSV also retains the full loss config, model source, initialization/resume path, learning rate, and LR multipliers. EoS trains the action expert on a frozen student VLM and is a supervised control, not another VLM-KD loss.

### Consistency / endpoint distillation

| Run | Directory | Objective | Epoch reached | Train data | Eval files |
|---|---|---|---|---|---|
| T01 | `output_cd_eos2b_endpoint_gt01_nav_e2_bs32_20260830` | 0*CD + 1*teacher endpoint + 0.1*GT endpoint; teacher=cached_full | 2 | D01 | 2 |
| T02 | `output_cd_eos2b_endpoint_gt03_nav_e2_bs32_20260830` | 0*CD + 1*teacher endpoint + 0.3*GT endpoint; teacher=cached_full | 2 | D01 | 2 |
| T03 | `output_cd_eos2b_endpoint_gt05_nav_e2_bs32_20260830` | 0*CD + 1*teacher endpoint + 0.5*GT endpoint; teacher=cached_full | 3 | D01 | 7 |
| T04 | `output_cd_eos2b_endpoint_gt05_term03_nav_e2_bs32_20260831` | 0*CD + 1*teacher endpoint + 0.5*GT endpoint; teacher=cached_full; terminal fraction=0.3 | 2 | D01 | 2 |
| T05 | `output_cd_eos2b_endpoint_gt05_term06_nav_e2_bs32_20260831` | 0*CD + 1*teacher endpoint + 0.5*GT endpoint; teacher=cached_full; terminal fraction=0.6 | 2 | D01 | 2 |
| T06 | `output_cd_eos2b_endpoint_gt10_nav_e2_bs32_20260830` | 0*CD + 1*teacher endpoint + 1*GT endpoint; teacher=cached_full | 2 | D01 | 3 |
| T07 | `output_cd_eos2b_fullteacher_cache_x0gtw0.1_nav_e2_bs32_20260828` | 1*CD + 0*teacher endpoint + 0.1*GT endpoint; teacher=cached_full | 2 | D01 | 4 |
| T08 | `output_cd_eos2b_fullteacher_endpoint_nav_e2_bs32_20260829` | 0*CD + 1*teacher endpoint + 0*GT endpoint; teacher=cached_full | 2 | D01 | 4 |
| T09 | `output_cd_eos2b_mix_endpoint_gt03_nav_e2` | 0*CD + 1*teacher endpoint + 0.3*GT endpoint; teacher=cached_full | 2 | D01 | 2 |
| T10 | `output_cd_eos2b_nav_e2_bs32_20260827` | 1*CD + 0*teacher endpoint + 0*GT endpoint; teacher=online/self | 2 | D01 | 2 |
| T11 | `output_cd_eos2b_x0gtw0.1_nav_e2_bs32_20260827` | 1*CD + 0*teacher endpoint + 0.1*GT endpoint; teacher=online/self | 2 | D01 | 4 |
| T12 | `output_cd_expert_2cam_nav_lcdrive` | 1*CD + 0*teacher endpoint + 0*GT endpoint; teacher=online/self | 1 | D01 | 5 |
| T13 | `output_cd_expert_2cam_nav_lcdrive_fixed_20260825_bs32` | 1*CD + 0*teacher endpoint + 0*GT endpoint; teacher=online/self | 1 | D01 | 2 |
| T14 | `output_cd_expert_2cam_nav_lcdrive_fixed_20260825_bs32_stage2` | 1*CD + 0*teacher endpoint + 0*GT endpoint; teacher=online/self | 1 | D01 | 4 |
| T15 | `output_cd_expert_pruned28_2cam_nav_lcdrive_bs32_e2_20260826` | 1*CD + 0*teacher endpoint + 0*GT endpoint; teacher=online/self | 2 | D01 | 6 |

### Expert-on-student / supervised controls

| Run | Directory | Objective | Epoch reached | Train data | Eval files |
|---|---|---|---|---|---|
| T16 | `output_eos_2b_lcdrive` | Supervised action flow matching; VLM frozen (EoS control) | 1 | D02 | 6 |
| T17 | `output_eos_2b_mix_nav_lcdrive` | Supervised action flow matching; VLM frozen (EoS control) | 5 | D01 | 4 |
| T18 | `output_eos_2b_nav_e3_clean_maskfix_e2_lcdrive` | Supervised action flow matching; VLM frozen (EoS control) | 2 | D01 | 4 |
| T19 | `output_eos_2b_nav_lcdrive` | Supervised action flow matching; VLM frozen (EoS control) | 5 | D01 | 3 |
| T20 | `output_expert_on_student_lcdrive` | Supervised action flow matching; VLM frozen (EoS control) | 1 | D02 | 5 |

### KAVA latent-slot distillation and controls

| Run | Directory | Objective | Epoch reached | Train data | Eval files |
|---|---|---|---|---|---|
| T21 | `output_kava_T2_bs48_lcdrive` | CE + 0*latent + 1*KV; M=8, T=2, l1 | 1 | D03 | 2 |
| T22 | `output_kava_noaux_T2_bs48_lcdrive` | CE + 0*latent + 0*KV; M=8, T=2, l1 | 1 | D03 | 2 |
| T50 | `output_stage1_kava_control_lcdrive` | CE + 0*latent + 0*KV; M=8, T=1, l1 | 1 | D03 | 1 |
| T51 | `output_stage1_kava_cosmos2b_lcdrive` | CE + 0*latent + 1*KV; M=8, T=1, l1 | 1 | D03 | 2 |
| T52 | `output_stage1_kava_cosmos2b_lcdrive` | CE + 0.01*latent + 1*KV; M=8, T=1, smooth_l1 | 3 | D03 | 3 |

### VLM cache / block / field distillation and CE controls

| Run | Directory | Objective | Epoch reached | Train data | Eval files |
|---|---|---|---|---|---|
| T23 | `output_kd_2b_block2b2cam_lcdrive` | 1*block; t=beta, span=1 | 3 | D04 | 3 |
| T24 | `output_kd_2b_block2b_lcdrive` | 1*block; t=beta, span=1 | 2.19 | D02 | 1 |
| T25 | `output_kd_2b_block2bdepth_e2_lcdrive` | 1*block; t=beta, span=1, block_layer_weights=ladder | 2 | D04 | 1 |
| T26 | `output_kd_2b_block2bdepth_ladder_add_e2_lcdrive` | 1*block; t=beta, span=1, block_layer_weights=ladder_add | 2 | D04 | 1 |
| T27 | `output_kd_2b_block2bmix_m7w1.0_e3_lcdrive` | 1*block; t=beta, span=1, block_span_mix=7, block_span_mix_weight=1.0 | 3 | D04 | 2 |
| T28 | `output_kd_2b_block2bspan_e2_lcdrive` | 1*block; t=beta, span=7 | 2 | D04 | 1 |
| T29 | `output_kd_2b_block2bspan_e3_lcdrive` | 1*block; t=beta, span=14 | 3 | D04 | 1 |
| T30 | `output_kd_2b_block2bspan_e4_lcdrive` | 1*block; t=beta, span=28 | 4 | D04 | 1 |
| T31 | `output_kd_2b_block2bspan_lcdrive` | 1*block; t=beta, span=1 | 1 | D04 | 1 |
| T32 | `output_kd_2b_field2b_lcdrive` | 1*field | 1 | D04 | 1 |
| T33 | `output_kd_2b_mix2bnav_m9w1.0_lcdrive` | 1*block; t=beta, span=1, block_span_mix=9, block_span_mix_weight=1.0, layer mixing | 5 | D01 | 5 |
| T34 | `output_kd_2b_mix2bnav_m9w1.0_plr100.0_lcdrive` | 1*block; t=beta, span=1, block_span_mix=9, block_span_mix_weight=1.0, layer mixing | 5 | D01 | 5 |
| T35 | `output_kd_2b_mixpin2bnav_m9w1.0_plr100.0_lcdrive` | 1*block; t=beta, span=1, block_span_mix=9, block_span_mix_weight=1.0, layer mixing | 5 | D01 | 5 |
| T36 | `output_kd_2b_nav2bmix_m7w1.0_e5_clean_maskfix_lcdrive` | 1*block; t=beta, span=1, block_span_mix=7, block_span_mix_weight=1.0 | 5 | D01 | 7 |
| T37 | `output_kd_2b_nav2bmix_m7w1.0_e6_efficient_lcdrive` | 1*block; t=beta, span=1, block_span_mix=7, block_span_mix_weight=1.0 | 6 | D01 | 1 |
| T38 | `output_kd_2b_nav2bmix_m7w1.0_lcdrive` | 1*block; t=beta, span=1, block_span_mix=7, block_span_mix_weight=1.0 | 5 | D01 | 3 |
| T39 | `output_kd_4b_blockonly_e3_lcdrive` | 1*block; t=zero, span=1 | 3 | D02 | 2 |
| T40 | `output_kd_4b_blockonly_lcdrive` | 1*block; t=zero, span=1 | 1 | D02 | 1 |
| T41 | `output_kd_4b_blockrandt_e3_lcdrive` | 1*block; t=beta, span=1 | 3 | D02 | 2 |
| T42 | `output_kd_4b_blockrandt_lcdrive` | 1*block; t=beta, span=1 | 1 | D02 | 1 |
| T43 | `output_kd_4b_ce_lcdrive` | 1*CE | 1 | D02 | 2 |
| T44 | `output_kd_4b_cekv_lcdrive` | 1*CE + 45.409*KV | 1 | D02 | 2 |
| T45 | `output_kd_4b_kd_lcdrive` | 1*CE + 0.70921*logit KD | 1 | D02 | 2 |
| T46 | `output_kd_4b_kv_lcdrive` | 1*CE + 0.70921*logit KD + 45.409*KV | 1 | D02 | 2 |
| T47 | `output_kd_4b_kvband_lcdrive` | 45.409*KV; kv_layer_bands=[0.01, 0.96, 2.03] | 1 | D02 | 2 |
| T48 | `output_kd_4b_kvonly_e3_lcdrive` | 45.409*KV | 3 | D02 | 2 |
| T49 | `output_kd_4b_kvonly_lcdrive` | 45.409*KV | 1 | D02 | 2 |

## Saved Evaluation Results

Every included evaluation is listed, including intermediate checkpoints >=1 epoch and controlled inference changes. Refer to the JSON filename for the specific variant; use the CSV for exact checkpoint and expert sources. `Txx` refers to the training-directory index above.

### Consistency / endpoint distillation

| Run | Epoch | Eval data | Head | NFE | n | ADE | minADE | Result JSON |
|---|---|---|---|---|---|---|---|---|
| T08 | 1 | D11 | action expert | 1 | 1000 | 2.8423 | 1.5877 | `cd_eos2b_endpoint_ep1_eos_checkpoint-1563_nfe1.json` |
| T08 | 1 | D11 | action expert | 2 | 1000 | 2.8573 | 1.3618 | `cd_eos2b_endpoint_ep1_eos_checkpoint-1563_nfe2.json` |
| T08 | 2 | D11 | action expert | 1 | 1000 | 2.8598 | 1.5928 | `cd_eos2b_endpoint_ep2_eos_checkpoint-3126_nfe1.json` |
| T08 | 2 | D11 | action expert | 2 | 1000 | 2.8433 | 1.3429 | `cd_eos2b_endpoint_ep2_eos_checkpoint-3126_nfe2.json` |
| T01 | 2 | D11 | action expert | 1 | 1000 | 2.7896 | 1.6094 | `cd_eos2b_endpoint_gt01_ep2_eos_checkpoint-3126_nfe1.json` |
| T01 | 2 | D11 | action expert | 2 | 1000 | 2.6996 | 1.3214 | `cd_eos2b_endpoint_gt01_ep2_eos_checkpoint-3126_nfe2.json` |
| T02 | 2 | D11 | action expert | 1 | 1000 | 2.6968 | 1.6629 | `cd_eos2b_endpoint_gt03_ep2_eos_checkpoint-3126_nfe1.json` |
| T02 | 2 | D11 | action expert | 2 | 1000 | 2.5141 | 1.3344 | `cd_eos2b_endpoint_gt03_ep2_eos_checkpoint-3126_nfe2.json` |
| T03 | 1 | D11 | action expert | 2 | 1000 | 2.3967 | 1.4590 | `cd_eos2b_endpoint_gt05_ep1_eos_checkpoint-1563_nfe2.json` |
| T03 | 2 | D11 | action expert | 1 | 1000 | 2.5720 | 1.7620 | `cd_eos2b_endpoint_gt05_ep2_eos_checkpoint-3126_nfe1.json` |
| T03 | 2 | D11 | action expert | 2 | 1000 | 2.3726 | 1.4276 | `cd_eos2b_endpoint_gt05_ep2_eos_checkpoint-3126_nfe2.json` |
| T03 | 3 | D11 | action expert | 1 | 1000 | 2.5643 | 1.7516 | `cd_eos2b_endpoint_gt05_ep3_eos_checkpoint-4689_nfe1.json` |
| T03 | 3 | D11 | action expert | 2 | 1000 | 2.3739 | 1.4299 | `cd_eos2b_endpoint_gt05_ep3_eos_checkpoint-4689_nfe2.json` |
| T06 | 1 | D11 | action expert | 2 | 1000 | 2.3792 | 1.6033 | `cd_eos2b_endpoint_gt10_ep1_eos_checkpoint-1563_nfe2.json` |
| T06 | 2 | D11 | action expert | 1 | 1000 | 2.5182 | 1.8731 | `cd_eos2b_endpoint_gt10_ep2_eos_checkpoint-3126_nfe1.json` |
| T06 | 2 | D11 | action expert | 2 | 1000 | 2.3107 | 1.5345 | `cd_eos2b_endpoint_gt10_ep2_eos_checkpoint-3126_nfe2.json` |
| T10 | 2 | D11 | action expert | 1 | 1000 | 2.5909 | 1.7932 | `cd_eos2b_final_eos_checkpoint-3126_nfe1.json` |
| T10 | 2 | D11 | action expert | 2 | 1000 | 2.6459 | 1.7038 | `cd_eos2b_final_eos_checkpoint-3126_nfe2.json` |
| T07 | 1 | D11 | action expert | 1 | 1000 | 2.5347 | 1.9206 | `cd_eos2b_fullteacher_ep1_eos_checkpoint-1563_nfe1.json` |
| T07 | 1 | D11 | action expert | 2 | 1000 | 2.4668 | 1.6564 | `cd_eos2b_fullteacher_ep1_eos_checkpoint-1563_nfe2.json` |
| T07 | 2 | D11 | action expert | 1 | 1000 | 2.5630 | 1.8028 | `cd_eos2b_fullteacher_ep2_eos_checkpoint-3126_nfe1.json` |
| T07 | 2 | D11 | action expert | 2 | 1000 | 2.5304 | 1.5837 | `cd_eos2b_fullteacher_ep2_eos_checkpoint-3126_nfe2.json` |
| T04 | 2 | D11 | action expert | 1 | 1000 | 2.6461 | 1.5591 | `cd_eos2b_gt05_term03_ep2_eos_checkpoint-3126_nfe1.json` |
| T04 | 2 | D11 | action expert | 2 | 1000 | 2.5172 | 1.3567 | `cd_eos2b_gt05_term03_ep2_eos_checkpoint-3126_nfe2.json` |
| T05 | 2 | D11 | action expert | 1 | 1000 | 2.6402 | 1.5571 | `cd_eos2b_gt05_term06_ep2_eos_checkpoint-3126_nfe1.json` |
| T05 | 2 | D11 | action expert | 2 | 1000 | 2.5582 | 1.3581 | `cd_eos2b_gt05_term06_ep2_eos_checkpoint-3126_nfe2.json` |
| T09 | 2 | D11 | action expert | 1 | 1000 | 2.1013 | 1.3513 | `cd_eos2b_mix_endpoint_gt03_ep2_eos_checkpoint-3126_nfe1.json` |
| T09 | 2 | D11 | action expert | 2 | 1000 | 2.1613 | 1.1336 | `cd_eos2b_mix_endpoint_gt03_ep2_eos_checkpoint-3126_nfe2.json` |
| T11 | 1 | D11 | action expert | 1 | 1000 | 2.5204 | 1.9165 | `cd_eos2b_x0gtw0.1_e1_eos_checkpoint-1563_nfe1.json` |
| T11 | 1 | D11 | action expert | 2 | 1000 | 2.4552 | 1.6682 | `cd_eos2b_x0gtw0.1_e1_eos_checkpoint-1563_nfe2.json` |
| T11 | 2 | D11 | action expert | 1 | 1000 | 2.5312 | 1.7900 | `cd_eos2b_x0gtw0.1_e2_eos_checkpoint-3126_nfe1.json` |
| T11 | 2 | D11 | action expert | 2 | 1000 | 2.5204 | 1.6101 | `cd_eos2b_x0gtw0.1_e2_eos_checkpoint-3126_nfe2.json` |
| T15 | 2 | D13 | action expert | 1 | 1000 | 2.8872 | 2.0570 | `stepsweep/nav_2b_e3_cdp28_e2_k1.json` |
| T15 | 2 | D13 | action expert | 2 | 1000 | 3.0210 | 1.8686 | `stepsweep/nav_2b_e3_cdp28_e2_k2.json` |
| T13 | 1 | D13 | action expert | 1 | 1000 | 1.4983 | 1.2002 | `stepsweep/nav_cd1563_k1.json` |
| T13 | 1 | D13 | action expert | 2 | 1000 | 1.4886 | 0.8939 | `stepsweep/nav_cd1563_k2.json` |
| T12 | 1 | D13 | action expert | 1 | 1000 | 1.9161 | 1.2617 | `stepsweep/nav_cd2084_k1.json` |
| T12 | 1 | D13 | action expert | 10 | 1000 | 2.1610 | 0.8954 | `stepsweep/nav_cd2084_k10.json` |
| T12 | 1 | D13 | action expert | 2 | 1000 | 1.8392 | 1.0283 | `stepsweep/nav_cd2084_k2.json` |
| T12 | 1 | D13 | action expert | 3 | 1000 | 1.8609 | 0.9607 | `stepsweep/nav_cd2084_k3.json` |
| T12 | 1 | D13 | action expert | 5 | 1000 | 1.9543 | 0.9154 | `stepsweep/nav_cd2084_k5.json` |
| T15 | 1.024 | D13 | action expert | 1 | 1000 | 2.5064 | 1.4840 | `stepsweep/nav_cdp28_e1_s1600_k1.json` |
| T15 | 1.024 | D13 | action expert | 2 | 1000 | 2.7891 | 1.9246 | `stepsweep/nav_cdp28_e1_s1600_k2.json` |
| T15 | 2 | D13 | action expert | 1 | 1000 | 2.4730 | 1.4703 | `stepsweep/nav_cdp28_e2_s3126_k1.json` |
| T15 | 2 | D13 | action expert | 2 | 1000 | 2.6862 | 1.8203 | `stepsweep/nav_cdp28_e2_s3126_k2.json` |
| T14 | 1 | D13 | action expert | 1 | 1000 | 1.4812 | 1.1923 | `stepsweep/nav_cds2_1563_k1.json` |
| T14 | 1 | D13 | action expert | 2 | 1000 | 1.4924 | 0.8945 | `stepsweep/nav_cds2_1563_k2.json` |
| T14 | 1 | D13 | action expert | 1 | 1000 | 2.8896 | 1.7223 | `stepsweep/nav_cds2_1563_pruned_4-10-13-15-19-25-27-34_k1.json` |
| T14 | 1 | D13 | action expert | 2 | 1000 | 3.2074 | 2.1873 | `stepsweep/nav_cds2_1563_pruned_4-10-13-15-19-25-27-34_k2.json` |
| T03 | 2 | D11 | action expert | 1 | 1000 | 17.8343 | 11.4905 | `untrainedvlm_cdgt05expert_nfe1.json` |
| T03 | 2 | D11 | action expert | 2 | 1000 | 20.3689 | 16.3022 | `untrainedvlm_cdgt05expert_nfe2.json` |

### Expert-on-student / supervised controls

| Run | Epoch | Eval data | Head | NFE | n | ADE | minADE | Result JSON |
|---|---|---|---|---|---|---|---|---|
| T16 | 1 | D06 | action expert | 10 | 1000 | 9.6678 | 4.9886 | `camsweep/eos2b_1cam.json` |
| T16 | 1 | D05 | action expert | 10 | 1000 | 4.8798 | 2.7861 | `camsweep/eos2b_2cam.json` |
| T16 | 1 | D07 | action expert | 10 | 1000 | 4.1863 | 2.4470 | `camsweep/eos2b_3cam.json` |
| T16 | 1 | D08 | action expert | 10 | 1000 | 3.7980 | 2.3611 | `camsweep/eos2b_4cam.json` |
| T20 | 1 | D06 | action expert | 10 | 1000 | 4.7588 | 2.5844 | `camsweep/eos4b_1cam.json` |
| T20 | 1 | D05 | action expert | 10 | 1000 | 3.2990 | 1.6512 | `camsweep/eos4b_2cam.json` |
| T20 | 1 | D07 | action expert | 10 | 1000 | 3.5659 | 2.0997 | `camsweep/eos4b_3cam.json` |
| T20 | 1 | D08 | action expert | 10 | 1000 | 2.5667 | 1.4066 | `camsweep/eos4b_4cam.json` |
| T18 | 2 | D11 | action expert | 1 | 1000 | 2.5830 | 2.0590 | `eos2b_baseline_eos_checkpoint-4168_nfe1.json` |
| T18 | 2 | D11 | action expert | 2 | 1000 | 2.4661 | 1.8148 | `eos2b_baseline_eos_checkpoint-4168_nfe2.json` |
| T17 | 5 | D11 | action expert | 10 | 1000 | 2.1092 | 1.1292 | `eos_2b_mix_nav_eos_checkpoint-10420.json` |
| T17 | 1 | D11 | action expert | 10 | 1000 | 2.1325 | 1.1445 | `eos_2b_mix_nav_eos_checkpoint-2084.json` |
| T17 | 2 | D11 | action expert | 10 | 1000 | 2.1045 | 1.1223 | `eos_2b_mix_nav_eos_checkpoint-4168.json` |
| T17 | 3 | D11 | action expert | 10 | 1000 | 2.1120 | 1.1328 | `eos_2b_mix_nav_eos_checkpoint-6252.json` |
| T19 | 5 | D11 | action expert | 10 | 1000 | 5.2087 | 2.6216 | `eos_2b_nav_control_checkpoint-7815.json` |
| T18 | 1 | D11 | action expert | 10 | 1000 | 2.4962 | 1.4092 | `eos_2b_nav_e3_clean_maskfix_e2_eos_checkpoint-2084.json` |
| T18 | 2 | D11 | action expert | 10 | 1000 | 2.4859 | 1.3974 | `eos_2b_nav_e3_clean_maskfix_e2_eos_checkpoint-4168.json` |
| T19 | 5 | D11 | action expert | 10 | 1000 | 3.2444 | 2.0753 | `eos_2b_nav_eos_checkpoint-7815.json` |
| T19 | 5 | D11 | action expert | 2 | 1000 | 3.3397 | 2.7387 | `eos_2b_nav_eos_checkpoint-7815_nfe2.json` |
| T16 | 1 | D08 | action expert | 10 | 240 | 4.0070 | 2.5609 | `stitch_2b_eos_ROLLOUT_subset.json` |
| T16 | 1 | D08 | action expert | 10 | 1000 | 3.8103 | 2.3668 | `stitch_2b_eos_checkpoint-1598.json` |
| T20 | 1 | D08 | action expert | 10 | 1000 | 2.5840 | 1.4044 | `stitch_4b_eos_checkpoint-1598.json` |

### KAVA latent-slot distillation and controls

| Run | Epoch | Eval data | Head | NFE | n | ADE | minADE | Result JSON |
|---|---|---|---|---|---|---|---|---|
| T50 | 1 | D12 | token | - | 500 | 5.1271 | 4.2966 | `e3_ctl.json` |
| T51 | 1 | D12 | token | - | 500 | 5.7642 | 4.0213 | `e3_kava.json` |
| T51 | 1 | D12 | token | - | 500 | 5.8176 | 4.0453 | `e3_zeroed.json` |
| T21 | 1 | D12 | token | - | 500 | 9.5696 | 4.1646 | `eT2_t2slots.json` |
| T21 | 1 | D12 | token | - | 500 | 27.1187 | 7.9402 | `eT2_t2zeroed.json` |
| T22 | 1 | D12 | token | - | 500 | 4.8383 | 4.3189 | `ectl_ctlslots.json` |
| T22 | 1 | D12 | token | - | 500 | 4.6628 | 4.2541 | `ectl_ctlzeroed.json` |
| T52 | 3 | D12 | token | - | 500 | 4.8581 | 4.0524 | `kava_perclip_slots.json` |
| T52 | 3 | D12 | token | - | 500 | 4.8581 | 4.0524 | `kava_perclip_zeroed.json` |
| T52 | 3 | D12 | token | - | 500 | 4.6696 | 4.0165 | `kava_perclip_zeroed2.json` |

### VLM cache / block / field distillation and CE controls

| Run | Epoch | Eval data | Head | NFE | n | ADE | minADE | Result JSON |
|---|---|---|---|---|---|---|---|---|
| T23 | 1.668 | D05 | action expert | 10 | 1000 | 6.2927 | 2.7348 | `camsweep/b2c2cam_step2000_2cam.json` |
| T23 | 2.085 | D05 | action expert | 10 | 1000 | 5.4669 | 2.7291 | `camsweep/b2c2cam_step2500_2cam.json` |
| T23 | 3 | D05 | action expert | 10 | 1000 | 6.4803 | 2.5603 | `camsweep/b2c2cam_step3597_2cam.json` |
| T43 | 1 | D08 | token | - | 1000 | 3.4749 | 2.4697 | `evalhf_4b_ce_checkpoint-1598.json` |
| T44 | 1 | D08 | token | - | 1000 | 3.6427 | 2.9080 | `evalhf_4b_cekv_checkpoint-1598.json` |
| T45 | 1 | D08 | token | - | 1000 | 4.5953 | 1.9007 | `evalhf_4b_kd_checkpoint-1598.json` |
| T46 | 1 | D08 | token | - | 1000 | 4.7516 | 3.1045 | `evalhf_4b_kv_checkpoint-1598.json` |
| T47 | 1 | D08 | token | - | 1000 | 37.5239 | 37.5239 | `evalhf_4b_kvband_checkpoint-1598.json` |
| T49 | 1 | D08 | token | - | 1000 | 37.5239 | 37.5239 | `evalhf_4b_kvonly_checkpoint-1598.json` |
| T24 | 1.878 | D08 | action expert | 10 | 1000 | 4.8203 | 2.6444 | `stitch_2b_block2b_checkpoint-3000.json` |
| T25 | 2 | D15 | action expert | 10 | 1000 | 5.9495 | 2.6846 | `stitch_2b_block2bdepth_e2_checkpoint-2398_cam13.json` |
| T26 | 2 | D15 | action expert | 10 | 1000 | 5.8496 | 2.7146 | `stitch_2b_block2bdepth_ladder_add_e2_checkpoint-2398_cam13.json` |
| T27 | 2 | D15 | action expert | 10 | 1000 | 5.4640 | 2.5615 | `stitch_2b_block2bmix_m7w1.0_e3_checkpoint-2398_cam13.json` |
| T27 | 3 | D15 | action expert | 10 | 1000 | 5.9268 | 2.5804 | `stitch_2b_block2bmix_m7w1.0_e3_checkpoint-3597_cam13.json` |
| T31 | 1 | D15 | action expert | 10 | 1000 | 5.8744 | 2.8367 | `stitch_2b_block2bspan_checkpoint-1199_cam13.json` |
| T28 | 2 | D15 | action expert | 10 | 1000 | 5.2581 | 2.5360 | `stitch_2b_block2bspan_e2_checkpoint-2398_cam13.json` |
| T29 | 3 | D15 | action expert | 10 | 1000 | 5.1292 | 2.4160 | `stitch_2b_block2bspan_e3_checkpoint-3597_cam13.json` |
| T30 | 4 | D15 | action expert | 10 | 1000 | 5.1550 | 2.3760 | `stitch_2b_block2bspan_e4_checkpoint-4796_cam13.json` |
| T32 | 1 | D15 | action expert | 10 | 1000 | 27.3783 | 22.0637 | `stitch_2b_field2b_checkpoint-1199_cam13.json` |
| T33 | 1 | D11 | action expert | 10 | 1000 | 3.5415 | 1.7423 | `stitch_2b_mix2bnav_m9w1.0_checkpoint-1563_cam13_nav.json` |
| T33 | 2 | D11 | action expert | 10 | 1000 | 3.1653 | 1.4191 | `stitch_2b_mix2bnav_m9w1.0_checkpoint-3126_cam13_nav.json` |
| T33 | 3 | D11 | action expert | 10 | 1000 | 2.7464 | 1.1965 | `stitch_2b_mix2bnav_m9w1.0_checkpoint-4689_cam13_nav.json` |
| T33 | 4 | D11 | action expert | 10 | 1000 | 2.5888 | 1.1719 | `stitch_2b_mix2bnav_m9w1.0_checkpoint-6252_cam13_nav.json` |
| T33 | 5 | D11 | action expert | 10 | 1000 | 2.4965 | 1.0662 | `stitch_2b_mix2bnav_m9w1.0_checkpoint-7815_cam13_nav.json` |
| T34 | 1 | D11 | action expert | 10 | 1000 | 3.5687 | 1.7813 | `stitch_2b_mix2bnav_m9w1.0_plr100.0_checkpoint-1563_cam13_nav.json` |
| T34 | 2 | D11 | action expert | 10 | 1000 | 3.2396 | 1.4543 | `stitch_2b_mix2bnav_m9w1.0_plr100.0_checkpoint-3126_cam13_nav.json` |
| T34 | 3 | D11 | action expert | 10 | 1000 | 2.8419 | 1.2289 | `stitch_2b_mix2bnav_m9w1.0_plr100.0_checkpoint-4689_cam13_nav.json` |
| T34 | 4 | D11 | action expert | 10 | 1000 | 2.6484 | 1.1830 | `stitch_2b_mix2bnav_m9w1.0_plr100.0_checkpoint-6252_cam13_nav.json` |
| T34 | 5 | D11 | action expert | 10 | 1000 | 2.5355 | 1.0904 | `stitch_2b_mix2bnav_m9w1.0_plr100.0_checkpoint-7815_cam13_nav.json` |
| T35 | 1 | D11 | action expert | 10 | 1000 | 3.6712 | 1.8373 | `stitch_2b_mixpin2bnav_m9w1.0_plr100.0_checkpoint-1563_cam13_nav.json` |
| T35 | 2 | D11 | action expert | 10 | 1000 | 3.2082 | 1.4589 | `stitch_2b_mixpin2bnav_m9w1.0_plr100.0_checkpoint-3126_cam13_nav.json` |
| T35 | 3 | D11 | action expert | 10 | 1000 | 2.8657 | 1.2552 | `stitch_2b_mixpin2bnav_m9w1.0_plr100.0_checkpoint-4689_cam13_nav.json` |
| T35 | 4 | D11 | action expert | 10 | 1000 | 2.6487 | 1.2018 | `stitch_2b_mixpin2bnav_m9w1.0_plr100.0_checkpoint-6252_cam13_nav.json` |
| T35 | 5 | D11 | action expert | 10 | 1000 | 2.5319 | 1.0923 | `stitch_2b_mixpin2bnav_m9w1.0_plr100.0_checkpoint-7815_cam13_nav.json` |
| T38 | 5 | D15 | action expert | 10 | 1000 | 5.4275 | 2.3310 | `stitch_2b_nav2bmix_m7w1.0_checkpoint-7815_cam13.json` |
| T38 | 5 | D11 | action expert | 10 | 1000 | 5.2087 | 2.6216 | `stitch_2b_nav2bmix_m7w1.0_checkpoint-7815_cam13_nav.json` |
| T38 | 5 | D16 | action expert | 10 | 1000 | 5.5757 | 2.7927 | `stitch_2b_nav2bmix_m7w1.0_checkpoint-7815_cam13_nonav.json` |
| T36 | 1 | D11 | action expert | 10 | 1000 | 4.0697 | 1.9367 | `stitch_2b_nav2bmix_m7w1.0_e5_clean_maskfix_checkpoint-1563_cam13_nav_trajnpz.json` |
| T36 | 2 | D11 | action expert | 10 | 1000 | 4.2819 | 2.1171 | `stitch_2b_nav2bmix_m7w1.0_e5_clean_maskfix_checkpoint-3126_cam13_nav_trajnpz.json` |
| T36 | 3 | D11 | action expert | 10 | 1000 | 3.2649 | 1.4621 | `stitch_2b_nav2bmix_m7w1.0_e5_clean_maskfix_checkpoint-4689_cam13_nav.json` |
| T36 | 3 | D11 | action expert | 10 | 1000 | 3.2649 | 1.4621 | `stitch_2b_nav2bmix_m7w1.0_e5_clean_maskfix_checkpoint-4689_cam13_nav_trajnpz.json` |
| T36 | 4 | D11 | action expert | 10 | 1000 | 3.3064 | 1.5769 | `stitch_2b_nav2bmix_m7w1.0_e5_clean_maskfix_checkpoint-6252_cam13_nav_trajnpz.json` |
| T36 | 5 | D11 | action expert | 10 | 1000 | 3.2645 | 1.5030 | `stitch_2b_nav2bmix_m7w1.0_e5_clean_maskfix_checkpoint-7815_cam13_nav.json` |
| T36 | 5 | D11 | action expert | 10 | 1000 | 3.2645 | 1.5030 | `stitch_2b_nav2bmix_m7w1.0_e5_clean_maskfix_checkpoint-7815_cam13_nav_trajnpz.json` |
| T37 | 6 | D11 | action expert | 10 | 1000 | 3.6169 | 1.7894 | `stitch_2b_nav2bmix_m7w1.0_e6_efficient_checkpoint-9378_cam13_nav.json` |
| T40 | 1 | D08 | action expert | 10 | 1000 | 3.5824 | 2.0293 | `stitch_4b_blockonly_checkpoint-1598.json` |
| T39 | 2 | D08 | action expert | 10 | 1000 | 3.0922 | 1.7006 | `stitch_4b_blockonly_e3_checkpoint-3196.json` |
| T39 | 3 | D08 | action expert | 10 | 1000 | 3.0033 | 1.6576 | `stitch_4b_blockonly_e3_checkpoint-4794.json` |
| T42 | 1 | D08 | action expert | 10 | 1000 | 3.4005 | 1.9518 | `stitch_4b_blockrandt_checkpoint-1598.json` |
| T41 | 2 | D08 | action expert | 10 | 1000 | 3.0357 | 1.6642 | `stitch_4b_blockrandt_e3_checkpoint-3196.json` |
| T41 | 3 | D08 | action expert | 10 | 1000 | 2.9885 | 1.6008 | `stitch_4b_blockrandt_e3_checkpoint-4794.json` |
| T43 | 1 | D08 | action expert | 10 | 1000 | 12.5500 | 6.9948 | `stitch_4b_ce.json` |
| T44 | 1 | D08 | action expert | 10 | 1000 | 4.9100 | 2.7601 | `stitch_4b_cekv.json` |
| T45 | 1 | D08 | action expert | 10 | 1000 | 17.4294 | 11.3750 | `stitch_4b_kd.json` |
| T46 | 1 | D08 | action expert | 10 | 1000 | 5.9970 | 2.9554 | `stitch_4b_kv.json` |
| T47 | 1 | D08 | action expert | 10 | 1000 | 5.5903 | 3.1763 | `stitch_4b_kvband_checkpoint-1598.json` |
| T49 | 1 | D08 | action expert | 10 | 1000 | 4.1085 | 2.6313 | `stitch_4b_kvonly.json` |
| T48 | 2 | D08 | action expert | 10 | 1000 | 3.8633 | 2.5061 | `stitch_4b_kvonly_e3_checkpoint-3196.json` |
| T48 | 3 | D08 | action expert | 10 | 1000 | 3.8176 | 2.4098 | `stitch_4b_kvonly_e3_checkpoint-4794.json` |

## Teacher And Baseline References

No local training dataset is asserted for released teacher weights. These references are not counted as training runs.

| Eval data | Head | NFE | n | ADE | minADE | Result JSON |
|---|---|---|---|---|---|---|
| D06 | action expert | 10 | 1000 | 7.8732 | 3.9363 | `camsweep/teacherPruneC_1cam.json` |
| D05 | action expert | 10 | 1000 | 2.4241 | 1.1871 | `camsweep/teacherPruneC_2cam.json` |
| D09 | action expert | 10 | 1000 | 2.6970 | 1.4527 | `camsweep/teacherPruneC_3cam.json` |
| D08 | action expert | 10 | 1000 | 1.7025 | 0.7950 | `camsweep/teacherPruneC_4cam.json` |
| D06 | action expert | 10 | 1000 | 3.2087 | 1.6798 | `camsweep/teacher_1cam.json` |
| D05 | action expert | 10 | 1000 | 1.6822 | 0.6981 | `camsweep/teacher_2cam.json` |
| D07 | action expert | 10 | 1000 | 2.3399 | 1.3647 | `camsweep/teacher_3cam.json` |
| D10 | action expert | 10 | 1000 | 1.3143 | 0.5918 | `camsweep/teacher_4cam.json` |
| D08 | token | - | 1000 | 1.2417 | 0.6259 | `evalhf_10b_cot.json` |
| D08 | token | - | 1000 | 1.2111 | 0.6413 | `evalhf_10b_nocot.json` |
| D13 | action expert | 1 | 1000 | 3.0971 | 1.7343 | `stepsweep/nav_teacher_pruned28_k1.json` |
| D13 | action expert | 10 | 1000 | 2.5607 | 1.1918 | `stepsweep/nav_teacher_pruned28_k10.json` |
| D13 | action expert | 2 | 1000 | 3.1377 | 2.1363 | `stepsweep/nav_teacher_pruned28_k2.json` |
| D05 | action expert | 1 | 1000 | 1.6116 | 1.2436 | `stepsweep/teacher_k1.json` |
| D05 | action expert | 10 | 1000 | 1.6736 | 0.7185 | `stepsweep/teacher_k10.json` |
| D05 | action expert | 2 | 1000 | 1.4236 | 0.8876 | `stepsweep/teacher_k2.json` |
| D05 | action expert | 3 | 1000 | 1.4767 | 0.8078 | `stepsweep/teacher_k3.json` |
| D05 | action expert | 5 | 1000 | 1.5623 | 0.7547 | `stepsweep/teacher_k5.json` |
| D13 | action expert | 1 | 1000 | 1.6362 | 1.2817 | `stepsweep/teachernav_k1.json` |
| D13 | action expert | 10 | 1000 | 1.6976 | 0.7441 | `stepsweep/teachernav_k10.json` |
| D13 | action expert | 2 | 1000 | 1.4459 | 0.9318 | `stepsweep/teachernav_k2.json` |
| D13 | action expert | 3 | 1000 | 1.4767 | 0.8412 | `stepsweep/teachernav_k3.json` |
| D13 | action expert | 4 | 1000 | 1.5081 | 0.7985 | `stepsweep/teachernav_k4.json` |
| D13 | action expert | 5 | 1000 | 1.5533 | 0.7812 | `stepsweep/teachernav_k5.json` |
| D13 | action expert | 6 | 1000 | 1.5875 | 0.7659 | `stepsweep/teachernav_k6.json` |
| D13 | action expert | 7 | 1000 | 1.6209 | 0.7555 | `stepsweep/teachernav_k7.json` |
| D13 | action expert | 8 | 1000 | 1.6557 | 0.7509 | `stepsweep/teachernav_k8.json` |
| D13 | action expert | 9 | 1000 | 1.6796 | 0.7473 | `stepsweep/teachernav_k9.json` |
| D14 | action expert | 1 | 1000 | 1.6609 | 1.3064 | `stepsweep/teachernonav_k1.json` |
| D14 | action expert | 10 | 1000 | 1.7188 | 0.7625 | `stepsweep/teachernonav_k10.json` |
| D14 | action expert | 2 | 1000 | 1.4822 | 0.9587 | `stepsweep/teachernonav_k2.json` |
| D14 | action expert | 3 | 1000 | 1.5137 | 0.8631 | `stepsweep/teachernonav_k3.json` |
| D14 | action expert | 4 | 1000 | 1.5446 | 0.8187 | `stepsweep/teachernonav_k4.json` |
| D14 | action expert | 5 | 1000 | 1.5850 | 0.8019 | `stepsweep/teachernonav_k5.json` |
| D14 | action expert | 6 | 1000 | 1.6162 | 0.7855 | `stepsweep/teachernonav_k6.json` |
| D14 | action expert | 7 | 1000 | 1.6452 | 0.7742 | `stepsweep/teachernonav_k7.json` |
| D14 | action expert | 8 | 1000 | 1.6756 | 0.7696 | `stepsweep/teachernonav_k8.json` |
| D14 | action expert | 9 | 1000 | 1.6991 | 0.7657 | `stepsweep/teachernonav_k9.json` |
| D08 | action expert | 10 | 1000 | 1.3039 | 0.5776 | `stitch_4b_teacher.json` |
| D08 | action expert | 10 | 1000 | 1.5398 | 0.9323 | `stitch_4b_teacher_pruneA.json` |
| D08 | action expert | 10 | 1000 | 1.7016 | 0.9306 | `stitch_4b_teacher_pruneB.json` |
| D08 | action expert | 10 | 1000 | 1.6112 | 0.7893 | `stitch_4b_teacher_pruneC.json` |

## Exclusions And Evidence Gaps

- Original root configs and logs are untouched. Full exclusions remain in the CSVs with status/reason, but are omitted from the study tables above.
- Historical pre-fix runs are retained as history, not certified as clean experiments. In particular, use the `clean_maskfix` navigation runs as the fixed baseline and consult [COMPARE_EVAL.md](../COMPARE_EVAL.md) for protocol retractions.
- `structinit_*` files are post-training weight-edit diagnostics, not separately completed training runs.

| Excluded training directory | Observed epoch | Reason |
|---|---|---|
| `.stale_noaux_T2_cancelled382` | 0 | excluded: below one epoch / no completion evidence |
| `.stale_noaux_T2_nccl383` | 0 | excluded: below one epoch / no completion evidence |
| `.stale_noaux_T2_nccl384` | 0 | excluded: below one epoch / no completion evidence |
| `b2c_smoke` | 0 | excluded: below one epoch / no completion evidence |
| `block2b_piprobe` | 0 | excluded: below one epoch / no completion evidence |
| `block2b_smoke` | 0 | excluded: below one epoch / no completion evidence |
| `blockfield_ck3500` | 0 | excluded: below one epoch / no completion evidence |
| `blockfield_smoke` | 0 | excluded: below one epoch / no completion evidence |
| `blockfr_smoke` | 0 | excluded: below one epoch / no completion evidence |
| `blockfr_smoke2` | 0 | excluded: below one epoch / no completion evidence |
| `blockfr_smoke3` | 0 | excluded: below one epoch / no completion evidence |
| `blockfr_smoke4` | 0 | excluded: below one epoch / no completion evidence |
| `expert_on_student_bs2` | 0 | excluded: below one epoch / no completion evidence |
| `expert_on_student_bs4` | 0 | excluded: below one epoch / no completion evidence |
| `expert_on_student_smoke` | 0 | excluded: below one epoch / no completion evidence |
| `freerun_probe` | 0 | excluded: below one epoch / no completion evidence |
| `lrtest_1e-4` | 0 | excluded: below one epoch / no completion evidence |
| `lrtest_5e-5` | 0 | excluded: below one epoch / no completion evidence |
| `output_cd_eos2b_fullteacher_cache_smoke_20260828` | 0 | excluded: below one epoch / no completion evidence |
| `output_cd_eos2b_smoke2_20260827` | 0 | excluded: below one epoch / no completion evidence |
| `output_cd_eos2b_x0gtw0.1_smoke_20260827` | 0 | excluded: below one epoch / no completion evidence |
| `output_cd_eos2b_x0gtw1.0_smoke_20260827` | 0 | excluded: below one epoch / no completion evidence |
| `output_cd_expert_10b_lcdrive` | 0 | excluded: below one epoch / no completion evidence |
| `output_cd_expert_2cam_nav_lcdrive_fixed_20260825` | 0 | excluded: below one epoch / no completion evidence |
| `output_kava_T2_bs2_lcdrive` | 0 | excluded: below one epoch / no completion evidence |
| `output_kava_T2_bs4_lcdrive` | 0 | excluded: below one epoch / no completion evidence |
| `output_kd_2b_block2bdepth_lcdrive` | 0 | excluded: below one epoch / no completion evidence |
| `output_kd_2b_block2bmix_m7w1.0_lcdrive` | 0 | excluded: below one epoch / no completion evidence |
| `output_kd_2b_blockfield_lcdrive` | 0.313 | excluded: below one epoch / no completion evidence |
| `output_kd_2b_nav2bmix_m7w1.0_e6_lcdrive` | 0 | excluded: below one epoch / no completion evidence |
| `output_kd_4b_blockfr_lcdrive` | 0 | excluded: below one epoch / no completion evidence |
| `output_kd_4b_blockrandt_e5_lcdrive` | 0 | excluded: below one epoch / no completion evidence |
| `output_prunedexpert_10b_lcdrive` | 0 | excluded: below one epoch / no completion evidence |
| `overfit_one` | 0 | excluded: below one epoch / no completion evidence |
| `probe_dbg` | 0 | excluded: below one epoch / no completion evidence |
| `probe_dbg2` | 0 | excluded: below one epoch / no completion evidence |
| `roll_smoke` | 0 | excluded: below one epoch / no completion evidence |
| `smoke_band` | 0 | excluded: below one epoch / no completion evidence |
| `smoke_block` | 0 | excluded: below one epoch / no completion evidence |
| `smoke_block2` | 0 | excluded: below one epoch / no completion evidence |
| `smoke_block3` | 0 | excluded: below one epoch / no completion evidence |
| `smoke_calib` | 0 | excluded: below one epoch / no completion evidence |
| `smoke_ce` | 0 | excluded: below one epoch / no completion evidence |
| `smoke_kd_4b` | 0 | excluded: below one epoch / no completion evidence |
| `smoke_kvonly` | 0 | excluded: below one epoch / no completion evidence |

### Source Warnings

- Malformed historical config: /home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/outputs/2026-09-06/09-48-14/.hydra/config.yaml

## Files And Regeneration

- [training_runs.csv](training_runs.csv): one record per retained or excluded training history.
- [evaluations.csv](evaluations.csv): all discovered per-clip evaluations, including exclusions and source paths.
- [audit_issues.json](audit_issues.json): malformed/unreadable evidence.

- [completed_log_audit.csv](completed_log_audit.csv): completion-log coverage, including smoke jobs reporting >1 epoch.

From the repository root:

```bash
recipes/alpamayo1_5_sft/.venv/bin/python recipes/alpamayo1_5_distill/scripts/summarize_ablation_runs.py
```
