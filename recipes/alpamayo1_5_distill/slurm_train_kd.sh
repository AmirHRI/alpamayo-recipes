#!/bin/bash
#SBATCH --job-name=a1_5_kd_train
#SBATCH --partition=debug
#SBATCH --output=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/kdtrain_%j.out
#SBATCH --error=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/kdtrain_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
# 120G: host RAM is only for the frame-decoding dataloader workers (the models live on
# the GPUs). Asking 160G left the KAVA job PENDING on (Resources) when co-tenants held
# 368G of the node's 503 GiB — a memory limit, not a GPU one.
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=amirhosein_chahe@honda-ri.com

# Qwen3-VL-4B student distilled from a co-resident, frozen Alpamayo-1.5-10B teacher.
#
#   ARM=ce  sbatch slurm_train_kd.sh     # control: CE only, no teacher loaded at all
#   ARM=kd  sbatch slurm_train_kd.sh     # + logit-KD over the 4000-token traj slice
#   ARM=kv  sbatch slurm_train_kd.sh     # + all-token/all-layer KV alignment (full recipe)
#   ARM=cekv   sbatch slurm_train_kd.sh  # CE + KV, no logit-KD
#   ARM=kvonly sbatch slurm_train_kd.sh  # KV alone, no CE and no KD
#   ARM=kvband sbatch slurm_train_kd.sh  # KV alone with depth-banded layer weights
#   ARM=blockonly sbatch slurm_train_kd.sh  # L_block alone (teacher-forced block match)
#   ARM=blockrandt sbatch slurm_train_kd.sh # L_block with t sampled, not pinned to 0
#   SMOKE=1 ARM=kv sbatch slurm_train_kd.sh
#   RESUME=<ckpt> EPOCHS=3 ARM=kvonly sbatch slurm_train_kd.sh   # continue for more epochs
#
# The three arms exist to attribute the result. Each writes its own output_dir and
# run_name, so they cannot overwrite one another — the failure mode the KAVA sweep hit
# when CONTROL and the sweep block both set paths.output_dir and hydra kept the last one.
#
# Weights are MEASURED, not guessed: scripts/calibrate_kd_weights.py put kd at 14.1% and
# kv at 0.22% of the CE gradient at weight 1.0, so the shipped kv_weight is 45.4. The
# in-training gradient probe that would normally re-check this CANNOT run here — DeepSpeed
# ZeRO-2's per-parameter grad hooks fire during its extra backwards (see
# trainer._probe_gradient_shares). KAVA_GRAD_PROBE_STEPS is therefore left at 0; re-measure
# mid-run by pointing the calibrator at a checkpoint instead.
#
# ⚠️ Unlike slurm_train_kava.sh, gradient_checkpointing here is TRUE and must stay true.
# That script had to forbid it because hook-captured K/V come back detached; this recipe
# recomputes the student's K/V from output_hidden_states instead, which is exactly what
# makes checkpointing available — and it is what fits a 4B student plus a frozen 8B
# teacher on one 80 GB card (measured 69.6 -> 38.6 GB, losses identical to 5 decimals).

set -euo pipefail

RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
OUT_DIR=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
GPUS="${GPUS:-2}"
SMOKE="${SMOKE:-0}"
ARM="${ARM:-kv}"
BS="${BS:-}"              # per-device batch; ZeRO-2 shards only ACROSS ranks, so bs>1 needs >=2
ACCUM="${ACCUM:-}"        # effective batch = GPUS x BS x ACCUM; the config ships 2 x 1 x 12 = 24
WARMUP="${WARMUP:-}"
# Continue a finished run for more epochs, preserving Adam moments and the dataloader
# position. RESUME is a checkpoint-* dir; EPOCHS is the TOTAL including epochs already
# trained (so a 1-epoch run continued by 2 needs EPOCHS=3).
RESUME="${RESUME:-}"
EPOCHS="${EPOCHS:-}"

cd "$RECIPE_DIR"
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
# The CE loss upcasts logits to fp32 over a 155,697 vocab; expandable segments reclaim the
# large reserved-but-unallocated blocks that otherwise trigger OOM at the loss step.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Off by design, not by oversight — see the header. A nonzero value costs a crash at
# step 0 (best case) or gradients silently double-reduced into the step (worst case).
export KAVA_GRAD_PROBE_STEPS=0
MASTER_PORT=$((29560 + SLURM_JOB_ID % 20000))

EXTRA=()
case "$ARM" in
    ce)
        # ⚠️ teacher_checkpoint_path=null, not "weights at 0". Zeroing the weights would
        # still load and run the 8B teacher every step: same wall-clock and 16 GB/GPU for
        # a term multiplied by zero. Null skips construction entirely, so this arm also
        # tells us what the student costs on its own.
        EXTRA+=(++model.teacher_checkpoint_path=null
                ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0) ;;
    kd)
        EXTRA+=(++model.kd.kv_weight=0.0) ;;
    kv)
        : ;;  # config defaults are the full recipe
    cekv)
        # CE + KV, no logit-KD. The arm the stitched result implies but that was never run:
        # both existing KV numbers INCLUDE logit-KD, and logit-KD measurably HURTS the expert
        # head (+4.38 min_ade vs control, z=+13.7). Removing it should let KV do better than
        # the 2.9554 the kv arm reached.
        EXTRA+=(++model.kd.kd_weight=0.0) ;;
    blockrandt)
        # L_block with t SAMPLED from the teacher's own training schedule (Beta(1.5,1.0)
        # rescaled by 0.999), instead of pinned to t=0. Same cost per step -- one expert
        # forward either way -- so any difference is attributable to WHERE on the flow the
        # cache is supervised, not to extra compute.
        # Motivation: CKA on the frozen expert shows it transforms its representation most
        # around t~0.2-0.4 and least at t=1, while the original L_block supervised only t=0.
        # Control: `blockonly` at 1 epoch (stitched min_ade 2.0293), same init, schedule and
        # budget.
        # ⚠️ model.kd.* -- NOT model.*. Hydra's `++` CREATES a missing key instead of
        # erroring, so the wrong prefix silently leaves every default weight in place and
        # trains the all-objectives config under this arm's name.
        EXTRA+=(++model.kd.ce_weight=0.0 ++model.kd.kd_weight=0.0 ++model.kd.kv_weight=0.0
                ++model.kd.block_weight=1.0 ++model.kd.block_timestep=beta) ;;
    blockonly)
        # L_block ALONE -- teacher-forced block-output matching, no CE, no KD, no L_KV.
        # Alone by design: every arm so far showed that adding objectives to a cache-matching
        # term HURTS the expert head (logit-KD +4.38, and dropping CE from cekv bought -0.13).
        # block_weight is nominally 1.0 and is NOT a tuning knob here: with a single loss and
        # max_grad_norm=1.0 clipping active on every step (observed pre-clip norms 5.6-126),
        # the gradient is renormalised to unit norm, so any uniform scaling of the sole loss
        # is erased. The magnitude lever is `learning_rate`.
        EXTRA+=(++model.kd.kd_weight=0.0 ++model.kd.ce_weight=0.0
                ++model.kd.kv_weight=0.0 ++model.kd.block_weight=1.0) ;;
    kvband)
        # kvonly with DEPTH-BANDED layer weights instead of uniform. Bands measured causally
        # by scripts/layer_importance.py (n=100): swapping one layer of the student's cache
        # for the teacher's and re-driving the frozen expert put 0.1% of the recoverable gain
        # in layers 0-11, 49% in 12-23, 105% in 24-35 (marginal, so they overcount).
        # Weights renormalised to mean 1.0 so the total loss scale is unchanged -- this is a
        # DIRECTION change, not a magnitude one.
        # ⚠️ Start from BASE, not from a kvonly checkpoint: the matched control is uniform
        # kvonly at 1 epoch (stitched min_ade 2.6313), same init, same schedule, same budget.
        EXTRA+=(++model.kd.kd_weight=0.0 ++model.kd.ce_weight=0.0
                "++model.kd.kv_layer_bands=[0.01,0.96,2.03]") ;;
    kvonly)
        # KV alone -- no CE, no KD. Asks whether the student needs token supervision at all
        # when the target is a cache read by the teacher's expert.
        # Safe for the stitched eval: `<|traj_future_start|>` is part of the PROMPT under
        # `components_prompt: [traj_future]`, not something the student must learn to emit,
        # which is why all four stitched arms logged zero "No <traj_future_start>" warnings.
        # The student's own TOKEN head will be destroyed -- that is the point of the arm, so
        # do not evaluate this one with slurm_eval_kd.sh, only slurm_eval_stitched.sh.
        # kv_weight stays 45.409 rather than being rescaled: AdamW normalises by the second
        # moment, so a uniform scaling of the only remaining loss barely moves the step size,
        # and holding it fixed keeps this arm comparable to the others.
        EXTRA+=(++model.kd.kd_weight=0.0 ++model.kd.ce_weight=0.0) ;;
    *)
        echo "[slurm] unknown ARM=$ARM (expected ce|kd|kv|cekv|kvonly|kvband|blockonly|blockrandt)" >&2; exit 1 ;;
esac
RUN_TAG="$ARM"
if [[ -n "$RESUME" ]]; then
    [[ -z "$EPOCHS" ]] && { echo "[slurm] RESUME needs EPOCHS (total, including epochs already done)" >&2; exit 1; }
    [[ -d "$RESUME" ]] || { echo "[slurm] no such checkpoint: $RESUME" >&2; exit 1; }
    # ⚠️ A DIFFERENT output_dir, deliberately. sft_base ships `save_total_limit: 2`, so
    # continuing in place would delete the very checkpoint being resumed from once two new
    # saves land -- and for kvonly that is the checkpoint behind the published 2.6313.
    RUN_TAG="${ARM}_e${EPOCHS}"
    # save_strategy=epoch (not the inherited save_steps=500) so there is exactly one
    # checkpoint per epoch, and save_total_limit high enough to keep all of them.
    EXTRA+=(++trainer.resume_from_checkpoint="$RESUME"
            ++trainer.num_train_epochs="$EPOCHS"
            ++trainer.save_strategy=epoch
            ++trainer.save_total_limit=10)
    echo "[slurm] RESUME from $RESUME -> total $EPOCHS epochs, one checkpoint each"
fi
EXTRA+=(paths.output_dir="$OUT_DIR/output_kd_4b_${RUN_TAG}_lcdrive"
        "run_name=kd_4b_${RUN_TAG}_$(date +%m%d-%H%M)")
echo "[slurm] ARM=$ARM -> $OUT_DIR/output_kd_4b_${RUN_TAG}_lcdrive"

if [[ "$SMOKE" == "1" ]]; then
    # Short, self-contained proof the path runs: teacher loads, sequences match, and all
    # three terms are finite AND falling. A term that is finite but flat is the failure
    # this recipe has to catch.
    EXTRA+=(++trainer.max_steps=20 ++trainer.logging_steps=2
            ++trainer.save_strategy=no ++trainer.eval_strategy=no
            ++trainer.warmup_steps=0 ++trainer.gradient_accumulation_steps=1
            ++data.train_dataset.chunk_ids="0-120" ++data.val_dataset.chunk_ids="0-120")
    echo "[slurm] SMOKE mode: 20 steps"
fi

[[ -n "$BS"     ]] && EXTRA+=(trainer.per_device_train_batch_size="$BS")
[[ -n "$ACCUM"  ]] && EXTRA+=(trainer.gradient_accumulation_steps="$ACCUM")
[[ -n "$WARMUP" ]] && EXTRA+=(trainer.warmup_steps="$WARMUP")

# The cgroup exposes roughly half of --cpus-per-task, so the config's default worker count
# (sized for a bigger allocation) thrashes here. Derive it from the actual allocation.
WORKERS="${WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-16} / (3 * GPUS) ))}"
[[ "$WORKERS" -lt 2 ]] && WORKERS=2
EXTRA+=(++trainer.dataloader_num_workers="$WORKERS")

# Free-form hydra overrides, word-split. For one-off knobs that do not deserve a named
# variable -- e.g. EXTRA_ARGS='++trainer.ddp_timeout=5400' after a NCCL collective timeout.
# shellcheck disable=SC2206,SC2086
[[ -n "${EXTRA_ARGS:-}" ]] && EXTRA+=($EXTRA_ARGS)

echo "[slurm] job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES workers=$WORKERS"

srun "$VENV/torchrun" \
    --nproc_per_node "$GPUS" \
    --master_port "$MASTER_PORT" \
    -m alpamayo1_5_distill.train_kd \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name sft_kd_qwen3_4b_lcdrive \
    "${EXTRA[@]}"
