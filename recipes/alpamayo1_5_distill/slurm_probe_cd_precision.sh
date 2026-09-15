#!/bin/bash
#SBATCH --job-name=cm_precision
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cm_precision_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cm_precision_%j.err
set -euo pipefail

PRECISION="${PRECISION:?Set PRECISION=fp32 or PRECISION=fp16}"
case "$PRECISION" in
    fp32) FP16=false ;;
    fp16) FP16=true ;;
    *) echo "Unsupported PRECISION=$PRECISION" >&2; exit 1 ;;
esac
RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
OUT=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training
RUN_OUT="$OUT/output_cm_precision_${PRECISION}_${SLURM_JOB_ID}"
if [[ -e "$RUN_OUT" ]]; then
    echo "Refusing existing output: $RUN_OUT" >&2
    exit 1
fi
unset PRUNE_EXPERT_LAYERS
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
export HF_HOME=/temp/achahe/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CD_VERIFY_SAVE=1
export KAVA_GRAD_PROBE_STEPS=0
export WANDB_MODE=disabled
export NVIDIA_TF32_OVERRIDE=0
cd "$RECIPE_DIR"
MASTER_PORT=$((30420 + SLURM_JOB_ID % 20000))
echo "[precision-probe] $PRECISION expert; fp32 weights/moments/EMA; frozen bf16 reference"
nvidia-smi -L
srun "$VENV/torchrun" --nproc_per_node=1 --master_port="$MASTER_PORT" \
    -m alpamayo1_5_distill.train_kd \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name sft_cd_eos_4b_consistency2to1_2cam_nav_lcdrive \
    "++model.expert_precision=$PRECISION" \
    "paths.output_dir=$RUN_OUT" \
    "run_name=cm_precision_${PRECISION}_${SLURM_JOB_ID}" \
    trainer.bf16=false "++trainer.fp16=$FP16" ++trainer.tf32=false \
    trainer.per_device_train_batch_size=1 trainer.gradient_accumulation_steps=1 \
    ++trainer.max_steps=16 trainer.warmup_steps=0 \
    trainer.lr_scheduler_type=constant '~trainer.lr_scheduler_kwargs' \
    trainer.learning_rate=2e-5 trainer.logging_steps=1 ++trainer.logging_nan_inf_filter=false \
    trainer.dataloader_num_workers=2 trainer.save_strategy=steps trainer.save_steps=16 \
    trainer.report_to=none data.train_dataset.chunk_ids=0-120 \
    callbacks.ema.liveness_check_at=16 \
    ++callbacks.precision_probe._target_=alpamayo1_5_distill.models.precision_probe.PrecisionProbeCallback