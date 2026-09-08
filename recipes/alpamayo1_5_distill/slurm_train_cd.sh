#!/bin/bash
#SBATCH --job-name=a1_5_cd_expert
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
# 24, not 16: 6 dataloader workers x 2 ranks, each decoding 8 images per sample.
#SBATCH --cpus-per-task=24
# ⚠️ --mem is REQUIRED. Without it this job takes the node's entire 515 GB and every other
# job queues on (Resources) with GPUs idle -- that is what blocked the blockrandt continuation.
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cd_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cd_%j.err
#
# Consistency distillation of the FULL 36-layer action expert, teacher VLM frozen.
# See configs/sft_cd_expert_teachercache.yaml for what this measures and the Gate C numbers
# it has to beat.
#
#   sbatch slurm_train_cd.sh                          # 2cam + nav, the default
#   CONFIG=sft_cd_expert_teachercache sbatch ...      # the 4cam no-nav variant
#   SMOKE=1 sbatch slurm_train_cd.sh                  # 20 steps
#   CD_SELFTEST=1 SMOKE=1 sbatch slurm_train_cd.sh    # Gate A self-tests, then 20 steps
#   M=40 EXTRA_ARGS='++model.cd.metric=pseudo_huber' sbatch slurm_train_cd.sh
set -euo pipefail

SMOKE="${SMOKE:-0}"
RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
OUT=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training

# ⚠️ MUST be unset. `_apply_expert_pruning` is a SILENT no-op when empty and bypasses 8 of
# 36 layers when set, so a stray export from a previous shell would train the set-C ablation
# under this arm's name -- the trap LATENCY_PROFILE.md:47-50 records. `init_cd` raises too;
# this refuses earlier and says why.
if [[ -n "${PRUNE_EXPERT_LAYERS:-}" ]]; then
    echo "[slurm] REFUSING: PRUNE_EXPERT_LAYERS='$PRUNE_EXPERT_LAYERS' is set." >&2
    echo "[slurm] This arm distils the FULL 36-layer expert. Unset it and resubmit." >&2
    exit 1
fi

cd "$RECIPE_DIR"
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
export PATH="$VENV:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ⚠️ Off by design. A nonzero value costs a crash at step 0 under ZeRO-2 (best case) or
# gradients silently double-reduced into the step (worst case). This arm runs deepspeed:null
# so the probe would work -- but leave it off unless deliberately measuring gradient shares.
export KAVA_GRAD_PROBE_STEPS="${KAVA_GRAD_PROBE_STEPS:-0}"
MASTER_PORT=$((29960 + SLURM_JOB_ID % 20000))

EXTRA=()
[[ -n "${M:-}" ]] && EXTRA+=("++model.cd.m_rungs=$M")
if [[ "$SMOKE" == "1" ]]; then
    EXTRA+=(++trainer.max_steps=20 ++trainer.logging_steps=2 ++trainer.save_strategy=no
            ++trainer.eval_strategy=no ++trainer.warmup_steps=0
            ++trainer.gradient_accumulation_steps=1
            ++data.train_dataset.chunk_ids=0-120 ++data.val_dataset.chunk_ids=0-120
            ++trainer.dataloader_num_workers=2
            # the liveness assert has to fire inside a 20-step smoke, not after it
            ++callbacks.ema.liveness_check_at=10 ++callbacks.ema.warmup_steps=2)
    echo "[slurm] SMOKE: 20 steps"
fi
[[ -n "${EXTRA_ARGS:-}" ]] && EXTRA+=($EXTRA_ARGS)

echo "[slurm] CD expert, PRUNE_EXPERT_LAYERS unset -> full depth expected"
nvidia-smi -L

# ⚠️ train_kd, not train_hf: it rebinds ReasoningVLA_Trainer to KaVaTrainer, which is what
# logs cd_loss / the per-rung bands separately AND what swaps the EMA in around the save.
srun "$VENV/torchrun" --nproc_per_node 2 --master_port "$MASTER_PORT" \
    -m alpamayo1_5_distill.train_kd \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name "${CONFIG:-sft_cd_expert_2cam_nav_lcdrive}" \
    run_name="cd_$(date +%m%d-%H%M)" \
    "${EXTRA[@]}"
