#!/bin/bash
#SBATCH --job-name=a1_5_eos_train
#SBATCH --partition=gpu
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/eostrain_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/eostrain_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=56
#SBATCH --mem=170G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=amirhosein_chahe@honda-ri.com

# EXPERT-ON-STUDENT: train the pruned action expert on a frozen student's cache.
# `from_student` sets requires_grad_(cotrain_vlm) on every VLM parameter and prints
# "VLM FROZEN, trainable X B" -- read that line, do not assume it.
#
#   CONFIG=sft_expert_on_student_2b_nav_lcdrive sbatch slurm_train_eos.sh
#   SMOKE=1 CONFIG=... sbatch slurm_train_eos.sh          # 20 steps
#
# ⚠️ NO PIN_GPUS here, deliberately. Pinning to a global index requires owning the whole node,
# and this arm is meant to coexist with another 2-GPU job. slurm's own assignment is correct
# WHEN the other occupant is a real slurm job -- which it is not always on this node, so the
# resolved uuid per rank is printed below. Read it against nvidia-smi -L before trusting a
# long run, and check no foreign process already sits on those cards.
#
# ⚠️ --mem is REQUIRED. Without it slurm hands over the node's entire RAM and every other job
# queues on (Resources) with GPUs idle. 170G is half the node, matching the 2-of-4 GPU share.
set -euo pipefail
# ⚠️ PRUNE_EXPERT_LAYERS is REQUIRED but may be the literal "none". The 2B students are 28
# layers deep, so the expert must be pruned to match and a MISSING pin would raise mid-load
# after ~5 min of weight loading -- hence the mandatory variable. The 4B student is already
# 36 deep, matching the teacher's expert, so pruning it would CREATE the mismatch. "none"
# says that explicitly: it is not the same as forgetting the variable, and it must not be
# exported (the loader keys off the variable's PRESENCE, so exporting an empty string is
# not equivalent).
: "${PRUNE_EXPERT_LAYERS:?set PRUNE_EXPERT_LAYERS=comma,separated,indices, or =none for an unpruned 36-layer student}"
if [[ "$PRUNE_EXPERT_LAYERS" == "none" ]]; then
    unset PRUNE_EXPERT_LAYERS
    echo "[slurm] PRUNE_EXPERT_LAYERS=none -> expert left UNPRUNED (36 layers)"
else
    export PRUNE_EXPERT_LAYERS
    echo "[slurm] PRUNE_EXPERT_LAYERS=$PRUNE_EXPERT_LAYERS"
fi
CONFIG="${CONFIG:?set CONFIG=<config name under configs/>}"
SMOKE="${SMOKE:-0}"
RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
cd "$RECIPE_DIR"
# Same guard slurm_train_kd.sh carries: sbatch inherits the submit shell's env, so a shell
# that never sourced .env produces a run that loads the model and only THEN dies in
# wandb.init with "No API key configured" (job 20721, ~6 min wasted per attempt). Read the
# key from the repo .env when the submit env did not carry one; ~/.netrc remains the fallback.
if [[ -z "${WANDB_API_KEY:-}" && -r /home/achahe/alpamayo-recipes/.env ]]; then
    WANDB_API_KEY=$(sed -n 's/^[[:space:]]*WANDB_API_KEY[[:space:]]*=[[:space:]]*//p' \
                    /home/achahe/alpamayo-recipes/.env | tail -1 | tr -d '"'\''[:space:]')
    [[ -n "$WANDB_API_KEY" ]] && export WANDB_API_KEY
fi
[[ -n "${WANDB_API_KEY:-}" ]] \
    && echo "[slurm] WANDB_API_KEY set (len ${#WANDB_API_KEY})" \
    || echo "[slurm] WANDB_API_KEY not set; falling back to ~/.netrc for W&B auth."
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MASTER_PORT=$((29880 + SLURM_JOB_ID % 20000))
GPUS=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES:-0}")

EXTRA=()
[[ -n "${BS:-}"     ]] && EXTRA+=(trainer.per_device_train_batch_size="$BS")
[[ -n "${ACCUM:-}"  ]] && EXTRA+=(trainer.gradient_accumulation_steps="$ACCUM")
[[ -n "${WARMUP:-}" ]] && EXTRA+=(trainer.warmup_steps="$WARMUP")
[[ -n "${EPOCHS:-}" ]] && EXTRA+=(++trainer.num_train_epochs="$EPOCHS")
# RESUME=<checkpoint-N|auto> continues a finished run IN PLACE: same output_dir, so new
# checkpoints land beside the old ones and the step counter keeps going.
# ⚠️ READ THIS BEFORE USING IT. The scheduler is cosine_warmup_with_min_lr, and it is REBUILT
# for the NEW total step count, then fast-forwarded to the restored step. So extending a
# COMPLETED run does NOT continue at the LR it ended on -- it is a WARM RESTART. Concretely,
# a 2-epoch run ends at min_lr (1e-6); resuming it with EPOCHS=4 rebuilds a 4-epoch cosine
# and lands step 3126 near its MIDPOINT, i.e. ~1e-5, a 10x jump back UP. That is standard
# practice for "train longer" but it is NOT a smooth continuation, and the first few hundred
# steps will move the weights considerably more than the last few hundred did.
# Also note the optimizer/dataloader state is restored, so the skipped epochs are not re-seen.
if [[ -n "${RESUME:-}" ]]; then
    if [[ "$RESUME" == "auto" ]]; then
        EXTRA+=(++trainer.resume_from_checkpoint=true)
        echo "[slurm] RESUME=auto -> newest checkpoint in output_dir"
    else
        # ⚠️ HF resolves resume_from_checkpoint as a FILESYSTEM PATH relative to CWD, NOT as a
        # name relative to output_dir. Passing a bare "checkpoint-3126" dies with
        # "Can't find a valid checkpoint at checkpoint-3126" ~3 min in, AFTER the model has
        # loaded and wandb has resumed the run. So resolve bare names here, against the
        # output_dir declared in the config, and fail LOUDLY and immediately if that misses.
        if [[ "$RESUME" != /* ]]; then
            OUT=$("$VENV/python" - "$RECIPE_DIR/configs/$CONFIG.yaml" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["paths"]["output_dir"])
PY
)
            RESUME="$OUT/$RESUME"
        fi
        [[ -d "$RESUME" ]] || { echo "[slurm] no such checkpoint dir: $RESUME" >&2; exit 1; }
        EXTRA+=(++trainer.resume_from_checkpoint="$RESUME")
        echo "[slurm] RESUME <- $RESUME"
    fi
    echo "[slurm] ⚠️ scheduler is REBUILT for the new total: expect an LR warm restart, not a continuation"
fi
# The cgroup exposes roughly half of --cpus-per-task, so derive workers from the allocation
# rather than trusting the config's default (sized for a bigger one).
WORKERS="${WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-16} / (3 * GPUS) ))}"
[[ "$WORKERS" -lt 2 ]] && WORKERS=2
EXTRA+=(++trainer.dataloader_num_workers="$WORKERS")
if [[ "$SMOKE" == "1" ]]; then
    EXTRA+=(++trainer.max_steps=20 ++trainer.logging_steps=2
            ++trainer.save_strategy=no ++trainer.eval_strategy=no
            ++trainer.warmup_steps=0 ++trainer.gradient_accumulation_steps=1
            ++data.train_dataset.chunk_ids="0-600" ++data.val_dataset.chunk_ids="0-600")
    echo "[slurm] SMOKE: 20 steps"
fi
[[ -n "${EXTRA_ARGS:-}" ]] && EXTRA+=($EXTRA_ARGS)

echo "[slurm] job=$SLURM_JOB_ID config=$CONFIG gpus=$CUDA_VISIBLE_DEVICES nproc=$GPUS workers=$WORKERS"
# Which PHYSICAL cards did slurm actually give us? Printed, not assumed.
"$VENV/python" - <<'PY'
import torch
for i in range(torch.cuda.device_count()):
    print(f"[slurm] rank{i} -> cuda:{i} uuid "
          f"{torch.cuda.get_device_properties(i).uuid}", flush=True)
PY
nvidia-smi -L

srun "$VENV/torchrun" --nproc_per_node "$GPUS" --master_port "$MASTER_PORT" \
    -m alpamayo1_5_sft.train_hf \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name "$CONFIG" \
    "${EXTRA[@]}"
