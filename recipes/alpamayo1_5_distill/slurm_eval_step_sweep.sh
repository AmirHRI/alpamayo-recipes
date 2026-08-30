#!/bin/bash
#SBATCH --job-name=a1_5_step_sweep
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
# ⚠️ --mem is REQUIRED. Without it the job takes the node's entire 515 GB and every other
# job queues on (Resources) with GPUs idle. 96G matches the other eval scripts here.
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/stepsweep_%j.out
#SBATCH --error=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/stepsweep_%j.err
#
# GATE C -- the teacher's accuracy as a function of denoising steps, at 2 cameras.
# See scripts/eval_step_sweep.py for what this measures and why it has never been run.
#
#   sbatch slurm_eval_step_sweep.sh                     # teacher, K=1,2,3,5,10, n=1000
#   LIMIT=40 sbatch slurm_eval_step_sweep.sh            # smoke, 40 clips
#   ONLY=k10 sbatch slurm_eval_step_sweep.sh            # one row (the self-check)
#   STEPS='[1,2,10]' CAMERAS='[0,1,2,3]' sbatch ...     # override the grid / cameras
#
# The k10 row is the harness self-check: it must land on the established 2cam teacher
# figures, min_ade 0.6981 / ade 1.6822 (LATENCY_PROFILE.md:136). The script prints the delta.
set -euo pipefail

TAG="${TAG:-teacher}"
LIMIT="${LIMIT:-0}"
ONLY="${ONLY:-}"
BS="${BS:-4}"
STEPS="${STEPS:-[1,2,3,5,10]}"
CAMERAS="${CAMERAS:-[1,3]}"

CONFIG_NAME="${CONFIG_NAME:-sft_eval_stitched_4b_lcdrive}"
RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
OUT=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/stepsweep
TEN_B=/data/achahe/alpasim/huggingface/hub/models--nvidia--Alpamayo-1.5-10B-A1-format
COSMOS_8B=/data/achahe/alpasim/huggingface/hub/models--nvidia--Cosmos-Reason2-8B/snapshots/a9fae2cf89dc64db96b12860417f0eb403013bb9

# ⚠️ MUST be unset. `from_teacher` calls `_apply_expert_pruning`, which is a silent no-op
# when the variable is empty and bypasses 8 of 36 layers when it is not -- so a stray export
# would profile the set-C ablation under the name "teacher". This is the trap
# LATENCY_PROFILE.md:47-50 records; the script also prints the live layer count.
if [[ -n "${PRUNE_EXPERT_LAYERS:-}" && "${ALLOW_PRUNED_TEACHER:-0}" != "1" ]]; then
    echo "[slurm] REFUSING: PRUNE_EXPERT_LAYERS='$PRUNE_EXPERT_LAYERS' is set." >&2
    echo "[slurm] Gate C is the UNPRUNED 36-layer teacher. Unset it and resubmit." >&2
    exit 1
fi
if [[ -n "${PRUNE_EXPERT_LAYERS:-}" ]]; then
    echo "[slurm] explicit pruned-teacher sweep: PRUNE_EXPERT_LAYERS=$PRUNE_EXPERT_LAYERS"
fi

cd "$RECIPE_DIR"
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
export PATH="$VENV:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

echo "[slurm] Gate C: config=$CONFIG_NAME tag=$TAG steps=$STEPS cameras=$CAMERAS limit=$LIMIT only=${ONLY:-<all>}"
nvidia-smi -L

EXTRA=()
[[ -n "$ONLY" ]] && EXTRA+=("++sweep.only=$ONLY")
[[ -n "${EXTRA_ARGS:-}" ]] && EXTRA+=($EXTRA_ARGS)

# ⚠️ sdpa is REQUIRED, not a preference: under flash_attention_2 the expert dies with a
# device-side assert in the unpadding path (alpamayo_r1.py:248-262).
"$VENV/python" -m alpamayo1_5_distill.scripts.eval_step_sweep \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name "$CONFIG_NAME" \
    ++model._target_=alpamayo1_5_distill.models.stitched_model.StitchedAlpamayoR1.from_teacher \
    ++model.vlm_name_or_path="$COSMOS_8B" \
    ++model.checkpoint_path="$TEN_B" \
    ++model.teacher_checkpoint_path="$TEN_B" \
    ++model.attn_implementation=sdpa \
    ++sweep.tag="$TAG" \
    ++sweep.steps="$STEPS" \
    ++sweep.cameras="$CAMERAS" \
    ++sweep.batch_size="$BS" \
    ++sweep.num_workers="${WORKERS:-4}" \
    ++sweep.seed="${SEED:-1234}" \
    ++sweep.limit="$LIMIT" \
    ++sweep.out_dir="$OUT" \
    "${EXTRA[@]}" \
    2>&1 | tee "$OUT/${TAG}_$(date +%m%d-%H%M).log"

echo "[slurm] done -> $OUT"
