#!/bin/bash
#SBATCH --job-name=a1_5_cd_eval
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cdeval_%j.out
#SBATCH --error=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cdeval_%j.err
#
# Score a CD checkpoint (or the untouched teacher) across denoising step counts, NAV-conditioned.
#
#   ARM=teacher sbatch slurm_eval_cd.sh                       # the baseline -- run this too
#   ARM=cd300 CKPT=<.../checkpoint-300> sbatch slurm_eval_cd.sh
#   LIMIT=200 STEPS='[1,2,10]' ARM=cd300 CKPT=... sbatch ...
#
# ⚠️ The CD checkpoint stores the EMA weights under the canonical `expert.*` names (the swap
# happens in KaVaTrainer._save_checkpoint), so BOTH model paths point at the same dir: the
# frozen VLM is unchanged from the teacher and comes back byte-identical, while `expert.*`
# carries the distilled weights via the teacher slot that `_load_teacher_non_vlm` reads.
#
# ⚠️ 1 NFE needs NO new sampler. f(x,tau=1) = x + 1*v_repo(x, s=0) IS `_euler` at
# inference_step=1, so the existing step sweep measures the consistency model directly.
set -euo pipefail

ARM="${ARM:?set ARM=teacher|cd<step>}"
LIMIT="${LIMIT:-0}"
BS="${BS:-1}"
STEPS="${STEPS:-[1,2,3,5,10]}"
CAMERAS="${CAMERAS:-[1,3]}"
TEN_B=/data/achahe/alpasim/huggingface/hub/models--nvidia--Alpamayo-1.5-10B-A1-format
COSMOS_8B=/data/achahe/alpasim/huggingface/hub/models--nvidia--Cosmos-Reason2-8B/snapshots/a9fae2cf89dc64db96b12860417f0eb403013bb9
RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
OUT=/data/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/stepsweep

if [[ -n "${PRUNE_EXPERT_LAYERS:-}" ]]; then
    echo "[slurm] REFUSING: PRUNE_EXPERT_LAYERS is set; this arm is the FULL 36-layer expert." >&2
    exit 1
fi

if [[ "$ARM" == "teacher" ]]; then
    EXPERT_SRC="$TEN_B"
else
    EXPERT_SRC="${CKPT:?set CKPT=<checkpoint dir> for a non-teacher ARM}"
fi

cd "$RECIPE_DIR"
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
export PATH="$VENV:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
echo "[slurm] ARM=$ARM expert<-$EXPERT_SRC steps=$STEPS cameras=$CAMERAS limit=$LIMIT bs=$BS"
nvidia-smi -L

# ⚠️ sdpa is REQUIRED at eval: flash_attention_2 dies on the expert's 4D mask with a
# device-side assert (alpamayo_r1.py:248-262). Training escapes it only because the training
# forward passes attention_mask=None.
"$VENV/python" -m alpamayo1_5_distill.scripts.eval_step_sweep \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name sft_eval_cd_2cam_nav \
    ++model._target_=alpamayo1_5_distill.models.stitched_model.StitchedAlpamayoR1.from_teacher \
    ++model.vlm_name_or_path="$COSMOS_8B" \
    ++model.checkpoint_path="$TEN_B" \
    ++model.teacher_checkpoint_path="$EXPERT_SRC" \
    ++model.attn_implementation=sdpa \
    ++sweep.tag="nav_$ARM" ++sweep.steps="$STEPS" ++sweep.cameras="$CAMERAS" \
    ++sweep.batch_size="$BS" ++sweep.limit="$LIMIT" ++sweep.num_workers=4 \
    ++sweep.seed=1234 ++sweep.out_dir="$OUT" \
    ${EXTRA_ARGS:-} \
    2>&1 | tee "$OUT/nav_${ARM}_$(date +%m%d-%H%M).log"
