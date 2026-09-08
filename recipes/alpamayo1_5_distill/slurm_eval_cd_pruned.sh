#!/bin/bash
#SBATCH --job-name=a1_5_cd_p28_eval
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cdeval_%j.out
#SBATCH --error=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/cdeval_%j.err
#
# Evaluate a cache-aligned consistency checkpoint with 28 active expert blocks in 36
# slots. The inference factory materialises the identity slots before strictly loading the
# sparse EMA expert checkpoint; the frozen VLM still comes from the original teacher.
#
#   ARM=cdp28_e1_s1600 CKPT=<.../checkpoint-1600> STEPS='[1,2]' \
#     sbatch slurm_eval_cd_pruned.sh
set -euo pipefail

ARM="${ARM:?set a unique ARM tag}"
CKPT="${CKPT:?set CKPT=<pruned consistency checkpoint dir>}"
VLM_CKPT="${VLM_CKPT:-}"
LIMIT="${LIMIT:-0}"
BS="${BS:-1}"
STEPS="${STEPS:-[1,2]}"
CAMERAS="${CAMERAS:-[1,3]}"
PRUNE_MAP=4,10,13,15,19,25,27,34
TEN_B=/temp/achahe/hf_cache/hub/models--nvidia--Alpamayo-1.5-10B-A1-format
COSMOS_8B=/temp/achahe/hf_cache/hub/models--nvidia--Cosmos-Reason2-8B/snapshots/a9fae2cf89dc64db96b12860417f0eb403013bb9
COSMOS_2B=/temp/achahe/hf_cache/hub/Cosmos-Reason2-2B
RECIPE_DIR=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
VENV=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/.venv/bin
OUT=/temp/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/training/stepsweep
TAG="nav_${ARM}"

[[ -d "$CKPT" ]] || { echo "[slurm] no checkpoint: $CKPT" >&2; exit 1; }
[[ -z "$VLM_CKPT" || -d "$VLM_CKPT" ]] || { echo "[slurm] no VLM checkpoint: $VLM_CKPT" >&2; exit 1; }
if [[ -n "${PRUNE_EXPERT_LAYERS:-}" && "$PRUNE_EXPERT_LAYERS" != "$PRUNE_MAP" ]]; then
    echo "[slurm] REFUSING mismatched PRUNE_EXPERT_LAYERS='$PRUNE_EXPERT_LAYERS'" >&2
    exit 1
fi
export PRUNE_EXPERT_LAYERS="$PRUNE_MAP"

cd "$RECIPE_DIR"
export PYTHONPATH=/home/achahe/alpamayo-recipes/recipes
export PATH="$VENV:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

# Build the ordinary inference model, but materialise the absent expert slots before loading
# the sparse EMA checkpoint. This retains the optimized prefill-only sampler.
MODEL_ARGS=(
    '~model'
    ++model._target_=alpamayo1_5_distill.models.stitched_model.StitchedAlpamayoR1.from_teacher
    "++model.vlm_name_or_path=$COSMOS_8B"
    "++model.checkpoint_path=$TEN_B"
    "++model.teacher_checkpoint_path=$CKPT"
    ++model.sparse_pruned_expert=true
    ++model.attn_implementation=sdpa
)
if [[ -n "$VLM_CKPT" ]]; then
    MODEL_ARGS=(
        '~model'
        ++model._target_=alpamayo1_5_distill.models.stitched_model.StitchedAlpamayoR1.from_stitch
        "++model.vlm_name_or_path=$COSMOS_2B"
        "++model.alpamayo_config_path=$TEN_B"
        "++model.checkpoint_path=$VLM_CKPT"
        "++model.teacher_checkpoint_path=$CKPT"
        ++model.attn_implementation=sdpa
    )
fi

echo "[slurm] ARM=$ARM checkpoint=$CKPT"
[[ -n "$VLM_CKPT" ]] && echo "[slurm] student VLM=$VLM_CKPT; compacting sparse expert 36->28"
echo "[slurm] consistency student=28/36 active, skipped=$PRUNE_MAP"
echo "[slurm] steps=$STEPS cameras=$CAMERAS limit=$LIMIT bs=$BS"
nvidia-smi -L

LOG="$OUT/${TAG}_$(date +%m%d-%H%M).log"
"$VENV/python" -m alpamayo1_5_distill.scripts.eval_step_sweep \
    --config-path pkg://alpamayo1_5_distill/configs \
    --config-name sft_eval_cd_2cam_nav \
    "${MODEL_ARGS[@]}" \
    "++sweep.tag=$TAG" "++sweep.steps=$STEPS" "++sweep.cameras=$CAMERAS" \
    "++sweep.batch_size=$BS" "++sweep.limit=$LIMIT" ++sweep.num_workers=4 \
    ++sweep.seed=1234 "++sweep.out_dir=$OUT" \
    ${EXTRA_ARGS:-} \
    2>&1 | tee "$LOG"

if [[ -n "$VLM_CKPT" ]]; then
    grep -aF "[stitch] expert REMAP 36->28, dropped [4, 10, 13, 15, 19, 25, 27, 34]" "$LOG"
    grep -aF "Loaded 626 VLM tensors from $VLM_CKPT" "$LOG"
else
    grep -aF "[stitch] PRUNED expert layers [4, 10, 13, 15, 19, 25, 27, 34]" "$LOG"
    grep -aF "[stitch] sparse-pruned expert loaded: 28/36 active" "$LOG"
fi
grep -aF "DONE_STEP_SWEEP" "$LOG"

"$VENV/python" - "$OUT" "$TAG" "$STEPS" "$LIMIT" <<'PY'
import ast
import os
import sys

import numpy as np

out, tag, raw_steps, raw_limit = sys.argv[1:]
steps = [int(x) for x in ast.literal_eval(raw_steps)]
limit = int(raw_limit)
expected = limit if limit else 1000
for step in steps:
    path = os.path.join(out, f"{tag}_k{step}.npz")
    z = np.load(path, allow_pickle=False)
    n = len(z["clip_ids"])
    assert n == expected, (path, n, expected)
    assert z["pred_xyz"].shape[:3] == (n, 1, 6), z["pred_xyz"].shape
    for metric in ("min_ade", "ade", "max_ade"):
        assert z[metric].shape == (n,), (metric, z[metric].shape)
    print(
        f"[tripwire] k{step}: n={n} min_ade={z['min_ade'].mean():.4f} "
        f"ade={z['ade'].mean():.4f} max_ade={z['max_ade'].mean():.4f} "
        f"pred={z['pred_xyz'].shape}"
    )
PY
