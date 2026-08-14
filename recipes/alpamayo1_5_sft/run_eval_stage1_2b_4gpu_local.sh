#!/usr/bin/env bash

# Local (no-Slurm) Stage-1 VLM eval of the fine-tuned Cosmos-Reason2-2B on the
# LCDrive val split, across 4 GPUs. Mirrors run_stage1_2b_4gpu_local.sh's path
# resolution so it works against the same checkpoint/model layout.

set -euo pipefail

WORKSPACE_DIR=/home/achahe/alpamayo-recipes
RECIPE_DIR="$WORKSPACE_DIR/recipes/alpamayo1_5_sft"
VENV_DIR="$RECIPE_DIR/.venv"

PAI_DIR="${PAI_DIR:-/data/02/achahe/physical_ai_av}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/data/01/achahe/hf_cache}"
OUT_DIR="${OUT_DIR:-/data/01/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/training}"
CKPT="${CKPT:-$OUT_DIR/output_stage1_cosmos2b_lcdrive_4gpu_10ep/checkpoint-11990}"
ALPAMAYO_A1_DIR="${ALPAMAYO_A1_DIR:-/data/01/achahe/models/Alpamayo-1.5-10B_a1_sft}"
COSMOS_VLM_PATH="${COSMOS_VLM_PATH:-$HF_CACHE_DIR/Cosmos-Reason2-2B}"

NPROC="${NPROC:-4}"
EVAL_BS="${EVAL_BS:-2}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:--1}"   # -1 = full val split

WANDB_TEAM="${WANDB_TEAM:-zrb20}"
WANDB_PROJECT="${WANDB_PROJECT:-alpamayo1_5-sft-cosmos2b}"

LC_MANIFEST_DIR="$WORKSPACE_DIR/lcdrive_physicalai_av_manifests"
LC_VAL_UUIDS="$LC_MANIFEST_DIR/lcdrive_val_clip_uuids.txt"

EVAL_OUT_DIR="${EVAL_OUT_DIR:-$(dirname "$CKPT")}"
mkdir -p "$EVAL_OUT_DIR"
cd "$RECIPE_DIR"

if [[ ! -x "$VENV_DIR/bin/python" || ! -x "$VENV_DIR/bin/torchrun" ]]; then
    echo "[eval] Missing venv or torchrun at $VENV_DIR."
    exit 1
fi

if [[ -d "$PAI_DIR/physical_ai_av" ]]; then
    PAI_DIR="$PAI_DIR/physical_ai_av"
fi
if [[ ! -d "$PAI_DIR" ]]; then
    echo "[eval] PAI dataset directory not found: $PAI_DIR"
    exit 1
fi

if [[ ! -f "$LC_VAL_UUIDS" ]]; then
    echo "[eval] LCDrive val UUID manifest missing: $LC_VAL_UUIDS"
    exit 1
fi

if [[ ! -f "$CKPT/config.json" ]]; then
    echo "[eval] Stage-1 checkpoint not found or incomplete: $CKPT"
    exit 1
fi

if [[ ! -f "$COSMOS_VLM_PATH/preprocessor_config.json" || ! -f "$COSMOS_VLM_PATH/config.json" ]]; then
    echo "[eval] Cosmos-Reason2-2B base model not found/incomplete at: $COSMOS_VLM_PATH"
    echo "[eval] Set COSMOS_VLM_PATH to a complete local directory."
    exit 1
fi

if [[ ! -f "$ALPAMAYO_A1_DIR/config.json" ]]; then
    echo "[eval] Converted Alpamayo A1 checkpoint not found: $ALPAMAYO_A1_DIR"
    exit 1
fi

export HF_HOME="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

gpu_is_usable() {
    local gpu_id="$1"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    sys.exit(1)

try:
    torch.cuda.set_device(0)
    _ = torch.zeros((1,), device="cuda:0")
except Exception:
    sys.exit(1)
PY
}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    mapfile -t FREE_GPU_IDS < <(
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F',' '($2 + 0) < 1024 {gsub(/ /, "", $1); print $1}'
    )

    PREFERRED_GPU_ORDER=(4 5 6 7 0 1 2 3)
    ORDERED_FREE_GPU_IDS=()
    for gid in "${PREFERRED_GPU_ORDER[@]}"; do
        for free_gid in "${FREE_GPU_IDS[@]}"; do
            if [[ "$gid" == "$free_gid" ]]; then
                ORDERED_FREE_GPU_IDS+=("$gid")
                break
            fi
        done
    done

    USABLE_GPU_IDS=()
    for gid in "${ORDERED_FREE_GPU_IDS[@]}"; do
        if gpu_is_usable "$gid"; then
            USABLE_GPU_IDS+=("$gid")
        else
            echo "[eval] Skipping GPU $gid: CUDA probe failed (busy/unavailable)."
        fi
    done

    if (( ${#USABLE_GPU_IDS[@]} < NPROC )); then
        echo "[eval] Requested NPROC=$NPROC but only ${#USABLE_GPU_IDS[@]} GPUs are free/usable."
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
        exit 1
    fi

    CUDA_VISIBLE_DEVICES="$(printf '%s\n' "${USABLE_GPU_IDS[@]}" | head -n "$NPROC" | paste -sd, -)"
    export CUDA_VISIBLE_DEVICES
fi

MASTER_PORT="${MASTER_PORT:-29602}"
LOG_FILE="$EVAL_OUT_DIR/eval_full_4gpu.log"

echo "[eval] CKPT=$CKPT"
echo "[eval] COSMOS_VLM_PATH=$COSMOS_VLM_PATH"
echo "[eval] PAI_DIR=$PAI_DIR"
echo "[eval] NPROC=$NPROC EVAL_BS=$EVAL_BS MAX_EVAL_STEPS=$MAX_EVAL_STEPS"
echo "[eval] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[eval] EVAL_OUT_DIR=$EVAL_OUT_DIR"
echo "[eval] Logging to: $LOG_FILE"

"$VENV_DIR/bin/torchrun" \
    --nproc_per_node "$NPROC" \
    --master_port "$MASTER_PORT" \
    -m alpamayo1_5_sft.evaluate_hf \
    --config-path pkg://alpamayo1_5_sft/configs \
    --config-name sft_stage1_cosmos2b_lcdrive \
    model.vlm_name_or_path="$COSMOS_VLM_PATH" \
    model.alpamayo_config_path="$ALPAMAYO_A1_DIR" \
    data.val_dataset.local_dir="$PAI_DIR" \
    data.val_dataset.clip_uuid_filter="$LC_VAL_UUIDS" \
    evaluate.eval_ckpt="$CKPT" \
    evaluate.max_eval_steps="$MAX_EVAL_STEPS" \
    trainer.per_device_eval_batch_size="$EVAL_BS" \
    trainer.dataloader_num_workers="$DATALOADER_WORKERS" \
    wandb.team="$WANDB_TEAM" \
    wandb.project="$WANDB_PROJECT" \
    paths.output_dir="$EVAL_OUT_DIR" \
    2>&1 | tee "$LOG_FILE"
