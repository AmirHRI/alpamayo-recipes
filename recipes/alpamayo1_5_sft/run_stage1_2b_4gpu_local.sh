#!/usr/bin/env bash

# Local (no-Slurm) Stage-1 VLM fine-tune of Cosmos-Reason2-2B on LCDrive.
# Defaults target 4x H100 80GB and 10 epochs.

set -euo pipefail

WORKSPACE_DIR=/home/achahe/alpamayo-recipes
RECIPE_DIR="$WORKSPACE_DIR/recipes/alpamayo1_5_sft"
VENV_DIR="$RECIPE_DIR/.venv"

PAI_DIR="${PAI_DIR:-/data/02/achahe/physical_ai_av}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/data/01/achahe/hf_cache}"
OUT_DIR="${OUT_DIR:-/data/01/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/training}"
ALPAMAYO_A1_DIR="${ALPAMAYO_A1_DIR:-/data/01/achahe/models/Alpamayo-1.5-10B_a1_sft}"
COSMOS_VLM_PATH="${COSMOS_VLM_PATH:-}"

EPOCHS="${EPOCHS:-10}"
NPROC="${NPROC:-4}"
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"
DATALOADER_PREFETCH="${DATALOADER_PREFETCH:-4}"

WANDB_TEAM="${WANDB_TEAM:-zrb20}"
WANDB_PROJECT="${WANDB_PROJECT:-alpamayo1_5-sft-cosmos2b}"
RUN_NAME="${RUN_NAME:-lcdrive_2b_a1_5_stage1_4gpu_10ep}"

LC_MANIFEST_DIR="$WORKSPACE_DIR/lcdrive_physicalai_av_manifests"
LC_TRAIN_UUIDS="$LC_MANIFEST_DIR/lcdrive_train_clip_uuids.txt"
LC_VAL_UUIDS="$LC_MANIFEST_DIR/lcdrive_val_clip_uuids.txt"

mkdir -p "$OUT_DIR"
cd "$RECIPE_DIR"

if [[ ! -x "$VENV_DIR/bin/python" || ! -x "$VENV_DIR/bin/torchrun" ]]; then
    echo "[local] Missing venv or torchrun at $VENV_DIR."
    echo "[local] Run: cd $RECIPE_DIR && uv venv .venv && source .venv/bin/activate && uv sync --active"
    exit 1
fi

# Accept either the dataset root itself (.../physical_ai_av) or its parent dir.
if [[ -d "$PAI_DIR/physical_ai_av" ]]; then
    PAI_DIR="$PAI_DIR/physical_ai_av"
fi

if [[ ! -d "$PAI_DIR" ]]; then
    for candidate in \
        /data/02/achahe/physical_ai_av \
        /data/01/achahe/physical_ai_av \
        /data/achahe/02/physical_ai_av; do
        if [[ -d "$candidate" ]]; then
            echo "[local] PAI_DIR '$PAI_DIR' not found. Falling back to: $candidate"
            PAI_DIR="$candidate"
            break
        fi
    done
fi

if [[ ! -d "$PAI_DIR" ]]; then
    echo "[local] PAI dataset directory not found: $PAI_DIR"
    exit 1
fi

if [[ ! -f "$LC_TRAIN_UUIDS" || ! -f "$LC_VAL_UUIDS" ]]; then
    echo "[local] LCDrive UUID manifest files are missing under: $LC_MANIFEST_DIR"
    exit 1
fi

if [[ ! -f "$ALPAMAYO_A1_DIR/config.json" ]]; then
    echo "[local] Converted Alpamayo A1 checkpoint not found: $ALPAMAYO_A1_DIR"
    echo "[local] Set ALPAMAYO_A1_DIR to your converted checkpoint directory."
    exit 1
fi

export HF_HOME="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

resolve_snapshot_dir() {
    local repo_cache_dir="$1"
    local refs_main
    local commit
    refs_main="$repo_cache_dir/refs/main"
    if [[ -f "$refs_main" ]]; then
        commit="$(<"$refs_main")"
        if [[ -n "$commit" && -d "$repo_cache_dir/snapshots/$commit" ]]; then
            echo "$repo_cache_dir/snapshots/$commit"
            return 0
        fi
    fi

    if [[ -d "$repo_cache_dir/snapshots" ]]; then
        local first_snapshot
        first_snapshot="$(find "$repo_cache_dir/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
        if [[ -n "$first_snapshot" ]]; then
            echo "$first_snapshot"
            return 0
        fi
    fi

    return 1
}

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

if [[ -z "$COSMOS_VLM_PATH" ]]; then
    COSMOS_REPO_DIR="$HF_CACHE_DIR/models--nvidia--Cosmos-Reason2-2B"
    if [[ ! -d "$COSMOS_REPO_DIR" ]]; then
        echo "[local] Cosmos-Reason2-2B not found in $HF_CACHE_DIR; downloading from Hugging Face..."
        if ! "$VENV_DIR/bin/huggingface-cli" download nvidia/Cosmos-Reason2-2B --cache-dir "$HF_CACHE_DIR"; then
            echo "[local] Failed to download Cosmos-Reason2-2B."
            echo "[local] If this is a gated-model error, run 'huggingface-cli login' and ensure your account has access."
            exit 1
        fi
    fi

    if ! COSMOS_VLM_PATH="$(resolve_snapshot_dir "$COSMOS_REPO_DIR")"; then
        echo "[local] Could not resolve a valid Cosmos-Reason2-2B snapshot under: $COSMOS_REPO_DIR"
        exit 1
    fi
fi

COSMOS_LOCAL_DIR="${COSMOS_LOCAL_DIR:-$HF_CACHE_DIR/Cosmos-Reason2-2B}"
if [[ ! -f "$COSMOS_VLM_PATH/preprocessor_config.json" || ! -f "$COSMOS_VLM_PATH/config.json" ]]; then
    echo "[local] Cosmos path appears incomplete (missing config/preprocessor files): $COSMOS_VLM_PATH"
    echo "[local] Downloading full model to: $COSMOS_LOCAL_DIR"
    if ! "$VENV_DIR/bin/huggingface-cli" download nvidia/Cosmos-Reason2-2B \
        --cache-dir "$HF_CACHE_DIR" \
        --local-dir "$COSMOS_LOCAL_DIR"; then
        echo "[local] Failed to download full Cosmos-Reason2-2B into $COSMOS_LOCAL_DIR"
        echo "[local] If this is a gated-model error, run 'huggingface-cli login' and ensure your account has access."
        echo "[local] Alternatively, set COSMOS_VLM_PATH to an existing local Cosmos-Reason2-2B directory."
        exit 1
    fi
    COSMOS_VLM_PATH="$COSMOS_LOCAL_DIR"
fi

if [[ ! -f "$COSMOS_VLM_PATH/preprocessor_config.json" || ! -f "$COSMOS_VLM_PATH/config.json" ]]; then
    echo "[local] Cosmos-Reason2-2B is still incomplete at: $COSMOS_VLM_PATH"
    echo "[local] Set COSMOS_VLM_PATH to a complete local directory containing at least config.json and preprocessor_config.json."
    exit 1
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY
else
    echo "[local] WANDB_API_KEY not set; continuing (W&B can still auth via ~/.netrc)."
fi

MASTER_PORT="${MASTER_PORT:-29540}"
DEEPSPEED_CFG="$RECIPE_DIR/configs/deepspeed/zero2.json"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # Auto-pick idle GPUs when CUDA_VISIBLE_DEVICES is not explicitly set.
    mapfile -t FREE_GPU_IDS < <(
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F',' '($2 + 0) < 1024 {gsub(/ /, "", $1); print $1}'
    )

    # Prefer one PCIe island first for better NCCL performance on this host.
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
            echo "[local] Skipping GPU $gid: CUDA probe failed (busy/unavailable)."
        fi
    done

    if (( ${#USABLE_GPU_IDS[@]} < NPROC )); then
        echo "[local] Requested NPROC=$NPROC but only ${#USABLE_GPU_IDS[@]} GPUs are both free and CUDA-usable."
        echo "[local] Current GPU usage:"
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
        echo "[local] Either lower NPROC or set CUDA_VISIBLE_DEVICES manually."
        exit 1
    fi

    CUDA_VISIBLE_DEVICES="$(printf '%s\n' "${USABLE_GPU_IDS[@]}" | head -n "$NPROC" | paste -sd, -)"
    export CUDA_VISIBLE_DEVICES
else
    # Validate manually requested GPUs before torchrun to fail fast with context.
    IFS=',' read -r -a REQUESTED_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
    for gid in "${REQUESTED_GPU_IDS[@]}"; do
        gid="${gid// /}"
        if [[ -z "$gid" ]]; then
            continue
        fi
        if ! gpu_is_usable "$gid"; then
            echo "[local] CUDA_VISIBLE_DEVICES contains unusable GPU: $gid"
            echo "[local] Pick different IDs (e.g., exclude busy/unavailable devices)."
            exit 1
        fi
    done
fi

echo "[local] Starting Stage-1 training"
echo "[local] PAI_DIR=$PAI_DIR"
echo "[local] HF_CACHE_DIR=$HF_CACHE_DIR"
echo "[local] COSMOS_VLM_PATH=$COSMOS_VLM_PATH"
echo "[local] ALPAMAYO_A1_DIR=$ALPAMAYO_A1_DIR"
echo "[local] OUT_DIR=$OUT_DIR"
echo "[local] NPROC=$NPROC PER_DEVICE_BS=$PER_DEVICE_BS GRAD_ACCUM=$GRAD_ACCUM EPOCHS=$EPOCHS"
echo "[local] DATALOADER_WORKERS=$DATALOADER_WORKERS DATALOADER_PREFETCH=$DATALOADER_PREFETCH"
echo "[local] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

"$VENV_DIR/bin/torchrun" \
    --nproc_per_node "$NPROC" \
    --master_port "$MASTER_PORT" \
    -m alpamayo1_5_sft.train_hf \
    --config-path pkg://alpamayo1_5_sft/configs \
    --config-name sft_stage1_cosmos2b_lcdrive \
    model.vlm_name_or_path="$COSMOS_VLM_PATH" \
    model.alpamayo_config_path="$ALPAMAYO_A1_DIR" \
    data.train_dataset.local_dir="$PAI_DIR" \
    data.val_dataset.local_dir="$PAI_DIR" \
    data.train_dataset.clip_uuid_filter="$LC_TRAIN_UUIDS" \
    data.val_dataset.clip_uuid_filter="$LC_VAL_UUIDS" \
    trainer.deepspeed="$DEEPSPEED_CFG" \
    trainer.per_device_train_batch_size="$PER_DEVICE_BS" \
    trainer.gradient_accumulation_steps="$GRAD_ACCUM" \
    trainer.dataloader_num_workers="$DATALOADER_WORKERS" \
    +trainer.dataloader_prefetch_factor="$DATALOADER_PREFETCH" \
    +trainer.dataloader_persistent_workers=true \
    trainer.num_train_epochs="$EPOCHS" \
    paths.output_dir="$OUT_DIR/output_stage1_cosmos2b_lcdrive_4gpu_10ep" \
    wandb.team="$WANDB_TEAM" \
    wandb.project="$WANDB_PROJECT" \
    run_name="$RUN_NAME"
