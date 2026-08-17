#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=kv-cache-smoke
#SBATCH --output=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/scripts/kv_cache_smoke_%j.log

set -euo pipefail

source /home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft/bin/activate
cd /home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill

python -m alpamayo1_5_distill.scripts.generate_teacher_kv \
  config=cache_teacher_kv_lcdrive teacher=teacher_ar1_5_10b \
  cache_root=/temp/achahe/kv_cache_smoke_test \
  model.checkpoint_path=/temp/achahe/hf_cache/hub/models--nvidia--Alpamayo-1.5-10B-A1-format \
  data.cache_dataset.local_dir=/temp/achahe/physical_ai_av/ \
  data.cache_dataset.clip_uuid_filter=/home/achahe/pai_download/lcdrive_physicalai_av_manifests/lcdrive_train_clip_uuids.txt \
  m=16 lam=0.1 eviction=rkv mode=generate limit=2
