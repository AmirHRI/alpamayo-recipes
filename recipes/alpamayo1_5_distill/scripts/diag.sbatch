#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=diag
#SBATCH --output=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/scripts/diag_%j.log

set -x
source /home/achahe/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft/bin/activate
which python
python --version
echo "VIRTUAL_ENV=$VIRTUAL_ENV"
pwd
python -c "import sys; print('\n'.join(sys.path))"
python -c "import alpamayo1_5_distill; print('OK', alpamayo1_5_distill.__file__)"
cd /home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill
python -m alpamayo1_5_distill.scripts.generate_teacher_kv --help 2>&1 | head -5 || true
