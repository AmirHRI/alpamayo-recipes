#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --job-name=diag2
#SBATCH --output=/home/achahe/alpamayo-recipes/recipes/alpamayo1_5_distill/scripts/diag2_%j.log

set -x
cat /etc/os-release
echo "---python search---"
ls -la /usr/bin/python*
which -a python3.12 python3.11 python3.10 2>&1
find / -maxdepth 4 -iname "python3.12" -type f 2>/dev/null
