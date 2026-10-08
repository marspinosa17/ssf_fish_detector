#!/bin/bash
#SBATCH --job-name=train_ul_rtdetr_l
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=hpc/slurm_logs/train_ul_rtdetr_l_%j.out
#SBATCH --error=hpc/slurm_logs/train_ul_rtdetr_l_%j.err

echo "Running on node(s): $SLURM_NODELIST"
date

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fd_train

python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available on this node/env'
print('GPU OK:', torch.cuda.get_device_name(0))
print('torch version:', torch.__version__)
"

FD_ROOT="${FD_ROOT:-$SLURM_SUBMIT_DIR}"
RUN_NAME=ul_rtdetr_l_v1
cd "$FD_ROOT" || exit 1

mkdir -p "runs/$RUN_NAME"
pip freeze > "runs/$RUN_NAME/pip_freeze.txt"

python training/ultralytics/train_ul_rtdetr.py \
    --model rtdetr-l.pt \
    --data "$FD_ROOT/data/spectrograms/dataset.yaml" \
    --project "$FD_ROOT/runs" \
    --name "$RUN_NAME" \
    --epochs 120 \
    --time 23.5 \
    --patience 20 \
    --batch 16 \
    --mosaic 0.0 \
    --auto-augment "" \
    --hsv-h 0.0 \
    --hsv-s 0.0 \
    --hsv-v 0.4 \
    --scale 0.2 \
    --fliplr 0.5

date
