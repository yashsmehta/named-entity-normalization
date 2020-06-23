#!/bin/bash
#SBATCH --job-name=np
#SBATCH --output=logs/ym_%A.out
#SBATCH --time=0-1:00
#SBATCH --nodes=1
#SBATCH --mem=7G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

srun -u python scraper.py
#srun -u python finetune_models/MLP_combined_features.py -embed 'bert-base' -mode '256_head_tail' -embed_mode 'mean' -epochs 10 -layer 11 -seed ${SLURM_ARRAY_TASK_ID}
