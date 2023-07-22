#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=marl_experiments_2
#SBATCH --mem=8000

module purge
module load Python/3.9.6-GCCcore-11.2.0
module load tqdm/4.64.0-GCCcore-11.3.0
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

python main_ex2.py
