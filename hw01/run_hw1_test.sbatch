#!/bin/bash
#SBATCH --job-name=hw1_cnn
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/hw1_%j.out
#SBATCH --error=logs/hw1_%j.err

set -eo pipefail
mkdir -p logs

# Activate environment
module purge
module load anaconda/2024.02

source ~/.bashrc
conda activate torch_env

# Optional: better CPU utilization
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Confirm GPU
nvidia-smi

# Run training
python -u HW1_v3.py
