#!/bin/bash
#SBATCH --job-name=graph-rnn
#SBATCH --partition=gpu_a100     # Specify the appropriate partition here
#SBATCH --gpus=1
#SBATCH --time=14:00:00
#SBATCH --output=slurm_logs/graphrnn_%j.out


# Create log directories if they don't exist
mkdir -p slurm_logs
mkdir -p runs

# Load modules
module purge
module load 2024
module load Anaconda3/2024.06-1

# Activate your environment
source activate  aig-rnn

# Print environment info
# Set output directory with job ID
OUTPUT_DIR="new_runs/graphrnn_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Config file to use (provide as parameter or default)
CONFIG_FILE=${1:-"configs/config_aig_base.yaml"}

# Run the main script
echo "Using config file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"

srun python -u src/main.py \
 --config_file configs/config_aig_base.yaml \
 --restore new_runs/graphrnn_11003135/base_checkpoints/checkpoint-10000.pth \
 --save_dir new_runs/graphrnn_11003135_continued

# Print job completion message
echo "Job finished at $(date)"
echo "Results saved to $OUTPUT_DIR"