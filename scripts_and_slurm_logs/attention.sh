#!/bin/bash
#SBATCH --job-name=attention
#SBATCH --partition=gpu_a100     # Specify the appropriate partition here
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/attention_%j.out

# Create log directories if they don't exist
mkdir -p slurm_logs

cd ..

# Load modules
module purge
module load 2024
module load Anaconda3/2024.06-1

# Activate your environment
conda activate aig-rnn  # Changed from 'source activate'

# Set output directory with job ID

# Config file to use (provide as parameter or default)
CONFIG_FILE=${1:-"configs/config_aig_attention.yaml"}

# Run the main script
echo "Using config file: $CONFIG_FILE"

srun python -u src/main.py \  # Changed from main.py to train.py
    --config_file=$CONFIG_FILE

# Print job completion message
echo "Job finished at $(date)"
