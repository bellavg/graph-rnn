#!/bin/bash
#SBATCH --job-name=graph-rnn
#SBATCH --partition=gpu_a100     # Specify the appropriate partition here
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/graphrnn_%j.out


# Change to the parent directory if needed
cd ..

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
echo "Job started at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "GPU information: $(nvidia-smi -L)"

# Set output directory with job ID
OUTPUT_DIR="runs/graphrnn_${SLURM_JOB_ID}"
mkdir -p $OUTPUT_DIR

# Config file to use (provide as parameter or default)
CONFIG_FILE=${1:-"configs/config_aig_base.yaml"}

# Run the main script
echo "Using config file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"

srun python src/main.py \
    --config_file=$CONFIG_FILE \
    --save_dir=$OUTPUT_DIR

# Print job completion message
echo "Job finished at $(date)"
echo "Results saved to $OUTPUT_DIR"