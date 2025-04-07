#!/bin/bash
#SBATCH --job-name=graph-rnn-ft  # Changed job name slightly
#SBATCH --partition=gpu_a100     # Specify the appropriate partition here
#SBATCH --gpus=1
#SBATCH --time=08:00:00          # Adjusted time estimate (fine-tuning might be shorter)
#SBATCH --output=slurm_logs/finetune_%j.out # Changed log name


# Create log directories if they don't exist
mkdir -p slurm_logs
mkdir -p runs

# Load modules (adjust if needed)
module purge
module load 2024
module load Anaconda3/2024.06-1

# Activate your environment
source activate aig-rnn # Make sure this is your correct environment name

# --- Configuration for Fine-Tuning ---

# Checkpoint to restore from (provided by user)
RESTORE_PATH="current_runs/rnn_11048520/checkpoints_GraphLevelRNN/checkpoint-50000.pth"

# Config file used for the *original* training run of the checkpoint
# IMPORTANT: This config defines the model architecture to load.
# Assumes the config for run 'rnn_11048520' was e.g. 'configs/config_aig_rnn_base.yaml'
# If passed as $1, use that, otherwise use a default (you might need to change this default)
ORIGINAL_CONFIG_FILE=${1:-"configs/config_aig_rnn_base.yaml"} #<-- CHANGE DEFAULT IF NEEDED

# Fine-tuning parameters (adjust as needed)
FT_LR="1e-5"            # New learning rate for fine-tuning
FT_ADD_STEPS="20000"      # How many *additional* steps to run
FT_SCHEDULER="cosine"   # 'cosine', 'constant', or 'step'
FT_ETA_MIN="1e-7"       # Min LR for cosine fine-tuning scheduler
# FT_STEP_SIZE="5000"   # Relevant if FT_SCHEDULER='step'
# FT_GAMMA="0.5"        # Relevant if FT_SCHEDULER='step'

# Set output directory with job ID and indicate fine-tuning
OUTPUT_DIR="current_runs/finetune_${SLURM_JOB_ID}_from_50k" #<-- Example name
mkdir -p $OUTPUT_DIR

# --- Check if files exist ---
if [ ! -f "$RESTORE_PATH" ]; then
    echo "Error: Restore checkpoint not found at $RESTORE_PATH"
    exit 1
fi
if [ ! -f "$ORIGINAL_CONFIG_FILE" ]; then
    echo "Error: Original config file not found at $ORIGINAL_CONFIG_FILE"
    echo "Please provide the correct config file path as the first argument or edit the script."
    exit 1
fi


# --- Run the main script ---
echo "Starting fine-tuning run..."
echo "Using original config file: $ORIGINAL_CONFIG_FILE"
echo "Restoring checkpoint: $RESTORE_PATH"
echo "Fine-tuning LR: $FT_LR for $FT_ADD_STEPS additional steps"
echo "Fine-tuning Scheduler: $FT_SCHEDULER"
echo "Output directory: $OUTPUT_DIR"

srun python -u src/main.py \
    --config_file=$ORIGINAL_CONFIG_FILE \
    --restore=$RESTORE_PATH \
    --save_dir=$OUTPUT_DIR \
    --fine_tune \
    --ft_lr=$FT_LR \
    --ft_add_steps=$FT_ADD_STEPS \
    --ft_scheduler=$FT_SCHEDULER \
    --ft_eta_min=$FT_ETA_MIN
    # Add --ft_step_size and --ft_gamma if using 'step' scheduler

# Print job completion message
echo "Job finished at $(date)"
echo "Results saved to $OUTPUT_DIR"