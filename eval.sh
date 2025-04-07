#!/bin/bash
#SBATCH --job-name=aig-multi-eval
#SBATCH --partition=gpu_a100     # Adjust partition if needed
#SBATCH --gpus=1                 # Adjust GPU count if needed
#SBATCH --time=05:00:00          # Adjust time (5 hours estimate, depends on 500 graphs * num_checkpoints)
#SBATCH --output=slurm_logs/aig_multi_eval_%j.out # Log file for the overall job

# --- Script Configuration ---
RUNS_DIR="current_runs"          # Directory containing different model runs
OUTPUT_BASE_DIR="current_evaluation_results" # Base directory for all evaluation outputs
PLOTS_BASE_DIR="current_evaluation_plots"    # Base directory for visualization plots
NUM_GRAPHS_PER_CHECKPOINT=500
NUM_CHECKPOINTS_TO_EVAL=5
NUM_PLOTS_TO_SAVE=5
TARGET_NODES=64                  # Target number of nodes for generation (adjust if needed)

# --- Environment Setup ---
mkdir -p slurm_logs
mkdir -p $OUTPUT_BASE_DIR
mkdir -p $PLOTS_BASE_DIR

echo "Starting AIG evaluation script at $(date)"
echo "Base runs directory: $RUNS_DIR"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "Plots directory: $PLOTS_BASE_DIR"

# Load necessary modules
module purge
module load 2024                # Adjust module loading as per your cluster
module load Anaconda3/2024.06-1

# Activate your conda environment
source activate aig-rnn # Make sure 'aig-rnn' is the correct environment name

# --- Find and Evaluate Models ---

# Find all potential model checkpoint directories
# This assumes a structure like current_runs/TYPE_ID/checkpoints_DETAILS/
find "$RUNS_DIR" -type d -name 'checkpoints_*' | while read -r CKPT_DIR; do
    MODEL_RUN_DIR=$(dirname "$CKPT_DIR")
    MODEL_RUN_NAME=$(basename "$MODEL_RUN_DIR")
    echo "-----------------------------------------------------"
    echo "Processing Model Run: $MODEL_RUN_NAME in directory $CKPT_DIR"

    # Find all checkpoint files, extract step number, sort numerically, get last N
    CHECKPOINTS=$(find "$CKPT_DIR" -name 'checkpoint-*.pth' -printf '%f\n' | \
                  grep -oP 'checkpoint-\K[0-9]+(?=\.pth)' | \
                  sort -nr | \
                  head -n $NUM_CHECKPOINTS_TO_EVAL)

    if [ -z "$CHECKPOINTS" ]; then
        echo "  No checkpoints found in $CKPT_DIR. Skipping."
        continue
    fi

    echo "  Found last $NUM_CHECKPOINTS_TO_EVAL checkpoint steps to evaluate: $CHECKPOINTS"

    # Loop through the selected checkpoints (steps) for this model run
    for STEP in $CHECKPOINTS; do
        CHECKPOINT_FILE="checkpoint-${STEP}.pth"
        CHECKPOINT_PATH="${CKPT_DIR}/${CHECKPOINT_FILE}"

        if [ ! -f "$CHECKPOINT_PATH" ]; then
            echo "  Checkpoint file not found: $CHECKPOINT_PATH. Skipping."
            continue
        fi

        echo "    Evaluating Checkpoint: $CHECKPOINT_FILE"

        # Define specific output and plot directories for this checkpoint
        EVAL_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_RUN_NAME}_step_${STEP}_eval_${SLURM_JOB_ID}"
        PLOT_DIR="${PLOTS_BASE_DIR}/${MODEL_RUN_NAME}_step_${STEP}_plots"
        mkdir -p $EVAL_OUTPUT_DIR
        mkdir -p $PLOT_DIR

        OUTPUT_CSV="${EVAL_OUTPUT_DIR}/results.csv"

        # Run the evaluation script for this specific checkpoint
        # NOTE: Assumes run_aig_eval.py accepts --model_path
        echo "      Launching evaluation job..."
        srun --job-name="${MODEL_RUN_NAME}_${STEP}" --output="slurm_logs/${MODEL_RUN_NAME}_${STEP}_%j.out" \
        python -u src/run_aig_eval.py \
            --model_path "$CHECKPOINT_PATH" \
            --num_graphs $NUM_GRAPHS_PER_CHECKPOINT \
            --nodes_target $TARGET_NODES \
            --output_csv "$OUTPUT_CSV" \
            --save_plots \
            --plot_dir "$PLOT_DIR" \
            --num_plots $NUM_PLOTS_TO_SAVE

        echo "      Evaluation launched for step $STEP. Results -> $EVAL_OUTPUT_DIR, Plots -> $PLOT_DIR"
        # Add a small sleep if submitting many srun jobs quickly causes issues
        # sleep 1
    done
    echo "-----------------------------------------------------"

done

echo "All evaluation jobs launched at $(date)"