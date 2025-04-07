#!/bin/bash
#SBATCH --job-name=aig-dir-eval  # Changed job name slightly
#SBATCH --partition=gpu_a100     # Adjust partition if needed
#SBATCH --gpus=1                 # Adjust GPU count if needed
#SBATCH --time=05:00:00          # Adjust time (might need more if python script evaluates all checkpoints)
#SBATCH --output=slurm_logs/aig_dir_eval_%j.out # Log file for the overall job

# --- Script Configuration ---
RUNS_DIR="current_runs"          # Directory containing different model runs
OUTPUT_BASE_DIR="current_evaluation_results" # Base directory for all evaluation outputs
PLOTS_BASE_DIR="current_evaluation_plots"    # Base directory for visualization plots
PYTHON_SCRIPT="src/evaluate_aig_generation.py" # Explicitly define python script path

# --- Parameters for the Python Script ---
NUM_GRAPHS_PER_CHECKPOINT=500 # Python script generates this many per checkpoint it finds
NUM_PLOTS_TO_SAVE=5           # Python script saves this many plots per checkpoint it finds
TARGET_NODES=64               # Target number of nodes for generation
SORT_PLOTS_BY="nodes"         # Sort criteria for plots ('nodes' or 'level')

# --- Environment Setup ---
mkdir -p slurm_logs
mkdir -p $OUTPUT_BASE_DIR
mkdir -p $PLOTS_BASE_DIR

echo "Starting AIG evaluation script at $(date)"
echo "Base runs directory: $RUNS_DIR"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "Plots directory: $PLOTS_BASE_DIR"
echo "Python script: $PYTHON_SCRIPT"

# Load necessary modules
module purge
module load 2024                # Adjust module loading as per your cluster
module load Anaconda3/2024.06-1

# Activate your conda environment
source activate aig-rnn # Make sure 'aig-rnn' is the correct environment name

# --- Find and Evaluate Model Run Directories ---

# Find all potential model checkpoint directories
# This assumes a structure like current_runs/TYPE_ID/checkpoints_DETAILS/
find "$RUNS_DIR" -type d -name 'checkpoints_*' | while read -r CKPT_DIR; do
    MODEL_RUN_DIR=$(dirname "$CKPT_DIR")
    MODEL_RUN_NAME=$(basename "$MODEL_RUN_DIR")
    echo "-----------------------------------------------------"
    echo "Processing Model Run: $MODEL_RUN_NAME, Checkpoint Dir: $CKPT_DIR"

    # Define specific output and plot directories for this model run
    # The python script will generate one CSV containing results for all checkpoints in the dir
    EVAL_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_RUN_NAME}_eval_${SLURM_JOB_ID}"
    PLOT_DIR="${PLOTS_BASE_DIR}/${MODEL_RUN_NAME}_plots_${SLURM_JOB_ID}" # Plots for the whole run go here
    mkdir -p $EVAL_OUTPUT_DIR
    mkdir -p $PLOT_DIR

    OUTPUT_CSV="${EVAL_OUTPUT_DIR}/results_summary.csv" # One summary CSV per run

    # Check if the directory actually contains checkpoints before launching
    if ! ls "${CKPT_DIR}/checkpoint-"*.pth 1> /dev/null 2>&1; then
        echo "  No checkpoint files (.pth) found in $CKPT_DIR. Skipping launch."
        continue
    fi

    # Run the evaluation script ONCE for this directory
    # Pass CKPT_DIR as the positional argument 'model_dir'
    echo "  Launching evaluation job for directory: $CKPT_DIR"
    srun --job-name="${MODEL_RUN_NAME}_eval" --output="slurm_logs/${MODEL_RUN_NAME}_eval_%j.out" \
    python -u $PYTHON_SCRIPT \
        "$CKPT_DIR" \
        --num_graphs $NUM_GRAPHS_PER_CHECKPOINT \
        --nodes_target $TARGET_NODES \
        --output_csv "$OUTPUT_CSV" \
        --save_plots \
        --plot_dir "$PLOT_DIR" \
        --num_plots $NUM_PLOTS_TO_SAVE \
        --plot_sort_by $SORT_PLOTS_BY \
        # Add --gpu -1 if you want CPU, otherwise default is 0 or controlled by SBATCH --gpus
        # Add --temp, --patience, --max_gen_steps if needed

    echo "  Evaluation job launched for run $MODEL_RUN_NAME. Results -> $EVAL_OUTPUT_DIR, Plots -> $PLOT_DIR"
    # Remove the sleep unless needed for cluster stability
    # sleep 1
    echo "-----------------------------------------------------"

done

echo "All evaluation jobs launched at $(date)"