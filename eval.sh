#!/bin/bash
#SBATCH --job-name=aig-dir-eval
#SBATCH --partition=gpu_a100     # Adjust partition if needed
#SBATCH --gpus=1                 # Adjust GPU count if needed
#SBATCH --time=05:00:00          # Adjust time
#SBATCH --output=slurm_logs/aig_dir_eval_%j.out

# --- Script Configuration ---
RUNS_DIR="current_runs"
OUTPUT_BASE_DIR="current_evaluation_results"
PLOTS_BASE_DIR="current_evaluation_plots"
PYTHON_SCRIPT="src/evaluate_aig_generation.py"

# --- Parameters for the Python Script ---
NUM_GRAPHS_PER_CHECKPOINT=50     # Reduced for debugging
NUM_PLOTS_TO_SAVE=5
TARGET_NODES=64
SORT_PLOTS_BY="nodes"
FORCE_MAX_NODES=64
FORCE_MAX_LEVEL=13
NUM_CHECKPOINTS_TO_EVAL=5

# --- New Parameters ---
TEMPERATURE=0.8                # Lower temperature for more deterministic sampling

# --- Environment Setup ---
mkdir -p slurm_logs
mkdir -p $OUTPUT_BASE_DIR
mkdir -p $PLOTS_BASE_DIR

echo "Starting AIG evaluation script at $(date)"
echo "Python script: $PYTHON_SCRIPT"
echo "Forcing Max Nodes Train: $FORCE_MAX_NODES"
echo "Forcing Max Level Train: $FORCE_MAX_LEVEL"
echo "Evaluating last $NUM_CHECKPOINTS_TO_EVAL checkpoints per directory."
echo "Using temperature: $TEMPERATURE"

module purge
module load 2024
module load Anaconda3/2024.06-1
source activate aig-rnn

# --- Find and Evaluate Model Run Directories ---
find "$RUNS_DIR" -type d -name 'checkpoints_*' | while read -r CKPT_DIR; do
    MODEL_RUN_DIR=$(dirname "$CKPT_DIR")
    MODEL_RUN_NAME=$(basename "$MODEL_RUN_DIR")
    echo "-----------------------------------------------------"
    echo "Processing Model Run: $MODEL_RUN_NAME, Checkpoint Dir: $CKPT_DIR"

    EVAL_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_RUN_NAME}_eval_${SLURM_JOB_ID}"
    PLOT_DIR="${PLOTS_BASE_DIR}/${MODEL_RUN_NAME}_plots_${SLURM_JOB_ID}"
    mkdir -p $EVAL_OUTPUT_DIR
    mkdir -p $PLOT_DIR

    OUTPUT_CSV="${EVAL_OUTPUT_DIR}/results_summary_last_${NUM_CHECKPOINTS_TO_EVAL}.csv"

    if ! ls "${CKPT_DIR}/checkpoint-"*.pth 1> /dev/null 2>&1; then
        echo "  No checkpoint files (.pth) found in $CKPT_DIR. Skipping launch."
        continue
    fi

    echo "  Launching evaluation job for directory: $CKPT_DIR"

    # Launch the evaluation job - REMOVED debug flags
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
        --force_max_nodes_train $FORCE_MAX_NODES \
        --force_max_level_train $FORCE_MAX_LEVEL \
        --num_checkpoints $NUM_CHECKPOINTS_TO_EVAL \
        --temp $TEMPERATURE

    echo "  Evaluation job launched for run $MODEL_RUN_NAME. Results -> $EVAL_OUTPUT_DIR, Plots -> $PLOT_DIR"
    echo "-----------------------------------------------------"

done

echo "All evaluation jobs launched at $(date)"