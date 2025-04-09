#!/bin/bash
#SBATCH --job-name=eval_all_aigs
#SBATCH --partition=gpu_a100 # Or your preferred partition
#SBATCH --gpus=1
#SBATCH --time=04:00:00       # Adjust time as needed
#SBATCH --output=slurm_logs/eval_all_%j.out


# --- User Configuration ---

# !! EDIT THIS ARRAY: List paths to directories containing each model's checkpoints !!
MODEL_DIRS=(
    "./checkpoints/checkpoints_gru_mhsa"
    "./checkpoints/checkpoints_gru_mhsa2"
    "./checkpoints/checkpoints_gru_mlp"
    "./checkpoints/checkpoints_gru_mlp2"
    "./checkpoints/checkpoints_gru_rnn"
    "./checkpoints/checkpoints_lstm_mhsa"
    "./checkpoints/checkpoints_lstm_rnn"
    # Add any other relevant model directories following this pattern
)

# !! EDIT THESE VALUES: Set the correct max nodes/level used during TRAINING !!
# !! This script assumes these are the SAME for all models in MODEL_DIRS !!
# This is for final_data.pkl
FORCE_MAX_NODES=89  # Replace with the actual max_node_count_train
FORCE_MAX_LEVEL=18  # Replace with the actual max_level_train

# --- Script Configuration ---
PYTHON_SCRIPT="src/all_gen_eval.py" # Path to your evaluation script
BASE_OUTPUT_DIR="./evaluation_results_all" # Base directory for all outputs

# Parameters for the Python script (adjust as needed)
NUM_GRAPHS=50         # Number of graphs to generate per checkpoint
NODES_TARGET=100      # Target number of nodes per graph
TEMPERATURES="0.8 1.0 1.2 1.5" # List of temperatures (space-separated)
NUM_PLOTS=3           # Number of best plots to save
PLOT_SORT_BY="nodes"  # Sort plots by 'nodes' (biggest) or 'level'
SAVE_PLOTS_FLAG="--save_plots" # Enable plot saving
FIND_CHECKPOINTS_FLAG="--find_checkpoints" # Use if checkpoints are in subdirs
# DEBUG_FLAG="--debug" # Uncomment for verbose python script output

# --- Environment Setup ---
echo "Loading environment..."
cd .. || { echo "Failed to cd .."; exit 1; } # Go to parent dir where src is

module purge
module load 2024
module load Anaconda3/2024.06-1

# Activate your environment
source activate aig-rnn || { echo "Failed to activate conda env aig-rnn"; exit 1; }

echo "Environment loaded."

# --- Check Python Script ---
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at ${PYTHON_SCRIPT}"
    exit 1
fi

# --- Main Loop ---
mkdir -p "$BASE_OUTPUT_DIR"

echo "Starting evaluation for ${#MODEL_DIRS[@]} models..."

for MODEL_CHECKPOINT_DIR in "${MODEL_DIRS[@]}"; do
    # Extract a short name for the model from its directory path for output naming
    MODEL_NAME=$(basename "$(dirname "$MODEL_CHECKPOINT_DIR")")_$(basename "$MODEL_CHECKPOINT_DIR")
    # Sanitize model name for use in paths (replace / with _)
    MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | tr '/' '_')

    echo "-----------------------------------------------------"
    echo "Processing Model: $MODEL_NAME"
    echo "Checkpoint Directory: $MODEL_CHECKPOINT_DIR"
    echo "-----------------------------------------------------"

    if [ ! -d "$MODEL_CHECKPOINT_DIR" ]; then
        echo "Warning: Checkpoint directory not found: ${MODEL_CHECKPOINT_DIR}. Skipping."
        continue
    fi

    # --- Define Output Paths for this Model ---
    CURRENT_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_NAME_SAFE}/results"
    CURRENT_PLOT_DIR="${BASE_OUTPUT_DIR}/${MODEL_NAME_SAFE}/plots"
    CURRENT_OUTPUT_CSV="${CURRENT_OUTPUT_DIR}/results_summary_${MODEL_NAME_SAFE}.csv"

    mkdir -p "$CURRENT_OUTPUT_DIR"
    mkdir -p "$CURRENT_PLOT_DIR"

    echo "Output CSV:    $CURRENT_OUTPUT_CSV"
    echo "Plot Dir:      $CURRENT_PLOT_DIR"
    echo "Num Graphs:    $NUM_GRAPHS"
    echo "Target Nodes:  $NODES_TARGET"
    echo "Temperatures:  $TEMPERATURES"
    echo "Num Plots:     $NUM_PLOTS (Sorted by: $PLOT_SORT_BY)"
    echo "Forcing Train Max Nodes: $FORCE_MAX_NODES"
    echo "Forcing Train Max Level: $FORCE_MAX_LEVEL"

    # --- Run the Python Evaluation Script for this model ---
    python -u "$PYTHON_SCRIPT" \
        "$MODEL_CHECKPOINT_DIR" \
        --output_csv "$CURRENT_OUTPUT_CSV" \
        --plot_dir "$CURRENT_PLOT_DIR" \
        --num_graphs $NUM_GRAPHS \
        --nodes_target $NODES_TARGET \
        --temperatures $TEMPERATURES \
        --num_plots $NUM_PLOTS \
        --plot_sort_by "$PLOT_SORT_BY" \
        $SAVE_PLOTS_FLAG \
        $FIND_CHECKPOINTS_FLAG \
        --force_max_nodes_train $FORCE_MAX_NODES \
        --force_max_level_train $FORCE_MAX_LEVEL \
        # $DEBUG_FLAG # Uncomment for debug

    # Check the exit status of the python script
    if [ $? -ne 0 ]; then
        echo "Error: Python script failed for model $MODEL_NAME. Check logs."
        # Decide whether to continue with the next model or exit
        # continue
        # exit 1 # Uncomment to stop the whole script on failure
    else
        echo "Successfully processed model $MODEL_NAME."
    fi

    echo "Finished processing $MODEL_NAME at $(date)"

done

echo "--------------------------------------"
echo "All model evaluations finished at $(date)"
echo "Results saved in subdirectories under $BASE_OUTPUT_DIR"
echo "--------------------------------------"

# Deactivate environment (optional)
# conda deactivate

exit 0