#!/bin/bash
#SBATCH --job-name=test_gen
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=00:59:00
#SBATCH --output=slurm_logs/gen_test_%j.out


cd ..

# Load modules
module purge
module load 2024
module load Anaconda3/2024.06-1

# Activate your environment
source activate  aig-rnn


# Path to the Python evaluation script
PYTHON_SCRIPT="src/all_gen_eval.py" # Make sure this path is correct relative to where you run the script

# *** SET THE PATH TO YOUR CHECKPOINT FILE HERE ***
CHECKPOINT_PATH="./old/current_runs/attention_11048420/checkpoints_NodeAttn_EdgeAttn"


# Output directories

OUTPUT_DIR="./old/evaluation_results_single"
PLOT_DIR="/old/evaluation_plots_single"
OUTPUT_CSV="${OUTPUT_DIR}/results_summary.csv"

# Parameters for the Python script (adjust values)
NUM_GRAPHS=50       # Number of graphs to generate
NODES_TARGET=64     # Target number of nodes per graph
TEMPERATURES="0.8 1.0 1.2" # List of temperatures (space-separated)
NUM_PLOTS=3         # Max plots to save if saving is enabled
SAVE_PLOTS_FLAG="--save_plots" # Add this flag to save plots, set to "" to disable
# SORT_PLOTS_BY="nodes" # Optional: 'nodes' or 'level'
# PATIENCE=15         # Optional: Generation patience
# MAX_GEN_STEPS=100   # Optional: Max steps for generation
# FORCE_MAX_NODES=64  # Optional: Override training max nodes if needed
# FORCE_MAX_LEVEL=13  # Optional: Override training max level if needed
# DEBUG_FLAG="--debug"  # Optional: Add --debug flag for more verbose python script output

# --- Setup directories ---
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PLOT_DIR"

echo "Starting single checkpoint evaluation..."
echo "Script:        $PYTHON_SCRIPT"
echo "Checkpoint:    $CHECKPOINT_PATH"
echo "Output CSV:    $OUTPUT_CSV"
echo "Plot Dir:      $PLOT_DIR"
echo "Num Graphs:    $NUM_GRAPHS"
echo "Target Nodes:  $NODES_TARGET"
echo "Temperatures:  $TEMPERATURES"

# --- Check if files exist ---
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at ${PYTHON_SCRIPT}"
    exit 1
fi
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found at ${CHECKPOINT_PATH}"
    # Optional: You could attempt to find it or exit
    # Example: find . -name "$(basename $CHECKPOINT_PATH)" -print -quit
    exit 1
fi

# --- Run the Python Evaluation Script ---
# The -u flag ensures unbuffered output from Python
python -u "$PYTHON_SCRIPT" \
    "$CHECKPOINT_PATH" \
    --num_graphs $NUM_GRAPHS \
    --nodes_target $NODES_TARGET \
    --output_csv "$OUTPUT_CSV" \
    --temperatures $TEMPERATURES \
    --plot_dir "$PLOT_DIR" \
    --num_plots $NUM_PLOTS \
    $SAVE_PLOTS_FLAG \
    # $DEBUG_FLAG # Uncomment to enable debug output from the python script
    # --plot_sort_by $SORT_PLOTS_BY     # Uncomment and set if needed
    # --patience $PATIENCE              # Uncomment and set if needed
    # --max_gen_steps $MAX_GEN_STEPS    # Uncomment and set if needed
    # --force_max_nodes_train $FORCE_MAX_NODES # Uncomment if needed
    # --force_max_level_train $FORCE_MAX_LEVEL # Uncomment if needed

echo "--------------------------------------"
echo "Evaluation finished at $(date)"
echo "Results saved to $OUTPUT_CSV"
if [[ "$SAVE_PLOTS_FLAG" == "--save_plots" ]]; then
    echo "Plots saved in $PLOT_DIR"
fi
echo "--------------------------------------"

exit 0
