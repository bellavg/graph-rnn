#!/bin/bash
#SBATCH --job-name=aig-dir-eval
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/aig_dir_eval_%j.out

set -e  # Exit immediately if a command exits with a non-zero status

# --- Script Configuration ---
BASE_DIR=$(pwd)
SCRIPT_DIR="${BASE_DIR}/src"  # Adjust if needed
DEBUG_SCRIPT="${SCRIPT_DIR}/all_gen_eval.py"
RUNS_DIR="${BASE_DIR}/monday_runs"
OUTPUT_BASE_DIR="${BASE_DIR}/monday_evaluation_results"
PLOTS_BASE_DIR="${BASE_DIR}/current_evaluation_plots"

# --- Parameters for the Python Script ---
NUM_GRAPHS_PER_CHECKPOINT=10
NUM_PLOTS_TO_SAVE=0
TARGET_NODES=64
SORT_PLOTS_BY="nodes"
FORCE_MAX_NODES=64
FORCE_MAX_LEVEL=13
MAX_GEN_STEPS=1000
PATIENCE=15
CHECKPOINT_PATTERN="checkpoint-[0-9]*.pth"  # Pattern to match checkpoint files

# --- Setup directories
mkdir -p slurm_logs
mkdir -p $OUTPUT_BASE_DIR
mkdir -p $PLOTS_BASE_DIR

echo "Starting AIG evaluation script at $(date)"
echo "Python script: $DEBUG_SCRIPT"
echo "Parameters:"
echo "  - Max Nodes Train: $FORCE_MAX_NODES"
echo "  - Max Level Train: $FORCE_MAX_LEVEL"
echo "  - Checkpoints to evaluate: $NUM_CHECKPOINTS_TO_EVAL"
echo "  - Temperature: $TEMPERATURE"
echo "  - Target nodes: $TARGET_NODES"
echo "  - Graphs per checkpoint: $NUM_GRAPHS_PER_CHECKPOINT"
echo "  - Max generation steps: $MAX_GEN_STEPS"
echo "  - Patience: $PATIENCE"

# --- Load environment
module purge
module load 2024
module load Anaconda3/2024.06-1
source activate aig-rnn

# --- Check if debug script exists
if [ ! -f "$DEBUG_SCRIPT" ]; then
    echo "Error: Debug script not found at ${DEBUG_SCRIPT}"
    echo "Copying from current directory"
    cp debug_evaluate_aig_generation.py $DEBUG_SCRIPT
    if [ ! -f "$DEBUG_SCRIPT" ]; then
        echo "Failed to copy script. Aborting."
        exit 1
    fi
fi

# --- Function to validate checkpoint directory
validate_checkpoint_dir() {
    local ckpt_dir=$1
    if [ ! -d "$ckpt_dir" ]; then
        echo "Directory does not exist: $ckpt_dir"
        return 1
    fi

    # Check for checkpoint files using find
    local count=$(find "$ckpt_dir" -name "checkpoint-*.pth" | wc -l)
    if [ "$count" -eq 0 ]; then
        echo "No checkpoint files found in: $ckpt_dir"
        return 1
    fi

    echo "Found $count checkpoint files in: $ckpt_dir"
    return 0
}

# --- Function to run evaluation
run_evaluation() {
        local ckpt_dir=$1
        local model_run_name=$2

        echo "-----------------------------------------------------"
        echo "Processing Model Run: $model_run_name, Checkpoint Dir: $ckpt_dir"

        EVAL_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${model_run_name}_eval_${SLURM_JOB_ID}"
        PLOT_DIR="${PLOTS_BASE_DIR}/${model_run_name}_plots_${SLURM_JOB_ID}"
        mkdir -p $EVAL_OUTPUT_DIR
        mkdir -p $PLOT_DIR

        OUTPUT_CSV="${EVAL_OUTPUT_DIR}/results_summary.csv"

        echo "  Launching evaluation job for directory: $ckpt_dir"
        echo "  Using Temperatures: 1.0 1.5 2.0 2.5" # Indicate new temps

        # --- MODIFIED PYTHON CALL ---
        srun python -u $DEBUG_SCRIPT \
            "$ckpt_dir" \
            --num_graphs $NUM_GRAPHS_PER_CHECKPOINT \
            --nodes_target $TARGET_NODES \
            --output_csv "$OUTPUT_CSV" \
            --save_plots \
            --plot_dir "$PLOT_DIR" \
            --num_plots $NUM_PLOTS_TO_SAVE \
            --plot_sort_by $SORT_PLOTS_BY \
            --force_max_nodes_train $FORCE_MAX_NODES \
            --force_max_level_train $FORCE_MAX_LEVEL \
            # --num_checkpoints $NUM_CHECKPOINTS_TO_EVAL \ # Removed this line
            --temperatures 1.0 1.5 2.0 2.5 \                 # Added this line (adjust values as needed)
            # --temp $TEMPERATURE \                       # Removed this line
            --max_gen_steps $MAX_GEN_STEPS \
            --patience $PATIENCE \
            --debug \                                     # Keep debug if you want logs from all_gen_eval
            # --try_temps \                               # Removed this line
            --checkpoint_pattern "$CHECKPOINT_PATTERN" \
            --find_checkpoints                            # Keep if needed

    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "  WARNING: Evaluation job for $model_run_name returned non-zero exit code: $exit_code"
    else
        echo "  Evaluation job completed successfully for run $model_run_name"
    fi
    echo "  Results -> $EVAL_OUTPUT_DIR, Plots -> $PLOT_DIR"
    echo "-----------------------------------------------------"
}

# --- List of checkpoint directories to evaluate
CHECKPOINT_DIRS=(
    "current_runs/rnn_11048520/checkpoints_GraphLevelRNN"
    "current_runs/attention_11048420/checkpoints_NodeAttn_EdgeAttn"
    "current_runs/mlp_11049142/checkpoints_GraphLevelRNN"
    "graph-rnn/current_runs/lstm_attention_11051065/lstm_attn_checkpoints"
    "current_runs/small_lstm_11053533/small_lstm_checkpoints"
)

# --- Loop through each checkpoint directory
for CKPT_DIR in "${CHECKPOINT_DIRS[@]}"; do
    # Get the full path if not already absolute
    if [[ "$CKPT_DIR" != /* ]]; then
        CKPT_DIR="${BASE_DIR}/${CKPT_DIR}"
    fi

    # Extract model name from path
    MODEL_RUN_NAME=$(basename "$(dirname "$CKPT_DIR")")_$(basename "$CKPT_DIR")

    # Validate checkpoint directory
    echo "Validating checkpoint directory: $CKPT_DIR"
    if validate_checkpoint_dir "$CKPT_DIR"; then
        run_evaluation "$CKPT_DIR" "$MODEL_RUN_NAME"
    else
        echo "Skipping invalid checkpoint directory: $CKPT_DIR"
    fi
done

# Check if we found any valid directories
if [ ${#CHECKPOINT_DIRS[@]} -eq 0 ]; then
    echo "No valid checkpoint directories found"
    exit 1
fi

echo "All evaluation jobs completed at $(date)"
echo "Results are in: $OUTPUT_BASE_DIR"
echo "Plots are in: $PLOTS_BASE_DIR"