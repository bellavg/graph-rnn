#!/bin/bash
#SBATCH --job-name=aig_test
#SBATCH --partition=gpu_a100         # Keep partition if appropriate
#SBATCH --gpus=1                     # Request 1 GPU per run
#SBATCH --time=12:00:00              # Increased time, adjust as needed per checkpoint             # Keep memory request
#SBATCH --output=slurm_logs/aig_eval_%j.out # Changed log file name pattern



cd ..


PYTHON_SCRIPT="src/get_aigs.py"      # Path to your main Python script
BASE_CHECKPOINT_DIR="./checkpoints" # Directory containing model subdirectories
BASE_OUTPUT_DIR="./evaluation_results_final" # Base directory for results
GRAPH_DATA_FILE="./dataset/final_data.pkl" # Path to the graph dataset for evaluation

# === PARAMETERS for get_aigs.py ===
NUM_GENERATE=500       # Number of graphs to generate per checkpoint
DO_VISUALIZE=true      # Set to true to enable visualization
NUM_VISUALIZE=5       # Number of graphs to visualize if visualization is enabled
NUM_LATEST_CHECKPOINTS=5 # Number of latest checkpoints to process per model

# --- Echo Parameters ---
echo "======================================================"
echo "Starting Multi-Checkpoint AIG Generation/Evaluation"
echo "Processing latest $NUM_LATEST_CHECKPOINTS checkpoints per model type."
echo "Timestamp: $(date)"
echo "Python Script: $PYTHON_SCRIPT"
echo "Base Checkpoint Dir: $BASE_CHECKPOINT_DIR"
echo "Base Output Dir: $BASE_OUTPUT_DIR"
echo "Graph Data File: $GRAPH_DATA_FILE"
echo "Num Generate per Checkpoint: $NUM_GENERATE"
echo "Visualize: $DO_VISUALIZE"
echo "Num Visualize: $NUM_VISUALIZE"
echo "======================================================"

# --- Load environment ---
echo "Loading environment modules..."
module purge
module load 2024                # Or your required environment modules
module load Anaconda3/2024.06-1 # Or your Anaconda/Python module
echo "Activating Conda environment..."
source activate aig-rnn         # Make sure this conda env name is correct

# --- Check if Python script exists ---
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at ${PYTHON_SCRIPT}"
    exit 1
fi

# --- Check if Graph Data file exists ---
if [ ! -f "$GRAPH_DATA_FILE" ]; then
    echo "Error: Graph data file not found at ${GRAPH_DATA_FILE}"
    exit 1
fi

# --- Set Visualization Flag ---
VISUALIZE_FLAG=""
if [ "$DO_VISUALIZE" = true ]; then
    VISUALIZE_FLAG="--visualize"
fi
# --- End Set Visualization Flag ---


# --- Find model type subdirectories ---
MODEL_DIRS=()
while IFS= read -r -d $'\0'; do
    # Ensure it's actually a directory before adding
    if [ -d "$REPLY" ]; then
        MODEL_DIRS+=("$REPLY")
    fi
done < <(find "$BASE_CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 -type d -print0)


if [ ${#MODEL_DIRS[@]} -eq 0 ]; then
    echo "Error: No model subdirectories found in $BASE_CHECKPOINT_DIR"
    exit 1
fi

echo "Found ${#MODEL_DIRS[@]} model directories to process."

# --- Loop through each model directory ---
for MODEL_DIR_PATH in "${MODEL_DIRS[@]}"; do
    MODEL_RUN_SUBDIR=$(basename "$MODEL_DIR_PATH")
    echo "======================================================"
    echo "Processing Model Type: $MODEL_RUN_SUBDIR"
    echo "======================================================"

    # --- Find, Sort, and Select last N checkpoints in this directory ---
    # Use find to get .pth files, sort naturally (version sort), take last N
    mapfile -t LATEST_CHECKPOINTS < <(find "$MODEL_DIR_PATH" -maxdepth 1 -name "*.pth" -printf "%f\n" | sort -V | tail -n "$NUM_LATEST_CHECKPOINTS")

    if [ ${#LATEST_CHECKPOINTS[@]} -eq 0 ]; then
        echo "  Warning: No checkpoint .pth files found in $MODEL_DIR_PATH. Skipping."
        continue # Skip to the next model directory
    fi

    echo "  Found ${#LATEST_CHECKPOINTS[@]} checkpoints for this model type (processing latest $NUM_LATEST_CHECKPOINTS):"
    printf "    %s\n" "${LATEST_CHECKPOINTS[@]}"

    # --- Loop through the selected checkpoints for this model type ---
    for CHECKPOINT_FILENAME in "${LATEST_CHECKPOINTS[@]}"; do
        CKPT_FILE_PATH="${MODEL_DIR_PATH}/${CHECKPOINT_FILENAME}"
        CHECKPOINT_NAME=$(basename "$CHECKPOINT_FILENAME" .pth) # Get name without extension

        echo "-----------------------------------------------------"
        echo "Processing Checkpoint File: $CKPT_FILE_PATH"

        # --- Create unique output directory for this checkpoint ---
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_RUN_SUBDIR}/${CHECKPOINT_NAME}"
        mkdir -p "$OUTPUT_DIR"
        if [ $? -ne 0 ]; then
            echo "  Error: Failed to create output directory $OUTPUT_DIR"
            echo "  Skipping checkpoint $CKPT_FILE_PATH"
            continue # Skip to the next checkpoint
        fi
        echo "  Output Dir: $OUTPUT_DIR"
        # --- ---

        # --- Construct and Execute the Python Command ---
        echo "  Launching get_aigs.py for $CHECKPOINT_NAME..."
        # Using srun as you had it before
        srun python -u "$PYTHON_SCRIPT" \
            --model-path "$CKPT_FILE_PATH" \
            --output-dir "$OUTPUT_DIR" \
            --num-generate "$NUM_GENERATE" \
            --graph-file "$GRAPH_DATA_FILE" \
            $VISUALIZE_FLAG \
            --num-visualize "$NUM_VISUALIZE"\
            --evaluate # Evaluate all generated graphs structurally

        exit_code=$?
        # --- ---

        if [ $exit_code -ne 0 ]; then
            echo "    WARNING: Python script for $CKPT_FILE_PATH returned non-zero exit code: $exit_code"
            # Optional: Add more error handling here
        else
            echo "    Processing completed successfully for $CKPT_FILE_PATH"
        fi
        echo "-----------------------------------------------------"
    done # End loop for checkpoints within a model type
done # End loop for model types


echo "======================================================"
echo "All selected checkpoint processing finished at $(date)"
echo "Results are located under: $BASE_OUTPUT_DIR"
echo "======================================================"

exit 0
