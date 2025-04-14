#!/bin/bash
#SBATCH --job-name=aig_test
#SBATCH --partition=gpu_a100         # Keep partition if appropriate
#SBATCH --gpus=1                     # Request 1 GPU per run
#SBATCH --time=04:00:00              # Increased time, adjust as needed per checkpoint             # Keep memory request
#SBATCH --output=slurm_logs/aig_get_multi_%j.out # Changed log file name pattern



cd ..


PYTHON_SCRIPT="src/get_aigs.py"      # <<< Path to your main Python script
# Directory containing subdirectories like 'checkpoints_gru_rmsp', 'checkpoints_gru_node2', etc.
BASE_CHECKPOINT_DIR="./checkpoints" # <<< CHANGE THIS if needed (based on image, this seems correct relative to PROJECT_BASE_DIR)
# Base directory where all results will be saved (new location)
BASE_OUTPUT_DIR="./evaluation_results_all" # <<< CHANGE THIS destination as desired

# === PARAMETERS for get_aigs.py ===
NUM_GENERATE=50       # Number of graphs to generate per checkpoint
DO_VISUALIZE=true      # Set to true to enable visualization, false to disable
NUM_VISUALIZE=10       # Number of graphs to visualize if visualization is enabled



echo "======================================================"
echo "Starting Multi-Checkpoint AIG Generation/Evaluation"
echo "Timestamp: $(date)"
echo "Python Script: $PYTHON_SCRIPT"
echo "Base Checkpoint Dir: $BASE_CHECKPOINT_DIR"
echo "Base Output Dir: $BASE_OUTPUT_DIR"
echo "Num Generate per Checkpoint: $NUM_GENERATE"
echo "Visualize: $DO_VISUALIZE"
echo "Num Visualize: $NUM_VISUALIZE"
echo "======================================================"

# --- Load environment ---
# Adjust module load and conda activate commands based on your cluster setup
echo "Loading environment modules..."
module purge
module load 2024                # Or your required environment modules
module load Anaconda3/2024.06-1 # Or your Anaconda/Python module
echo "Activating Conda environment..."
source activate aig-rnn         # <<< Make sure this conda env name is correct


#pip install scipy==1.14.0
#pip install numpy
#pip install EMD-signal
#pip install pyemd
#conda install pyemd


# --- Check if Python script exists ---
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at ${PYTHON_SCRIPT}"
    exit 1
fi

# --- Find all checkpoint files ---
# Use find to search for .pth files within the subdirectories of BASE_CHECKPOINT_DIR
# -maxdepth 2 assumes checkpoints are directly inside subdirs like checkpoints_gru_rmsp/
# Adjust maxdepth if they are nested deeper.
CHECKPOINT_FILES=()
while IFS= read -r -d $'\0'; do
    CHECKPOINT_FILES+=("$REPLY")
done < <(find "$BASE_CHECKPOINT_DIR" -maxdepth 2 -name "*.pth" -print0)

# Check if any files were found
if [ ${#CHECKPOINT_FILES[@]} -eq 0 ]; then
    echo "Error: No checkpoint .pth files found in subdirectories of $BASE_CHECKPOINT_DIR"
    exit 1
fi

echo "Found ${#CHECKPOINT_FILES[@]} checkpoint files to process."

# --- Set Visualization Flag ---
VISUALIZE_FLAG=""
if [ "$DO_VISUALIZE" = true ]; then
    VISUALIZE_FLAG="--visualize"
fi

# --- Loop through each checkpoint file ---
for CKPT_FILE_PATH in "${CHECKPOINT_FILES[@]}"; do
    echo "-----------------------------------------------------"
    echo "Processing Checkpoint File: $CKPT_FILE_PATH"

    # --- Create unique output directory for this checkpoint ---
    # Extract the specific model run dir name (e.g., checkpoints_gru_rmsp)
    MODEL_RUN_SUBDIR=$(basename "$(dirname "$CKPT_FILE_PATH")")
    # Extract the checkpoint filename without extension (e.g., checkpoint-100)
    CHECKPOINT_NAME=$(basename "$CKPT_FILE_PATH" .pth)
    # Define the specific output directory for this checkpoint's results
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_RUN_SUBDIR}/${CHECKPOINT_NAME}"
    mkdir -p "$OUTPUT_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create output directory $OUTPUT_DIR"
        # Decide whether to skip or exit
        echo "Skipping checkpoint $CKPT_FILE_PATH"
        continue # Skip to the next checkpoint
    fi
    echo "Output Dir: $OUTPUT_DIR"
    # --- ---

    # --- Construct and Execute the Python Command ---
    echo "Launching get_aigs.py for $CHECKPOINT_NAME..."
    # Use srun if you need SLURM to manage the python process resource allocation specifically
    # If the main script handles resource usage well, you might just run python directly
    python -u "$PYTHON_SCRIPT" \
        --model-path "$CKPT_FILE_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --num-generate "$NUM_GENERATE" \
        $VISUALIZE_FLAG \
        --num-visualize "$NUM_VISUALIZE"\
        --evaluate

    exit_code=$?
    # --- ---

    if [ $exit_code -ne 0 ]; then
        echo "  WARNING: Python script for $CKPT_FILE_PATH returned non-zero exit code: $exit_code"
        # Optional: Add more error handling here, e.g., stop the whole script
    else
        echo "  Processing completed successfully for $CKPT_FILE_PATH"
    fi
    echo "-----------------------------------------------------"

done

echo "======================================================"
echo "All checkpoint processing finished at $(date)"
echo "Results are located under: $BASE_OUTPUT_DIR"
echo "======================================================"

exit 0