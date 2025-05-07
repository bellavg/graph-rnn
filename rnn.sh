#!/bin/bash
#SBATCH --job-name=aig_rnn_pipeline
#SBATCH --partition=gpu_h100     # Specify the appropriate partition here
#SBATCH --gpus=1
#SBATCH --time=12:00:00          # Adjust time as needed
#SBATCH --output=slurm_logs/aig_pipeline_%j.out


# --- Configuration ---
CONFIG_FILE=${1:-"src/config_aig_base.yaml"} # Config file path (pass as arg or default)
NUM_GENERATE=${2:-1000}                         # Number of graphs to generate (pass as arg or default)
BASE_OUTPUT_DIR="aig_run_${SLURM_JOB_ID}"       # Unique output directory for this run

# --- Derived Paths ---
CHECKPOINT_DIR="${BASE_OUTPUT_DIR}/checkpoints"
GENERATION_DIR="${BASE_OUTPUT_DIR}/generated"
GENERATED_GRAPHS_FILENAME="generated_aigs.pkl"
GENERATED_GRAPHS_PATH="${GENERATION_DIR}/${GENERATED_GRAPHS_FILENAME}"

# --- Stop script on any error ---
set -e

# --- Setup ---
echo "========================================================"
echo "Starting AIG Train -> Generate -> Evaluate Pipeline"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Timestamp: $(date)"
echo "Using Config File: ${CONFIG_FILE}"
echo "Graphs to Generate: ${NUM_GENERATE}"
echo "Base Output Directory: ${BASE_OUTPUT_DIR}"
echo "========================================================"

# Create directories
mkdir -p slurm_logs
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${GENERATION_DIR}"
# Evaluation script outputs to its own directory, no need to pre-create

# Navigate to the script's parent directory (assuming script is in a 'scripts' folder)
# Adjust if your structure is different
# cd .. # Uncomment if needed

echo "Loading modules..."
module purge
module load 2024 # Adjust module environment as needed
module load Anaconda3/2024.06-1
echo "Activating Conda environment..."
source activate aig-rnn # Replace with your actual environment name



echo "--------------------------------------------------------"
echo "Step 1: Starting Training (src/main.py)..."
echo "--------------------------------------------------------"
srun python -u src/main.py \
    --config_file="${CONFIG_FILE}" \
    --save_dir="${CHECKPOINT_DIR}" \
    # Add --restore argument here if needed, e.g.:
    # --restore path/to/previous/checkpoint.pth

echo "Training finished."

# Check if final checkpoint exists
if [ ! -f "$FINAL_CHECKPOINT_PATH" ]; then
    echo "ERROR: Expected final checkpoint ${FINAL_CHECKPOINT_PATH} not found after training!"
    # List available checkpoints for debugging
    echo "Available files in ${CHECKPOINT_DIR}:"
    ls -lh "${CHECKPOINT_DIR}"
    exit 1
fi
echo "Final checkpoint found: ${FINAL_CHECKPOINT_PATH}"

# --- Step 2: Generation ---
echo "--------------------------------------------------------"
echo "Step 2: Starting Generation (src/get_aigs.py)..."
echo "--------------------------------------------------------"
srun python -u src/get_aigs.py \
    --model-path="${FINAL_CHECKPOINT_PATH}" \
    --output-dir="${GENERATION_DIR}" \
    --output-graphs-file="${GENERATED_GRAPHS_FILENAME}" \
    --num-generate=${NUM_GENERATE} \
    # Add other generation parameters like --gen-temp if needed

echo "Generation finished."

# Check if generated graphs file exists
if [ ! -f "$GENERATED_GRAPHS_PATH" ]; then
    echo "ERROR: Generated graphs file ${GENERATED_GRAPHS_PATH} not found after generation!"
    exit 1
fi
echo "Generated graphs saved to: ${GENERATED_GRAPHS_PATH}"

# --- Step 3: Evaluation ---
echo "--------------------------------------------------------"
echo "Step 3: Starting Evaluation (src/evaluate_aigs.py)..."
echo "--------------------------------------------------------"
# Note: evaluate_aigs.py will output results to stdout
srun python -u src/evaluate_aigs.py \
    "${GENERATED_GRAPHS_PATH}" \
    --train_pkl_files ${TRAIN_PKL_FILES_STR} # Pass training files for novelty

echo "Evaluation finished."

# --- Pipeline Complete ---
echo "========================================================"
echo "AIG Pipeline Completed Successfully!"
echo "Timestamp: $(date)"
echo "Outputs saved in: ${BASE_OUTPUT_DIR}"
echo "========================================================"

# Deactivate environment (optional)
# conda deactivate

exit 0
```