#!/bin/bash
#SBATCH --job-name=aig_rnn_pipeline
#SBATCH --partition=gpu_h100     # Specify the appropriate partition here
#SBATCH --gpus=1
#SBATCH --time=12:00:00          # Adjust time as needed
#SBATCH --output=slurm_logs/aig_pipeline_%j.out


# --- Configuration ---
CONFIG_FILE=${1:-"src/config_aig_rnn.yaml"} # Config file path (pass as arg or default)
NUM_GENERATE=${2:-1000}                         # Number of graphs to generate (pass as arg or default)
#BASE_OUTPUT_DIR="aig_run_${SLURM_JOB_ID}"       # Unique output directory for this run

## --- Stop script on any error ---
#set -e
#set -o pipefail # Ensure errors in pipes are caught
#
## --- Setup ---
#echo "========================================================"
#echo "Starting AIG Train -> Generate -> Evaluate Pipeline"
#echo "Job ID: ${SLURM_JOB_ID}"
#echo "Timestamp: $(date)"
#echo "Using Config File: ${CONFIG_FILE}"
#echo "Graphs to Generate: ${NUM_GENERATE}"
#echo "Base Output Directory: ${BASE_OUTPUT_DIR}"
#echo "========================================================"
#
## Validate config file exists
#if [ ! -f "$CONFIG_FILE" ]; then
#    echo "ERROR: Config file not found at ${CONFIG_FILE}"
#    exit 1
#fi
#
## --- Parse Config File using Python ---
#echo "Parsing configuration from ${CONFIG_FILE}..."
#CONFIG_DATA=$(python -c "
#import yaml, json, sys
#try:
#    with open('$CONFIG_FILE', 'r') as f:
#        config = yaml.safe_load(f)
#    # Extract needed values with defaults
#    steps = config.get('train', {}).get('steps', 0)
#    checkpoint_dir_rel = config.get('train', {}).get('checkpoint_dir', 'checkpoints/default_checkpoints') # Relative path from config
#    graph_files = config.get('data', {}).get('graph_files', [])
#    if not isinstance(graph_files, list): graph_files = [] # Ensure it's a list
#
#    # Construct absolute checkpoint dir based on BASE_OUTPUT_DIR for training save path
#    # Note: main.py uses --save_dir, which becomes the base for checkpoint_dir_rel
#    train_save_dir = '$BASE_OUTPUT_DIR' # Training saves checkpoints relative to this base
#    full_checkpoint_dir = f\"{train_save_dir}/{checkpoint_dir_rel}\" # Where checkpoints will actually be saved
#
#    output = {
#        'steps': steps,
#        'full_checkpoint_dir': full_checkpoint_dir, # The actual directory containing checkpoints
#        'graph_files': graph_files
#    }
#    print(json.dumps(output))
#except Exception as e:
#    print(f\"Error parsing config: {e}\", file=sys.stderr)
#    sys.exit(1)
#")
#
#if [ $? -ne 0 ]; then
#    echo "ERROR: Failed to parse config file using Python."
#    exit 1
#fi
#
## Extract values from JSON output
#TOTAL_STEPS=$(echo "$CONFIG_DATA" | python -c "import sys, json; print(json.load(sys.stdin)['steps'])")
#FULL_CHECKPOINT_DIR=$(echo "$CONFIG_DATA" | python -c "import sys, json; print(json.load(sys.stdin)['full_checkpoint_dir'])")
## Read graph files into a bash array
#readarray -t TRAIN_PKL_FILES < <(echo "$CONFIG_DATA" | python -c "import sys, json; [print(f) for f in json.load(sys.stdin)['graph_files']]")
#
## Check if steps were parsed correctly
#if [ "$TOTAL_STEPS" -le 0 ]; then
#    echo "ERROR: Could not parse valid 'train.steps' from config file."
#    exit 1
#fi
#
## Construct the final checkpoint path
#FINAL_CHECKPOINT_PATH="${FULL_CHECKPOINT_DIR}/checkpoint-${TOTAL_STEPS}.pth"
#
## Construct the string for evaluate_aigs.py (space-separated paths)
#TRAIN_PKL_FILES_STR=$(printf "%s " "${TRAIN_PKL_FILES[@]}")
#TRAIN_PKL_FILES_STR=${TRAIN_PKL_FILES_STR% } # Remove trailing space
#
#echo "Parsed Config:"
#echo "  Total Training Steps: ${TOTAL_STEPS}"
#echo "  Full Checkpoint Dir: ${FULL_CHECKPOINT_DIR}"
#echo "  Expected Final Checkpoint: ${FINAL_CHECKPOINT_PATH}"
#echo "  Training PKL Files: ${TRAIN_PKL_FILES_STR}"
## --- End Config Parsing ---
#
#
## --- Derived Paths ---
## Checkpoint dir for main.py's --save_dir argument (base for relative path in config)
#TRAIN_SAVE_DIR="${BASE_OUTPUT_DIR}" # Training script saves relative to this
GENERATION_DIR="./generated"
#GENERATED_GRAPHS_FILENAME="generated_aigs.pkl"
#GENERATED_GRAPHS_PATH="${GENERATION_DIR}/${GENERATED_GRAPHS_FILENAME}"
#
## Create directories
#mkdir -p slurm_logs
#mkdir -p "${TRAIN_SAVE_DIR}" # Create base save dir for training
#mkdir -p "${GENERATION_DIR}"
## The actual checkpoint dir will be created by main.py relative to TRAIN_SAVE_DIR
#
## --- Environment Setup ---
echo "Loading modules..."
module purge
module load 2024 # Adjust module environment as needed
module load Anaconda3/2024.06-1
echo "Activating Conda environment..."
## Make sure the environment name is correct
#source activate aig-rnn || { echo "ERROR: Failed to activate Conda environment 'aig-rnn'"; exit 1; }
#
#
## --- Step 1: Training ---
#echo "--------------------------------------------------------"
#echo "Step 1: Starting Training (src/main.py)..."
#echo "Saving checkpoints relative to: ${TRAIN_SAVE_DIR}"
#echo "--------------------------------------------------------"
#srun python -u main.py \
#    --config_file="${CONFIG_FILE}" \
#    --save_dir="${TRAIN_SAVE_DIR}" \
#    # Add --restore argument here if needed, e.g.:
#    # --restore path/to/previous/checkpoint.pth
#
#echo "Training finished."
#
## Check if final checkpoint exists
#if [ ! -f "$FINAL_CHECKPOINT_PATH" ]; then
#    echo "ERROR: Expected final checkpoint ${FINAL_CHECKPOINT_PATH} not found after training!"
#    echo "Checking directory: ${FULL_CHECKPOINT_DIR}"
#    ls -lh "${FULL_CHECKPOINT_DIR}" || echo "  Directory not found or empty."
#    exit 1
#fi
#echo "Final checkpoint found: ${FINAL_CHECKPOINT_PATH}"

## --- Step 2: Generation ---
#echo "--------------------------------------------------------"
#echo "Step 2: Starting Generation (src/get_aigs.py)..."
#echo "Using model: ${FINAL_CHECKPOINT_PATH}"
#echo "Saving generated graphs in: ${GENERATION_DIR}"
#echo "--------------------------------------------------------"
#srun python -u src/get_aigs.py \
#    --model-path="aig_run_11655841/checkpoints/checkpoints_gru_rnn_node/checkpoint-75000.pth"\
#    --output-dir="${GENERATION_DIR}" \
#    --output-graphs-file="generated_graphs.pkl" \
#    --num-generate=${NUM_GENERATE} \
#    # Add other generation parameters like --gen-temp if needed
#
#echo "Generation finished."
#
## Check if generated graphs file exists
#if [ ! -f "$GENERATED_GRAPHS_PATH" ]; then
#    echo "ERROR: Generated graphs file ${GENERATED_GRAPHS_PATH} not found after generation!"
#    exit 1
#fi
#echo "Generated graphs saved to: ${GENERATED_GRAPHS_PATH}"

# --- Step 3: Evaluation ---
echo "--------------------------------------------------------"
echo "Step 3: Starting Evaluation (src/evaluate_aigs.py)..."
echo "Evaluating: ${GENERATED_GRAPHS_PATH}"
echo "Against Training Files: ${TRAIN_PKL_FILES_STR}"
echo "--------------------------------------------------------"
# Note: evaluate_aigs.py will output results to stdout
srun python -u src/evaluate_aigs.py \
    "./generated/generated_graphs.pkl" \
    --train_pkl_files ${TRAIN_PKL_FILES_STR} # Pass training files for novelty

echo "Evaluation finished."

# --- Pipeline Complete ---
echo "========================================================"
echo "AIG Pipeline Completed Successfully!"
echo "Timestamp: $(date)"
echo "========================================================"

# Deactivate environment (optional)
# conda deactivate

exit 0
