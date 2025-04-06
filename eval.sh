#!/bin/bash
#SBATCH --job-name=aig-eval
#SBATCH --partition=gpu_a100     # Specify the appropriate partition here (adjust if needed)
#SBATCH --gpus=1                 # Request 1 GPU (adjust if evaluation is faster/slower)
#SBATCH --time=00:59:00          # Adjust time based on expected evaluation duration (e.g., 1 hour)
#SBATCH --output=slurm_logs/aig_eval_%j.out # Specific log file for evaluation


# Create log directories if they don't exist
mkdir -p slurm_logs
mkdir -p evaluation_results # Create a directory for evaluation results

# Load modules (same as training)
module purge
module load 2024
module load Anaconda3/2024.06-1

# Activate your environment (same as training)
source activate aig-rnn
# --- Evaluation Specific Settings ---

# Define the checkpoint to evaluate
CHECKPOINT_PATH="runs/graphrnn_10999759/base_checkpoints/checkpoint-10000.pth"

# Define the output directory for this evaluation run
# You might want to make this more specific, e.g., include checkpoint step
EVAL_OUTPUT_DIR="evaluation_results/eval_run_${SLURM_JOB_ID}"
mkdir -p $EVAL_OUTPUT_DIR

TEST_DATASET_PATH="dataset/inputs8_outputs8max_nodes128max.pkl"

# Optional: Path to test dataset for comparison metrics (if needed by aig_evaluate.py)
# TEST_DATASET_PATH="dataset/inputs8_outputs8max_nodes128max.pkl" # Example path

echo "Evaluating checkpoint: $CHECKPOINT_PATH"
echo "Evaluation results will be saved to: $EVAL_OUTPUT_DIR"

# Run the AIG evaluation script using srun
# Make sure src/aig_evaluate.py accepts these arguments (check its argparse setup)
srun python -u src/aig_evaluate.py \
    --model_paths $CHECKPOINT_PATH \
    --output_dir $EVAL_OUTPUT_DIR \
    --num_graphs 1000 \
    --min_nodes 12 \
    --max_nodes 128 \
    --test_dataset $TEST_DATASET_PATH \


# Print job completion message
echo "Evaluation job finished at $(date)"
echo "Results saved to $EVAL_OUTPUT_DIR"
