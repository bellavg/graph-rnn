#!/bin/bash
#SBATCH --job-name=graph-rnn
#SBATCH --partition=gpu_a100     # Specify the appropriate partition here
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/graphrnn_%j.out





module purge

module load 2024
module load Anaconda3/2024.06-1

conda env create aig-rnn
source activate aig-rnn
pip install -r requirements.txt

echo "Environment created and activated"