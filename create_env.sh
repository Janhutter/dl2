#!/bin/bash
#SBATCH --nodes=1                           # node count
#SBATCH --ntasks=1                          # total number of tasks across all nodes
#SBATCH --cpus-per-task=1                   # cpu-cores per task (>1 if multi-threaded tasks), 4 is default
#SBATCH --gpus=1                            # number of gpus per node
#SBATCH --partition=gpu_mig                # partition
#SBATCH --time=00:20:00                     # total run time limit (HH:MM:SS)
#SBATCH --output=logs/create_env_%A.log

source .venv/bin/activate
alias python=~/tea/.venv/bin/python

python -m pip install -r requirements.txt
