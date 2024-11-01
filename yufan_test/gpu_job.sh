#!/bin/bash
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=pawsey0799-gpu
#SBATCH --mem=28.75GB

# Load necessary modules
# ~/.bashrc
# ca dxtb-dev

# Run the job
