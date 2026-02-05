#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=name

module --ignore_cache load "transformers/4.57.1"
module --ignore_cache load "pandas/2.3.3"
module load transformers/4.57.1
module load pandas/2.3.3
module load torch/2.9.0

source ../env_llm/bin/activate
python3 infer_script_batching.py
