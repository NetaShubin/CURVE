#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=persona_multi
#SBATCH --array=0-59%10
#SBATCH --output=logs/%A_%a.out

# Create a logs directory if it doesn't exist
mkdir -p logs

module --ignore_cache load "transformers/4.57.1"
module --ignore_cache load "pandas/2.3.3"
module load transformers/4.57.1
module load pandas/2.3.3
module load torch/2.9.0

source ../env_llm/bin/activate

# Define the list of countries
COUNTRIES=("USA" "Japan" "India" "Brazil" "Saudi Arabia" "South Africa")

# Calculate which country and which iteration this specific task is
COUNTRY_IDX=$((SLURM_ARRAY_TASK_ID / 10))
ITERATION=$(( (SLURM_ARRAY_TASK_ID % 10) + 1 ))
SELECTED_COUNTRY=${COUNTRIES[$COUNTRY_IDX]}

echo "🌍 Country: $SELECTED_COUNTRY | 🔄 Iteration: $ITERATION"

time python3 param_script_batching.py \
        --country "$SELECTED_COUNTRY" \
        --iteration "$ITERATION"
