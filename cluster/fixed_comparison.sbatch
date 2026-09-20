#!/bin/bash
#SBATCH --job-name=cs760-fixed-120
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%2
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
project_dir="/data/${USER}/bertfinetune"
cd "$project_dir"
source .venv/bin/activate

export BERT_MODEL_PATH="$project_dir/model_cache/bert-base-uncased"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python run_fixed_comparison.py --batch-index "$SLURM_ARRAY_TASK_ID" --device cuda
