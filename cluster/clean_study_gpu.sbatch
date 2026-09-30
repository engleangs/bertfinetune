#!/usr/bin/env bash
#SBATCH --job-name=cs760-clean
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the new project directory}"
phase="${1:?Expected smoke, main, or completion}"
shared_project="${CS760_SHARED_PROJECT:-/data/${USER}/bertfinetune}"
python_bin="${CS760_PYTHON:-${shared_project}/.venv/bin/python}"
export BERT_MODEL_PATH="${BERT_MODEL_PATH:-${shared_project}/model_cache/bert-base-uncased}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

"$python_bin" run_clean_study.py run --phase "$phase" \
  --task-index "${SLURM_ARRAY_TASK_ID:?Expected array task}" --device cuda --resume
