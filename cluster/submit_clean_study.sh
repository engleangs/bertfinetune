#!/usr/bin/env bash
# Usage: bash cluster/submit_clean_study.sh [--submit]
# Without --submit, print the exact job graph without contacting Slurm.
set -euo pipefail
script_path="${BASH_SOURCE[0]}"
script_dir="."
if [[ "$script_path" == */* ]]; then script_dir="${script_path%/*}"; fi
cd "$script_dir/.."
concurrency="${CS760_CONCURRENCY:-2}"
train_time="${CS760_TRAIN_TIME:-04:00:00}"
shared_project="${CS760_SHARED_PROJECT:-/data/${USER}/bertfinetune}"
python_bin="${CS760_PYTHON:-${shared_project}/.venv/bin/python}"

echo "smoke: 2 small source train/dev jobs"
echo "main: 116 interleaved jobs (20 baseline + 96 search), concurrency ${concurrency}"
echo "selection: source-dev selection separately in each fold; prune 64 unselected models"
echo "completion: 80 jobs (32 checkpoint evaluations + 48 new training runs)"
echo "finalize: verify all results, export small ZIP, retain 16 models, refresh ZIP"
echo "GPU time request per task: ${train_time}"
if [[ "${1:-}" != "--submit" ]]; then
  echo "Dry run only. Check quota/time and smoke logs before a full submission."
  exit 0
fi
if [[ ! "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
  echo "CS760_CONCURRENCY must be a positive integer" >&2; exit 2
fi
if [[ -e artifacts/clean_study_v1/submission_jobs.txt ]]; then
  echo "A submission record already exists. Inspect jobs and retry individual failed indices." >&2
  exit 2
fi
export BERT_MODEL_PATH="${BERT_MODEL_PATH:-${shared_project}/model_cache/bert-base-uncased}"
"$python_bin" run_clean_study.py prepare
record="artifacts/clean_study_v1/submission_jobs.txt"
trap 'echo "Submission interrupted. Inspect partial job IDs in artifacts/clean_study_v1/submission_jobs.txt before retrying." >&2' ERR
smoke=$(sbatch --parsable --array="0-1%${concurrency}" --time="$train_time" cluster/clean_study_gpu.sbatch smoke)
echo "smoke=${smoke}" >> "$record"
main=$(sbatch --parsable --dependency="afterok:${smoke}" --array="0-115%${concurrency}" --time="$train_time" cluster/clean_study_gpu.sbatch main)
echo "main=${main}" >> "$record"
selection=$(sbatch --parsable --dependency="afterok:${main}" cluster/clean_study_cpu.sbatch select)
echo "selection=${selection}" >> "$record"
completion=$(sbatch --parsable --dependency="afterok:${selection}" --array="0-79%${concurrency}" --time="$train_time" cluster/clean_study_gpu.sbatch completion)
echo "completion=${completion}" >> "$record"
finalize=$(sbatch --parsable --dependency="afterok:${completion}" cluster/clean_study_cpu.sbatch finalize)
echo "finalize=${finalize}" >> "$record"
cat "$record"
