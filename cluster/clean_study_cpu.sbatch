#!/usr/bin/env bash
#SBATCH --job-name=cs760-clean-report
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the new project directory}"
shared_project="${CS760_SHARED_PROJECT:-/data/${USER}/bertfinetune}"
python_bin="${CS760_PYTHON:-${shared_project}/.venv/bin/python}"
case "${1:?Expected select or finalize}" in
  select)
    "$python_bin" run_clean_study.py select
    # Only the 64 unselected NEW search checkpoints can be removed here.
    "$python_bin" run_clean_study.py prune --scope unselected --apply
    ;;
  finalize)
    "$python_bin" run_clean_study.py summarize
    "$python_bin" run_clean_study.py export
    # Preserve one development-selected model per fold/loss on the cluster (16).
    "$python_bin" run_clean_study.py prune --scope archive --apply
    # Refresh the export so its manifests describe the actual retention state.
    "$python_bin" run_clean_study.py export
    ;;
  *) echo "Unknown CPU phase" >&2; exit 2 ;;
esac
