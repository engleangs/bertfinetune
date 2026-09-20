#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

# Keep large wheel downloads and extraction off the cluster's small /tmp quota.
mkdir -p .install-tmp .pip-cache
export TMPDIR="$project_dir/.install-tmp"
export PIP_CACHE_DIR="$project_dir/.pip-cache"
export HTTPS_PROXY="${HTTPS_PROXY:-http://squid.auckland.ac.nz:3128}"
export HTTP_PROXY="${HTTP_PROXY:-http://squid.auckland.ac.nz:3128}"

python3.12 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check \
    'torch==2.7.0+cu118' --index-url https://download.pytorch.org/whl/cu118
.venv/bin/python -m pip install --disable-pip-version-check \
    -r cluster/requirements-pilot.txt
.venv/bin/python -m pip check
.venv/bin/python - <<'PY'
import numpy
import torch
import tqdm
import transformers

print("Environment ready:",
      "torch", torch.__version__,
      "transformers", transformers.__version__,
      "numpy", numpy.__version__,
      "tqdm", tqdm.__version__)
PY
