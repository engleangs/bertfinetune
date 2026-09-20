#!/bin/bash
#SBATCH --job-name=cs760-gpu-check
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
project_dir="/data/${USER}/bertfinetune"
cd "$project_dir"
source .venv/bin/activate
export BERT_MODEL_PATH="$project_dir/model_cache/bert-base-uncased"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - <<'PY'
import sys
import os
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("Slurm job did not receive a usable CUDA GPU")
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA runtime:", torch.version.cuda)
model_dir = os.environ["BERT_MODEL_PATH"]
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir).to("cuda").eval()
inputs = tokenizer("A short GPU smoke test.", return_tensors="pt")
with torch.inference_mode():
    output = model(**{name: tensor.to("cuda") for name, tensor in inputs.items()})
print("BERT output shape:", tuple(output.last_hidden_state.shape))
PY
