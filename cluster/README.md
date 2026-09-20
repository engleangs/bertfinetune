# COMPSCI 760 experiments on UoA ML Cluster

This directory contains Slurm job scripts for the new experiment version. The
existing `results.csv`, `results_lodo.csv`, and their checkpoints are historical
results and must remain separate from the fixed comparison and earlier pilot.

## Fixed 120-run comparison (chosen 2026-09-20)

The final comparison uses one preselected recipe for all three losses: learning
rate `3e-5`, 8 epochs, 10% warmup, and gradient norm cap 1.0. No development
pilot is required to choose between recipes. The matrix is 1 mixed in-domain
setting plus 7 leave-one-domain-out settings, each with 5 matched seeds and 3
losses: `(1 + 7) × 5 × 3 = 120` training runs. The primary test outcome is
complete-triplet micro-F1. Each run still selects its checkpoint using
development complete-triplet micro-F1 within its fixed eight-epoch budget.

`run_fixed_comparison.py` assigns two entire data settings to each of four
batches, for 30 runs per batch. Each batch writes to its own CSV under
`artifacts/fixed_3e-5_8epochs/`; individual artifacts use the same separate
root. This avoids concurrent writes to a shared results CSV and keeps earlier
experiments and any partial paired-pilot output untouched. A repeated batch
skips completed runs; an incomplete run stops for inspection rather than being
overwritten automatically.

Inspect the plan locally without training:

```bash
python run_fixed_comparison.py --dry-run
```

On a personal computer, install PyTorch for that computer and the packages in
`cluster/requirements-pilot.txt`, place the same 21 English data files under
`data/m-absa/`, and use the same
`bert-base-uncased` weights and tokenizer. Run batches 0, 1, 2, and 3 in turn;
for example:

```bash
python run_fixed_comparison.py --batch-index 0 --device auto
```

`--device auto` selects the locally available accelerator or CPU. Running the
batches sequentially avoids competing for memory on a single GPU. If the model
is stored locally, point `BERT_MODEL_PATH` at its directory before starting.
The Slurm script below is only for the university cluster.

After the updated source is transferred and the cluster's time limit and GPU
quota are checked, the planned single submission is:

```bash
sbatch cluster/fixed_comparison.sbatch
```

That submission creates four Slurm array tasks and allows at most two to run
concurrently. Each task requests one GPU and up to 24 hours. The time and
concurrency settings must be checked against the cluster policy and actual
runtime before submission. Training must run through Slurm, not in a login
terminal.

## Earlier paired tuning pilot (superseded by the fixed recipe)

## Pilot design

- Loss methods: `standard`, `weighted`, and `category_weighted`.
- Matched conditions: identical data fold, seed, BERT model, batch size,
  optimizer, learning rate, epoch budget, warmup, and gradient clipping.
- Primary selection metric: development-set complete-triplet micro-F1.
- Warmup: 10% of optimizer steps; gradient norm cap: 1.0.
- Candidate common recipes: learning rate `2e-5` for 5 epochs and `3e-5` for
  8 epochs. The pilot compares whole recipes; it does not attribute a change to
  learning rate or epoch count individually.
- Default pilot folds and seeds: in-domain plus Restaurant held out, seeds 13
  and 42. This gives 2 recipes × 3 losses × 2 folds × 2 seeds = 24 tasks.

`run_paired_pilot.py` reads only source-domain training and development files.
It never reads a test file. Each task has its own output directory. Selection
requires all 24 matching development summaries and chooses one recipe from the
equal-weight mean across folds, seeds, and losses. The final test comparison
must use the same selected recipe for all three methods.

## Files needed on the cluster

`cluster/make_transfer_bundle.py` creates a `bertfinetune-cluster.tar` outside
the project directory from tracked code, new experiment files, the 21 English M-ABSA
splits, and five portable BERT files. It excludes Git metadata, virtual
environments, and previous outputs. Upload the archive into `/data/$USER`, then
run these lightweight setup commands there:

```bash
tar -xf bertfinetune-cluster.tar
cd bertfinetune
sha256sum -c cluster/transfer_manifest.sha256
bash cluster/setup_env.sh
```

The ignored `data/m-absa` directory contains the 21 split files. The ignored
`model_cache/bert-base-uncased` directory contains the five portable model and
tokenizer files. The login node's default Python 3.6.8 is too old; use the
available Python 3.12.14. The Windows virtual environment cannot be copied to
Linux. The setup script uses the verified university proxy and places temporary
wheel files on `/data`, since `/tmp` has a small quota. It installs the CUDA
11.8 PyTorch build used locally; the cluster's 610 driver supports older CUDA
applications. Check the install report and the Slurm GPU smoke log before
training. Do not run training in the login terminal.

## Run order

From the project directory on the cluster, after the environment and assets
are ready:

```bash
sbatch cluster/gpu_smoke.sbatch
```

Read the resulting `slurm-cs760-gpu-check-<jobid>.out` and confirm CUDA is
available. Then submit **one** real BERT task by overriding the array range:

```bash
sbatch --array=0 cluster/paired_pilot.sbatch
```

Check its Slurm output and `artifacts/paired_pilot/.../dev_summary.json`.
When it works, submit the remaining tasks with a small concurrency cap:

```bash
sbatch --array=1-23%2 cluster/paired_pilot.sbatch
```

After all 24 tasks finish, run the lightweight selection command in the same
environment (or put it in a short CPU Slurm job if required by local policy):

```bash
python run_paired_pilot.py --summarize
```

The selected common recipe is saved at `artifacts/paired_pilot/selection.json`.
Do not run training directly in the SSH login terminal. These pilot commands
describe the earlier recipe-selection plan and are not part of the chosen
fixed-recipe comparison.
