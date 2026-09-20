"""Run the fixed 120-run comparison in four independent Slurm batches.

The matrix is one in-domain setting plus seven held-out domains, each with five
matched seeds and three loss configurations. The learning rate, epoch budget,
warmup, and gradient clipping are fixed before inspecting final test results.
"""

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import config as cfg
import run_study


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "fixed_3e-5_8epochs"
DOMAINS = ("coursera", "hotel", "laptop", "restaurant", "phone", "sight", "food")
SEEDS = (13, 42, 123, 2024, 777)
LOSSES = ("standard", "weighted", "category_weighted")
BATCH_COUNT = 4
LEARNING_RATE = 3e-5
EPOCHS = 8
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0
CONFIGS = {item.name: item for item in (*cfg.EXPERIMENTS, *cfg.ABLATION_EXPERIMENTS)}


@dataclass(frozen=True)
class FinalTask:
    mode: str
    held_out_domain: str | None
    seed: int
    config_name: str


def fixed_config(config_name):
    return replace(
        CONFIGS[config_name],
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
    )


def build_plan():
    if tuple(cfg.DOMAINS) != DOMAINS or tuple(cfg.SEEDS) != SEEDS:
        raise ValueError("The configured domains or seeds differ from the frozen 120-run plan")
    if set(CONFIGS) != set(LOSSES):
        raise ValueError("The loss configurations differ from the frozen plan")
    shared_settings = [
        {key: value for key, value in asdict(fixed_config(name)).items()
         if key not in {"name", "loss_type"}}
        for name in LOSSES
    ]
    if shared_settings[1:] != shared_settings[:-1]:
        raise ValueError("The three losses must share every other training setting")
    conditions = [("indomain", None), *(("crossdomain", domain) for domain in DOMAINS)]
    plan = [
        FinalTask(mode, held_out_domain, seed, config_name)
        for mode, held_out_domain in conditions
        for seed in SEEDS
        for config_name in LOSSES
    ]
    if len(plan) != 120 or len(set(plan)) != 120:
        raise ValueError("The frozen comparison must contain 120 unique runs")
    return plan


def split_batches(plan):
    runs_per_condition = len(SEEDS) * len(LOSSES)
    runs_per_batch = 2 * runs_per_condition
    if len(plan) != BATCH_COUNT * runs_per_batch:
        raise ValueError("The plan cannot be divided into four 30-run batches")
    return [
        plan[index * runs_per_batch:(index + 1) * runs_per_batch]
        for index in range(BATCH_COUNT)
    ]


def run_batch(batch_index, device="cuda", output_root=DEFAULT_OUTPUT_ROOT):
    batches = split_batches(build_plan())
    if not 0 <= batch_index < BATCH_COUNT:
        raise ValueError(f"batch_index must be between 0 and {BATCH_COUNT - 1}")
    output_root = Path(output_root)
    results_csv = output_root / f"batch_{batch_index}.csv"
    for position, task in enumerate(batches[batch_index], start=1):
        label = task.held_out_domain or "indomain"
        print(
            f"[batch {batch_index}: {position}/30] "
            f"{label} / seed {task.seed} / {task.config_name}",
            flush=True,
        )
        run_study.run(
            mode=task.mode,
            config_name=task.config_name,
            seed=task.seed,
            device=device,
            output_dir=output_root,
            results_csv=results_csv,
            held_out_domain=task.held_out_domain,
            experiment_override=fixed_config(task.config_name),
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--batch-index", type=int, choices=range(BATCH_COUNT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    batches = split_batches(build_plan())
    if args.dry_run:
        print("Fixed comparison: 8 data settings x 5 seeds x 3 losses = 120 runs")
        print("Recipe: lr=3e-5, epochs=8, warmup=10%, max_grad_norm=1.0")
        for index, tasks in enumerate(batches):
            settings = list(dict.fromkeys(task.held_out_domain or "indomain" for task in tasks))
            print(f"batch {index}: {', '.join(settings)}; {len(tasks)} runs")
        return 0
    run_batch(args.batch_index, args.device, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
