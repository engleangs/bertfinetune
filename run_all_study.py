"""
S1: runs the entire study — 2 modes x 2 configs x len(config.SEEDS) seeds.
With 5 seeds that's 20 runs total. Cut config.SEEDS to 3 if week 4 is tight;
everything downstream (analyze_results.py) adapts automatically since it
just reads whatever's in results.csv.
"""
import config as cfg
from run_study import run

if __name__ == "__main__":
    total = len(cfg.MODES) * len(cfg.EXPERIMENTS) * len(cfg.SEEDS)
    done = 0
    for mode in cfg.MODES:
        for experiment_cfg in cfg.EXPERIMENTS:
            for seed in cfg.SEEDS:
                done += 1
                print(f"\n=== [{done}/{total}] mode={mode} config={experiment_cfg.name} seed={seed} ===")
                run(mode, experiment_cfg.name, seed)

    print(f"\nAll {total} runs complete. Run analyze_results.py next.")
