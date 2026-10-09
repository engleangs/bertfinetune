"""Small development dashboard, parameter ledger and portable report export."""

from collections import defaultdict
import json
from pathlib import Path
import shutil
import statistics

from src.artifacts import atomic_write_json
from .config import VERSION


def collect_runs(output, smoke=False):
    patterns = ["smoke_*/*/seed_*/manifest.json"] if smoke else ["research/*/seed_*/manifest.json"]
    rows = []
    for pattern in patterns:
        for path in sorted(output.glob(pattern)):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            metrics = None
            if (path.parent / "dev_metrics.json").exists():
                raw = json.loads((path.parent / "dev_metrics.json").read_text(encoding="utf-8"))
                metrics = {key: raw[key] for key in ("explicit", "null", "combined", "null_threshold",
                           "null_threshold_curve", "null_diagnostics", "per_domain", "unweighted_loss")}
                metrics["primary_errors"] = raw["taxonomy"]["primary_outcome_counts"]
            test = None
            if manifest["status"] == "complete":
                raw = json.loads((path.parent / "test_metrics.json").read_text(encoding="utf-8"))
                test = {key: raw[key] for key in ("explicit", "null", "combined", "null_threshold", "null_diagnostics")}
            rows.append({"trial_id": path.parent.parent.name, "seed": manifest["identity"]["seed"],
                         "status": manifest["status"], "config": manifest["identity"]["config"],
                         "stage": manifest.get("stage", "custom"), "smoke": manifest["identity"]["smoke"],
                         "training": manifest.get("training"), "metrics": metrics, "test": test,
                         "error": manifest.get("error"),
                         "directory": str(path.parent.resolve()), "fingerprint": manifest["fingerprint"]})
    return rows


def plot_comparison(rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    groups = defaultdict(list)
    for row in rows:
        if row["status"] in ("dev_complete", "complete"):
            groups[row["trial_id"]].append(row)
    if not groups:
        return False
    groups = sorted(groups.values(), key=lambda rs: (rs[0]["config"]["epochs"], rs[0]["config"]["name"]))
    labels = [f"{rs[0]['config']['name']}  (e{rs[0]['config']['epochs']}, n={len(rs)})" for rs in groups]
    fig, ax = plt.subplots(figsize=(12, max(4, len(groups) * .5 + 1.5)))
    for offset, scope, color in ((-.24, "explicit", "#879aaf"), (0, "null", "#cf802c"), (.24, "combined", "#247e82")):
        values = [statistics.mean(r["metrics"][scope]["micro_f1"] for r in rs) for rs in groups]
        ax.barh([index + offset for index in range(len(groups))], values, height=.22, label=scope.capitalize(), color=color)
    ax.set_yticks(range(len(groups)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Development micro F1 (mean across listed seeds)")
    ax.set_title("SMOKE ONLY: integration scores" if any(r["smoke"] for r in rows) else
                 "NULL fast track: keep epoch budgets and seed counts visible", loc="left")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def write_report(output, share=None, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = collect_runs(output, smoke)
    suffix = "_smoke" if smoke else ""
    freeze_path = output / "selection.json"
    frozen = json.loads(freeze_path.read_text(encoding="utf-8")) if freeze_path.exists() else None
    if frozen and frozen.get("smoke") != smoke:
        frozen = None
    calibrations = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(output.glob("calibration/seed_*/calibration.json"))]
    chart = f"comparison{suffix}.png"
    has_chart = plot_comparison(rows, output / chart)
    report = {"version": VERSION, "smoke": smoke, "runs": rows, "frozen_selection": frozen,
              "calibrations": calibrations}
    atomic_write_json(output / f"report_data{suffix}.json", report)
    lines = ["# NULL fast-track report", "", "SMOKE CHECK ONLY: scores are not research evidence." if smoke else
             "Exploratory study. Search and checkpoint/threshold selection use development data; test evaluation is a separate command.", "",
             "Shared explicit objective: .75 CE + .25 inverse-frequency CE. Frequencies and vocabulary use training annotations only. Four sentiment classes are retained.", "",
             "All new arms reserve one IA slot and share the same text budget. CUDA AMP and cached tokenization are enabled for throughput; these arms have a different numerical/input protocol from the historical study.", "",
             "## Technique tracking", "", "All five techniques are implemented. The completed-run counts below show what has actually been measured.", "",
             "| Technique | Implemented experiment | Where it appears |", "|---|---|---|",
             "| 1 Calibration/slices | Fine threshold grid + abstention; presence/category/pair scores; NULL-only/mixed/no-NULL, rare/unseen and domain slices | Every dev_metrics.json; optional legacy calibration |",
             "| 2 NULL focal | Sigmoid focal using unweighted probabilities, gamma recorded | Loss screen |",
             "| 3 ASL | Separate positive/negative focusing and negative margin, without stacked BCE weights | Loss screen |",
             "| 4 Implicit representation | Dedicated [IA], then learned category-conditioned attention | Representation stage |",
             "| 5 Informative negatives | All positives + adaptive hard/random negatives; random control with identical budget/reduction | Negative stage |", ""]
    completed = [r for r in rows if r["status"] in ("dev_complete", "complete")]
    coverage = {"1 Calibration/slices": len(completed),
                "2 Binary focal": sum(r["config"]["null_head"] and r["config"]["null_loss"] == "focal" for r in completed),
                "3 ASL": sum(r["config"]["null_head"] and r["config"]["null_loss"] == "asl" for r in completed),
                "4 Dedicated IA": sum(r["config"]["null_head"] and r["config"]["representation"] != "cls" for r in completed),
                "4 Category attention": sum(r["config"]["null_head"] and r["config"]["representation"] == "ia_attention" for r in completed),
                "5 Random control": sum(r["config"]["negative_sampling"] == "random" for r in completed),
                "5 Hard negatives": sum(r["config"]["negative_sampling"] == "hard" for r in completed)}
    lines += ["| Method | Completed development runs |", "|---|---:|"]
    lines += [f"| {method} | {count} |" for method, count in coverage.items()]
    lines += ["", "Counts include confirmation runs; stages remain separate in the run table. Legacy calibration is reported separately.", ""]
    timing = [s for r in completed if r["training"] for s in r["training"].get("epoch_seconds", [])]
    if timing:
        lines += [f"Observed median train + dev evaluation time: {statistics.median(timing)/60:.2f} min/epoch. This excludes checkpoint writes, model loading, and tokenization; smoke timing does not estimate full-corpus training.", ""]
    if calibrations:
        lines += ["## Historical development calibration", "", "These are separate diagnostics of existing checkpoints, not a controlled comparison with the new input/AMP protocol.", ""]
        for cal in calibrations:
            saved, new = cal["saved"], cal["calibrated"]
            lines += [f"Seed {cal['seed']}: threshold {saved['threshold']:g} -> {new['threshold']:g}; combined dev F1 {saved['combined']['micro_f1']:.4f} -> {new['combined']['micro_f1']:.4f}; NULL F1 {new['null']['micro_f1']:.4f}.", ""]
    if frozen:
        lines += ["## Frozen shortlist", "", f"Seeds: {frozen['seeds']}; selection: {frozen['selection_rule']}.", ""]
        for variant in frozen["variants"]:
            cfg = variant["config"]
            scopes = {scope: statistics.mean(r["development"][scope]["micro_f1"] for r in variant["runs"]) for scope in ("explicit", "null", "combined")}
            deviations = {scope: statistics.stdev(r["development"][scope]["micro_f1"] for r in variant["runs"]) if len(variant["runs"]) > 1 else None for scope in scopes}
            lines += [f"**{variant['role']}**: `{variant['trial_id']}`; explicit {scopes['explicit']:.4f}, NULL {scopes['null']:.4f}, combined {scopes['combined']:.4f}.",
                      f"Sample SD: {deviations}. LR={cfg['lr']:g}, NULL loss={cfg['null_loss']}, representation={cfg['representation']}, sampling={cfg['negative_sampling']}, NULL outer weight={cfg['null_loss_weight']:g}.", ""]
        lines += ["A selected threshold of 1 means abstention. Nonzero NULL extraction and a combined-score gain are separate findings; neither is guaranteed.", ""]
        lines += paired_comparison(rows, frozen, "metrics", "development")
    else:
        lines += ["The final shortlist is not frozen yet. Short-budget results select candidates for confirmation; they do not establish the five-epoch winner.", ""]
    lines += ["## Individual development runs", "", "| Recipe | Stage | Seed | Status | Epoch budget | LR | Explicit F1 | NULL P | NULL R | NULL F1 | Combined F1 | Threshold |", "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        cfg, m = row["config"], row["metrics"]
        values = ([f"{m['explicit']['micro_f1']:.4f}", f"{m['null']['micro_precision']:.4f}", f"{m['null']['micro_recall']:.4f}",
                   f"{m['null']['micro_f1']:.4f}", f"{m['combined']['micro_f1']:.4f}", f"{m['null_threshold']:g}"] if m else ["pending"] * 6)
        lines.append(f"| {cfg['name']} | {row['stage']} | {row['seed']} | {row['status']} | {cfg['epochs']} | {cfg['lr']:g} | " + " | ".join(values) + " |")
    for row in rows:
        if row["error"]:
            lines += ["", f"Run `{row['trial_id']}`, seed {row['seed']}: {row['error']}"]
    if has_chart:
        lines += ["", f"![Development comparison]({chart})", ""]
    lines += ["## Hyperparameters and interpretation", "", "| Trial ID | NULL loss | Representation | Sampling | Weight/cap | Focal gamma | ASL positive/negative gamma; margin | Negative ratio/minimum/hard fraction | Attention size | Batch/max length | Precision |", "|---|---|---|---|---|---:|---|---|---:|---|---|"]
    seen = set()
    for row in rows:
        if row["trial_id"] in seen:
            continue
        seen.add(row["trial_id"])
        c = row["config"]
        lines.append(f"| {row['trial_id']} | {c['null_loss']} | {c['representation']} | {c['negative_sampling']} | {c['null_loss_weight']}/{c['null_pos_weight_cap']} | {c['null_focal_gamma']} | {c['asl_gamma_positive']}/{c['asl_gamma_negative']}; {c['asl_margin']} | {c['negative_ratio']}/{c['minimum_negatives']}/{c['hard_fraction']} | {c['attention_size']} | {c['batch_size']}/{c['max_len']} | {c['precision']} |")
    lines += ["", "ASL ignores the BCE positive cap. Random and hard sampling use a balanced positive/negative-group reduction; all-cell loss uses a cell mean. Compare random versus hard to isolate mining; all-cell versus sampled also changes normalization.", "",
              "Historical test sets were inspected before this study. These are exploratory results. Exact F1 covers eligible explicit annotations plus deduplicated NULL; ambiguous/conflicting/overlapping explicit exclusions remain. No claim of new LODO improvement is made.", "",
              "Development and epoch CSVs append rows; report.md/report_data.json are refreshable summaries. Immutable selection archives preserve previous seed cohorts. Interrupted training resumes at an epoch boundary, replaying an unfinished epoch.", ""]
    completed_tests = [r for r in rows if r["test"]]
    if completed_tests:
        lines += ["## Frozen test evaluation", "", "| Trial | Seed | Explicit F1 | NULL F1 | Combined F1 |", "|---|---:|---:|---:|---:|"]
        for row in completed_tests:
            m = row["test"]
            lines.append(f"| {row['trial_id']} | {row['seed']} | {m['explicit']['micro_f1']:.4f} | {m['null']['micro_f1']:.4f} | {m['combined']['micro_f1']:.4f} |")
        if frozen:
            lines += paired_comparison(rows, frozen, "test", "test")
    target = output / f"report{suffix}.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if share and not smoke:
        share = Path(share)
        share.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, share / "report.md")
        shutil.copy2(output / "report_data.json", share / "report_data.json")
        if has_chart:
            shutil.copy2(output / chart, share / chart)
        if frozen:
            shutil.copy2(freeze_path, share / "selection.json")
    return target


def paired_comparison(rows, frozen, field, label):
    """Compare only frozen arms, matched seeds and equal epoch budgets."""
    variants = {v["role"]: v for v in frozen["variants"]}
    if "Best NULL on" not in variants or "NULL off control" not in variants:
        return []
    head, control = variants["Best NULL on"], variants["NULL off control"]
    by_arm = [{r["seed"]: r[field] for r in rows if r["trial_id"] == v["trial_id"] and r[field]
               and r["status"] in ("dev_complete", "complete")} for v in (head, control)]
    seeds = sorted(set(by_arm[0]) & set(by_arm[1]) & set(frozen["seeds"]))
    if not seeds:
        return []
    lines = ["", f"### Matched {label} comparison", "",
             f"Frozen arms, {len(seeds)}/{len(frozen['seeds'])} matched seeds. Values are micro F1 mean +/- sample SD; SD requires at least two seeds.", "",
             "| Arm | Explicit F1 | NULL F1 | Combined F1 |", "|---|---:|---:|---:|"]
    for name, arm in zip(("NULL on", "NULL off"), by_arm):
        values = []
        for scope in ("explicit", "null", "combined"):
            scores = [arm[s][scope]["micro_f1"] for s in seeds]
            sd = f"{statistics.stdev(scores):.4f}" if len(scores) > 1 else "NA"
            values.append(f"{statistics.mean(scores):.4f} +/- {sd}")
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    differences = [by_arm[0][s]["combined"]["micro_f1"] - by_arm[1][s]["combined"]["micro_f1"] for s in seeds]
    lines += ["", f"Mean paired combined-F1 difference (NULL on - NULL off): {100*statistics.mean(differences):+.2f} percentage points.", "",
              "| Seed | Combined-F1 difference (percentage points) |", "|---:|---:|"]
    lines += [f"| {seed} | {100*delta:+.2f} |" for seed, delta in zip(seeds, differences)]
    lines += ["", "This is an exploratory comparison; a small seed cohort and adaptive development selection limit statistical conclusions.", ""]
    return lines
