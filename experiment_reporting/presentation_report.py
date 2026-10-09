"""Combine saved studies for presentation; never choose models from test scores."""

from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics

from experiment_reporting.analysis import flatten_metrics

SCOPES = ("explicit", "null", "combined")
PARAMETERS = ("loss_type", "ce_weight", "weighted_ce_weight", "focal_gamma", "lr", "epochs",
              "batch_size", "max_len", "weight_decay", "warmup_ratio", "max_grad_norm", "null_head",
              "vocabulary_scope", "selection_metric", "null_loss", "null_loss_weight", "null_pos_weight_cap",
              "null_focal_gamma", "asl_gamma_positive", "asl_gamma_negative", "asl_margin", "representation",
              "negative_sampling", "negative_ratio", "minimum_negatives", "hard_fraction", "attention_size",
              "precision", "amp_initial_scale")


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def snapshot(study, name, optional=False):
    frozen = read(study / "selection.json")
    if frozen is None:
        if optional:
            return None, []
        raise ValueError(f"Missing frozen study: {study}")
    if frozen.get("smoke"):
        raise ValueError("Smoke studies cannot appear as final research results")
    if not frozen["seeds"] or len(set(frozen["seeds"])) != len(frozen["seeds"]):
        raise ValueError("Frozen seeds are missing or duplicated")
    rows, trials = [], set()
    for variant in frozen["variants"]:
        if variant["trial_id"] in trials or sorted(r["seed"] for r in variant["runs"]) != sorted(frozen["seeds"]):
            raise ValueError("Frozen variants have duplicate trials or inconsistent cohorts")
        trials.add(variant["trial_id"])
        role = variant.get("role") or "; ".join(variant["roles"])
        for run in variant["runs"]:
            directory = Path(run["directory"])
            manifest = read(directory / "manifest.json")
            if manifest is None or manifest["status"] not in ("dev_complete", "complete"):
                raise ValueError(f"Frozen run is incomplete: {directory}")
            if (manifest["identity"].get("smoke") or manifest["fingerprint"] != run["fingerprint"]
                    or manifest["checkpoint_sha256"] != run["checkpoint_sha256"]
                    or manifest["identity"]["config"] != variant["config"]
                    or manifest["identity"]["seed"] != run["seed"]
                    or manifest["training"]["best_epoch"] != run["best_epoch"]):
                raise ValueError(f"Frozen metadata changed: {directory}")
            dev = read(directory / "dev_metrics.json")
            if dev is None or dev["null_threshold"] != run["null_threshold"]:
                raise ValueError(f"Frozen development threshold changed: {directory}")
            for split in ("dev", "test"):
                if split == "test" and manifest["status"] != "complete":
                    continue
                raw = dev if split == "dev" else read(directory / "test_metrics.json")
                if raw is None or raw["null_threshold"] != run["null_threshold"]:
                    raise ValueError(f"Missing metrics or changed frozen test threshold: {directory}")
                flat = flatten_metrics(raw)
                if sum(flat["outcomes"].values()) != flat["gold_triplets"]:
                    raise ValueError(f"Taxonomy does not partition gold: {directory}")
                rows.append({"study": name, "trial_id": variant["trial_id"], "role": role, "seed": run["seed"],
                             "split": split, "config": variant["config"], "best_epoch": run["best_epoch"],
                             "threshold": run["null_threshold"], "metrics": raw, "flat": flat,
                             "expected_seeds": frozen["seeds"], "fingerprint": run["fingerprint"],
                             "checkpoint_sha256": run["checkpoint_sha256"], "directory": str(directory)})
                rows[-1]["status"] = manifest["status"]
    return frozen, rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["study"], row["trial_id"], row["split"])].append(row)
    summaries = []
    for (study, trial, split), members in groups.items():
        scores = {}
        for scope in SCOPES:
            values = [r["metrics"][scope]["micro_f1"] for r in members]
            scores[scope] = {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else None,
                             "precision": statistics.mean(r["metrics"][scope]["micro_precision"] for r in members),
                             "recall": statistics.mean(r["metrics"][scope]["micro_recall"] for r in members)}
        summaries.append({"study": study, "trial_id": trial, "role": members[0]["role"], "split": split,
                          "seeds": sorted(r["seed"] for r in members), "expected_seeds": members[0]["expected_seeds"],
                          "scores": scores})
    return summaries


def parameter_ledger(quick, null_study, rows):
    ledger = {}

    def add(study, trial, seed, stage, cfg, scores, threshold, status):
        key = study, trial, seed
        item = ledger.get(key)
        if item:
            if stage not in item["stages"]:
                item["stages"].append(stage)
            if status == "complete":
                item["status"] = status
            return
        ledger[key] = {"study": study, "trial_id": trial, "seed": seed, "stages": [stage], "config": cfg,
                       "dev": scores, "threshold": threshold, "status": status}

    for item in quick.get("search_results", []):
        add("Quick tuning", item["trial_id"], item["seed"], item["stage"], item["config"],
            {s: item[s + "_f1"] for s in SCOPES}, item["null_threshold"], "dev_complete")
    for row in rows:
        if row["split"] == "dev":
            study = {"Quick screens": "Quick tuning", "NULL screens": "NULL adaptation"}.get(row["study"], row["study"])
            add(study, row["trial_id"], row["seed"], "frozen screen" if row.get("result_type") == "screen" else "frozen finalist", row["config"],
                {s: row["metrics"][s]["micro_f1"] for s in SCOPES}, row["threshold"], row["status"])
    for path in sorted((null_study / "research").glob("*/seed_*/manifest.json")):
        manifest = read(path)
        if manifest["identity"].get("smoke"):
            continue
        metric = read(path.parent / "dev_metrics.json") if manifest["status"] in ("dev_complete", "complete") else None
        add("NULL adaptation", path.parent.parent.name, manifest["identity"]["seed"], manifest.get("stage", "custom"),
            manifest["identity"]["config"], {s: metric[s]["micro_f1"] for s in SCOPES} if metric else None,
            metric["null_threshold"] if metric else None, manifest["status"])
    return list(ledger.values())


def save_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return f"{value:.4f}" if value is not None else "NA"


def mean_sd(score):
    return f"{score['mean']:.4f} +/- " + (f"{score['sd']:.4f}" if score["sd"] is not None else "NA")


def matched_null(rows, split):
    by_role = {}
    for role in ("Best NULL on", "NULL off control"):
        by_role[role] = {r["seed"]: r for r in rows if r["study"] == "NULL adaptation" and r["role"] == role and r["split"] == split}
    seeds = sorted(set(by_role["Best NULL on"]) & set(by_role["NULL off control"]))
    if not seeds:
        return []
    diffs = []
    for seed in seeds:
        on, off = (by_role[role][seed] for role in ("Best NULL on", "NULL off control"))
        for scope in SCOPES:
            if sum(on["metrics"][scope][k] for k in ("true_positives", "false_negatives")) != sum(off["metrics"][scope][k] for k in ("true_positives", "false_negatives")):
                raise ValueError("NULL on/off evaluation gold support differs")
        diffs.append(on["metrics"]["combined"]["micro_f1"] - off["metrics"]["combined"]["micro_f1"])
    return ["", f"Matched NULL {split} comparison: {len(seeds)} seeds; mean paired combined-F1 difference (on - off) **{100*statistics.mean(diffs):+.2f} percentage points**.", "",
            "| Seed | Combined-F1 difference (pp) |", "|---:|---:|"] + [f"| {s} | {100*d:+.2f} |" for s, d in zip(seeds, diffs)]


def diagnostics(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["split"] == "test":
            groups[(row["study"], row["trial_id"], row["role"])].append(row)
    lines = ["", "## Error taxonomy and component evaluation", "",
             "Primary term/category/sentiment failures partition explicit gold. They are first-failed components, not sole causal explanations. Projected scores are diagnostics; exact-triplet F1 remains the benchmark.", "",
             "| Study / arm | Seeds | Term failures | Category failures | Sentiment failures | Correct term+sentiment / wrong category | Term F1 | Term+category F1 | Term+sentiment F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (study, _, role), members in groups.items():
        failures = [statistics.mean(r["flat"]["outcomes"].get(k, 0) for r in members) for k in ("term", "category", "sentiment")]
        components = [statistics.mean(r["flat"][k] for r in members) for k in ("term_f1", "term_category_f1", "term_sentiment_f1")]
        mismatch = statistics.mean(r["metrics"]["taxonomy"].get("category_mismatch_with_correct_term_sentiment", 0) for r in members)
        lines.append(f"| {study}: {role} | {len(members)} | " + " | ".join(f"{n:.1f}" for n in failures) + f" | {mismatch:.1f} | " + " | ".join(fmt(n) for n in components) + " |")
    lines += ["", "## Rare/unseen labels and hardest domain", "",
              "Explicit rarity uses each arm's training counts. Unseen and rare labels are distinct. Domain scores are in-domain results.", "",
              "| Study / arm | Bucket | Mean gold | Exact recall | Hardest explicit domain | Domain F1 |", "|---|---|---:|---:|---|---:|"]
    for (study, _, role), members in groups.items():
        domains = members[0]["metrics"]["per_domain"]
        hardest = min(domains, key=lambda d: statistics.mean(r["metrics"]["per_domain"][d]["explicit"]["micro_f1"] for r in members))
        f1 = statistics.mean(r["metrics"]["per_domain"][hardest]["explicit"]["micro_f1"] for r in members)
        for bucket in ("rare", "unseen"):
            items = [b for r in members for b in r["metrics"]["taxonomy"]["by_rarity"] if b["rarity"] == bucket]
            gold = statistics.mean(b["gold_triplets"] for b in items)
            recalls = [1 - b["error_rate"] for b in items if b["error_rate"] is not None]
            recall = statistics.mean(recalls) if recalls else None
            lines.append(f"| {study}: {role} | {bucket} | {gold:.1f} | {fmt(recall)} | {hardest} | {f1:.4f} |")
    return lines


def comparison_plot(summaries, phase1, path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    data = [s for s in summaries if s["split"] == "test"]
    if phase1:
        base = phase1["summaries"][0]
        data.insert(0, {"study": "Phase 1 rescore", "role": "standard CE", "seeds": base["seeds"],
                       "scores": {s: {"mean": base[s]["f1"], "sd": base[s]["sd"]} for s in SCOPES}})
    if not data:
        return False
    fig, ax = plt.subplots(figsize=(13, max(4.5, len(data) * .7 + 1.5)))
    for offset, scope, color in ((-.23, "explicit", "#376b86"), (0, "null", "#ce832a"), (.23, "combined", "#358782")):
        values = [r["scores"][scope]["mean"] for r in data]
        ax.barh([i + offset for i in range(len(data))], values, height=.21,
                xerr=[r["scores"][scope]["sd"] or 0 for r in data], capsize=2, color=color, label=scope.capitalize())
    ax.set_yticks(range(len(data)), [f"{r['study']}: {r['role']} (n={len(r['seeds'])})" for r in data])
    ax.invert_yaxis()
    ax.set_title("Frozen test results: descriptive across studies", loc="left")
    ax.set_xlabel("Exact micro F1; error bars show sample SD across seeds")
    ax.legend(ncol=3, frameon=False, loc="upper right", bbox_to_anchor=(1, -.14))
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def write_report(quick_study, null_study, output):
    output.mkdir(parents=True, exist_ok=True)
    quick, quick_rows = snapshot(quick_study, "Quick tuning")
    null, null_rows = snapshot(null_study, "NULL adaptation", optional=True)
    rows = quick_rows + null_rows
    quick_ablation, quick_screens = snapshot(output / "ablation_freezes/quick", "Quick screens", optional=True)
    null_ablation, null_screens = snapshot(output / "ablation_freezes/null", "NULL screens", optional=True)
    screen_rows = quick_screens + null_screens
    for r in rows:
        r["result_type"] = "finalist"
    for r in screen_rows:
        r["result_type"] = "screen"
    summaries = summarize(rows)
    ledger = parameter_ledger(quick, null_study, rows + screen_rows)
    documentation = read(quick_study / "final_documentation.json", {})
    phase1 = documentation.get("phase1")
    expected = sum(len(v["runs"]) for v in quick["variants"]) + (sum(len(v["runs"]) for v in null["variants"]) if null else 0)
    actual = sum(r["split"] == "test" for r in rows)
    finalists_complete = null is not None and actual == expected
    ablation_expected = sum(len(v["runs"]) for f in (quick_ablation, null_ablation) if f for v in f["variants"])
    ablation_actual = sum(r["split"] == "test" for r in screen_rows)
    ablations_complete = bool(quick_ablation and null_ablation) and ablation_actual == ablation_expected
    complete = finalists_complete and ablations_complete
    generated = datetime.now(timezone.utc).isoformat()
    payload = {"generated_at_utc": generated, "complete": complete, "test_runs": actual, "expected_frozen_test_runs": expected,
               "quick_selection": quick, "null_selection": null, "phase1": phase1, "summaries": summaries,
               "runs": rows, "screen_runs": screen_rows, "parameter_ledger": ledger,
               "finalists_complete": finalists_complete, "ablations_complete": ablations_complete,
               "screen_test_runs": ablation_actual, "expected_screen_test_runs": ablation_expected,
               "verification_scope": "Saved frozen metadata and metrics checked; model hash checks are performed by the evaluation runners."}
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    parameters = []
    for item in ledger:
        parameters.append({"study": item["study"], "trial_id": item["trial_id"], "seed": item["seed"],
            "stages": "; ".join(item["stages"]), "status": item["status"], **{k: item["config"].get(k) for k in PARAMETERS},
            **{s + "_dev_f1": item["dev"][s] if item["dev"] else None for s in SCOPES},
            "threshold": item["threshold"], "parameters_json": json.dumps(item["config"], sort_keys=True)})
    save_csv(output / "parameters.csv", parameters, ["study", "trial_id", "seed", "stages", "status", *PARAMETERS,
                                                   *[s + "_dev_f1" for s in SCOPES], "threshold", "parameters_json"])
    exports = [{"study": r["study"], "result_type": r["result_type"], "arm": r["role"], "trial_id": r["trial_id"], "seed": r["seed"], "split": r["split"],
                "best_epoch": r["best_epoch"], "threshold": r["threshold"],
                **{f"{s}_{metric}": r["metrics"][s]["micro_" + metric] for s in SCOPES for metric in ("precision", "recall", "f1")}}
               for r in rows + screen_rows]
    save_csv(output / "seed_results.csv", exports, ["study", "result_type", "arm", "trial_id", "seed", "split", "best_epoch", "threshold",
                                                  *[f"{s}_{m}" for s in SCOPES for m in ("precision", "recall", "f1")]])
    lines = ["# Final presentation: fine-tuning and NULL adaptation", "", f"Generated {generated}. Status: **{'complete finalists and screen ablations' if complete else 'in progress: finalist or screen test evaluation pending'}**.", "",
             "The completed quick-tuning study supplies the established fine-tuning result. The NULL follow-up retains its selected explicit CE mixture/LR and studies implicit extraction. This report combines evidence; it does not merge model weights or pool results across protocols.", "",
             f"Saved finalist test evaluations: {actual}/{expected} currently frozen runs. Saved screen test evaluations: {ablation_actual}/{ablation_expected} currently frozen candidates. If a freeze is absent, its denominator is not yet known. Every parameter tried in the two study searches appears in parameters.csv.", "",
             "## Frozen results", "", "F1 is calculated per seed, then averaged. SD is sample SD, not a confidence interval. Partial cohorts are labeled by observed/requested seeds.", "",
             "| Study | Arm | Split | Seeds | Explicit F1 +/- SD | NULL F1 +/- SD | Combined F1 +/- SD | NULL P | NULL R |", "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for summary in summaries:
        scores = summary["scores"]
        lines.append(f"| {summary['study']} | {summary['role']} | {summary['split']} | {len(summary['seeds'])}/{len(summary['expected_seeds'])} | " +
                     " | ".join(mean_sd(scores[s]) for s in SCOPES) + f" | {scores['null']['precision']:.4f} | {scores['null']['recall']:.4f} |")
    lines += matched_null(rows, "dev") + matched_null(rows, "test")
    if comparison_plot(summaries, phase1, output / "test_comparison.png"):
        lines += ["", "![Frozen test comparison](test_comparison.png)", ""]
    lines += ["## Exploratory screen test ablations", "",
              "All registered quick/NULL screen candidates are frozen before new test inference. Each uses its saved development-selected checkpoint and threshold. These mostly one-seed results keep epoch budgets visible; they are not pooled with the finalist cohorts and do not replace development selection.", "",
              "| Study | Recipe | Seed | Epoch budget | LR | Explicit test F1 | NULL P | NULL R | NULL test F1 | Combined test F1 | Dev threshold |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in screen_rows:
        if r["split"] != "test":
            continue
        m, c = r["metrics"], r["config"]
        lines.append(f"| {r['study']} | {c['name']} | {r['seed']} | {c['epochs']} | {c['lr']:g} | {m['explicit']['micro_f1']:.4f} | {m['null']['micro_precision']:.4f} | {m['null']['micro_recall']:.4f} | {m['null']['micro_f1']:.4f} | {m['combined']['micro_f1']:.4f} | {r['threshold']:g} |")
    if comparison_plot(summarize(screen_rows), None, output / "screen_test_comparison.png"):
        lines += ["", "![Screen test ablations](screen_test_comparison.png)", ""]
    if phase1:
        baseline = phase1["summaries"][0]
        primary, = [s for s in summaries if s["study"] == "Quick tuning" and s["role"] == "Best NULL off" and s["split"] == "test"]
        gain = 100 * (primary["scores"]["explicit"]["mean"] - baseline["explicit"]["f1"])
        lines += ["## Established Phase 1 comparison", "",
                  f"On matched seeds and the same corrected eligible gold, Phase 1 explicit F1 is {baseline['explicit']['f1']:.4f}; the development-selected quick-tuning model scores {primary['scores']['explicit']['mean']:.4f}: **{gain:+.2f} percentage points**. This is a pipeline comparison, not an isolated loss effect.", ""]
    lines += ["## Parameters and development search", "", "The full values, including unused flags, are saved in parameters.csv and results.json. The table below shows the selected settings; inactive loss flags do not describe an additional training objective.", "",
              "| Study / arm | Explicit loss | LR | CE / weighted CE | Epoch budget | Batch / length | NULL loss / representation / negatives | NULL outer weight / cap | Checkpoint objective | Thresholds by seed |", "|---|---|---:|---|---:|---|---|---|---|---|"]
    for study, frozen in (("Quick tuning", quick), ("NULL adaptation", null)):
        for v in frozen["variants"] if frozen else []:
            cfg = v["config"]
            role = v.get("role") or "; ".join(v["roles"])
            null_settings = f"{cfg.get('null_loss', 'BCE')}/{cfg.get('representation', 'CLS')}/{cfg.get('negative_sampling', 'all')}" if cfg["null_head"] else "head off"
            thresholds = ", ".join(f"{r['seed']}:{r['null_threshold']:g}" for r in v["runs"]) if cfg["null_head"] else "abstain"
            lines.append(f"| {study}: {role} | {cfg['loss_type']} | {cfg['lr']:g} | {cfg['ce_weight']:g}/{cfg['weighted_ce_weight']:g} | {cfg['epochs']} | {cfg['batch_size']}/{cfg['max_len']} | {null_settings} | {cfg['null_loss_weight']:g}/{cfg['null_pos_weight_cap']:g} | {cfg['selection_metric']} | {thresholds} |")
    lines += ["", "| Study | Stage | Trial | Seed | Status | Explicit dev F1 | NULL dev F1 | Combined dev F1 |", "|---|---|---|---:|---|---:|---:|---:|"]
    for item in ledger:
        values = [fmt(item["dev"][s]) if item["dev"] else "pending" for s in SCOPES]
        lines.append(f"| {item['study']} | {'; '.join(item['stages'])} | {item['trial_id']} | {item['seed']} | {item['status']} | " + " | ".join(values) + " |")
    lines += diagnostics(rows)
    lines += ["", "## Reporting boundaries", "",
              "- All four sentiments, including Conflict, are retained. Exact F1 uses eligible explicit annotations plus deduplicated NULL gold; excluded ambiguous/conflicting/overlapping explicit annotations remain outside this benchmark.",
              "- Standard CE, inverse-frequency CE, mixtures and explicit focal were covered by quick tuning. NULL focal and ASL are separate head-specific experiments. ASL does not stack the BCE positive cap.",
              "- The quick study uses historical input/FP32 and explicit-epoch/NULL-threshold objectives. The new study reserves one extra token slot and defaults to AMP and combined dev selection. Cross-study comparisons are descriptive; the new matched NULL-on/off comparison is the controlled joint-score comparison.",
              "- Three-epoch NULL screens, five-epoch quick screens and finalist cohorts stay separate. Screen test ablations are generally seed 13 only; they are not each a five-seed test experiment.",
              "- Test scores are not used to replace development selection or tune thresholds. A threshold of 1 means NULL abstention. A nonzero NULL F1 alone does not establish a combined-score gain.",
              "- Historical test inspection makes both studies exploratory. In-domain domain rankings do not establish new LODO improvement.",
              "- Reporting reads saved metrics and validates frozen metadata; checkpoint hashes are checked by the test runners. No raw test data or model is loaded by this report builder.",
              "- These report exports are refreshable snapshots. Original development/epoch/test logs remain in their study folders and retain append history.", "",
              "Files: [all parameters](parameters.csv), [individual dev/test scores](seed_results.csv), [complete saved evidence](results.json). Model weights are excluded.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    for name, freeze in (("quick_selection.json", quick), ("null_selection.json", null)):
        if freeze:
            (output / name).write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    for name in ("phase1_comparison", "taxonomy_comparison", "domain_comparison"):
        source = quick_study / f"{name}.png"
        if source.exists():
            shutil.copy2(source, output / f"historical_{name}.png")
    (output / "README.md").write_text("# Final presentation results\n\nRead [the combined report](report.md). All parameters, individual scores and saved evidence are included; models are excluded.\n\nRegenerate without training:\n\n```powershell\n.venv/Scripts/python.exe finalize_finetuning_report.py\n```\n", encoding="utf-8")
    return output / "report.md"
