"""Rebuild the team's Phase 2 comparisons from bundled results, without training."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import zipfile

try:
    from .analysis import compare_lodo, mean_sd
except ImportError:
    from analysis import compare_lodo, mean_sd

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(root):
    manifest = read(root / "bundle_manifest.json")
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or unsafe bundle input: {entry['path']}")
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Bundle input changed: {entry['path']}; rebuild/export a new snapshot")
    return len(manifest["files"])


def save_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def in_domain_comparison(data):
    groups = defaultdict(list)
    for row in data["runs"]:
        if row["split"] == "test":
            groups[row["study"], row["trial_id"], row["role"]].append(row)
    result = []
    baseline = next(r for r in data["phase1"]["summaries"] if r["model"] == "phase1")
    result.append({"study": "Phase 1", "role": "Standard CE, same-gold rescore", "trial_id": "phase1_standard",
                   "n": len(baseline["seeds"]), "seeds": " ".join(map(str, baseline["seeds"])),
                   **{scope + "_f1": baseline[scope]["f1"] for scope in ("explicit", "null", "combined")},
                   **{scope + "_sd": baseline[scope]["sd"] for scope in ("explicit", "null", "combined")},
                   "explicit_gain_vs_phase1_pp": 0., "scope": "Corrected eligible explicit + deduplicated NULL gold"})
    for (study, trial, role), cohort in groups.items():
        seeds = sorted(row["seed"] for row in cohort)
        expected = sorted(cohort[0]["expected_seeds"])
        if seeds != expected or len(set(seeds)) != len(seeds):
            raise ValueError(f"Incomplete or duplicated in-domain cohort: {study}/{role}")
        cfg = cohort[0]["config"]
        row = {"study": study, "role": role, "trial_id": trial, "n": len(cohort),
               "seeds": " ".join(map(str, seeds))}
        for scope in ("explicit", "null", "combined"):
            summary = mean_sd([r["metrics"][scope]["micro_f1"] for r in cohort])
            row[scope + "_f1"], row[scope + "_sd"] = summary["mean"], summary["sd"]
        row.update(explicit_gain_vs_phase1_pp=100*(row["explicit_f1"]-baseline["explicit"]["f1"]),
                   lr=cfg["lr"], epochs=cfg["epochs"], batch_size=cfg["batch_size"], max_len=cfg["max_len"],
                   ce_weight=cfg["ce_weight"], weighted_ce_weight=cfg["weighted_ce_weight"],
                   null_head=cfg["null_head"], vocabulary_scope=cfg["vocabulary_scope"],
                   selection_metric=cfg["selection_metric"],
                   thresholds=" ".join(f"{r['seed']}:{r['threshold']}" for r in sorted(cohort,key=lambda r:r["seed"])))
        result.append(row)
    return result


def screens(data, split="test"):
    result = []
    for row in data["screen_runs"]:
        if row["split"] != split:
            continue
        cfg = row["config"]
        result.append({"study": row["study"], "role": row["role"], "recipe": cfg["name"], "trial_id": row["trial_id"],
                       "split": split, "selection_metric": cfg["selection_metric"],
                       "seed": row["seed"], "epochs": cfg["epochs"], "lr": cfg["lr"],
                       "best_epoch": row["best_epoch"], "loss_type": cfg["loss_type"],
                       "ce_weight": cfg["ce_weight"], "weighted_ce_weight": cfg["weighted_ce_weight"],
                       "focal_gamma": cfg["focal_gamma"], "vocabulary_scope": cfg["vocabulary_scope"],
                       "null_loss": cfg.get("null_loss", "bce" if cfg["null_head"] else "inactive"),
                       "representation": cfg.get("representation", "cls" if cfg["null_head"] else "inactive"),
                       "null_loss_weight": cfg["null_loss_weight"],
                       "null_focal_gamma": cfg.get("null_focal_gamma"),
                       "negative_sampling": cfg.get("negative_sampling", "all"),
                       "null_head": cfg["null_head"], "threshold": row["threshold"],
                       "null_precision": row["metrics"]["null"]["micro_precision"],
                       "null_recall": row["metrics"]["null"]["micro_recall"],
                       **{scope + "_f1": row["metrics"][scope]["micro_f1"] for scope in ("explicit", "null", "combined")}})
    if len(result) != data["expected_screen_test_runs"]:
        raise ValueError("Screen ablation matrix is incomplete")
    return result


def lodo_tables(comparison):
    summaries = []
    for row in comparison["summaries"]:
        entry = {"held_out_domain": row["domain"], "model": row["model"], "n": len(row["seeds"]),
                 "seeds": " ".join(map(str,row["seeds"])), "eligible_gold": row["gold"],
                 "source_category_coverage": row["coverage"]}
        for key, score in row["scores"].items():
            entry[key + "_f1"], entry[key + "_sd"] = score["mean"], score["sd"]
        entry.update({"primary_" + key: value for key,value in row["taxonomy"].items()})
        summaries.append(entry)
    deltas = []
    for domain in comparison["domains"]:
        for baseline in ("phase1_standard", "phase1_weighted"):
            cohort = [r for r in comparison["paired"] if r["domain"] == domain and r["baseline"] == baseline]
            row = {"held_out_domain": domain, "baseline": baseline, "n": len(cohort)}
            for key in ("explicit_delta_pp", "term_delta_pp", "term_sentiment_delta_pp"):
                summary = mean_sd([r[key] for r in cohort])
                row[key], row[key + "_sd"] = summary["mean"], summary["sd"]
            deltas.append(row)
    return summaries, deltas


def diagnostic_tables(data, lodo):
    """Flatten saved test diagnostics; counts describe one seed, not new data."""
    taxonomy_rows, domain_rows, rarity_rows, sentiment_rows = [], [], [], []
    records = [r for r in data["runs"] if r["split"] == "test"]
    records += [{**r, "study": "LODO", "role": r["domain"],
                 "trial_id": lodo["freeze"]["trial_id"]} for r in lodo["runs"]]
    for record in records:
        identity = {k: record[k] for k in ("study", "role", "trial_id", "seed")}
        metrics, tax = record["metrics"], record["metrics"]["taxonomy"]
        outcomes, gold = tax["primary_outcome_counts"], tax["gold_triplets"]
        if sum(outcomes.values()) != gold:
            raise ValueError("Saved taxonomy does not partition eligible gold")
        missed = gold - outcomes["correct"]
        row = {**identity, "eligible_gold": gold, "missed_gold": missed, **outcomes}
        row.update({name + "_share_of_missed": outcomes[name]/missed if missed else None
                    for name in ("term", "category", "sentiment")})
        row["category_mismatch_with_correct_term_sentiment"] = tax["category_mismatch_with_correct_term_sentiment"]
        taxonomy_rows.append(row)
        for domain, result in metrics["per_domain"].items():
            domain_rows.append({**identity, "domain": domain,
                                "explicit_f1": result["explicit"]["micro_f1"],
                                "null_f1": result["null"]["micro_f1"],
                                "term_f1": result["components"]["aspect_span"]["f1"],
                                "term_sentiment_f1": result["components"]["aspect_plus_sentiment"]["f1"],
                                "category_coverage": result["components"]["gold_category_coverage"],
                                "eligible_gold": result["components"]["gold_triplets"],
                                **result["primary_outcome_counts"]})
        for bucket in tax["by_rarity"]:
            rarity_rows.append({**identity, "rarity": bucket["rarity"],
                                "eligible_gold": bucket["gold_triplets"],
                                "error_rate": bucket["error_rate"], **bucket["primary_outcome_counts"]})
        for sentiment, result in metrics["sentiment_by_label"].items():
            sentiment_rows.append({**identity, "sentiment": sentiment, **result})
    return {"taxonomy_by_seed": taxonomy_rows, "domain_by_seed": domain_rows,
            "rarity_by_seed": rarity_rows, "sentiment_by_seed": sentiment_rows}


def figures(indomain, lodo, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12,5), constrained_layout=True)
    labels = [r["study"] + "\n" + r["role"] for r in indomain]
    x = list(range(len(indomain)))
    for offset, scope, color in [(-.25,"explicit","#2563eb"),(0,"null","#14b8a6"),(.25,"combined","#eab308")]:
        ax.bar([i+offset for i in x], [100*r[scope+"_f1"] for r in indomain], .25,
               yerr=[100*(r[scope+"_sd"] or 0) for r in indomain], color=color, label=scope, capsize=3)
    ax.set_xticks(x, labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Test F1 (%)")
    ax.set_title("In-domain finalists: five-seed means and sample SD")
    ax.legend(); ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    fig.savefig(output/"in_domain_comparison.png",dpi=180); plt.close(fig)
    domains = list(dict.fromkeys(r["held_out_domain"] for r in lodo))
    fig, axes = plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    x = list(range(len(domains)))
    for offset, model, label, color in [(-.25,"phase1_standard","Phase 1 standard","#94a3b8"),
                                      (0,"phase1_weighted","Phase 1 weighted","#14b8a6"),
                                      (.25,"phase2_mixed_025","Phase 2 mixed","#2563eb")]:
        cohort = [next(r for r in lodo if r["held_out_domain"]==d and r["model"]==model) for d in domains]
        for ax,key in zip(axes,("explicit","term_sentiment")):
            ax.bar([i+offset for i in x],[100*r[key+"_f1"] for r in cohort],.25,
                   yerr=[100*(r[key+"_sd"] or 0) for r in cohort],color=color,label=label,capsize=2)
    for ax,title in zip(axes,("Exact-triplet F1 on the same corrected gold","Term + sentiment transfer on the same gold")):
        ax.set_xticks(x,domains,rotation=35,ha="right"); ax.set_ylabel("Test F1 (%)")
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(axis="y",alpha=.15); ax.set_axisbelow(True)
    fig.savefig(output/"lodo_same_gold_comparison.png",dpi=180); plt.close(fig)


def extract_evidence(root):
    destination = (root / "evidence/extracted").resolve()
    for archive_path in sorted((root/"evidence").glob("*_outputs.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            for item in archive.infolist():
                target = (destination/item.filename).resolve()
                if not target.is_relative_to(destination) or target.suffix.lower() in {".pt",".pth",".bin",".safetensors",".ckpt"}:
                    raise ValueError(f"Unsafe or model-weight archive entry: {item.filename}")
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(item) as source, target.open("wb") as handle:
                    import shutil
                    shutil.copyfileobj(source,handle)
    print(f"Optional raw evidence extracted to {destination}; it is excluded from Git.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plots",action="store_true",help="Use only Python's standard library; retain packaged charts")
    parser.add_argument("--extract-evidence",action="store_true",help="Also unpack compressed metrics/predictions for inspection")
    args = parser.parse_args(argv)
    checked = verify_bundle(ROOT)
    data = read(ROOT/"results/in_domain/results.json")
    saved_lodo = read(ROOT/"results/lodo/results.json")
    if not data["complete"] or not saved_lodo["complete"]:
        raise ValueError("Only completed studies can be reported")
    output = ROOT/"comparison"; output.mkdir(exist_ok=True)
    indomain, ablations = in_domain_comparison(data), screens(data)
    with zipfile.ZipFile(ROOT/"evidence/phase2_outputs.zip") as current, zipfile.ZipFile(ROOT/"evidence/phase1_outputs.zip") as legacy:
        matched = compare_lodo(saved_lodo,current,legacy)
    lodo, deltas = lodo_tables(matched)
    save_csv(output/"in_domain_comparison.csv",indomain)
    save_csv(output/"development_screens.csv",screens(data, "dev"))
    save_csv(output/"screen_ablations.csv",ablations)
    save_csv(output/"lodo_same_gold.csv",lodo)
    save_csv(output/"lodo_paired_seed_deltas.csv",matched["paired"])
    save_csv(output/"lodo_mean_deltas.csv",deltas)
    for name, rows in diagnostic_tables(data, saved_lodo).items():
        save_csv(output/(name + ".csv"), rows)
    (output/"lodo_rescoring_evidence.json").write_text(json.dumps(matched,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if not args.no_plots:
        try:
            figures(indomain,lodo,output)
        except ImportError:
            print("Tables generated. Install requirements-analysis.txt to regenerate charts.")
    pct = lambda value: f"{100*value:.2f}"
    lines = ["# Phase 2 team comparison", "",
             f"Verified {checked} bundled input files. This analysis reads saved metrics and predictions; it does not train models or select parameters from test scores.", "",
             "## In-domain: completed five-seed finalists", "",
             "F1 values below are percentages. CSV scores use 0–1; delta_pp means percentage points. Sample SD measures seed variation, not a confidence interval.", "",
             "| Study | Arm | Seeds | Explicit F1 | NULL F1 | Combined F1 |",
             "|---|---|---:|---:|---:|---:|"]
    for row in indomain:
        lines.append(f"| {row['study']} | {row['role']} | {row['n']} | {pct(row['explicit_f1'])} | {pct(row['null_f1'])} | {pct(row['combined_f1'])} |")
    selected = next(r for r in indomain if r["study"]=="Quick tuning" and r["role"]=="Best NULL off")
    lines += ["", f"Primary in-domain result: **{pct(selected['explicit_f1'])}% explicit F1, +{selected['explicit_gain_vs_phase1_pp']:.2f} percentage points versus re-scored Phase 1 standard**. Keep the pre-test development selection.", "",
              "![In-domain comparison](in_domain_comparison.png)", "",
              "## LODO: Phase 1 predictions re-scored on Phase 2 gold", "",
              "Every historical prediction is retained, including predictions on annotations excluded by the corrected gold. Records are aligned by domain and example ID, with exact sentence text checks. Each baseline and Phase 2 use the same gold within each domain/seed. Category coverage uses each model's own original source vocabulary.", "",
              "| Held-out domain | Phase 1 standard F1 | Phase 1 weighted F1 | Phase 2 mixed F1 | Mixed − standard (pp) | Mixed − weighted (pp) |",
              "|---|---:|---:|---:|---:|---:|"]
    for domain in matched["domains"]:
        rows = {r["model"]:r for r in lodo if r["held_out_domain"]==domain}
        diff = {r["baseline"]:r for r in deltas if r["held_out_domain"]==domain}
        lines.append(f"| {domain} | {pct(rows['phase1_standard']['explicit_f1'])} | {pct(rows['phase1_weighted']['explicit_f1'])} | {pct(rows['phase2_mixed_025']['explicit_f1'])} | {diff['phase1_standard']['explicit_delta_pp']:+.2f} | {diff['phase1_weighted']['explicit_delta_pp']:+.2f} |")
    lines += ["", "![LODO comparison on corrected gold](lodo_same_gold_comparison.png)", "",
              "## Interpretation for the team", "",
              "- The in-domain gain and the LODO changes are pipeline comparisons: eligibility, optimizer settings, LR and loss differ from Phase 1. They do not isolate a loss effect.",
              "- Zero source-category coverage prevents this closed-set classifier from recovering exact triples. Compare term and term+sentiment F1 to study transfer; keep exact-triplet F1 as the benchmark.",
              "- New LODO parameters came from all-domain in-domain development selection. Label the LODO study exploratory transfer, not a strict source-only hyperparameter search.",
              "- Historical NULL-on recovery has low precision and a combined-score cost. The newer attention finalist selected abstention on development and ties its matching NULL-off control. Higher combined F1 does not demonstrate better NULL extraction.",
              "- Screening ablations use one seed and different epoch budgets; they stay separate from five-seed finalist means. Standard/weighted/mixed/focal loss screening was done at LR 2e-5; only its selected mixture received the LR sweep.",
              "- Report rare (source count 1–5) and unseen (count 0) separately. First-failed term/category/sentiment counts are ordered diagnostics, not proof of a single cause.", "",
              "## Files for report writing", "",
              "- [In-domain means and settings](in_domain_comparison.csv), [development screening](development_screens.csv), [separate test screen ablations](screen_ablations.csv).",
              "- [LODO same-gold means, components, coverage and taxonomy](lodo_same_gold.csv).",
              "- [Individual paired seed differences](lodo_paired_seed_deltas.csv), [mean differences and sample SD](lodo_mean_deltas.csv), [complete rescoring evidence](lodo_rescoring_evidence.json).",
              "- [Taxonomy counts](taxonomy_by_seed.csv), [domain metrics](domain_by_seed.csv), [source-category rarity](rarity_by_seed.csv), [triplet scores by sentiment](sentiment_by_seed.csv); these are per-seed rows, not independent test datasets.",
              "- [Original in-domain/NULL snapshot](../results/in_domain/report.md), [original LODO snapshot](../results/lodo/report.md), [NULL findings](../results/in_domain/null_findings.md).",
              "- [Report task guide](../REPORT_TASKS.md), [parameters and methodology](../PARAMETERS.md)."]
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if args.extract_evidence:
        extract_evidence(ROOT)
    print(f"Ready: {output/'report.md'}")
    print(f"In-domain gain: +{selected['explicit_gain_vs_phase1_pp']:.2f} pp; LODO same-gold comparison: 7 domains × 5 seeds × 3 arms.")


if __name__ == "__main__":
    main()
