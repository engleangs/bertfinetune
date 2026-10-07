"""Copy historical results into a portable, model-free Phase 1 folder.

Original artifacts and current training runs are never changed. Rebuild with:
    .venv/Scripts/python.exe consolidate_phase1.py --validate
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import config


ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "phase1_analysis"
RUN_FILES = (
    "manifest.json", "metrics.json", "history.json",
    "dev_predictions.jsonl", "test_predictions.jsonl",
)
DOCS = (
    "protocol.md", "protocol-amendment-2026-10-04.md",
    "finetuning-readiness-2026-10-04.md", "experiment-guide-2026-10-05.md",
    "tasks.md",
)


def copy_file(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def write_text(destination: Path, text: str):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8", newline="\n")


def replace_required(text: str, before: str, after: str):
    """Fail if the source changes rather than silently exporting broken code."""
    if before not in text:
        raise ValueError(f"Snapshot adapter needs review: {before!r}")
    return text.replace(before, after)


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def collect_runs():
    """Include Restaurant once, although its source index is results.csv."""
    runs = []
    for filename in ("results.csv", "results_lodo.csv"):
        for row in read_rows(ROOT / filename):
            if row["status"] != "complete":
                continue
            source = Path(row["artifact_dir"])
            if not source.is_dir():
                source = ROOT / "artifacts" / (
                    Path("lodo_runs") / row["held_out_domain"]
                    if filename == "results_lodo.csv"
                    else Path("runs") / row["mode"]
                ) / row["config"] / f"seed_{int(row['seed'])}"
            manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
            if (manifest["status"], manifest["seed"], manifest["config_name"], manifest["mode"]) != (
                "complete", int(row["seed"]), row["config"], row["mode"]
            ):
                raise ValueError(f"Index/manifest mismatch: {source}")
            if row["mode"] == "indomain":
                held_out = "all"
                relative = Path("indomain_runs")
            elif row["mode"] == "crossdomain":
                held_out = manifest.get("held_out_domain") or row.get("held_out_domain") or config.HOLD_OUT_DOMAIN
                if row.get("held_out_domain") and row["held_out_domain"] != held_out:
                    raise ValueError(f"Held-out domain mismatch: {source}")
                relative = Path("lodo_runs") / held_out
            else:
                raise ValueError(f"Unexpected historical mode: {row['mode']}")
            relative /= Path(row["config"]) / f"seed_{int(row['seed'])}"
            copied = {**row, "held_out_domain": held_out, "artifact_dir": relative.as_posix()}
            runs.append((source, copied))

    expected = {
        (mode, domain, loss, seed)
        for mode, domains in [("indomain", ["all"]), ("crossdomain", config.DOMAINS)]
        for domain in domains for loss in ("standard", "weighted") for seed in config.SEEDS
    }
    keys = [(r["mode"], r["held_out_domain"], r["config"], int(r["seed"])) for _, r in runs]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Historical run matrix is incomplete, unexpected, or duplicated")
    for source, _ in runs:
        for filename in RUN_FILES:
            if not (source / filename).is_file():
                raise FileNotFoundError(source / filename)
    return sorted(runs, key=lambda item: (
        item[1]["mode"], item[1]["held_out_domain"], item[1]["config"], int(item[1]["seed"])
    ))


def portable_report(text: str):
    """Update only locations and packaging instructions in copied reports."""
    text = text.replace(
        "Restaurant runs come from the original cross-domain rows in `results.csv`;\n"
        "the other six folds come from `results_lodo.csv`. They are included once.",
        "All seven LODO folds, including Restaurant, are in `results_lodo.csv`.\n"
        "The 10 in-domain controls are in `results.csv`. No run is duplicated.",
    )
    text = text.replace(
        "`artifacts/team_analysis/original_notebook.ipynb`",
        "`result_analysis/archive/team_review/original_notebook.ipynb`",
    )
    text = text.replace("artifacts/team_analysis/", "result_analysis/current/")
    text = text.replace("Initial_Evaluation_and_Error_Analysis_CS_760.ipynb", "phase1_analysis.ipynb")
    text = text.replace(".venv/Scripts/python.exe analyze_team_notebook.py", "python analyze_phase1.py")
    text = text.replace("analyze_team_notebook.py", "analyze_phase1.py")
    text = text.replace("src/team_diagnostics.py", "analysis_helpers/team_diagnostics.py")
    text = text.replace("src/team_report.py", "analysis_helpers/team_report.py")
    text = text.replace(
        "It requires this repository and its local artifacts/data. It is not a\nstandalone Colab archive notebook.",
        "The Phase 1 folder includes its corpus, indexes, predictions and analysis code.\n"
        "It works after cloning without the original artifacts directory.",
    )
    text = text.replace("Run from the repository with its virtual environment:",
                        "Run from the Phase 1 folder with the analysis environment:")
    text = text.replace("Run from the repository: python analyze_phase1.py",
                        "Run from the Phase 1 folder: python analyze_phase1.py")
    text = text.replace(
        "- `gold_error_records.jsonl`: every retained unique gold triplet with primary",
        "- `gold_error_records.jsonl` (optional `--export-errors`): every retained unique gold triplet with primary",
    )
    return text


def copy_analysis_code():
    """Snapshot the reviewed analysis, removing its training-only dependencies."""
    settings = (ROOT / "config.py").read_text(encoding="utf-8")
    settings = replace_required(settings, 'Path(__file__).resolve().parent / "data" / "m-absa"',
                                'Path(__file__).resolve().parent / "corpus"')
    settings = settings.replace("# Local destination populated by download_data.py.",
                                "# Small corpus included in this analysis snapshot.")
    write_text(BUNDLE / "phase1_config.py", settings)
    write_text(BUNDLE / "analysis_helpers" / "__init__.py", '"""Frozen Phase 1 analysis helpers; no training dependencies."""\n')
    for name in ("result_analysis", "team_diagnostics", "team_taxonomy", "team_report"):
        text = (ROOT / "src" / f"{name}.py").read_text(encoding="utf-8")
        text = text.replace("from src.", "from analysis_helpers.")
        text = text.replace("from config import", "from phase1_config import")
        if name == "team_report":
            text = portable_report(text)
        write_text(BUNDLE / "analysis_helpers" / f"{name}.py", text)

    # Copy function bodies verbatim so eligibility stays identical to version 1.
    source = (ROOT / "src" / "data.py").read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    selected = {
        "Example", "SENTIMENT_ALIASES", "normalize_sentiment", "parse_line",
        "load_domain_file", "label_frequencies", "find_span",
        "_select_baseline_triplets", "aligned_explicit_triplets",
    }
    pieces, found = [], set()
    for node in ast.parse(source).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in selected:
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            pieces.append("".join(lines[start - 1:node.end_lineno]).rstrip())
            found.add(name)
    if found != selected:
        raise ValueError(f"Analysis-only parser snapshot is missing: {selected - found}")
    header = (
        '"""Version-1 parsing and retained-gold rules, copied without training classes."""\n\n'
        "import ast\nfrom dataclasses import dataclass\nfrom typing import List, Tuple\n\n"
    )
    write_text(BUNDLE / "analysis_helpers" / "data.py", header + "\n\n\n".join(pieces) + "\n")

    text = (ROOT / "analyze_team_notebook.py").read_text(encoding="utf-8")
    text = text.replace("import config\n", "import phase1_config as config\n")
    text = text.replace("from src.", "from analysis_helpers.")
    text = text.replace('root / "data" / "m-absa"', 'root / "corpus"')
    start = text.index('            run_dir = Path(row["artifact_dir"])')
    end = text.index('            manifest = json.loads', start)
    text = text[:start] + (
        '            relative = Path(row["artifact_dir"])\n'
        '            if relative.is_absolute() or ".." in relative.parts:\n'
        '                raise ValueError("Phase 1 artifact paths must be relative to the bundle")\n'
        '            run_dir = root / relative\n'
    ) + text[end:]
    text = replace_required(text, 'str(path.relative_to(root))', 'path.relative_to(root).as_posix()')
    text = replace_required(text, '"artifact_dir": str(run_dir)', '"artifact_dir": run_dir.relative_to(root).as_posix()')
    text = replace_required(text, 'ROOT / "artifacts" / "team_analysis"', 'ROOT / "result_analysis" / "current"')
    text = replace_required(text, 'def write_outputs(analysis, output: Path):',
                            'def write_outputs(analysis, output: Path, *, export_errors: bool = False):')
    text = replace_required(text, '            if name == "errors":\n',
                            '            if name == "errors":\n                if not export_errors:\n                    continue\n')
    text = replace_required(text, '    args = parser.parse_args()\n',
                            '    parser.add_argument("--export-errors", action="store_true",\n'
                            '                        help="Write the large, locally regenerated per-gold JSONL file")\n'
                            '    args = parser.parse_args()\n')
    text = replace_required(text, 'write_outputs(result, args.output)',
                            'write_outputs(result, args.output, export_errors=args.export_errors)')
    text = text.replace('"Writing analysis tables, diagnostic records, and the report..."',
                        '"Writing analysis tables and the report..."')
    write_text(BUNDLE / "analyze_phase1.py", portable_report(text))


def copy_notebook():
    notebook = json.loads((ROOT / "Initial_Evaluation_and_Error_Analysis_CS_760.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        text = portable_report("".join(cell["source"]))
        if cell["cell_type"] == "code":
            text = text.replace("from analyze_team_notebook import", "from analyze_phase1 import")
            text = text.replace(
                'ROOT = Path.cwd().resolve()  # Change to the repository folder if needed.',
                'ROOT = Path.cwd().resolve()\n'
                'if (ROOT / "phase1_analysis" / "analyze_phase1.py").is_file():\n'
                '    ROOT = ROOT / "phase1_analysis"',
            )
            text = text.replace('ROOT / "artifacts" / "team_analysis"', 'ROOT / "result_analysis" / "current"')
            text = text.replace("bertfinetune repository folder", "phase1_analysis folder")
            cell["outputs"] = []
            cell["execution_count"] = None
        else:
            text = text.replace(
                "Run the cells in order from this repository using its virtual environment. It reads local `data/`, `results.csv`, `results_lodo.csv`, and `artifacts/`; no training or downloads are performed.",
                "Run cells in order with the Phase 1 analysis environment. This folder includes `corpus/`, both indexes and predictions; no training or downloads are performed.",
            )
            text = text.replace(
                "Restaurant is included from the original cross-domain index once; other folds come from the LODO index.",
                "All seven folds, including Restaurant, are consolidated in the LODO index exactly once.",
            )
            text = text.replace(
                "The trainer does not yet implement these new losses.",
                "The separate version-2 runner now implements these losses; see experiment-guide-2026-10-05.md in the main repository. This folder contains historical Phase 1 evidence.",
            )
            text = text.replace("- Start with seed 13, then compare the selected designs with all five matched seeds.",
                                "- The current continuation uses seeds 13 and 42 for the development pilot; promote selected designs to five matched seeds.")
            text = text.replace("## 9. Professor's loss experiments - proposed next stage", "## 9. Professor's loss experiments - separate continuation")
        cell["source"] = text.splitlines(keepends=True)
    write_text(BUNDLE / "phase1_analysis.ipynb", json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


def copy_reports(runs):
    path_map = {str(source).replace("\\", "/").casefold(): row["artifact_dir"] for source, row in runs}
    current = ROOT / "artifacts" / "team_analysis"
    for source in sorted(current.rglob("*")):
        if not source.is_file():
            continue
        if source.name in {"gold_error_records.jsonl", "atomic_probe.json"}:
            continue
        if source.name in {"original_notebook.ipynb", "notebook_validation.json", "test_validation.json"}:
            destination = BUNDLE / "result_analysis" / "archive" / "team_review" / source.name
        else:
            destination = BUNDLE / "result_analysis" / "current" / source.relative_to(current)
        if source.suffix == ".csv":
            rows = read_rows(source)
            for row in rows:
                for key, value in row.items():
                    if key.startswith("artifact_dir"):
                        row[key] = path_map[value.replace("\\", "/").casefold()]
            write_rows(destination, rows)
        elif source.name == "metadata.json":
            metadata = json.loads(source.read_text(encoding="utf-8"))
            metadata["data_sha256"] = {
                key.replace("\\", "/").replace("data/m-absa/", "corpus/"): value
                for key, value in metadata["data_sha256"].items()
            }
            write_text(destination, json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        else:
            copy_file(source, destination)

    for directory, label in (("result_analysis", "initial_20_runs"), ("lodo_analysis", "interim_48_runs")):
        source_dir = ROOT / "artifacts" / directory
        for source in sorted(source_dir.rglob("*")):
            if source.is_file() and source.suffix in {".csv", ".json", ".md", ".png"}:
                copy_file(source, BUNDLE / "result_analysis" / "archive" / label / source.relative_to(source_dir))
    for name in DOCS:
        copy_file(ROOT / name, BUNDLE / name)
    write_text(BUNDLE / "detailed-analysis.md", portable_report((ROOT / "detailed-analysis.md").read_text(encoding="utf-8")))


def write_manifest(runs):
    files = []
    for path in sorted(BUNDLE.rglob("*")):
        if not path.is_file() or path.name in {"bundle_manifest.json", "gold_error_records.jsonl"}:
            continue
        relative = path.relative_to(BUNDLE)
        if any(part in {"__pycache__", ".venv", ".ipynb_checkpoints"} for part in relative.parts):
            continue
        if path.stat().st_size >= 25 * 1024 * 1024:
            raise ValueError(f"Unexpected large file in shareable folder: {relative}")
        files.append({
            "path": relative.as_posix(), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "generated_report": relative.parts[:2] == ("result_analysis", "current") or relative.as_posix() == "detailed-analysis.md",
        })
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    sources = ["config.py", "analyze_team_notebook.py"] + [f"src/{name}.py" for name in (
        "data", "result_analysis", "team_diagnostics", "team_taxonomy", "team_report"
    )]
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "source_code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        "training_protocol": "historical version 1.0; reporting amendment 1.1 (2026-10-04)",
        "run_counts": {"lodo": 70, "indomain": 10, "total": len(runs)},
        "domains": config.DOMAINS, "seeds": config.SEEDS, "losses": ["standard", "weighted"],
        "source_indexes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ("results.csv", "results_lodo.csv")},
        "excluded": [
            "All model weights/checkpoints and tokenizer directories",
            "ZIP archives, including the duplicate flat LODO export",
            "Regenerable gold_error_records.jsonl (about 50 MB)",
            "Version-2 pilot runs: separate development study",
        ],
        "snapshot_adaptations": [
            "Relative artifact paths; Restaurant consolidated into the LODO index",
            "Unchanged version-1 parser function bodies, without torch/transformers imports",
            "Analysis-only module names and portable report/notebook locations",
            "Large per-gold JSONL export is opt-in; taxonomy definitions are unchanged",
        ],
        "payload_bytes": sum(item["bytes"] for item in files), "files": files,
    }
    write_text(BUNDLE / "bundle_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Shareable payload: {len(files)} files, {manifest['payload_bytes'] / 1024**2:.1f} MiB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="Recompute the 80-run analysis using this Python environment")
    args = parser.parse_args()
    # These hand-written entry points remain editable when rebuilding the snapshot.
    for name in ("README.md", "TEAM_FINDINGS.md", "verify_bundle.py", "requirements.txt", ".gitignore", ".gitattributes"):
        if not (BUNDLE / name).is_file():
            raise FileNotFoundError(f"Missing Phase 1 entry point: {BUNDLE / name}")
    runs = collect_runs()
    print("Copying 70 LODO runs and 10 in-domain controls (predictions and metadata only)...", flush=True)
    for source, row in runs:
        for name in RUN_FILES:
            copy_file(source / name, BUNDLE / row["artifact_dir"] / name)
    for mode, filename in (("indomain", "results.csv"), ("crossdomain", "results_lodo.csv")):
        write_rows(BUNDLE / filename, [row for _, row in runs if row["mode"] == mode])
    for domain in config.DOMAINS:
        for split in ("train", "dev", "test"):
            relative = config.DOMAIN_FILES[domain][split]
            copy_file(ROOT / "data" / "m-absa" / relative, BUNDLE / "corpus" / relative)
    copy_reports(runs)
    copy_analysis_code()
    copy_notebook()
    if args.validate:
        subprocess.run([sys.executable, str(BUNDLE / "analyze_phase1.py")], cwd=BUNDLE, check=True)
    write_manifest(runs)
    subprocess.run([sys.executable, str(BUNDLE / "verify_bundle.py"), "--include-reports"], check=True)
    print(f"Ready to review and add to Git: {BUNDLE}", flush=True)


if __name__ == "__main__":
    main()
