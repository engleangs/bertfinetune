"""Check the model-free Phase 1 snapshot without third-party dependencies."""

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
FORBIDDEN = {".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".h5", ".onnx", ".zip", ".tar", ".gz"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-reports", action="store_true", help="Also check the exported report snapshot; regenerated reports may differ")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "bundle_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Nonportable manifest path: {relative}")
        path = ROOT.joinpath(*relative.parts)
        if path.suffix.casefold() in FORBIDDEN:
            raise ValueError(f"Excluded artifact in manifest: {relative}")
        if item["generated_report"] and not args.include_reports:
            continue
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Missing or changed snapshot file: {relative}")

    keys = []
    for filename, mode in (("results.csv", "indomain"), ("results_lodo.csv", "crossdomain")):
        with (ROOT / filename).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            relative = PurePosixPath(row["artifact_dir"])
            if relative.is_absolute() or ".." in relative.parts or "\\" in row["artifact_dir"] or ":" in row["artifact_dir"]:
                raise ValueError(f"Nonportable artifact directory: {relative}")
            if row["status"] != "complete" or row["mode"] != mode:
                raise ValueError(f"Unexpected indexed run: {row}")
            for name in ("manifest.json", "metrics.json", "history.json", "dev_predictions.jsonl", "test_predictions.jsonl"):
                if not ROOT.joinpath(*relative.parts, name).is_file():
                    raise FileNotFoundError(relative / name)
            keys.append((mode, row["held_out_domain"], row["config"], int(row["seed"])))
    expected = {
        (mode, domain, loss, seed)
        for mode, domains in (("indomain", ["all"]), ("crossdomain", manifest["domains"]))
        for domain in domains for loss in manifest["losses"] for seed in manifest["seeds"]
    }
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Incomplete, unexpected, or duplicate run matrix")
    for path in ROOT.rglob("*"):
        if path.is_file() and ".venv" not in path.relative_to(ROOT).parts and path.suffix.casefold() in FORBIDDEN:
            raise ValueError(f"Excluded artifact present in folder: {path.relative_to(ROOT)}")
    print(f"Verified {len(keys)} runs: 70 LODO + 10 in-domain; snapshot hashes and relative paths are valid.")


if __name__ == "__main__":
    main()
