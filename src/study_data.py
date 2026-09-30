"""Versioned sentence-level conflict filtering; raw files are never edited."""

import ast
import hashlib
import json
from pathlib import Path

import config
from src.artifacts import atomic_write_json


POLICY = "drop-whole-training-row-containing-conflict-v1"


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def split_paths(domains=None, splits=("train", "dev", "test")):
    return [f"{domain}/en/{split}.txt" for domain in (domains or config.DOMAINS) for split in splits]


def parse_annotation(line):
    sentence, annotations = line.strip().split("####", 1)
    triplets = ast.literal_eval(annotations)
    if not isinstance(triplets, (list, tuple)) or any(
        not isinstance(t, (list, tuple)) or len(t) != 3 or not all(isinstance(v, str) for v in t)
        for t in triplets
    ):
        raise ValueError("Invalid M-ABSA annotation")
    return sentence, triplets


def prepare_clean_data(source, destination, expected_removed=5):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Raw and cleaned datasets must be separate sibling locations")
    report_path = destination / "cleaning_audit.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("policy") != POLICY:
            raise ValueError("Existing cleaned dataset uses a different policy")
        if set(report["files"]) != set(split_paths()) or (expected_removed is not None and report["removed_sentence_count"] != expected_removed):
            raise ValueError("Existing cleaned dataset has an unexpected file or removal count")
        for relative, row in report["files"].items():
            if file_digest(source / relative) != row["raw_sha256"] or file_digest(destination / relative) != row["clean_sha256"]:
                raise ValueError(f"Dataset differs from saved cleaning audit: {relative}")
        return report
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Incomplete cleaned data directory exists; use a new destination")
    report = {"policy": POLICY, "files": {}, "removed_rows": [],
              "removed_sentence_count": 0, "removed_conflict_triplets": 0,
              "removed_other_triplets": 0, "dev_test_unchanged": True}
    outputs = {}
    for relative in split_paths():
        raw = (source / relative).read_bytes()
        retained = []
        counts = {"raw_sentences": 0, "clean_sentences": 0, "raw_triplets": 0, "clean_triplets": 0}
        for number, raw_line in enumerate(raw.splitlines(keepends=True), start=1):
            if not raw_line.strip():
                retained.append(raw_line)
                continue
            sentence, triplets = parse_annotation(raw_line.decode("utf-8"))
            conflicts = sum(t[2].strip().casefold() == "conflict" for t in triplets)
            counts["raw_sentences"] += 1
            counts["raw_triplets"] += len(triplets)
            if conflicts:
                if not relative.endswith("/train.txt"):
                    raise ValueError(f"Conflict appeared outside training: {relative}:{number}; review the protocol")
                report["removed_rows"].append({"file": relative, "line": number, "sentence": sentence, "triplets": triplets})
                report["removed_sentence_count"] += 1
                report["removed_conflict_triplets"] += conflicts
                report["removed_other_triplets"] += len(triplets) - conflicts
            else:
                retained.append(raw_line)
                counts["clean_sentences"] += 1
                counts["clean_triplets"] += len(triplets)
        clean = b"".join(retained)
        if not relative.endswith("/train.txt") and clean != raw:
            raise ValueError("Development/test data changed")
        outputs[relative] = clean
        report["files"][relative] = {**counts, "raw_sha256": hashlib.sha256(raw).hexdigest(),
                                     "clean_sha256": hashlib.sha256(clean).hexdigest()}
    if expected_removed is not None and report["removed_sentence_count"] != expected_removed:
        raise ValueError(f"Expected {expected_removed} removed sentences, got {report['removed_sentence_count']}")
    for relative, content in outputs.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    atomic_write_json(report_path, report)
    return report


def verify_source_splits(data_root, audit, data_version, fold, splits=("train", "dev")):
    """Search calls this on source train/dev only, never target or test files."""
    domains = [d for d in config.DOMAINS if d != fold] if fold != "indomain" else config.DOMAINS
    hashes = {}
    for relative in split_paths(domains, splits):
        actual = file_digest(Path(data_root) / relative)
        if actual != audit["files"][relative][f"{data_version}_sha256"]:
            raise ValueError(f"Data changed after preparation: {relative}")
        hashes[relative] = actual
    return hashes
