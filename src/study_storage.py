"""Small result exports and reproducibility snapshots, with no model binaries."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

from src.artifacts import atomic_write_json
from src.study_data import file_digest, json_digest


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def code_files(project):
    project = Path(project)
    files = list(project.glob("*.py"))
    for folder in ("src", "tests"):
        files.extend((project / folder).rglob("*.py"))
    files.extend(p for p in (project / "cluster").iterdir()
                 if p.is_file() and p.suffix in (".py", ".sh", ".sbatch", ".txt"))
    for name in ("requirements.txt", "CLEAN_STUDY.md", "protocol.md", "config.py"):
        if (project / name).is_file():
            files.append(project / name)
    return sorted(set(files))


def code_fingerprint(project):
    project = Path(project)
    files = {p.relative_to(project).as_posix(): file_digest(p) for p in code_files(project)}
    return {"sha256": json_digest(files), "files": files}


def create_code_snapshot(project, output):
    project, output = Path(project), Path(output)
    for path in code_files(project):
        target = output / path.relative_to(project)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())


def lightweight_export(root, destination, legacy=False):
    """Export data/evidence/code only. Legacy models are read neither copied nor deleted."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    allowed_legacy = {"manifest.json", "metrics.json", "history.json", "dev_summary.json",
                      "dev_predictions.jsonl", "test_predictions.jsonl",
                      "dev_predictions.jsonl.gz", "test_predictions.jsonl.gz"}
    text_extensions = {".json", ".jsonl", ".gz", ".csv", ".md", ".txt", ".py", ".sh", ".sbatch", ".sha256"}
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if path.resolve() in (destination, destination.with_suffix(destination.suffix + ".json")):
            continue
        relative = path.relative_to(root)
        if any(part in ("tokenizer", "model_cache", ".venv", ".git", "__pycache__") for part in relative.parts):
            continue
        if legacy:
            include = path.name in allowed_legacy or path.suffix == ".csv"
        else:
            include = path.suffix in text_extensions and (path.suffix != ".gz" or path.name.endswith(".jsonl.gz"))
        if include:
            if path.stat().st_size > 50 * 1024 * 1024:
                raise ValueError(f"Unexpectedly large evidence file: {relative}")
            files.append(path)
    if not files:
        raise ValueError("No result evidence found")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".zip.tmp")
    os.close(fd)
    inventory = []
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                relative = path.relative_to(root).as_posix()
                content = path.read_bytes()
                archive.writestr(relative, content)
                inventory.append({"path": relative, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
            archive.writestr("EXPORT_INVENTORY.json", json.dumps({
                "includes_model_weights": False, "includes_optimizer_states": False,
                "legacy_export": legacy, "files": inventory,
                "limitations": "Predictions support re-scoring and error analysis. Inference on new data requires a retained checkpoint or retraining.",
            }, ensure_ascii=False, indent=2))
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("Result archive failed CRC verification")
            for row in inventory:
                if hashlib.sha256(archive.read(row["path"])).hexdigest() != row["sha256"]:
                    raise RuntimeError("Result archive failed SHA256 verification")
        os.replace(temporary, destination)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    receipt = {"archive": destination.name, "bytes": destination.stat().st_size,
               "uncompressed_evidence_bytes": sum(row["bytes"] for row in inventory),
               "files": len(inventory), "sha256": file_digest(destination), "model_weights_included": False}
    atomic_write_json(destination.with_suffix(destination.suffix + ".json"), receipt)
    return receipt


def safe_checkpoint_path(root, relative):
    """Resolve an explicitly planned checkpoint, never a recursive-delete target."""
    root = Path(root).resolve()
    candidate = root / relative
    if candidate.name not in ("best_model.pt", "resume.pt") or candidate.is_symlink():
        raise ValueError("Only named study checkpoints may be pruned")
    resolved = candidate.resolve()
    if root not in resolved.parents:
        raise ValueError("Checkpoint escapes the study directory")
    for parent in candidate.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("Checkpoint path contains a symbolic link")
    return resolved
