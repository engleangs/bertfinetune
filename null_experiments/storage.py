"""Provenance, append-only logs and single-run protection."""

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket

from src.artifacts import atomic_write_json
from src.experiment_runtime import code_inventory, file_hash


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def source_inventory(root):
    result = code_inventory(root)
    paths = [root / "run_null_fast_track.py", *sorted((root / "null_experiments").glob("*.py"))]
    result["source_sha256"].update({p.relative_to(root).as_posix(): file_hash(p) for p in paths})
    return result


def append_csv(path, row, key_fields=None):
    """Append a new logical row; never replace previous experiment results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(row):
                raise ValueError(f"CSV schema changed: {path}; use a new study output")
            if key_fields and any(all(str(old[k]) == str(row[k]) for k in key_fields) for old in reader):
                return
    header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def event(output, kind, **fields):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": utc_now(), "event": kind, **fields}, sort_keys=True) + "\n")


def process_alive(pid):
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            status = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(status))) and status.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextmanager
def study_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "active.lock"
    payload = {"pid": os.getpid(), "host": socket.gethostname(), "started_at": utc_now()}
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["host"] != payload["host"] or process_alive(old["pid"]):
            raise ValueError(f"Another runner holds {path} (PID {old['pid']}); do not launch overlapping queues")
        path.unlink()  # Only this runner's stale, single lock file.
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except FileExistsError as exc:
        raise ValueError("Another runner started concurrently") from exc
    try:
        yield
    finally:
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) == payload:
            path.unlink()


def verify_completed(directory, identity):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["fingerprint"] != digest(identity):
        raise ValueError(f"Run provenance changed: {directory}. Choose a new output; completed results are preserved.")
    if manifest["status"] in ("dev_complete", "complete"):
        for name in ("best.pt", "dev_metrics.json", "dev_predictions.jsonl", "dev_probabilities.pt", "history.json", "audit.json"):
            if not (directory / name).is_file():
                raise ValueError(f"Completed run is missing {name}: {directory}")
        if file_hash(directory / "best.pt") != manifest["checkpoint_sha256"]:
            raise ValueError(f"Completed checkpoint changed: {directory}")
    return manifest
