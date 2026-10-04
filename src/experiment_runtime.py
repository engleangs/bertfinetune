"""Small provenance helpers; pilot fingerprints include train/dev only."""

import hashlib
from importlib import metadata
import platform
from pathlib import Path
import subprocess

import torch

import config


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_inventory():
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata.get("Name"))
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def code_inventory(root):
    paths = [Path("run_experiments.py"), Path("config.py"),
             *[p.relative_to(root) for p in sorted((Path(root) / "src").glob("*.py"))]]
    hashes = {str(p).replace("\\", "/"): file_hash(Path(root) / p) for p in paths}
    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"source_sha256": hashes, "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain"))}


def data_inventory(root, domains, splits=("train", "dev")):
    return {config.DOMAIN_FILES[domain][split]: file_hash(Path(root) / "data" / "m-absa" / config.DOMAIN_FILES[domain][split])
            for domain in domains for split in splits}
