"""Bundle the local experiment code, M-ABSA data, and offline BERT files."""

import argparse
import hashlib
import io
import subprocess
import tarfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIRS = (Path("data/m-absa"), Path("model_cache/bert-base-uncased"))


def git_paths(*args):
    output = subprocess.check_output(["git", *args, "-z"], cwd=PROJECT_ROOT)
    return {Path(item.decode("utf-8")) for item in output.split(b"\0") if item}


def source_files():
    paths = git_paths("ls-files") | git_paths("ls-files", "--others", "--exclude-standard")
    for asset_dir in ASSET_DIRS:
        if not (PROJECT_ROOT / asset_dir).is_dir():
            raise FileNotFoundError(f"Missing asset directory: {asset_dir}")
        paths.update(path.relative_to(PROJECT_ROOT) for path in (PROJECT_ROOT / asset_dir).rglob("*") if path.is_file())
    files = sorted(path for path in paths if (PROJECT_ROOT / path).is_file())
    if not files:
        raise RuntimeError("No files found to bundle")
    return files


def make_bundle(destination):
    files = source_files()
    destination = Path(destination).resolve()
    if destination == PROJECT_ROOT or PROJECT_ROOT in destination.parents:
        raise ValueError("Bundle destination must be outside the project directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    hashes = []
    with tarfile.open(destination, "w", format=tarfile.GNU_FORMAT) as archive:
        for relative_path in files:
            source = PROJECT_ROOT / relative_path
            digest = hashlib.sha256()
            with source.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes.append(f"{digest.hexdigest()}  {relative_path.as_posix()}\n")
            archive.add(source, arcname=f"bertfinetune/{relative_path.as_posix()}", recursive=False)
        manifest = "".join(hashes).encode("utf-8")
        info = tarfile.TarInfo("bertfinetune/cluster/transfer_manifest.sha256")
        info.size = len(manifest)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(manifest))
    archive_digest = hashlib.sha256()
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            archive_digest.update(chunk)
    print(f"Bundle: {destination}")
    print(f"Files: {len(files)}")
    print(f"Bytes: {destination.stat().st_size}")
    print(f"SHA256: {archive_digest.hexdigest()}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    make_bundle(parser.parse_args().destination)
