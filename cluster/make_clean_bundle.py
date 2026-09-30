"""Build the NEW study transfer package. Model weights are optional and shared."""

import argparse
import io
from pathlib import Path
import sys
import tarfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.study_data import file_digest, split_paths
from src.study_storage import code_files
from run_clean_study import DEFAULT_OUTPUT, load_study, MODEL_FILES, model_directory


def bundle(destination, include_model=False):
    load_study(DEFAULT_OUTPUT, verify_code=True)
    paths = set(code_files(PROJECT))
    for folder in ("data/m-absa", "data/m-absa-clean-v1"):
        paths.update(PROJECT / folder / relative for relative in split_paths())
    paths.add(PROJECT / "data/m-absa-clean-v1/cleaning_audit.json")
    paths.update(DEFAULT_OUTPUT / name for name in ("study.json", "cleaning_audit.json"))
    paths.update(p for p in (DEFAULT_OUTPUT / "reproducibility").rglob("*") if p.is_file())
    destination = Path(destination).resolve()
    if PROJECT == destination or PROJECT in destination.parents:
        raise ValueError("Place the transfer archive outside the project")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(paths):
            relative = path.relative_to(PROJECT).as_posix()
            archive.add(path, arcname=f"bertfinetune-clean-v1/{relative}", recursive=False)
            manifest.append(f"{file_digest(path)}  {relative}\n")
        if include_model:
            for name in MODEL_FILES:
                path = model_directory() / name
                relative = f"model_cache/bert-base-uncased/{name}"
                archive.add(path, arcname=f"bertfinetune-clean-v1/{relative}", recursive=False)
                manifest.append(f"{file_digest(path)}  {relative}\n")
        payload = "".join(manifest).encode("utf-8")
        info = tarfile.TarInfo("bertfinetune-clean-v1/cluster/transfer_manifest.sha256")
        info.size = len(payload)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(payload))
    digest = file_digest(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="utf-8")
    print(f"Transfer package: {destination}\nBytes: {destination.stat().st_size}\nSHA256: {digest}\nShared model included: {include_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--include-model", action="store_true")
    args = parser.parse_args()
    bundle(args.destination, args.include_model)
