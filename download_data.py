"""Download the English M-ABSA splits that are missing locally.

Usage:
    python download_data.py
    python download_data.py --data-dir /path/to/training-data
"""

import argparse
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import config as cfg


RAW_DATA_URL = "https://raw.githubusercontent.com/swaggy66/M-ABSA/main/data"


def expected_files():
    """Yield each repository-relative file required by the experiment."""
    for domain in cfg.DOMAINS:
        for split in ("train", "dev", "test"):
            yield Path(cfg.DOMAIN_FILES[domain][split])


def download_one(relative_path: Path, data_dir: Path) -> str:
    """Atomically download one file, or leave an existing file unchanged."""
    destination = data_dir / relative_path
    if destination.is_file():
        return f"skip     {relative_path}"

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RAW_DATA_URL}/{relative_path.as_posix()}"
    request = Request(url, headers={"User-Agent": "absa-triplet-data-downloader/1.0"})
    temporary_path = None

    try:
        with urlopen(request, timeout=60) as response:
            payload = response.read()

        if not payload or b"####" not in payload:
            raise ValueError(f"Downloaded file does not look like M-ABSA data: {url}")

        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.",
            suffix=".part", delete=False,
        ) as temporary_file:
            temporary_file.write(payload)
            temporary_path = Path(temporary_file.name)

        os.replace(temporary_path, destination)
        return f"download {relative_path}"
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def download_missing_data(data_dir: Path, workers: int = 8) -> None:
    files = list(expected_files())
    missing = [path for path in files if not (data_dir / path).is_file()]

    if not missing:
        print(f"M-ABSA data already exists at {data_dir}")
        return

    print(f"Downloading {len(missing)} missing M-ABSA files to {data_dir}")
    errors = []
    with ThreadPoolExecutor(max_workers=min(workers, len(missing))) as executor:
        futures = {
            executor.submit(download_one, relative_path, data_dir): relative_path
            for relative_path in missing
        }
        for future in as_completed(futures):
            relative_path = futures[future]
            try:
                print(future.result())
            except (HTTPError, URLError, OSError, ValueError) as exc:
                errors.append((relative_path, exc))
                print(f"failed   {relative_path}: {exc}")

    still_missing = [path for path in files if not (data_dir / path).is_file()]
    if errors or still_missing:
        missing_names = ", ".join(str(path) for path in still_missing)
        raise RuntimeError(f"M-ABSA download incomplete; missing: {missing_names}")

    print(f"M-ABSA English training data is ready at {data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(cfg.DATA_DIR),
        help=f"destination directory (default: {cfg.DATA_DIR})",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="number of parallel downloads (default: 8)",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    download_missing_data(args.data_dir.resolve(), args.workers)


if __name__ == "__main__":
    main()
