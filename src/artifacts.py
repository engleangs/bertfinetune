"""Atomic persistence helpers for checkpoints, predictions, and result rows."""

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    os.close(handle)
    return Path(name)


def atomic_write_json(destination, payload) -> None:
    destination = Path(destination)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False, sort_keys=True)
            file.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_jsonl(destination, rows: Iterable[Mapping]) -> None:
    destination = Path(destination)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                file.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(destination, payload) -> None:
    destination = Path(destination)
    temporary = _temporary_path(destination)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def upsert_csv_row(
    destination,
    row: Mapping,
    fieldnames: Sequence[str],
    key_fields: Sequence[str],
) -> int:
    """Atomically replace every existing row with the same logical key."""
    destination = Path(destination)
    existing = []
    if destination.exists():
        with destination.open("r", encoding="utf-8-sig", newline="") as file:
            existing = list(csv.DictReader(file))

    key = tuple(str(row.get(field, "")) for field in key_fields)
    retained = []
    replaced = 0
    for existing_row in existing:
        existing_key = tuple(str(existing_row.get(field, "")) for field in key_fields)
        if existing_key == key:
            replaced += 1
        else:
            retained.append(existing_row)
    retained.append(dict(row))

    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=fieldnames, extrasaction="ignore",
            )
            writer.writeheader()
            for existing_row in retained:
                writer.writerow(
                    {field: existing_row.get(field, "") for field in fieldnames}
                )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return replaced
