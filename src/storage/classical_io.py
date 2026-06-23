"""Crash-safe file writing and logging helpers."""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd


def atomic_write(path: Path, writer: Callable[[Path], None], suffix: str) -> None:
    """Write to a temporary file, then atomically replace the final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.stem}.tmp{suffix}")
    writer(tmp_path)
    tmp_path.replace(path)


def atomic_write_json(path: Path, data: Any) -> None:
    def writer(tmp_path: Path) -> None:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)

    atomic_write(path, writer, path.suffix or ".json")


def atomic_write_pickle(path: Path, data: Any) -> None:
    def writer(tmp_path: Path) -> None:
        with tmp_path.open("wb") as handle:
            pickle.dump(data, handle)

    atomic_write(path, writer, path.suffix or ".pkl")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    def writer(tmp_path: Path) -> None:
        frame.to_csv(tmp_path, index=False)

    atomic_write(path, writer, path.suffix or ".csv")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)
