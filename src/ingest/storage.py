"""Parquet read/write helpers with atomic writes.

All project parquet files are read/written through here so IO concerns
(missing files, atomic replace, dtypes) are centralized.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_parquet(path: Path) -> pd.DataFrame | None:
    """Return the parquet at `path`, or None if it doesn't exist yet."""
    if not path.exists():
        return None
    return pd.read_parquet(path)


def save_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write `df` to `path` via a temp file + rename so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
