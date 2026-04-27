"""Canonical project paths.

Resolved from the project root so that scripts work regardless of CWD,
as long as the package is importable.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
SERIES_DIR = DATA_DIR / "series"
STATS_DIR = DATA_DIR / "stats"
METADATA_PATH = DATA_DIR / "metadata.parquet"
UPDATE_LOG_PATH = DATA_DIR / "update_log.parquet"
RELEASE_CALENDAR_PATH = DATA_DIR / "release_calendar.parquet"
RELEASE_SERIES_PATH = DATA_DIR / "release_series.parquet"
DISCOVERY_PATH = DATA_DIR / "discovery.parquet"
SEP_DOTS_PATH = DATA_DIR / "sep_dots.parquet"

KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
STATE_DIR = PROJECT_ROOT / "state"


# FRED `frequency_short` codes we map to storage partitions.
FREQ_TO_PARTITION = {
    "D": "daily",
    "W": "weekly",
    "BW": "weekly",   # bi-weekly — store with weekly, rare
    "M": "monthly",
    "Q": "quarterly",
    "SA": "annual",   # semiannual — rare, store with annual
    "A": "annual",
}


def series_path(partition: str) -> Path:
    return SERIES_DIR / f"{partition}.parquet"


def ensure_dirs() -> None:
    for d in (DATA_DIR, SERIES_DIR, STATS_DIR, KNOWLEDGE_DIR, STATE_DIR):
        d.mkdir(parents=True, exist_ok=True)
