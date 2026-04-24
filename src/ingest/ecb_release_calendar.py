"""Static ECB / Eurostat release calendar.

Merges ECB Governing Council meeting dates and Eurostat flash HICP release dates
into release_calendar.parquet alongside the FRED-sourced releases.

FRED's release calendar API covers only US data releases; the ECB publishes its
calendar separately (https://www.ecb.europa.eu/press/calendars/html/index.en.html).
Eurostat publishes flash HICP dates roughly the last week of each reference month.

Release IDs used (negative to avoid collisions with FRED integer IDs):
  -10 : ECB Governing Council meeting
  -11 : Eurostat Flash HICP
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from .paths import RELEASE_CALENDAR_PATH, ensure_dirs
from .storage import load_parquet, save_parquet_atomic

logger = logging.getLogger(__name__)

# ── ECB Governing Council meeting dates ──────────────────────────────────────
# Source: https://www.ecb.europa.eu/press/calendars/html/index.en.html
# Dates are the monetary policy decision day (Thursday of the meeting week).
# Only the decision/press conference day is listed; non-monetary-policy meetings omitted.

_ECB_GC_DATES: list[date] = [
    # 2025
    date(2025, 1, 30),
    date(2025, 3, 6),
    date(2025, 4, 17),
    date(2025, 6, 5),
    date(2025, 7, 24),
    date(2025, 9, 11),
    date(2025, 10, 30),
    date(2025, 12, 18),
    # 2026
    date(2026, 1, 29),
    date(2026, 3, 5),
    date(2026, 4, 16),
    date(2026, 6, 4),
    date(2026, 7, 23),
    date(2026, 9, 10),
    date(2026, 10, 29),
    date(2026, 12, 10),
    # 2027 (approximate; update when ECB publishes official calendar)
    date(2027, 1, 28),
    date(2027, 3, 11),
    date(2027, 4, 22),
    date(2027, 6, 10),
    date(2027, 7, 22),
    date(2027, 9, 9),
    date(2027, 10, 28),
    date(2027, 12, 16),
]

# ── Eurostat Flash HICP release dates ────────────────────────────────────────
# Eurostat publishes flash HICP estimates for the reference month roughly the
# last week of that month (often the last Wednesday or Thursday).
# Source: https://ec.europa.eu/eurostat/news/release-calendar

_EUROSTAT_HICP_FLASH_DATES: list[date] = [
    # 2025
    date(2025, 1, 31),
    date(2025, 2, 28),
    date(2025, 3, 31),
    date(2025, 4, 30),
    date(2025, 5, 30),
    date(2025, 6, 30),
    date(2025, 7, 31),
    date(2025, 8, 29),
    date(2025, 9, 30),
    date(2025, 10, 31),
    date(2025, 11, 28),
    date(2025, 12, 17),
    # 2026
    date(2026, 1, 30),
    date(2026, 2, 27),
    date(2026, 3, 31),
    date(2026, 4, 30),
    date(2026, 5, 29),
    date(2026, 6, 30),
    date(2026, 7, 31),
    date(2026, 8, 28),
    date(2026, 9, 30),
    date(2026, 10, 30),
    date(2026, 11, 27),
    date(2026, 12, 16),
    # 2027 (approximate)
    date(2027, 1, 29),
    date(2027, 2, 26),
    date(2027, 3, 31),
    date(2027, 4, 30),
    date(2027, 5, 28),
    date(2027, 6, 30),
]

_ECB_GC_RELEASE_ID = -10
_EUROSTAT_HICP_RELEASE_ID = -11


def _build_ecb_rows() -> pd.DataFrame:
    rows = []
    for d in _ECB_GC_DATES:
        rows.append({
            "release_id": _ECB_GC_RELEASE_ID,
            "release_name": "ECB Governing Council",
            "release_date": d.isoformat(),
        })
    for d in _EUROSTAT_HICP_FLASH_DATES:
        rows.append({
            "release_id": _EUROSTAT_HICP_RELEASE_ID,
            "release_name": "Eurostat Flash HICP",
            "release_date": d.isoformat(),
        })
    return pd.DataFrame(rows)


def refresh_ecb_calendar() -> pd.DataFrame:
    """Merge ECB and Eurostat dates into release_calendar.parquet.

    Removes any previously-written ECB rows (negative release_ids), then
    appends the current static list. Safe to call on every data refresh.
    """
    ensure_dirs()
    existing = load_parquet(RELEASE_CALENDAR_PATH)
    if existing is not None and not existing.empty:
        # Drop old ECB rows; keep all FRED rows (positive release_ids).
        existing = existing[existing["release_id"] > 0].copy()
    else:
        existing = pd.DataFrame(columns=["release_id", "release_name", "release_date"])

    ecb_rows = _build_ecb_rows()
    combined = pd.concat([existing, ecb_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=["release_id", "release_date"])
    # Enforce dtypes to match the FRED-sourced schema.
    combined["release_id"] = combined["release_id"].astype("int64")
    combined["release_name"] = combined["release_name"].astype(str)
    combined["release_date"] = combined["release_date"].astype(str)
    save_parquet_atomic(combined, RELEASE_CALENDAR_PATH)
    logger.info(
        "ECB calendar: added %d GC meetings + %d HICP flash dates",
        len(_ECB_GC_DATES),
        len(_EUROSTAT_HICP_FLASH_DATES),
    )
    return ecb_rows
