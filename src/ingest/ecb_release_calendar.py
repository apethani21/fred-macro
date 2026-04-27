"""G7 + Japan central bank rate-decision calendar.

Merges ECB, BoE, BoJ, BoC, and Eurostat HICP dates into
release_calendar.parquet alongside the FRED-sourced US releases.

Release IDs used (negative to avoid collisions with FRED integer IDs):
  -10 : ECB Governing Council meeting
  -11 : Eurostat Flash HICP
  -20 : Bank of England MPC
  -21 : Bank of Japan Monetary Policy Meeting
  -22 : Bank of Canada rate decision
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

# ── Bank of England MPC decision dates ───────────────────────────────────────
# Source: https://www.bankofengland.co.uk/monetary-policy/the-interest-rate-decisions-of-the-mpc
# Date is the decision/publication day (typically Thursday of the meeting week).
# "Super Thursday" meetings (Feb, May, Aug, Nov) also publish the Monetary Policy Report.

_BOE_MPC_DATES: list[date] = [
    # 2025
    date(2025, 2, 6),
    date(2025, 3, 20),
    date(2025, 5, 8),
    date(2025, 6, 19),
    date(2025, 8, 7),
    date(2025, 9, 18),
    date(2025, 11, 6),
    date(2025, 12, 18),
    # 2026
    date(2026, 2, 5),
    date(2026, 3, 19),
    date(2026, 5, 7),
    date(2026, 6, 18),
    date(2026, 8, 6),
    date(2026, 9, 17),
    date(2026, 11, 5),
    date(2026, 12, 17),
    # 2027 (approximate; update when BoE publishes official schedule)
    date(2027, 2, 4),
    date(2027, 3, 18),
    date(2027, 5, 6),
    date(2027, 6, 17),
    date(2027, 8, 5),
    date(2027, 9, 16),
    date(2027, 11, 4),
    date(2027, 12, 16),
]

# ── Bank of Japan Monetary Policy Meeting (MPM) decision dates ───────────────
# Source: https://www.boj.or.jp/en/mopo/mpmsche_mpi/
# Date is the last (decision) day of the 2-day meeting.

_BOJ_MPM_DATES: list[date] = [
    # 2025
    date(2025, 1, 24),
    date(2025, 3, 19),
    date(2025, 5, 1),
    date(2025, 6, 17),
    date(2025, 7, 31),
    date(2025, 9, 23),
    date(2025, 10, 29),
    date(2025, 12, 19),
    # 2026 (approximate; verify against official BoJ calendar)
    date(2026, 1, 23),
    date(2026, 3, 19),
    date(2026, 4, 30),
    date(2026, 6, 17),
    date(2026, 7, 30),
    date(2026, 9, 18),
    date(2026, 10, 29),
    date(2026, 12, 18),
    # 2027 (approximate)
    date(2027, 1, 22),
    date(2027, 3, 18),
    date(2027, 4, 30),
    date(2027, 6, 16),
    date(2027, 7, 29),
    date(2027, 9, 17),
    date(2027, 10, 28),
    date(2027, 12, 17),
]

# ── Bank of Canada fixed announcement dates ───────────────────────────────────
# Source: https://www.bankofcanada.ca/core-functions/monetary-policy/key-interest-rate/
# 8 scheduled decisions per year; the January, April, July, and October meetings
# also publish a full Monetary Policy Report.

_BOC_DATES: list[date] = [
    # 2025
    date(2025, 1, 29),
    date(2025, 3, 12),
    date(2025, 4, 16),
    date(2025, 6, 4),
    date(2025, 7, 30),
    date(2025, 9, 17),
    date(2025, 10, 29),
    date(2025, 12, 10),
    # 2026
    date(2026, 1, 21),
    date(2026, 3, 4),
    date(2026, 4, 15),
    date(2026, 6, 3),
    date(2026, 7, 15),
    date(2026, 9, 9),
    date(2026, 10, 28),
    date(2026, 12, 9),
    # 2027 (approximate; update when BoC publishes official schedule)
    date(2027, 1, 20),
    date(2027, 3, 3),
    date(2027, 4, 14),
    date(2027, 6, 2),
    date(2027, 7, 14),
    date(2027, 9, 8),
    date(2027, 10, 27),
    date(2027, 12, 8),
]

_ECB_GC_RELEASE_ID       = -10
_EUROSTAT_HICP_RELEASE_ID = -11
_BOE_MPC_RELEASE_ID       = -20
_BOJ_MPM_RELEASE_ID       = -21
_BOC_RELEASE_ID           = -22

_CB_EVENTS: list[tuple[int, str, list[date]]] = [
    (_ECB_GC_RELEASE_ID,       "ECB Governing Council",          _ECB_GC_DATES),
    (_EUROSTAT_HICP_RELEASE_ID,"Eurostat Flash HICP",            _EUROSTAT_HICP_FLASH_DATES),
    (_BOE_MPC_RELEASE_ID,      "Bank of England MPC",            _BOE_MPC_DATES),
    (_BOJ_MPM_RELEASE_ID,      "Bank of Japan MPM",              _BOJ_MPM_DATES),
    (_BOC_RELEASE_ID,          "Bank of Canada",                 _BOC_DATES),
]


def _build_cb_rows() -> pd.DataFrame:
    rows = []
    for release_id, release_name, dates in _CB_EVENTS:
        for d in dates:
            rows.append({
                "release_id": release_id,
                "release_name": release_name,
                "release_date": d.isoformat(),
            })
    return pd.DataFrame(rows)


def refresh_cb_calendar() -> pd.DataFrame:
    """Merge G7+ central bank dates into release_calendar.parquet.

    Removes any previously-written CB rows (negative release_ids), then
    appends the current static lists. Safe to call on every data refresh.
    """
    ensure_dirs()
    existing = load_parquet(RELEASE_CALENDAR_PATH)
    if existing is not None and not existing.empty:
        existing = existing[existing["release_id"] > 0].copy()
    else:
        existing = pd.DataFrame(columns=["release_id", "release_name", "release_date"])

    cb_rows = _build_cb_rows()
    combined = pd.concat([existing, cb_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=["release_id", "release_date"])
    combined["release_id"] = combined["release_id"].astype("int64")
    combined["release_name"] = combined["release_name"].astype(str)
    combined["release_date"] = combined["release_date"].astype(str)
    save_parquet_atomic(combined, RELEASE_CALENDAR_PATH)
    logger.info(
        "CB calendar: %d events written (%d sources)",
        len(cb_rows),
        len(_CB_EVENTS),
    )
    return cb_rows


# Backward-compat alias — refresh_data.py imports this name.
refresh_ecb_calendar = refresh_cb_calendar
