"""Release calendar: when each FRED release publishes, and what series it covers.

Two separate tables so each has a single purpose:

  data/release_calendar.parquet  — (release_id, release_name, release_date)
  data/release_series.parquet    — (release_id, series_id)

Dates change frequently (new publications, schedule shifts); series-to-release
membership mostly does not. Refreshing them independently avoids rewriting
the series-membership table on every calendar refresh.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

import pandas as pd

from .fred_client import FredClient
from .paths import RELEASE_CALENDAR_PATH, RELEASE_SERIES_PATH, ensure_dirs
from .storage import save_parquet_atomic

# Default window: enough past context to see recent releases and enough
# future to drive the email's "what to watch" section.
DEFAULT_PAST_DAYS = 60
DEFAULT_FUTURE_DAYS = 120

logger = logging.getLogger(__name__)


def refresh_release_calendar(
    client: FredClient,
    release_ids: Iterable[int] | None = None,
    past_days: int = DEFAULT_PAST_DAYS,
    future_days: int = DEFAULT_FUTURE_DAYS,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Pull release dates in a window around `as_of` and write the calendar.

    FRED's default for /releases/dates is DESC future-first; we want a bounded
    window that covers recent + upcoming releases, so pass explicit realtime
    params.
    """
    ensure_dirs()
    releases_meta = {int(r["id"]): r for r in client.releases()}

    ref = as_of or datetime.now(timezone.utc).date()
    start = (ref - timedelta(days=past_days)).isoformat()
    end = (ref + timedelta(days=future_days)).isoformat()

    if release_ids is None:
        raw = client.release_dates(
            realtime_start=start, realtime_end=end, sort_order="asc"
        )
    else:
        raw = []
        for rid in release_ids:
            raw.extend(
                client.release_dates(
                    release_id=rid,
                    realtime_start=start,
                    realtime_end=end,
                    sort_order="asc",
                )
            )

    if not raw:
        logger.warning("No release dates returned")
        return pd.DataFrame(columns=["release_id", "release_name", "release_date"])

    df = pd.DataFrame(raw)
    df["release_id"] = df["release_id"].astype(int)
    df["release_date"] = pd.to_datetime(df["date"]).dt.date
    df["release_name"] = df["release_id"].map(
        lambda rid: releases_meta.get(rid, {}).get("name", "")
    )
    df = df[["release_id", "release_name", "release_date"]].sort_values(
        ["release_date", "release_id"]
    ).reset_index(drop=True)

    save_parquet_atomic(df, RELEASE_CALENDAR_PATH)
    logger.info(
        "Wrote release calendar: %d rows, %d releases, range %s → %s",
        len(df), df["release_id"].nunique(), df["release_date"].min(), df["release_date"].max(),
    )
    return df


def refresh_release_series(
    client: FredClient,
    release_ids: Iterable[int],
    limit_per_release: int = 1000,
) -> pd.DataFrame:
    """For each release, pull its series list and write (release_id, series_id) rows."""
    ensure_dirs()
    rows: list[dict] = []
    for rid in release_ids:
        try:
            series = client.release_series(rid, limit=limit_per_release)
        except Exception as e:
            logger.warning("release_series failed for release %d: %s", rid, e)
            continue
        for s in series:
            sid = s.get("id")
            if sid:
                rows.append({"release_id": int(rid), "series_id": sid})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates().sort_values(["release_id", "series_id"]).reset_index(drop=True)
    save_parquet_atomic(df, RELEASE_SERIES_PATH)
    logger.info(
        "Wrote release-series membership: %d rows across %d releases",
        len(df), df["release_id"].nunique() if not df.empty else 0,
    )
    return df


def upcoming_releases(
    calendar_df: pd.DataFrame,
    as_of: date | None = None,
    days_ahead: int = 14,
) -> pd.DataFrame:
    """Return calendar rows between `as_of` and `as_of + days_ahead`."""
    if calendar_df.empty:
        return calendar_df
    ref = as_of or datetime.now(timezone.utc).date()
    end = ref + pd.Timedelta(days=days_ahead)
    mask = (calendar_df["release_date"] >= ref) & (calendar_df["release_date"] <= end.date() if hasattr(end, "date") else end)
    return calendar_df.loc[mask].sort_values("release_date").reset_index(drop=True)
