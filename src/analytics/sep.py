"""SEP (Summary of Economic Projections) analytics helpers.

Reads from data/sep_dots.parquet and exposes the dot-plot distribution
in forms useful for charts and finding detection.

Typical usage
-------------
    from src.analytics.sep import load_sep_dots, sep_median, sep_dispersion
    latest = load_sep_dots()                    # full dataset
    latest = load_sep_dots("2026-03-18")        # single meeting
    medians = sep_median(latest)                # {year: median_rate}
    iqrs    = sep_dispersion(latest)            # {year: IQR in bp}
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.ingest.paths import SEP_DOTS_PATH
from src.ingest.storage import load_parquet

_EMPTY_COLS = ["meeting_date", "forecast_year", "rate", "participant_count"]


def load_sep_dots(meeting_date: str | date | None = None) -> pd.DataFrame:
    """Load SEP dot data, optionally filtered to a single meeting.

    Returns columns: meeting_date, forecast_year, rate, participant_count.
    Returns an empty DataFrame (with correct columns) if no data exists.
    """
    df = load_parquet(SEP_DOTS_PATH)
    if df is None or df.empty:
        return pd.DataFrame(columns=_EMPTY_COLS)
    if meeting_date is not None:
        target = meeting_date.isoformat() if isinstance(meeting_date, date) else str(meeting_date)
        df = df[df["meeting_date"] == target]
    return df.copy()


def latest_sep_date(df: pd.DataFrame | None = None) -> str | None:
    """Return the most recent meeting_date string, or None if no data."""
    if df is None:
        df = load_sep_dots()
    return df["meeting_date"].max() if not df.empty else None


def previous_sep_date(df: pd.DataFrame | None = None) -> str | None:
    """Return the second-most-recent meeting_date string, or None."""
    if df is None:
        df = load_sep_dots()
    dates = sorted(df["meeting_date"].unique()) if not df.empty else []
    return dates[-2] if len(dates) >= 2 else None


def expand_dots(df: pd.DataFrame) -> pd.DataFrame:
    """Expand participant_count to one row per participant.

    Input:  columns meeting_date, forecast_year, rate, participant_count
    Output: columns meeting_date, forecast_year, rate  (one row per dot)
    """
    rows: list[dict] = []
    for _, row in df.iterrows():
        base = {"meeting_date": row["meeting_date"], "forecast_year": row["forecast_year"], "rate": float(row["rate"])}
        rows.extend([base.copy() for _ in range(int(row["participant_count"]))])
    return pd.DataFrame(rows)


def sep_median(df: pd.DataFrame) -> dict[str, float]:
    """Return {forecast_year: median_rate_pct} for the given meeting's dots."""
    expanded = expand_dots(df)
    return {
        year: float(np.median(grp["rate"].values))
        for year, grp in expanded.groupby("forecast_year")
    }


def sep_dispersion(df: pd.DataFrame) -> dict[str, float]:
    """Return {forecast_year: IQR_in_bp} for the given meeting's dots."""
    expanded = expand_dots(df)
    result: dict[str, float] = {}
    for year, grp in expanded.groupby("forecast_year"):
        q75, q25 = np.percentile(grp["rate"].values, [75, 25])
        result[year] = float((q75 - q25) * 100)
    return result


def sep_shift(
    meeting_date: str,
    prev_date: str,
    df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Return {forecast_year: median_shift_bp} between two meetings.

    Positive = median moved higher (hawkish shift).
    """
    if df is None:
        df = load_sep_dots()
    med_now = sep_median(df[df["meeting_date"] == meeting_date])
    med_prev = sep_median(df[df["meeting_date"] == prev_date])
    return {
        year: (med_now[year] - med_prev[year]) * 100
        for year in med_now
        if year in med_prev
    }
