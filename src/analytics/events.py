"""FOMC meeting calendar and event-study helpers.

FOMC decision dates are published years in advance at
https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm. This module
ships a hardcoded list covering the common analysis window (2015-present)
and exposes helpers to run event studies around decision days.

For dates before 2015 or beyond the hardcoded list, add rows to
`FOMC_MEETING_DATES` as needed; there is no public FRED release that enumerates
FOMC meeting days (FRED's release calendar covers statistical releases, not
policy events).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .data import load_series


# ---------- FOMC calendar ----------
# Dates are the day the statement is released (day 2 of a two-day meeting).
# Scheduled meetings only; unscheduled emergency actions (e.g., 2020-03-15)
# are in UNSCHEDULED_ACTIONS for separate handling.
FOMC_MEETING_DATES: tuple[str, ...] = (
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29",
    "2015-09-17", "2015-10-28", "2015-12-16",
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27",
    "2016-09-21", "2016-11-02", "2016-12-14",
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26",
    "2017-09-20", "2017-11-01", "2017-12-13",
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01",
    "2018-09-26", "2018-11-08", "2018-12-19",
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31",
    "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16",
    "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
    "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
    "2026-09-16", "2026-10-28", "2026-12-09",
)

# Known off-cycle actions worth studying separately.
UNSCHEDULED_ACTIONS: tuple[tuple[str, str], ...] = (
    ("2020-03-03", "Emergency 50bp cut (pandemic)"),
    ("2020-03-15", "Emergency 100bp cut + restart QE"),
    ("2023-03-12", "BTFP announcement (SVB weekend)"),
)


def fomc_meeting_dates(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    include_unscheduled: bool = False,
) -> pd.DatetimeIndex:
    """Return scheduled FOMC decision dates within [start, end]."""
    dates = list(FOMC_MEETING_DATES)
    if include_unscheduled:
        dates += [d for d, _ in UNSCHEDULED_ACTIONS]
    idx = pd.to_datetime(sorted(dates))
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    return idx


# ---------- event study ----------

@dataclass
class EventStudyResult:
    event_dates: pd.DatetimeIndex
    window_offsets: np.ndarray          # integer day offsets around each event
    per_event: pd.DataFrame             # rows = offsets, columns = event dates, values = delta-from-baseline
    mean: pd.Series                     # mean across events at each offset
    median: pd.Series                   # median across events at each offset
    p25: pd.Series                      # 25th pct across events
    p75: pd.Series                      # 75th pct across events


def event_study(
    series: pd.Series,
    event_dates: pd.DatetimeIndex | list[str],
    window_days: tuple[int, int] = (-5, 20),
    rebase: bool = True,
) -> EventStudyResult:
    """Stack `series` around every date in `event_dates`, return per-event +
    cross-event mean/median/quartile response.

    `series` assumed to be a daily (or near-daily) series. For each event we
    align on day-offset from the event; if `rebase`, subtract the value at
    (or last available before) the event date so responses measure change
    from the event baseline.

    Suitable for FOMC event studies on DGS10, SOFR, VIXCLS, etc. Outputs feed
    `charts.event_study_panel` for the classic median-with-IQR-ribbon chart.
    """
    events = pd.DatetimeIndex(pd.to_datetime(list(event_dates)))
    lo, hi = window_days
    offsets = np.arange(lo, hi + 1)
    s = series.sort_index()

    # Build a calendar-day frame (business-day assumption: series skips weekends).
    # We align by calendar-day offsets and leave NaNs where the market was closed.
    daily_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s_daily = s.reindex(daily_idx).ffill()  # ffill across weekends to handle daily market series

    cols: dict[pd.Timestamp, pd.Series] = {}
    for ev in events:
        if ev < s_daily.index[0] or ev > s_daily.index[-1]:
            continue
        window = s_daily.loc[ev + pd.Timedelta(days=lo): ev + pd.Timedelta(days=hi)]
        if window.empty:
            continue
        day_offset = (window.index - ev).days
        vals = pd.Series(window.values, index=day_offset)
        if rebase:
            baseline = s_daily.asof(ev)
            if pd.notna(baseline):
                vals = vals - baseline
        cols[ev] = vals.reindex(offsets)

    per_event = pd.DataFrame(cols, index=offsets)
    return EventStudyResult(
        event_dates=events,
        window_offsets=offsets,
        per_event=per_event,
        mean=per_event.mean(axis=1).rename("mean"),
        median=per_event.median(axis=1).rename("median"),
        p25=per_event.quantile(0.25, axis=1).rename("p25"),
        p75=per_event.quantile(0.75, axis=1).rename("p75"),
    )


def fomc_event_study(
    series_id: str,
    window_days: tuple[int, int] = (-5, 20),
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    scope: Literal["scheduled", "all"] = "scheduled",
    rebase: bool = True,
) -> EventStudyResult:
    """Event study of `series_id` around FOMC decision days.

    Defaults capture the week before through a month after each meeting,
    rebased to the day before the statement (T-1) so the y-axis reads as
    cumulative response.
    """
    s = load_series(series_id)
    events = fomc_meeting_dates(
        start=start, end=end, include_unscheduled=(scope == "all")
    )
    return event_study(s, events, window_days=window_days, rebase=rebase)
