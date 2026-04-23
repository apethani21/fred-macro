"""FOMC meeting dates and event-window utilities.

Meeting dates are the statement-release day (last day of the 2-day meeting,
or the single meeting day for older unscheduled meetings). Source:
https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

Coverage: 2000-2026 hardcoded. Pre-2000 use DFF changes as a proxy.
Emergency/intermeeting actions (e.g., March 2020) are included where
material; they are flagged `is_scheduled=False` in `fomc_meetings()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

# (date_str, is_scheduled).  Unscheduled = emergency / intermeeting.
_FOMC_DATES_RAW: tuple[tuple[str, bool], ...] = (
    # 2000
    ("2000-02-02", True), ("2000-03-21", True), ("2000-05-16", True),
    ("2000-06-28", True), ("2000-08-22", True), ("2000-10-03", True),
    ("2000-11-15", True), ("2000-12-19", True),
    # 2001
    ("2001-01-03", False),  # intermeeting cut after dot-com/Nasdaq collapse
    ("2001-01-31", True), ("2001-03-20", True), ("2001-04-18", False),
    ("2001-05-15", True), ("2001-06-27", True), ("2001-08-21", True),
    ("2001-09-17", False),  # emergency cut post-9/11
    ("2001-10-02", True), ("2001-11-06", True), ("2001-12-11", True),
    # 2002
    ("2002-01-30", True), ("2002-03-19", True), ("2002-05-07", True),
    ("2002-06-26", True), ("2002-08-13", True), ("2002-09-24", True),
    ("2002-11-06", True), ("2002-12-10", True),
    # 2003
    ("2003-01-29", True), ("2003-03-18", True), ("2003-05-06", True),
    ("2003-06-25", True), ("2003-08-12", True), ("2003-09-16", True),
    ("2003-10-28", True), ("2003-12-09", True),
    # 2004
    ("2004-01-28", True), ("2004-03-16", True), ("2004-05-04", True),
    ("2004-06-30", True), ("2004-08-10", True), ("2004-09-21", True),
    ("2004-11-10", True), ("2004-12-14", True),
    # 2005
    ("2005-02-02", True), ("2005-03-22", True), ("2005-05-03", True),
    ("2005-06-30", True), ("2005-08-09", True), ("2005-09-20", True),
    ("2005-11-01", True), ("2005-12-13", True),
    # 2006
    ("2006-01-31", True), ("2006-03-28", True), ("2006-05-10", True),
    ("2006-06-29", True), ("2006-08-08", True), ("2006-09-20", True),
    ("2006-10-25", True), ("2006-12-12", True),
    # 2007
    ("2007-01-31", True), ("2007-03-21", True), ("2007-05-09", True),
    ("2007-06-28", True), ("2007-08-07", True), ("2007-08-17", False),
    ("2007-09-18", True), ("2007-10-31", True), ("2007-12-11", True),
    # 2008
    ("2008-01-22", False),  # emergency cut, Bear Stearns stress
    ("2008-01-30", True), ("2008-03-18", True), ("2008-04-30", True),
    ("2008-06-25", True), ("2008-08-05", True), ("2008-09-16", True),
    ("2008-10-08", False),  # emergency coordinated global cut
    ("2008-10-29", True), ("2008-12-16", True),
    # 2009
    ("2009-01-28", True), ("2009-03-18", True), ("2009-04-29", True),
    ("2009-06-24", True), ("2009-08-12", True), ("2009-09-23", True),
    ("2009-11-04", True), ("2009-12-16", True),
    # 2010
    ("2010-01-27", True), ("2010-03-16", True), ("2010-04-28", True),
    ("2010-06-23", True), ("2010-08-10", True), ("2010-09-21", True),
    ("2010-11-03", True), ("2010-12-14", True),
    # 2011
    ("2011-01-26", True), ("2011-03-15", True), ("2011-04-27", True),
    ("2011-06-22", True), ("2011-08-09", True), ("2011-09-21", True),
    ("2011-11-02", True), ("2011-12-13", True),
    # 2012
    ("2012-01-25", True), ("2012-03-13", True), ("2012-04-25", True),
    ("2012-06-20", True), ("2012-08-01", True), ("2012-09-13", True),
    ("2012-10-24", True), ("2012-12-12", True),
    # 2013
    ("2013-01-30", True), ("2013-03-20", True), ("2013-05-01", True),
    ("2013-06-19", True), ("2013-07-31", True), ("2013-09-18", True),
    ("2013-10-30", True), ("2013-12-18", True),
    # 2014
    ("2014-01-29", True), ("2014-03-19", True), ("2014-04-30", True),
    ("2014-06-18", True), ("2014-07-30", True), ("2014-09-17", True),
    ("2014-10-29", True), ("2014-12-17", True),
    # 2015
    ("2015-01-28", True), ("2015-03-18", True), ("2015-04-29", True),
    ("2015-06-17", True), ("2015-07-29", True), ("2015-09-17", True),
    ("2015-10-28", True), ("2015-12-16", True),
    # 2016
    ("2016-01-27", True), ("2016-03-16", True), ("2016-04-27", True),
    ("2016-06-15", True), ("2016-07-27", True), ("2016-09-21", True),
    ("2016-11-02", True), ("2016-12-14", True),
    # 2017
    ("2017-02-01", True), ("2017-03-15", True), ("2017-05-03", True),
    ("2017-06-14", True), ("2017-07-26", True), ("2017-09-20", True),
    ("2017-11-01", True), ("2017-12-13", True),
    # 2018
    ("2018-01-31", True), ("2018-03-21", True), ("2018-05-02", True),
    ("2018-06-13", True), ("2018-08-01", True), ("2018-09-26", True),
    ("2018-11-08", True), ("2018-12-19", True),
    # 2019
    ("2019-01-30", True), ("2019-03-20", True), ("2019-05-01", True),
    ("2019-06-19", True), ("2019-07-31", True), ("2019-09-18", True),
    ("2019-10-30", True), ("2019-12-11", True),
    # 2020
    ("2020-01-29", True),
    ("2020-03-03", False),  # emergency 50bp cut, COVID
    ("2020-03-15", False),  # emergency 100bp cut + QE restart, COVID
    ("2020-04-29", True), ("2020-06-10", True), ("2020-07-29", True),
    ("2020-09-16", True), ("2020-11-05", True), ("2020-12-16", True),
    # 2021
    ("2021-01-27", True), ("2021-03-17", True), ("2021-04-28", True),
    ("2021-06-16", True), ("2021-07-28", True), ("2021-09-22", True),
    ("2021-11-03", True), ("2021-12-15", True),
    # 2022
    ("2022-01-26", True), ("2022-03-16", True), ("2022-05-04", True),
    ("2022-06-15", True), ("2022-07-27", True), ("2022-09-21", True),
    ("2022-11-02", True), ("2022-12-14", True),
    # 2023
    ("2023-02-01", True), ("2023-03-22", True), ("2023-05-03", True),
    ("2023-06-14", True), ("2023-07-26", True), ("2023-09-20", True),
    ("2023-11-01", True), ("2023-12-13", True),
    # 2024
    ("2024-01-31", True), ("2024-03-20", True), ("2024-05-01", True),
    ("2024-06-12", True), ("2024-07-31", True), ("2024-09-18", True),
    ("2024-11-07", True), ("2024-12-18", True),
    # 2025
    ("2025-01-29", True), ("2025-03-19", True), ("2025-05-07", True),
    ("2025-06-18", True), ("2025-07-30", True), ("2025-09-17", True),
    ("2025-10-29", True), ("2025-12-10", True),
    # 2026 (scheduled through writing date April 2026)
    ("2026-01-28", True), ("2026-03-18", True),
)


@dataclass(frozen=True)
class FOMCMeeting:
    date: pd.Timestamp
    is_scheduled: bool


def fomc_meetings(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    scheduled_only: bool = False,
) -> list[FOMCMeeting]:
    """Return FOMC meeting records filtered by date range.

    Coverage starts 2000-01-01; earlier meetings not hardcoded (use DFF
    changes as proxy for pre-2000 rate-decision dates).
    """
    meetings = [
        FOMCMeeting(date=pd.Timestamp(d), is_scheduled=s)
        for d, s in _FOMC_DATES_RAW
    ]
    if scheduled_only:
        meetings = [m for m in meetings if m.is_scheduled]
    if start is not None:
        ts = pd.Timestamp(start)
        meetings = [m for m in meetings if m.date >= ts]
    if end is not None:
        te = pd.Timestamp(end)
        meetings = [m for m in meetings if m.date <= te]
    return meetings


def fomc_meeting_dates(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    scheduled_only: bool = False,
) -> list[pd.Timestamp]:
    """Convenience wrapper returning just Timestamps."""
    return [m.date for m in fomc_meetings(start, end, scheduled_only)]


def event_panel(
    daily_series: dict[str, pd.Series],
    event_dates: Sequence[pd.Timestamp],
    pre_days: int = 1,
    post_days: int = 5,
) -> pd.DataFrame:
    """Build a panel of daily values around each event date.

    For each (event_date, lag) pair in [-pre_days, +post_days], looks up the
    nearest available trading day within ±1 calendar day of the target date.

    Returns a DataFrame with columns:
        event_date, lag, {series_id}_level, {series_id}_change
    where _change is the 1-day change in that series (level_t - level_{t-1}).

    Rows where any series is missing are kept — callers can dropna by column.
    """
    if not daily_series:
        raise ValueError("daily_series must not be empty")

    # Build a unified daily index from the union of all series indices.
    all_idx = sorted(set().union(*[s.index for s in daily_series.values()]))
    all_idx_ts = pd.DatetimeIndex(all_idx)

    records: list[dict] = []
    for ev_date in event_dates:
        for lag in range(-pre_days, post_days + 1):
            target = ev_date + pd.Timedelta(days=lag)
            # Find nearest index value within ±1 calendar day of target.
            candidates = all_idx_ts[
                (all_idx_ts >= target - pd.Timedelta(days=1)) &
                (all_idx_ts <= target + pd.Timedelta(days=1))
            ]
            if len(candidates) == 0:
                continue
            actual = candidates[abs(candidates - target).argmin()]
            row: dict = {"event_date": ev_date, "lag": lag, "actual_date": actual}
            for sid, s in daily_series.items():
                # Align on the series' own index.
                s_idx = s.index
                pos = s_idx.get_indexer([actual], method="nearest")[0]
                if pos < 0 or pos >= len(s):
                    row[f"{sid}_level"] = float("nan")
                    row[f"{sid}_change"] = float("nan")
                    continue
                level = float(s.iloc[pos])
                prev_pos = pos - 1
                prev_level = float(s.iloc[prev_pos]) if prev_pos >= 0 else float("nan")
                row[f"{sid}_level"] = level
                row[f"{sid}_change"] = level - prev_level
            records.append(row)

    if not records:
        cols = ["event_date", "lag", "actual_date"]
        for sid in daily_series:
            cols += [f"{sid}_level", f"{sid}_change"]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(records)
