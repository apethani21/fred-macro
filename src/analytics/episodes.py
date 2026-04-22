"""Historical episode windows and NBER recession dates.

Two sources of truth:

  - NBER recessions: authoritative monthly dates from NBER, mirrored here as
    a hardcoded fallback; `nber_recessions_from_data()` derives the same
    ranges from FRED's USREC series if tracked.
  - Named episodes (Volcker, 1994 bond massacre, 2013 taper, etc.): hand-
    curated windows worth highlighting in charts and narratives.

Dates are (start, end) tuples of `pandas.Timestamp`, inclusive of start and
end (end-of-month convention for recessions, which is how NBER publishes).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

# Source: https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
# (peak, trough). We treat the window [peak_month, trough_month] as "in recession".
NBER_RECESSIONS: tuple[tuple[str, str], ...] = (
    ("1945-02-01", "1945-10-01"),
    ("1948-11-01", "1949-10-01"),
    ("1953-07-01", "1954-05-01"),
    ("1957-08-01", "1958-04-01"),
    ("1960-04-01", "1961-02-01"),
    ("1969-12-01", "1970-11-01"),
    ("1973-11-01", "1975-03-01"),
    ("1980-01-01", "1980-07-01"),
    ("1981-07-01", "1982-11-01"),
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
)


@dataclass(frozen=True)
class Episode:
    name: str
    start: str
    end: str
    description: str


# Windows tuned for charting/narrative context, not academic precision.
NAMED_EPISODES: tuple[Episode, ...] = (
    Episode(
        "1970s_inflation",
        "1973-10-01", "1982-11-01",
        "The Great Inflation: oil shock through Volcker disinflation.",
    ),
    Episode(
        "volcker_disinflation",
        "1979-10-01", "1982-11-01",
        "Volcker's regime shift — deliberate high rates to break inflation expectations.",
    ),
    Episode(
        "1994_bond_massacre",
        "1994-02-01", "1994-12-01",
        "Fed surprise tightening cycle; long-end yields surged, MBS convexity hedging amplified moves.",
    ),
    Episode(
        "1998_ltcm",
        "1998-08-01", "1998-10-31",
        "LTCM collapse following Russia default; Fed coordinated rescue and eased.",
    ),
    Episode(
        "gfc",
        "2007-08-01", "2009-06-30",
        "Global Financial Crisis: BNP liquidity halt → Lehman → QE1.",
    ),
    Episode(
        "2013_taper_tantrum",
        "2013-05-22", "2013-09-30",
        "Bernanke's taper comments; real yields jumped and EM assets sold off.",
    ),
    Episode(
        "covid_2020",
        "2020-02-20", "2020-12-31",
        "COVID shutdown, dash-for-cash, Fed emergency facilities + QE.",
    ),
    Episode(
        "2022_inflation_surge",
        "2022-03-01", "2023-12-31",
        "Fastest Fed hiking cycle since Volcker; yield curve inverted deeply.",
    ),
    Episode(
        "svb_2023",
        "2023-03-08", "2023-05-01",
        "Regional bank stress; SVB/Signature/First Republic failures; Fed BTFP.",
    ),
)
EPISODES_BY_NAME: dict[str, Episode] = {e.name: e for e in NAMED_EPISODES}


def _to_ts(x: str | pd.Timestamp) -> pd.Timestamp:
    return x if isinstance(x, pd.Timestamp) else pd.Timestamp(x)


def nber_recession_ranges() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """The hardcoded NBER list as (start, end) Timestamp pairs."""
    return [(_to_ts(s), _to_ts(e)) for s, e in NBER_RECESSIONS]


def nber_recession_ranges_from_data() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Derive recession ranges from FRED `USREC` if tracked, else fall back."""
    from .data import load_series  # local import avoids circulars at module load
    try:
        s = load_series("USREC").dropna()
    except (FileNotFoundError, KeyError):
        return nber_recession_ranges()
    # USREC is 1 during recession months, 0 otherwise. Find run boundaries.
    flips = s.ne(s.shift()).cumsum()
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for _, run in s.groupby(flips):
        if run.iloc[0] == 1:
            ranges.append((run.index[0], run.index[-1]))
    return ranges


def in_recession(d: str | pd.Timestamp) -> bool:
    ts = _to_ts(d)
    return any(s <= ts <= e for s, e in nber_recession_ranges())


def episode_dates(name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if name not in EPISODES_BY_NAME:
        raise KeyError(f"Unknown episode {name!r}. Known: {list(EPISODES_BY_NAME)}")
    ep = EPISODES_BY_NAME[name]
    return _to_ts(ep.start), _to_ts(ep.end)


def slice_to_episode(series_or_df, name: str):
    s_ts, e_ts = episode_dates(name)
    return series_or_df.loc[s_ts:e_ts]


def compare_to_episodes(
    current_value: float,
    historical: pd.Series,
    episode_names: Iterable[str],
) -> pd.DataFrame:
    """For each named episode, report the historical mean/min/max of `historical` in that window.

    Useful for 'current X is Y vs Z during the 2008 GFC' framings. Caller provides the
    series to compare against; this just does the slicing and stats.
    """
    rows = []
    for name in episode_names:
        try:
            window = slice_to_episode(historical.dropna(), name)
        except KeyError:
            continue
        if window.empty:
            rows.append({"episode": name, "n": 0, "mean": None, "min": None, "max": None, "current_vs_mean": None})
            continue
        mean = float(window.mean())
        rows.append({
            "episode": name,
            "n": int(window.size),
            "mean": mean,
            "min": float(window.min()),
            "max": float(window.max()),
            "current_vs_mean": current_value - mean,
        })
    return pd.DataFrame(rows)
