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


# ---- M1: Inflation Episode Cross-Sectional Analysis ----

@dataclass(frozen=True)
class InflationEpisode:
    idx: int
    start: pd.Timestamp
    end: pd.Timestamp
    peak_date: pd.Timestamp
    peak_value: float
    duration_months: int
    driver: str              # "demand" | "supply" | "mixed"
    real_ff_min: float       # min real fed funds during episode (FEDFUNDS - CPI YoY)
    unrate_change: float     # unemployment rate change from episode start to end


def identify_inflation_episodes(
    cpi_yoy: pd.Series | None = None,
    threshold: float = 4.0,
    min_duration: int = 3,
) -> list[InflationEpisode]:
    """Identify US CPI inflation episodes above `threshold` YoY for >= min_duration months.

    If cpi_yoy is None, loads CPIAUCSL and computes 12-month YoY internally.
    Ongoing episode (not yet ended) is included if active.
    """
    from .data import load_aligned, load_series  # avoid circular at module load

    import numpy as np

    if cpi_yoy is None:
        raw = load_series("CPIAUCSL").dropna()
        cpi_yoy = (raw.pct_change(12) * 100).rename("cpi_yoy")

    s = cpi_yoy.dropna().resample("ME").last()
    above = (s >= threshold).astype(int)

    # Supporting series for driver classification and real rates.
    try:
        df_supp = load_aligned(["FEDFUNDS", "UNRATE"], freq="M", how="coarsest", dropna="none")
    except (FileNotFoundError, KeyError):
        df_supp = pd.DataFrame()

    flips = above.ne(above.shift()).cumsum()
    episodes: list[InflationEpisode] = []
    ep_idx = 0

    for _, run in s.groupby(flips):
        if above.loc[run.index[0]] == 0:
            continue
        if len(run) < min_duration:
            continue

        start, end = run.index[0], run.index[-1]
        peak_dt = run.idxmax()
        peak_val = float(run.max())

        # Driver: demand shock = unemployment fell; supply = unemployment rose.
        driver = "mixed"
        unrate_change = float("nan")
        if not df_supp.empty and "UNRATE" in df_supp.columns:
            unrate_ep = df_supp["UNRATE"].loc[start:end].dropna()
            if len(unrate_ep) >= 3:
                unrate_change = float(unrate_ep.iloc[-1] - unrate_ep.iloc[0])
                if unrate_change < -0.5:
                    driver = "demand"
                elif unrate_change > 0.5:
                    driver = "supply"

        # Real fed funds minimum during episode.
        real_ff_min = float("nan")
        if not df_supp.empty and "FEDFUNDS" in df_supp.columns:
            ff_ep = df_supp["FEDFUNDS"].loc[start:end].dropna()
            cpi_ep = s.loc[start:end].dropna()
            aligned = pd.concat([ff_ep.rename("ff"), cpi_ep.rename("cpi")], axis=1).dropna()
            if not aligned.empty:
                real_ff_min = float((aligned["ff"] - aligned["cpi"]).min())

        episodes.append(InflationEpisode(
            idx=ep_idx,
            start=start,
            end=end,
            peak_date=peak_dt,
            peak_value=peak_val,
            duration_months=len(run),
            driver=driver,
            real_ff_min=real_ff_min,
            unrate_change=unrate_change,
        ))
        ep_idx += 1

    return episodes


def current_inflation_episode(
    cpi_yoy: pd.Series | None = None,
    threshold: float = 4.0,
    min_duration: int = 3,
) -> InflationEpisode | None:
    """Return the ongoing inflation episode if one is active (ended within 6 months), else None."""
    episodes = identify_inflation_episodes(cpi_yoy, threshold=threshold, min_duration=min_duration)
    if not episodes:
        return None
    last = episodes[-1]
    six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
    return last if last.end >= six_months_ago else None


def inflation_episode_distribution(episodes: list[InflationEpisode]) -> dict:
    """Cross-sectional summary statistics across a list of inflation episodes."""
    import numpy as np

    if not episodes:
        return {}

    def _qs(arr: list[float]) -> dict:
        s = pd.Series([v for v in arr if not (isinstance(v, float) and (v != v))])
        if s.empty:
            return {}
        return {
            "median": float(s.median()),
            "p25": float(s.quantile(0.25)),
            "p75": float(s.quantile(0.75)),
            "max": float(s.max()),
            "n": int(s.size),
        }

    driver_counts: dict[str, int] = {}
    for e in episodes:
        driver_counts[e.driver] = driver_counts.get(e.driver, 0) + 1

    real_ff_vals = [e.real_ff_min for e in episodes if not (e.real_ff_min != e.real_ff_min)]

    return {
        "n": len(episodes),
        "duration_months": _qs([e.duration_months for e in episodes]),
        "peak_value": _qs([e.peak_value for e in episodes]),
        "unrate_change": _qs([e.unrate_change for e in episodes]),
        "real_ff_min": _qs(real_ff_vals),
        "driver_counts": driver_counts,
    }


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
