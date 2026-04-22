"""Loading and aligning FRED series from our partitioned parquet store.

Single entry points used by research, composition, and ad-hoc analysis so
frequency-mismatch and missing-data policy is consistent everywhere.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Literal

import pandas as pd

from src.ingest.paths import METADATA_PATH, series_path
from src.ingest.storage import load_parquet

# FRED `frequency_short` -> pandas offset alias used when resampling. Chosen
# to match FRED's own storage anchor (period start for M/Q/A) so downsampled
# series align with untouched ones at the same timestamp — mixing start- and
# end-anchored monthlies silently produces zero overlap on align.
_FREQ_ALIAS = {
    "D": "B",      # business-daily; FRED daily series skip weekends already
    "W": "W-FRI",
    "BW": "2W-FRI",
    "M": "MS",
    "Q": "QS",
    "SA": "2QS",
    "A": "YS",
}

# Ordering coarsest (smallest index) -> finest (largest). Used to pick a
# common frequency when aligning across series.
_FREQ_ORDER = ["A", "SA", "Q", "M", "BW", "W", "D"]


@lru_cache(maxsize=1)
def _metadata() -> pd.DataFrame:
    meta = load_parquet(METADATA_PATH)
    if meta is None:
        raise FileNotFoundError(
            f"metadata.parquet not found at {METADATA_PATH}. Run scripts/refresh_data.py --discover first."
        )
    return meta


def clear_cache() -> None:
    """Drop the metadata cache. Call after a refresh in a long-running process."""
    _metadata.cache_clear()


def series_metadata(series_id: str) -> pd.Series:
    meta = _metadata()
    row = meta[meta["series_id"] == series_id]
    if row.empty:
        raise KeyError(f"Series {series_id!r} not in metadata. Refresh first.")
    return row.iloc[0]


def available_series(partition: str | None = None) -> list[str]:
    meta = _metadata()
    if partition is not None:
        meta = meta[meta["partition"] == partition]
    return meta["series_id"].tolist()


def load_series(series_id: str) -> pd.Series:
    """Return the series as a pd.Series indexed by date (ascending), float values.

    Missing values (FRED '.') are NaN. Duplicates are dropped keeping last.
    """
    meta = series_metadata(series_id)
    partition = meta["partition"]
    df = load_parquet(series_path(partition))
    if df is None:
        raise FileNotFoundError(f"Partition file missing for {series_id}: {partition}")
    rows = df[df["series_id"] == series_id]
    if rows.empty:
        raise KeyError(f"No observations stored for {series_id}. Refresh first.")
    s = (
        rows.sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")["value"]
        .astype(float)
    )
    s.index = pd.to_datetime(s.index)
    s.name = series_id
    return s


def _pick_common_frequency(series_ids: Iterable[str]) -> str:
    """Pick the coarsest FRED frequency_short among the given series."""
    freqs = []
    for sid in series_ids:
        f = series_metadata(sid).get("frequency_short")
        if f in _FREQ_ORDER:
            freqs.append(f)
    if not freqs:
        return "M"  # sensible fallback for macro work
    return min(freqs, key=_FREQ_ORDER.index)


def load_aligned(
    series_ids: list[str],
    freq: str | None = None,
    how: Literal["coarsest", "finest"] = "coarsest",
    agg: Literal["last", "mean"] = "last",
    ffill: bool = True,
    dropna: Literal["any", "all", "none"] = "any",
) -> pd.DataFrame:
    """Load several series aligned on a common DatetimeIndex.

    Arguments:
      freq   — explicit FRED code ('D','W','M','Q','A') to target; if None,
               chosen from `how`.
      how    — 'coarsest' resamples finer series down (e.g., daily→monthly using
               `agg`). 'finest' upsamples and forward-fills coarser series.
      agg    — aggregation used when downsampling.
      ffill  — forward-fill missing values within each column after alignment.
      dropna — drop rows where 'any' or 'all' columns are NaN, or 'none'.

    The output columns are in the order of `series_ids`.
    """
    if not series_ids:
        raise ValueError("series_ids must be non-empty")

    target = freq
    if target is None:
        if how == "coarsest":
            target = _pick_common_frequency(series_ids)
        else:
            # finest: pick the highest-resolution frequency present
            present = [series_metadata(s).get("frequency_short") for s in series_ids]
            present = [f for f in present if f in _FREQ_ORDER]
            target = max(present, key=_FREQ_ORDER.index) if present else "M"
    if target not in _FREQ_ALIAS:
        raise ValueError(f"Unknown target frequency {target!r}")

    alias = _FREQ_ALIAS[target]
    resampler = "mean" if agg == "mean" else "last"

    cols = {}
    for sid in series_ids:
        s = load_series(sid).sort_index()
        src_freq = series_metadata(sid).get("frequency_short")
        src_rank = _FREQ_ORDER.index(src_freq) if src_freq in _FREQ_ORDER else -1
        tgt_rank = _FREQ_ORDER.index(target)

        if src_rank == tgt_rank:
            out = s
        elif src_rank > tgt_rank:  # source is finer → downsample
            out = getattr(s.resample(alias), resampler)()
        else:  # source is coarser → upsample
            out = s.resample(alias).asfreq()
            if ffill:
                out = out.ffill()
        cols[sid] = out

    df = pd.concat(cols, axis=1)
    df.columns = series_ids  # preserve caller order

    if ffill and how == "finest":
        df = df.ffill()
    if dropna == "any":
        df = df.dropna(how="any")
    elif dropna == "all":
        df = df.dropna(how="all")
    return df


def to_returns(df_or_series, kind: Literal["pct", "log", "diff"] = "pct"):
    """Returns used for correlating non-stationary series (yields, prices).

    - 'pct' — simple percent change (default for prices)
    - 'log' — log returns
    - 'diff' — first difference in level (right for rates/yields in pct points)
    """
    if kind == "pct":
        return df_or_series.pct_change()
    if kind == "log":
        import numpy as np
        return pd.DataFrame(np.log(df_or_series)).diff() if isinstance(df_or_series, pd.DataFrame) else pd.Series(np.log(df_or_series)).diff()
    if kind == "diff":
        return df_or_series.diff()
    raise ValueError(f"Unknown returns kind: {kind!r}")
