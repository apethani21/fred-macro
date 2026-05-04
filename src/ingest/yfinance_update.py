"""Fetch market data from Yahoo Finance and write to the daily parquet store.

Called by refresh_data.py --yfinance. Downloads full available history for
each registered ticker and merges into data/series/daily.parquet and
data/metadata.parquet using the same (series_id, date, value) long format
as FRED series, so load_series() and all analytics helpers work unchanged.
"""
from __future__ import annotations

import logging
import warnings
from datetime import datetime, timezone

import pandas as pd

from src.ingest.paths import METADATA_PATH, series_path
from src.ingest.storage import load_parquet, save_parquet_atomic

logger = logging.getLogger(__name__)

# Registry of tickers to maintain. series_id in parquet = ticker string.
YF_SERIES: dict[str, dict] = {
    "CL=F": {
        "title": "WTI Crude Oil - Front Month Futures (NYMEX)",
        "units": "Dollars per Barrel",
        "units_short": "USD/bbl",
    },
    "EURUSD=X": {
        "title": "EUR/USD Exchange Rate",
        "units": "US Dollars per Euro",
        "units_short": "USD",
    },
    "JPY=X": {
        "title": "USD/JPY Exchange Rate",
        "units": "Japanese Yen per US Dollar",
        "units_short": "JPY",
    },
    "GBPUSD=X": {
        "title": "GBP/USD Exchange Rate",
        "units": "US Dollars per British Pound",
        "units_short": "USD",
    },
    "AUDUSD=X": {
        "title": "AUD/USD Exchange Rate",
        "units": "US Dollars per Australian Dollar",
        "units_short": "USD",
    },
    "DX-Y.NYB": {
        "title": "US Dollar Index (DXY) - ICE 6-Currency Basket",
        "units": "Index",
        "units_short": "Index",
    },
    "^VIX": {
        "title": "CBOE Volatility Index (VIX)",
        "units": "Index",
        "units_short": "Index",
    },
}


def fetch_yf_history(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Download history for one ticker; return (series_id, date, value) DataFrame."""
    import yfinance as yf  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(ticker, period=period, progress=False)["Close"]
    s = raw.squeeze().dropna()
    s.index = pd.to_datetime(s.index).normalize()
    df = s.rename("value").reset_index()
    df.columns = ["date", "value"]
    df.insert(0, "series_id", ticker)
    df["date"] = pd.to_datetime(df["date"])
    return df[["series_id", "date", "value"]]


def run_yf_refresh() -> dict[str, int]:
    """Fetch all registered tickers; merge into daily.parquet + metadata.parquet.

    Returns {ticker: row_count} for each ticker attempted.
    """
    daily_path = series_path("daily")
    existing = load_parquet(daily_path)

    yf_ids = set(YF_SERIES)
    if existing is not None and not existing.empty:
        base = existing[~existing["series_id"].isin(yf_ids)].copy()
    else:
        base = pd.DataFrame(columns=["series_id", "date", "value"])

    counts: dict[str, int] = {}
    frames: list[pd.DataFrame] = [base]
    for ticker in YF_SERIES:
        try:
            df = fetch_yf_history(ticker)
            frames.append(df)
            counts[ticker] = len(df)
            logger.info("[yfinance] %s: %d rows", ticker, len(df))
        except Exception as exc:
            logger.error("[yfinance] %s failed: %s", ticker, exc)
            counts[ticker] = 0

    combined = (
        pd.concat(frames, ignore_index=True)
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .drop_duplicates(["series_id", "date"], keep="last")
        .sort_values(["series_id", "date"])
    )
    save_parquet_atomic(combined, daily_path)
    logger.info("[yfinance] daily.parquet written: %d total rows", len(combined))

    _upsert_metadata()
    return counts


def _upsert_metadata() -> None:
    now = datetime.now(timezone.utc).isoformat()
    new_rows = pd.DataFrame([
        {
            "series_id": ticker,
            "title": info["title"],
            "units": info["units"],
            "units_short": info["units_short"],
            "frequency": "Daily",
            "frequency_short": "D",
            "seasonal_adjustment_short": "NSA",
            "observation_start": None,
            "observation_end": None,
            "last_updated": now,
            "popularity": None,
            "notes": None,
            "partition": "daily",
            "last_refreshed": now,
            "source": "yfinance",
        }
        for ticker, info in YF_SERIES.items()
    ])

    existing = load_parquet(METADATA_PATH)
    if existing is not None and not existing.empty:
        base = existing[~existing["series_id"].isin(set(YF_SERIES))].copy()
        combined = pd.concat([base, new_rows], ignore_index=True)
    else:
        combined = new_rows
    save_parquet_atomic(combined, METADATA_PATH)
    logger.info("[yfinance] metadata.parquet updated for %d tickers", len(YF_SERIES))
