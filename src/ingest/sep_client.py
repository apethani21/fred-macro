"""FOMC Summary of Economic Projections (SEP) scraper.

Discovers published SEP tables from the FOMC calendar page, parses the
fed-funds-rate dot-plot HTML, and stores in data/sep_dots.parquet.

Schema
------
meeting_date      : str   YYYY-MM-DD  (the decision day, from the URL)
forecast_year     : str   "2026" | "Longer run" | etc.
rate              : float midpoint in percent (e.g. 3.625)
participant_count : int   number of participants projecting this rate

Usage
-----
    from src.ingest.sep_client import SepClient
    client = SepClient()
    df = client.refresh_all()          # incremental; skips already-fetched meetings
    df = client.refresh_all(force=True) # re-fetch everything
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .paths import SEP_DOTS_PATH, ensure_dirs
from .storage import load_parquet, save_parquet_atomic

logger = logging.getLogger(__name__)

_BASE = "https://www.federalreserve.gov"
_CALENDAR_URL = f"{_BASE}/monetarypolicy/fomccalendars.htm"
_HTM_RE = re.compile(r"/monetarypolicy/fomcprojtabl(\d{8})\.htm")
_MIN_INTERVAL = 1.5  # seconds between requests
_last_request: float = 0.0


def _get(url: str) -> requests.Response:
    global _last_request
    elapsed = time.monotonic() - _last_request
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    resp = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "fred-macro-bot/1.0 (personal research)"},
    )
    _last_request = time.monotonic()
    resp.raise_for_status()
    return resp


def discover_sep_meetings() -> list[tuple[date, str]]:
    """Return sorted (decision_date, html_url) pairs for all published SEPs.

    Discovers by scanning the FOMC calendar page for links matching the
    ``fomcprojtabl{YYYYMMDD}.htm`` pattern.
    """
    resp = _get(_CALENDAR_URL)
    results: list[tuple[date, str]] = []
    for date_str in _HTM_RE.findall(resp.text):
        try:
            d = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
            url = f"{_BASE}/monetarypolicy/fomcprojtabl{date_str}.htm"
            results.append((d, url))
        except ValueError:
            logger.warning("Unrecognised SEP date string: %s", date_str)
    return sorted(results)


def parse_sep_page(meeting_date: date, url: str) -> pd.DataFrame:
    """Parse the fed-funds-rate dot-plot table from a SEP HTML page.

    The Fed's accessible HTML (Figure 2) has a ``<table class="pubtables">``
    where rows are rate levels (``<th class="stub">``) and columns are
    forecast years (``<th class="colhead">``).  Cells with data have
    ``class="data"``; empty cells have ``class="emptydata"``.

    Returns columns: meeting_date, forecast_year, rate, participant_count.
    Rows with no participants are omitted.
    """
    resp = _get(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    heading = soup.find("h4", {"id": "xt6"})
    if heading is None:
        raise ValueError(f"Figure 2 heading not found: {url}")
    table = heading.find_next("table", class_="pubtables")
    if table is None:
        raise ValueError(f"dot-plot table not found after Figure 2: {url}")

    # First colhead is the "Midpoint..." row label — skip it.
    header_cells = table.thead.find_all("th", class_="colhead")
    forecast_years = [th.get_text(strip=True) for th in header_cells[1:]]

    rows: list[dict] = []
    for tr in table.tbody.find_all("tr"):
        stub = tr.find("th", class_="stub")
        if stub is None:
            continue
        try:
            rate = float(stub.get_text(strip=True))
        except ValueError:
            continue
        for year, td in zip(forecast_years, tr.find_all("td")):
            if "emptydata" in (td.get("class") or []):
                continue
            text = td.get_text(strip=True)
            if not text or text == "\xa0":
                continue
            try:
                rows.append({
                    "meeting_date": meeting_date.isoformat(),
                    "forecast_year": year,
                    "rate": rate,
                    "participant_count": int(text),
                })
            except ValueError:
                pass

    df = pd.DataFrame(rows, columns=["meeting_date", "forecast_year", "rate", "participant_count"])
    return df


class SepClient:
    """Incremental refresh of FOMC SEP dot-plot data."""

    def _already_fetched(self) -> set[str]:
        existing = load_parquet(SEP_DOTS_PATH)
        if existing is None or existing.empty:
            return set()
        return set(existing["meeting_date"].unique())

    def refresh_all(self, *, force: bool = False) -> pd.DataFrame:
        """Fetch all available SEP meetings and upsert into sep_dots.parquet.

        With ``force=False`` (default) meetings already present in the parquet
        are skipped.  Returns the full merged DataFrame.
        """
        ensure_dirs()
        meetings = discover_sep_meetings()
        if not meetings:
            logger.warning("No SEP meetings discovered — calendar page may have changed.")
            return pd.DataFrame()

        fetched = self._already_fetched()
        new_frames: list[pd.DataFrame] = []
        n_skipped = 0

        for meeting_date, url in meetings:
            if not force and meeting_date.isoformat() in fetched:
                n_skipped += 1
                continue
            try:
                df = parse_sep_page(meeting_date, url)
                if df.empty:
                    logger.warning("Empty parse for SEP %s", meeting_date.isoformat())
                else:
                    new_frames.append(df)
                    logger.info("SEP fetched: %s (%d rows)", meeting_date.isoformat(), len(df))
            except Exception as exc:
                logger.error("SEP fetch failed for %s: %s", meeting_date.isoformat(), exc)

        logger.info("SEP: %d new, %d skipped (already stored)", len(new_frames), n_skipped)

        existing = load_parquet(SEP_DOTS_PATH)
        all_frames = ([existing] if existing is not None and not existing.empty else []) + new_frames
        if not all_frames:
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["meeting_date", "forecast_year", "rate"])
        combined["participant_count"] = combined["participant_count"].astype("int32")
        combined = combined.sort_values(["meeting_date", "forecast_year", "rate"]).reset_index(drop=True)

        save_parquet_atomic(combined, SEP_DOTS_PATH)
        logger.info(
            "SEP stored: %d rows, %d meetings",
            len(combined),
            combined["meeting_date"].nunique(),
        )
        return combined
