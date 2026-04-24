"""ECB Statistical Data Warehouse (SDW) client.

Wraps the ECB SDMX 2.1 REST API at data-api.ecb.europa.eu.
No authentication required. Rate-limited to 1 req/sec.

Response format: SDMX-JSON (format=jsondata). The structure is:
  data["structure"]["dimensions"]["observation"][0]["values"]
      → list of {"id": "2026-04-23", ...} (the time periods)
  data["dataSets"][0]["series"][series_key]["observations"]
      → {"0": [value, ...], "1": [value, ...], ...}
      where the integer index maps into the time period list.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

ECB_BASE_URL = "https://data-api.ecb.europa.eu/service"

# ECB has no published hard rate limit, but 1 req/sec is safe and polite.
DEFAULT_MIN_INTERVAL_S = 1.0

RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 5


class EcbAPIError(RuntimeError):
    """Raised when ECB SDW returns an error we can't recover from."""


class EcbClient:
    def __init__(
        self,
        base_url: str = ECB_BASE_URL,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.min_interval_s = min_interval_s
        self.session = session or requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self._last_request_time = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_request_time = time.monotonic()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        p = {"format": "jsondata", **(params or {})}

        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                resp = self.session.get(url, params=p, timeout=60)
            except requests.RequestException as e:
                wait = 2 ** attempt
                logger.warning("ECB request error on %s (%s); retrying in %ds", path, e, wait)
                time.sleep(wait)
                continue

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 404:
                raise EcbAPIError(f"ECB 404 — unknown series key: {path}")

            if resp.status_code in RETRY_STATUS and attempt < MAX_RETRIES - 1:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else 2 ** attempt
                logger.warning(
                    "ECB %s on %s; retry %d/%d in %.1fs",
                    resp.status_code, path, attempt + 1, MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue

            raise EcbAPIError(f"ECB {resp.status_code} on {path}: {resp.text[:500]}")

        raise EcbAPIError(f"ECB request failed after {MAX_RETRIES} retries: {path}")

    # ---- public API ----

    def series_observations(
        self,
        flow: str,
        key: str,
        start_period: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch observations for one ECB series.

        Args:
            flow: ECB dataflow code, e.g. "FM", "ICP", "EST".
            key: SDMX series key, e.g. "B.U2.EUR.4F.KR.DFR.LEV".
            start_period: ISO period string (YYYY-MM-DD / YYYY-MM / YYYY-Qn).

        Returns:
            List of {"date": "YYYY-MM-DD", "value": float | None}.
            Dates are always normalized to YYYY-MM-DD (first day of period).
        """
        path = f"data/{flow}/{key}"
        params: dict[str, Any] = {}
        if start_period:
            params["startPeriod"] = start_period
        data = self._get(path, params)
        return _parse_sdmx_observations(data)


# ---- SDMX parsing ----

def _normalize_period(period_id: str) -> str:
    """Normalize an ECB SDMX period ID to YYYY-MM-DD.

    Handles:
      "2026-04-23"  → "2026-04-23"  (daily, unchanged)
      "2026-01"     → "2026-01-01"  (monthly → first of month)
      "2025-Q4"     → "2025-10-01"  (quarterly → first day of quarter)
    """
    if len(period_id) == 10:  # YYYY-MM-DD
        return period_id
    if len(period_id) == 7 and period_id[4] == "-" and period_id[5] != "Q":
        return period_id + "-01"
    if "Q" in period_id:
        year, q = period_id.split("-Q")
        month = {"1": "01", "2": "04", "3": "07", "4": "10"}[q]
        return f"{year}-{month}-01"
    # fallback: return as-is and let pandas handle it
    return period_id


def _parse_sdmx_observations(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse SDMX-JSON response into a list of {date, value} dicts.

    The ECB SDMX-JSON format stores observations with integer string keys
    mapping to positions in the TIME_PERIOD dimension's values list.
    """
    try:
        time_periods = data["structure"]["dimensions"]["observation"][0]["values"]
        datasets = data["dataSets"]
        if not datasets:
            return []
        series_map = datasets[0]["series"]
        if not series_map:
            return []
        # There's exactly one series when a fully-specified key is fetched.
        series_obj = next(iter(series_map.values()))
        observations = series_obj.get("observations", {})
    except (KeyError, IndexError, StopIteration) as e:
        raise EcbAPIError(f"Unexpected SDMX-JSON structure: {e}") from e

    rows: list[dict[str, Any]] = []
    for idx_str, obs_array in observations.items():
        idx = int(idx_str)
        raw_value = obs_array[0] if obs_array else None
        value = float(raw_value) if raw_value is not None else None
        period_id = time_periods[idx]["id"]
        rows.append({"date": _normalize_period(period_id), "value": value})
    return rows
