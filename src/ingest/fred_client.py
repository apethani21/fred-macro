"""FRED API client.

Minimal wrapper around the endpoints this project needs. Handles auth,
rate limiting (120 req/min by default), and retries on transient errors.
"""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

_API_KEY_RE = re.compile(r"api_key=[^&\s\"']+")


def _redact(s: Any) -> str:
    return _API_KEY_RE.sub("api_key=***", str(s))

logger = logging.getLogger(__name__)

FRED_BASE_URL = "https://api.stlouisfed.org/fred"
DEFAULT_KEY_PATH = Path.home() / "keys" / "fred" / "key.txt"

# FRED allows 120 requests per minute. Leave headroom.
DEFAULT_MIN_INTERVAL_S = 60.0 / 110.0

RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 5


class FredAPIError(RuntimeError):
    """Raised when FRED returns an error response we can't recover from."""


def load_api_key(path: Path | str | None = None) -> str:
    p = Path(path) if path else DEFAULT_KEY_PATH
    key = p.read_text().strip()
    if not key:
        raise FredAPIError(f"FRED API key file is empty: {p}")
    return key


class FredClient:
    def __init__(
        self,
        api_key: str | None = None,
        key_path: Path | str | None = None,
        base_url: str = FRED_BASE_URL,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key or load_api_key(key_path)
        self.base_url = base_url.rstrip("/")
        self.min_interval_s = min_interval_s
        self.session = session or requests.Session()
        self._last_request_time = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_request_time = time.monotonic()

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        q = {"api_key": self.api_key, "file_type": "json", **params}

        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                resp = self.session.get(url, params=q, timeout=30)
            except requests.RequestException as e:
                wait = 2 ** attempt
                logger.warning(
                    "FRED request error on %s (%s); retrying in %ds",
                    path, _redact(e), wait,
                )
                time.sleep(wait)
                continue

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in RETRY_STATUS and attempt < MAX_RETRIES - 1:
                # FRED sometimes returns 429 with Retry-After
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else 2 ** attempt
                logger.warning(
                    "FRED %s on %s; retry %d/%d in %.1fs",
                    resp.status_code, path, attempt + 1, MAX_RETRIES, wait,
                )
                time.sleep(wait)
                continue

            raise FredAPIError(
                f"FRED {resp.status_code} on {path}: {_redact(resp.text[:500])}"
            )

        raise FredAPIError(f"FRED request failed after {MAX_RETRIES} retries: {path}")

    # ---- endpoints ----

    def series(self, series_id: str) -> dict[str, Any]:
        """Return metadata for one series."""
        data = self._get("series", {"series_id": series_id})
        seriess = data.get("seriess") or []
        if not seriess:
            raise FredAPIError(f"No metadata returned for series {series_id}")
        return seriess[0]

    def series_observations(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return raw observation records. Dates are 'YYYY-MM-DD' strings; value may be '.'."""
        params: dict[str, Any] = {"series_id": series_id}
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end
        data = self._get("series/observations", params)
        return data.get("observations", [])

    def category_children(self, category_id: int) -> list[dict[str, Any]]:
        data = self._get("category/children", {"category_id": category_id})
        return data.get("categories", [])

    def category_series(
        self,
        category_id: int,
        limit: int = 1000,
        order_by: str = "popularity",
        sort_order: str = "desc",
    ) -> list[dict[str, Any]]:
        data = self._get(
            "category/series",
            {
                "category_id": category_id,
                "limit": limit,
                "order_by": order_by,
                "sort_order": sort_order,
            },
        )
        return data.get("seriess", [])

    def releases(self) -> list[dict[str, Any]]:
        data = self._get("releases", {"limit": 1000})
        return data.get("releases", [])

    def release_series(self, release_id: int, limit: int = 1000) -> list[dict[str, Any]]:
        data = self._get(
            "release/series", {"release_id": release_id, "limit": limit}
        )
        return data.get("seriess", [])

    def release_dates(
        self,
        release_id: int | None = None,
        include_release_dates_with_no_data: bool = True,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
        sort_order: str = "asc",
    ) -> list[dict[str, Any]]:
        """All release dates matching the filter, paginating transparently.

        FRED caps `limit` at 1000, so we loop on `offset` until a page comes
        back shorter than the page size.
        """
        page_size = 1000
        offset = 0
        out: list[dict[str, Any]] = []
        base: dict[str, Any] = {
            "include_release_dates_with_no_data": str(include_release_dates_with_no_data).lower(),
            "limit": page_size,
            "sort_order": sort_order,
        }
        if realtime_start:
            base["realtime_start"] = realtime_start
        if realtime_end:
            base["realtime_end"] = realtime_end
        if release_id is not None:
            path = "release/dates"
            base["release_id"] = release_id
        else:
            path = "releases/dates"

        while True:
            params = {**base, "offset": offset}
            data = self._get(path, params)
            page = data.get("release_dates", [])
            out.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
        return out
