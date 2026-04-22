"""Series discovery policies.

The tracked universe is derived from policies, not a hand-picked list:

  1. Walk the FRED category tree; for each category, take the top-N most
     popular series.
  2. For each release in a curated major-releases list, take all its series.
  3. (Later) reference-following from knowledge files.

`discover()` runs all configured policies and returns a DataFrame with one
row per (series_id, source) pair, so we keep provenance for auditing the
universe later.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .fred_client import FredClient

logger = logging.getLogger(__name__)

ROOT_CATEGORY_ID = 0

# Names match the release-name string returned by /fred/releases. The IDs are
# resolved at runtime by fuzzy match so they don't rot if FRED renumbers.
MAJOR_RELEASE_NAMES: tuple[str, ...] = (
    "H.4.1",                           # Factors Affecting Reserve Balances
    "H.6",                             # Money Stock Measures
    "H.8",                             # Assets and Liabilities of Commercial Banks
    "H.15",                            # Selected Interest Rates
    "Employment Situation",
    "Consumer Price Index",
    "Personal Income and Outlays",     # PCE
    "Gross Domestic Product",
    "Industrial Production",
    "Advance Monthly Sales for Retail",
)


@dataclass(frozen=True)
class DiscoveryConfig:
    top_n_per_category: int = 10
    max_category_depth: int = 3
    max_series_per_release: int = 200
    include_categories: bool = True
    include_releases: bool = True


def _walk_categories(
    client: FredClient, root_id: int, max_depth: int
) -> list[tuple[int, str, int]]:
    """BFS the category tree from `root_id`. Returns (category_id, name, depth)."""
    seen: set[int] = set()
    out: list[tuple[int, str, int]] = []
    # Seed with children of root — root itself has no series.
    frontier: list[tuple[int, int]] = [(root_id, 0)]
    while frontier:
        import time as _time
        cid, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        _time.sleep(0.6)  # stay well under FRED's rate limit during BFS
        children = client.category_children(cid)
        for child in children:
            child_id = int(child["id"])
            if child_id in seen:
                continue
            seen.add(child_id)
            out.append((child_id, child.get("name", ""), depth + 1))
            frontier.append((child_id, depth + 1))
    return out


def discover_by_category(
    client: FredClient, top_n: int, max_depth: int
) -> pd.DataFrame:
    """Top-N popular series per category, BFS from root to `max_depth`."""
    cats = _walk_categories(client, ROOT_CATEGORY_ID, max_depth)
    logger.info("Discovered %d categories (max_depth=%d)", len(cats), max_depth)
    rows: list[dict] = []
    for cid, cname, _depth in cats:
        try:
            series = client.category_series(cid, limit=top_n)
        except Exception as e:
            logger.warning("category_series failed for %s (%s): %s", cid, cname, e)
            continue
        for s in series:
            rows.append(
                {
                    "series_id": s.get("id"),
                    "title": s.get("title"),
                    "frequency_short": s.get("frequency_short"),
                    "popularity": s.get("popularity"),
                    "source_type": "category",
                    "source_id": str(cid),
                    "source_name": cname,
                }
            )
    return pd.DataFrame(rows)


def _resolve_release_ids(
    client: FredClient, wanted_names: Iterable[str]
) -> list[tuple[int, str]]:
    """Map each wanted substring to the best-matching release_id from /fred/releases."""
    releases = client.releases()
    wanted_norm = [(w, w.lower()) for w in wanted_names]
    resolved: list[tuple[int, str]] = []
    for w_orig, w_lower in wanted_norm:
        matches = [r for r in releases if w_lower in r.get("name", "").lower()]
        if not matches:
            logger.warning("No release found for %r", w_orig)
            continue
        best = matches[0]  # /releases is already name-ordered; first match is fine for now
        resolved.append((int(best["id"]), best["name"]))
    return resolved


def discover_by_releases(
    client: FredClient,
    release_names: Iterable[str] = MAJOR_RELEASE_NAMES,
    max_series_per_release: int = 200,
) -> pd.DataFrame:
    releases = _resolve_release_ids(client, release_names)
    logger.info("Discovered %d major releases", len(releases))
    rows: list[dict] = []
    for rid, rname in releases:
        try:
            series = client.release_series(rid, limit=max_series_per_release)
        except Exception as e:
            logger.warning("release_series failed for %s (%s): %s", rid, rname, e)
            continue
        for s in series:
            rows.append(
                {
                    "series_id": s.get("id"),
                    "title": s.get("title"),
                    "frequency_short": s.get("frequency_short"),
                    "popularity": s.get("popularity"),
                    "source_type": "release",
                    "source_id": str(rid),
                    "source_name": rname,
                }
            )
    return pd.DataFrame(rows)


def discover(client: FredClient, config: DiscoveryConfig | None = None) -> pd.DataFrame:
    """Run all configured policies; return a long-format provenance DataFrame.

    Rows are *not* deduped — a series discovered via two categories and one
    release appears three times so we keep full provenance. Downstream code
    dedupes on `series_id` when it just needs the universe list.
    """
    config = config or DiscoveryConfig()
    frames: list[pd.DataFrame] = []
    if config.include_categories:
        frames.append(
            discover_by_category(
                client, top_n=config.top_n_per_category, max_depth=config.max_category_depth
            )
        )
    if config.include_releases:
        frames.append(
            discover_by_releases(
                client, max_series_per_release=config.max_series_per_release
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=["series_id", "title", "frequency_short", "popularity", "source_type", "source_id", "source_name"]
        )
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["series_id"]).reset_index(drop=True)
    return df


def universe_by_frequency(discovery_df: pd.DataFrame) -> dict[str, list[str]]:
    """Collapse a discovery DataFrame to {frequency_short: [series_id, ...]}.

    Deduplicates on series_id, keeping the most popular instance's frequency
    (which should be identical across rows anyway — FRED frequency is a
    property of the series, not the source).
    """
    if discovery_df.empty:
        return {}
    deduped = (
        discovery_df.sort_values("popularity", ascending=False, na_position="last")
        .drop_duplicates("series_id", keep="first")
    )
    out: dict[str, list[str]] = {}
    for freq, group in deduped.groupby("frequency_short"):
        out[freq] = group["series_id"].tolist()
    return out
