"""Incremental refresh of FRED series into partitioned parquet files.

Algorithm per frequency partition:
  1. Load existing `data/series/{partition}.parquet` (if any).
  2. For each tracked series in that partition, find the most recent stored date.
  3. Re-fetch from FRED starting `buffer` days before that date — this catches revisions.
     If no stored data, fetch full history.
  4. Concat old + new, drop duplicates on (series_id, date) keeping the newer row.
  5. Write atomically.
  6. Update metadata and update_log.

Revision buffer: 90 days for daily/weekly, ~2 years for lower frequencies.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .fred_client import FredClient
from .paths import (
    FREQ_TO_PARTITION,
    METADATA_PATH,
    UPDATE_LOG_PATH,
    ensure_dirs,
    series_path,
)
from .storage import load_parquet, save_parquet_atomic

logger = logging.getLogger(__name__)


def partition_for_frequency(frequency_short: str) -> str:
    """Map a FRED `frequency_short` code to a storage partition name."""
    if frequency_short not in FREQ_TO_PARTITION:
        raise ValueError(f"Unknown FRED frequency_short: {frequency_short!r}")
    return FREQ_TO_PARTITION[frequency_short]


def refresh_universe(
    universe_by_freq: dict[str, list[str]],
    client: FredClient,
) -> dict[str, "RefreshSummary"]:
    """Refresh every series in `universe_by_freq`, grouped by storage partition.

    `universe_by_freq` maps a FRED frequency_short code to the series that have
    that frequency. Series with unknown frequency codes are skipped with a warning.
    """
    # Group by partition, since multiple frequencies can map to the same file
    # (e.g., weekly + bi-weekly both land in weekly.parquet).
    by_partition: dict[str, list[str]] = {}
    skipped: list[tuple[str, list[str]]] = []
    for freq, series_ids in universe_by_freq.items():
        try:
            partition = partition_for_frequency(freq)
        except ValueError:
            skipped.append((freq, series_ids))
            continue
        by_partition.setdefault(partition, []).extend(series_ids)

    if skipped:
        for freq, ids in skipped:
            logger.warning(
                "Skipping %d series with unsupported frequency %r: %s",
                len(ids), freq, ids[:5],
            )

    results: dict[str, RefreshSummary] = {}
    for partition, ids in by_partition.items():
        # Dedupe — a series can appear more than once if two frequencies collapse
        # to the same partition.
        unique_ids = list(dict.fromkeys(ids))
        logger.info("Refreshing %d series in partition %s", len(unique_ids), partition)
        results[partition] = refresh_series(unique_ids, client, partition=partition)
    return results

# Revision buffers by partition. Revisions to monthly/quarterly series can
# reach back multiple years (e.g., annual GDP benchmark revisions).
BUFFER_DAYS: dict[str, int] = {
    "daily": 90,
    "weekly": 90,
    "monthly": 730,
    "quarterly": 730,
    "annual": 730,
}


@dataclass
class SeriesUpdateResult:
    series_id: str
    partition: str
    rows_added: int
    rows_changed: int
    last_observation_date: date | None
    error: str | None = None


@dataclass
class RefreshSummary:
    per_series: list[SeriesUpdateResult] = field(default_factory=list)

    @property
    def errors(self) -> list[SeriesUpdateResult]:
        return [r for r in self.per_series if r.error]

    def report(self) -> str:
        lines = []
        for r in self.per_series:
            if r.error:
                lines.append(f"  {r.series_id} [{r.partition}]: ERROR — {r.error}")
            else:
                lines.append(
                    f"  {r.series_id} [{r.partition}]: +{r.rows_added} rows, "
                    f"{r.rows_changed} revisions, last={r.last_observation_date}"
                )
        return "\n".join(lines) or "  (nothing refreshed)"


def _observations_to_df(series_id: str, observations: list[dict]) -> pd.DataFrame:
    """Turn raw FRED observations into a clean long-format DataFrame.

    FRED uses '.' for missing values. We preserve the row so the gap is visible
    in the stored history, but the value is NaN.
    """
    if not observations:
        return pd.DataFrame(columns=["series_id", "date", "value"])
    df = pd.DataFrame(observations)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["value"] = pd.to_numeric(df["value"].where(df["value"] != ".", other=pd.NA), errors="coerce")
    df["series_id"] = series_id
    return df[["series_id", "date", "value"]].copy()


def _value_hash(df_for_series: pd.DataFrame) -> str:
    """Stable hash of a series' (date, value) contents for change detection."""
    # Sort to ensure deterministic hash regardless of row order.
    ordered = df_for_series.sort_values("date")
    payload = (
        ordered["date"].astype(str).str.cat(ordered["value"].astype(str), sep=",").str.cat(sep="\n")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _merge_with_revisions(
    existing: pd.DataFrame | None,
    incoming: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    """Combine existing + incoming for a single series.

    Returns (merged_df_for_series, rows_added, rows_changed).

    `incoming` wins on conflict because FRED's default observations endpoint
    returns the latest vintage.
    """
    if existing is None or existing.empty:
        return incoming.copy(), len(incoming), 0

    incoming_dates = set(incoming["date"])
    existing_for_series = existing
    existing_dates_values = dict(zip(existing_for_series["date"], existing_for_series["value"]))

    rows_added = 0
    rows_changed = 0
    for d, v in zip(incoming["date"], incoming["value"]):
        if d not in existing_dates_values:
            rows_added += 1
        else:
            old = existing_dates_values[d]
            # Both NaN counts as no change.
            both_nan = pd.isna(old) and pd.isna(v)
            if not both_nan and old != v:
                rows_changed += 1

    # Prefer incoming rows on overlap: append incoming after existing-minus-overlap
    existing_kept = existing_for_series[~existing_for_series["date"].isin(incoming_dates)]
    merged = pd.concat([existing_kept, incoming], ignore_index=True)
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged, rows_added, rows_changed


def _start_date_for_fetch(
    existing_for_series: pd.DataFrame | None, partition: str
) -> str | None:
    """Return the FRED observation_start query param, or None for full history."""
    if existing_for_series is None or existing_for_series.empty:
        return None
    last = max(existing_for_series["date"])
    buffer = timedelta(days=BUFFER_DAYS[partition])
    return (last - buffer).isoformat()


def refresh_series(
    series_ids: Iterable[str],
    client: FredClient,
    partition: str = "daily",
) -> RefreshSummary:
    """Refresh `series_ids`, all of which must belong to `partition`."""
    ensure_dirs()
    target_path = series_path(partition)
    existing_all = load_parquet(target_path)
    metadata = load_parquet(METADATA_PATH)
    update_log = load_parquet(UPDATE_LOG_PATH)

    now_utc = datetime.now(timezone.utc)
    summary = RefreshSummary()

    # Accumulate changes for a single bulk write per partition.
    new_series_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict] = []
    log_rows: list[dict] = []

    # Carry forward any other series already stored in the partition.
    if existing_all is not None:
        untouched = existing_all[~existing_all["series_id"].isin(list(series_ids))]
    else:
        untouched = None

    for sid in series_ids:
        try:
            existing_for_series = (
                existing_all[existing_all["series_id"] == sid].copy()
                if existing_all is not None
                else None
            )
            start = _start_date_for_fetch(existing_for_series, partition)
            observations = client.series_observations(sid, observation_start=start)
            incoming = _observations_to_df(sid, observations)

            merged, added, changed = _merge_with_revisions(existing_for_series, incoming)
            new_series_frames.append(merged)

            last_obs = max(merged["date"]) if not merged.empty else None
            summary.per_series.append(
                SeriesUpdateResult(
                    series_id=sid,
                    partition=partition,
                    rows_added=added,
                    rows_changed=changed,
                    last_observation_date=last_obs,
                )
            )

            meta = client.series(sid)
            metadata_rows.append(
                {
                    "series_id": sid,
                    "title": meta.get("title"),
                    "units": meta.get("units"),
                    "units_short": meta.get("units_short"),
                    "frequency": meta.get("frequency"),
                    "frequency_short": meta.get("frequency_short"),
                    "seasonal_adjustment_short": meta.get("seasonal_adjustment_short"),
                    "observation_start": meta.get("observation_start"),
                    "observation_end": meta.get("observation_end"),
                    "last_updated": meta.get("last_updated"),
                    "popularity": meta.get("popularity"),
                    "notes": meta.get("notes"),
                    "partition": partition,
                    "last_refreshed": now_utc.isoformat(),
                }
            )

            log_rows.append(
                {
                    "series_id": sid,
                    "last_fetched_at": now_utc.isoformat(),
                    "last_observation_date": last_obs.isoformat() if last_obs else None,
                    "last_value_hash": _value_hash(merged),
                }
            )
        except Exception as e:  # log and continue; one bad series shouldn't stop the batch
            logger.exception("Failed to refresh %s", sid)
            summary.per_series.append(
                SeriesUpdateResult(
                    series_id=sid,
                    partition=partition,
                    rows_added=0,
                    rows_changed=0,
                    last_observation_date=None,
                    error=str(e),
                )
            )

    # Assemble and write the partition file.
    frames = []
    if untouched is not None and not untouched.empty:
        frames.append(untouched)
    frames.extend(new_series_frames)
    if frames:
        out = pd.concat(frames, ignore_index=True).sort_values(["series_id", "date"]).reset_index(drop=True)
        save_parquet_atomic(out, target_path)

    # Upsert metadata and update_log.
    if metadata_rows:
        new_meta = pd.DataFrame(metadata_rows)
        if metadata is not None:
            metadata = metadata[~metadata["series_id"].isin(new_meta["series_id"])]
            new_meta = pd.concat([metadata, new_meta], ignore_index=True)
        save_parquet_atomic(new_meta.sort_values("series_id").reset_index(drop=True), METADATA_PATH)

    if log_rows:
        new_log = pd.DataFrame(log_rows)
        if update_log is not None:
            update_log = update_log[~update_log["series_id"].isin(new_log["series_id"])]
            new_log = pd.concat([update_log, new_log], ignore_index=True)
        save_parquet_atomic(new_log.sort_values("series_id").reset_index(drop=True), UPDATE_LOG_PATH)

    return summary
