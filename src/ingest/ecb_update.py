"""Incremental refresh of ECB SDW series into the shared partitioned parquet files.

Algorithm mirrors update.py (FRED):
  1. For each raw (non-derived) ECB series, find its last stored date.
  2. Fetch from ECB SDW starting buffer days before that date.
  3. Merge with existing, preferring newer values (handles revisions).
  4. Write each affected partition atomically.
  5. Compute derived series (BTP-Bund spread, Bund slope) from the merged data.
  6. Upsert metadata.parquet and update_log.parquet.

Revision buffers match the FRED convention: 90 days for daily series,
730 days for monthly/quarterly.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .ecb_client import EcbClient, EcbAPIError
from .ecb_registry import EcbSeriesSpec, get_derived_series, get_raw_series
from .paths import METADATA_PATH, UPDATE_LOG_PATH, series_path
from .storage import load_parquet, save_parquet_atomic
from .update import _merge_with_revisions, _value_hash

logger = logging.getLogger(__name__)

BUFFER_DAYS: dict[str, int] = {
    "daily": 90,
    "weekly": 90,
    "monthly": 730,
    "quarterly": 730,
    "annual": 730,
}


@dataclass
class EcbSeriesUpdateResult:
    series_id: str
    partition: str
    rows_added: int
    rows_changed: int
    last_observation_date: date | None
    error: str | None = None


@dataclass
class EcbRefreshSummary:
    per_series: list[EcbSeriesUpdateResult] = field(default_factory=list)

    @property
    def errors(self) -> list[EcbSeriesUpdateResult]:
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
    """Convert ECB SDW observation records to long-format DataFrame.

    Dates are already normalized to YYYY-MM-DD by the client.
    """
    if not observations:
        return pd.DataFrame(columns=["series_id", "date", "value"])
    df = pd.DataFrame(observations)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["series_id"] = series_id
    return df[["series_id", "date", "value"]].copy()


def _start_period_for_fetch(
    existing_for_series: pd.DataFrame | None, spec: EcbSeriesSpec
) -> str | None:
    """Return the ECB startPeriod query param, or None for full history."""
    if existing_for_series is None or existing_for_series.empty:
        return None
    last: date = max(existing_for_series["date"])
    buffer = timedelta(days=BUFFER_DAYS[spec.partition])
    start = last - buffer

    # Format start date per ECB SDMX period notation.
    if spec.frequency_short == "D":
        return start.isoformat()
    if spec.frequency_short == "M":
        return start.strftime("%Y-%m")
    if spec.frequency_short == "Q":
        q = (start.month - 1) // 3 + 1
        return f"{start.year}-Q{q}"
    return start.isoformat()


def _compute_derived(
    spec: EcbSeriesSpec,
    all_data: dict[str, pd.DataFrame],
) -> pd.DataFrame | None:
    """Compute a derived series as a linear combination of raw series.

    Returns a DataFrame in long format, or None if any component is missing.
    """
    frames: list[pd.DataFrame] = []
    signs: list[int] = []
    for src_id, sign in spec.derived_from:
        if src_id not in all_data or all_data[src_id].empty:
            logger.warning("Derived %s: component %s not available", spec.series_id, src_id)
            return None
        frames.append(all_data[src_id].set_index("date")["value"])
        signs.append(sign)

    # Align on common dates, drop any row with a NaN component.
    combined = pd.concat(frames, axis=1, keys=[f"c{i}" for i in range(len(frames))])
    combined = combined.dropna()
    if combined.empty:
        return None

    result = sum(combined[f"c{i}"] * s for i, s in enumerate(signs))
    out = result.reset_index()
    out.columns = ["date", "value"]
    out["series_id"] = spec.series_id
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out[["series_id", "date", "value"]].copy()


def _build_metadata_row(spec: EcbSeriesSpec, last_obs: date | None, now_utc: datetime) -> dict:
    return {
        "series_id": spec.series_id,
        "title": spec.title,
        "units": spec.units,
        "units_short": spec.units_short,
        "frequency": {"D": "Daily", "M": "Monthly", "Q": "Quarterly"}.get(
            spec.frequency_short, spec.frequency_short
        ),
        "frequency_short": spec.frequency_short,
        "seasonal_adjustment_short": "NSA",
        "observation_start": None,
        "observation_end": last_obs.isoformat() if last_obs else None,
        "last_updated": now_utc.isoformat(),
        "popularity": None,
        "notes": spec.notes,
        "partition": spec.partition,
        "source": "ECB",
        "last_refreshed": now_utc.isoformat(),
    }


def run_ecb_refresh(client: EcbClient | None = None) -> EcbRefreshSummary:
    """Fetch all raw ECB series, compute derived series, write to parquet.

    Returns a summary of results per series (including derived).
    """
    if client is None:
        client = EcbClient()

    now_utc = datetime.now(timezone.utc)
    summary = EcbRefreshSummary()

    raw_specs = get_raw_series()
    derived_specs = get_derived_series()
    all_partitions = sorted({s.partition for s in raw_specs + derived_specs})

    # Load existing data for every affected partition once.
    existing_by_partition: dict[str, pd.DataFrame | None] = {
        p: load_parquet(series_path(p)) for p in all_partitions
    }

    # --- Fetch raw series ---

    # Track newly-merged data per series for derived computation.
    merged_by_id: dict[str, pd.DataFrame] = {}

    # Accumulate rows to write per partition and for metadata/log.
    new_frames_by_partition: dict[str, list[pd.DataFrame]] = {p: [] for p in all_partitions}
    metadata_rows: list[dict] = []
    log_rows: list[dict] = []

    for spec in raw_specs:
        existing_all = existing_by_partition[spec.partition]
        existing_for_series = (
            existing_all[existing_all["series_id"] == spec.series_id].copy()
            if existing_all is not None and not existing_all.empty
            else None
        )

        try:
            start_period = _start_period_for_fetch(existing_for_series, spec)
            observations = client.series_observations(spec.flow, spec.key, start_period=start_period)
            incoming = _observations_to_df(spec.series_id, observations)

            merged, added, changed = _merge_with_revisions(existing_for_series, incoming)
            merged_by_id[spec.series_id] = merged
            new_frames_by_partition[spec.partition].append(merged)

            last_obs = max(merged["date"]) if not merged.empty else None
            summary.per_series.append(
                EcbSeriesUpdateResult(
                    series_id=spec.series_id,
                    partition=spec.partition,
                    rows_added=added,
                    rows_changed=changed,
                    last_observation_date=last_obs,
                )
            )
            metadata_rows.append(_build_metadata_row(spec, last_obs, now_utc))
            log_rows.append({
                "series_id": spec.series_id,
                "last_fetched_at": now_utc.isoformat(),
                "last_observation_date": last_obs.isoformat() if last_obs else None,
                "last_value_hash": _value_hash(merged),
            })

        except Exception as e:
            logger.exception("Failed to refresh ECB series %s", spec.series_id)
            summary.per_series.append(
                EcbSeriesUpdateResult(
                    series_id=spec.series_id,
                    partition=spec.partition,
                    rows_added=0,
                    rows_changed=0,
                    last_observation_date=None,
                    error=str(e),
                )
            )

    # --- Compute derived series ---

    # Populate merged_by_id with pre-existing data for any series not just fetched
    # (e.g., if only some raw series were refreshed, the others may still be in parquet).
    for spec in raw_specs:
        if spec.series_id not in merged_by_id:
            existing_all = existing_by_partition[spec.partition]
            if existing_all is not None and not existing_all.empty:
                rows = existing_all[existing_all["series_id"] == spec.series_id]
                if not rows.empty:
                    merged_by_id[spec.series_id] = rows.copy()

    for spec in derived_specs:
        try:
            derived_df = _compute_derived(spec, merged_by_id)
            if derived_df is None or derived_df.empty:
                logger.warning("Derived series %s produced no data", spec.series_id)
                continue

            # Merge derived with any existing data in the partition.
            existing_all = existing_by_partition[spec.partition]
            existing_for_series = (
                existing_all[existing_all["series_id"] == spec.series_id].copy()
                if existing_all is not None and not existing_all.empty
                else None
            )
            merged, added, changed = _merge_with_revisions(existing_for_series, derived_df)
            new_frames_by_partition[spec.partition].append(merged)

            last_obs = max(merged["date"]) if not merged.empty else None
            summary.per_series.append(
                EcbSeriesUpdateResult(
                    series_id=spec.series_id,
                    partition=spec.partition,
                    rows_added=added,
                    rows_changed=changed,
                    last_observation_date=last_obs,
                )
            )
            metadata_rows.append(_build_metadata_row(spec, last_obs, now_utc))
            log_rows.append({
                "series_id": spec.series_id,
                "last_fetched_at": now_utc.isoformat(),
                "last_observation_date": last_obs.isoformat() if last_obs else None,
                "last_value_hash": _value_hash(merged),
            })

        except Exception as e:
            logger.exception("Failed to compute derived ECB series %s", spec.series_id)
            summary.per_series.append(
                EcbSeriesUpdateResult(
                    series_id=spec.series_id,
                    partition=spec.partition,
                    rows_added=0,
                    rows_changed=0,
                    last_observation_date=None,
                    error=str(e),
                )
            )

    # --- Write partitions ---

    all_ecb_ids = {s.series_id for s in raw_specs + derived_specs}

    for partition in all_partitions:
        new_frames = new_frames_by_partition[partition]
        if not new_frames:
            continue

        existing_all = existing_by_partition[partition]
        # Carry forward non-ECB rows unchanged.
        untouched = (
            existing_all[~existing_all["series_id"].isin(all_ecb_ids)]
            if existing_all is not None and not existing_all.empty
            else None
        )

        frames = []
        if untouched is not None and not untouched.empty:
            frames.append(untouched)
        frames.extend(new_frames)

        out = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["series_id", "date"])
            .reset_index(drop=True)
        )
        save_parquet_atomic(out, series_path(partition))
        logger.info("Wrote %d rows to %s.parquet", len(out), partition)

    # --- Upsert metadata and update_log ---

    if metadata_rows:
        new_meta = pd.DataFrame(metadata_rows)
        meta = load_parquet(METADATA_PATH)
        if meta is not None:
            meta = meta[~meta["series_id"].isin(new_meta["series_id"])]
            new_meta = pd.concat([meta, new_meta], ignore_index=True)
        save_parquet_atomic(
            new_meta.sort_values("series_id").reset_index(drop=True), METADATA_PATH
        )

    if log_rows:
        new_log = pd.DataFrame(log_rows)
        update_log = load_parquet(UPDATE_LOG_PATH)
        if update_log is not None:
            update_log = update_log[~update_log["series_id"].isin(new_log["series_id"])]
            new_log = pd.concat([update_log, new_log], ignore_index=True)
        save_parquet_atomic(
            new_log.sort_values("series_id").reset_index(drop=True), UPDATE_LOG_PATH
        )

    return summary
