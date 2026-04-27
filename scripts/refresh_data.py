#!/usr/bin/env python3
"""Refresh FRED and ECB series data.

Common modes:
  python scripts/refresh_data.py --series DGS10 DGS2 --partition daily
      Explicit list (dev / debug).

  python scripts/refresh_data.py --discover
      Re-run discovery policies, save the universe, refresh every series in it.

  python scripts/refresh_data.py
      Incremental refresh using the previously-discovered universe in
      data/metadata.parquet. No-op if discovery has never run.

  python scripts/refresh_data.py --ecb
      Refresh ECB SDW series (deposit rate, HICP, yield curve, M3, wages, etc.).
      Combinable with the FRED modes above.

  python scripts/refresh_data.py --release-calendar
      Refresh data/release_calendar.parquet and data/release_series.parquet.
      Combinable with the modes above.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ingest.discovery import DiscoveryConfig, discover, universe_by_frequency
from src.ingest.ecb_client import EcbClient
from src.ingest.ecb_release_calendar import refresh_cb_calendar
from src.ingest.sep_client import SepClient
from src.ingest.ecb_update import run_ecb_refresh
from src.ingest.fred_client import FredClient
from src.ingest.paths import DISCOVERY_PATH, METADATA_PATH
from src.ingest.release_calendar import refresh_release_calendar, refresh_release_series
from src.ingest.storage import load_parquet, save_parquet_atomic
from src.ingest.update import refresh_series, refresh_universe
from src.monitor.health import build_health_snapshot
from src.monitor.run_log import RunLogger


def _universe_from_metadata() -> dict[str, list[str]]:
    meta = load_parquet(METADATA_PATH)
    if meta is None or meta.empty:
        return {}
    # Exclude ECB series — the FRED client can't fetch them.
    if "source" in meta.columns:
        meta = meta[meta["source"].isna() | (meta["source"] == "FRED")]
    out: dict[str, list[str]] = {}
    for freq, group in meta.dropna(subset=["frequency_short"]).groupby("frequency_short"):
        out[freq] = group["series_id"].tolist()
    return out


def _run_discovery(client: FredClient, args) -> dict[str, list[str]]:
    cfg = DiscoveryConfig(
        top_n_per_category=args.top_n,
        max_category_depth=args.max_depth,
        max_series_per_release=args.max_per_release,
    )
    df = discover(client, cfg)
    save_parquet_atomic(df, DISCOVERY_PATH)
    uni = universe_by_frequency(df)
    total = sum(len(v) for v in uni.values())
    print(
        f"Discovery: {len(df)} provenance rows, {total} unique series across "
        f"{len(uni)} frequencies."
    )
    for freq, ids in sorted(uni.items()):
        print(f"  {freq}: {len(ids)} series")
    return uni


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--series", nargs="+", help="Explicit series IDs (dev mode).")
    parser.add_argument(
        "--partition",
        default="daily",
        choices=["daily", "weekly", "monthly", "quarterly", "annual"],
        help="Partition for --series.",
    )
    parser.add_argument("--discover", action="store_true", help="Re-run discovery before refresh.")
    parser.add_argument("--top-n", type=int, default=10, help="Top-N series per category (with --discover).")
    parser.add_argument("--max-depth", type=int, default=3, help="Max category tree depth (with --discover).")
    parser.add_argument("--max-per-release", type=int, default=200, help="Max series per release (with --discover).")
    parser.add_argument("--ecb", action="store_true", help="Refresh ECB SDW series (deposit rate, HICP, yield curve, M3, wages, etc.).")
    parser.add_argument("--release-calendar", action="store_true", help="Also refresh the release calendar.")
    parser.add_argument("--sep", action="store_true", help="Refresh FOMC SEP dot-plot data (sep_dots.parquet).")
    parser.add_argument("--skip-refresh", action="store_true", help="Run discovery/calendar but don't refresh series data.")
    args = parser.parse_args()

    exit_code = 0
    with RunLogger("refresh_data") as run:
        client = FredClient()

        # --- release calendar (cheap; do first so it's independent of refresh success) ---
        if args.release_calendar:
            cal = refresh_release_calendar(client)
            release_ids = sorted({int(r) for r in cal["release_id"].unique()}) if not cal.empty else []
            if release_ids:
                refresh_release_series(client, release_ids)
            run.set("calendar_updated", True)
            run.set("release_count", len(release_ids))

        # Always merge static G7+ central bank calendar dates (fast, no network call).
        refresh_cb_calendar()

        # --- determine universe ---
        if args.series:
            universe = {args.partition: list(args.series)}
        elif args.discover:
            universe = _run_discovery(client, args)
        else:
            universe = _universe_from_metadata()
            if not universe:
                print("No universe found. Run with --discover first, or pass --series explicitly.")
                exit_code = 1 if not args.release_calendar else 0
                run.set("exit_code", exit_code)
                build_health_snapshot()
                return exit_code

        if args.skip_refresh:
            build_health_snapshot()
            return 0

        # --- refresh ---
        if args.series:
            summary = refresh_series(args.series, client, partition=args.partition)
            print(f"Refreshed {len(summary.per_series)} series [{args.partition}]:")
            print(summary.report())
            errs = len(summary.errors)
            run.set("series_ok", len(summary.per_series) - errs)
            run.set("series_errors", errs)
            exit_code = 1 if summary.errors else 0
        else:
            results = refresh_universe(universe, client)
            total_errors = 0
            total_ok = 0
            for partition, summary in results.items():
                errs = len(summary.errors)
                ok = len(summary.per_series) - errs
                total_errors += errs
                total_ok += ok
                print(f"[{partition}] {ok} ok, {errs} errors")
                if errs:
                    for r in summary.errors:
                        print(f"    {r.series_id}: {r.error}")
            run.set("series_ok", total_ok)
            run.set("series_errors", total_errors)
            run.set("partitions", len(results))
            exit_code = 1 if total_errors else 0

        # --- ECB refresh ---
        if args.ecb:
            ecb_client = EcbClient()
            ecb_summary = run_ecb_refresh(ecb_client)
            ecb_errors = len(ecb_summary.errors)
            ecb_ok = len(ecb_summary.per_series) - ecb_errors
            print(f"[ECB] {ecb_ok} ok, {ecb_errors} errors")
            print(ecb_summary.report())
            run.set("ecb_series_ok", ecb_ok)
            run.set("ecb_series_errors", ecb_errors)
            if ecb_errors:
                exit_code = 1

        # --- SEP dot-plot refresh ---
        if args.sep:
            sep_client = SepClient()
            sep_df = sep_client.refresh_all()
            sep_meetings = sep_df["meeting_date"].nunique() if not sep_df.empty else 0
            print(f"[SEP] {sep_meetings} meetings stored")
            run.set("sep_meetings", sep_meetings)

        run.set("exit_code", exit_code)

    build_health_snapshot()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
