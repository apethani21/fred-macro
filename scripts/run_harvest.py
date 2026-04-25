#!/usr/bin/env python3
"""Entry point for the knowledge harvester.

Scrapes FEDS Notes, Liberty Street Economics, and SF Fed Economic Letter for
macro-relevant empirical findings and writes them to knowledge/findings.md.

Usage:
  python scripts/run_harvest.py
      Harvest all three sources, up to 10 relevant articles each.

  python scripts/run_harvest.py --dry-run
      Print extracted findings; don't write anything.

  python scripts/run_harvest.py --source feds_notes
      Harvest only one source. Choices: feds_notes, liberty_street, sf_fed

  python scripts/run_harvest.py --max 5
      Process at most 5 relevant articles per source.

  python scripts/run_harvest.py --seeds
      Enrich unused topic seeds with pre-fetched primary sources.

  python scripts/run_harvest.py --seeds --seed-id notable_move_level-DGS10
      Enrich only the seed whose ID contains this substring.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Harvest macro findings from Fed research sources.")
    parser.add_argument("--dry-run", action="store_true", help="Print findings; don't write.")
    parser.add_argument("--source", choices=["feds_notes", "liberty_street", "sf_fed"],
                        help="Harvest only this source.")
    parser.add_argument("--max", type=int, default=10, dest="max_per_source",
                        help="Max relevant articles to process per source (default: 10).")
    parser.add_argument("--seeds", action="store_true",
                        help="Enrich unused topic seeds with pre-fetched primary sources.")
    parser.add_argument("--seed-id", metavar="SUBSTRING",
                        help="With --seeds: enrich only seeds whose ID contains this substring.")
    parser.add_argument("--max-seeds", type=int, default=5,
                        help="Max seeds to enrich per run (default: 5).")
    args = parser.parse_args()

    from src.monitor.health import build_health_snapshot
    from src.monitor.run_log import RunLogger
    from src.research.harvest import harvest_all

    with RunLogger("run_harvest", dry_run=args.dry_run) as run:
        if args.seeds:
            from src.research.enrich import enrich_seeds
            run.set("mode", "seed_enrich")
            n = enrich_seeds(
                dry_run=args.dry_run,
                seed_id_filter=args.seed_id,
                max_seeds=args.max_seeds,
            )
            run.set("seeds_enriched", n)
            print(f"\nSeed enrichment complete. {n} seed(s) enriched.")
        else:
            sources = [args.source] if args.source else None
            run.set("sources", sources or ["feds_notes", "liberty_street", "sf_fed"])
            run.set("max_per_source", args.max_per_source)
            n = harvest_all(dry_run=args.dry_run, max_per_source=args.max_per_source, sources=sources)
            run.set("new_findings", n)
            print(f"\nHarvest complete. {n} new finding(s) {'would be ' if args.dry_run else ''}written.")

    build_health_snapshot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
