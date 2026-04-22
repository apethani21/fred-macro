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
    args = parser.parse_args()

    from src.research.harvest import harvest_all

    sources = [args.source] if args.source else None
    n = harvest_all(dry_run=args.dry_run, max_per_source=args.max_per_source, sources=sources)
    print(f"\nHarvest complete. {n} new finding(s) {'would be ' if args.dry_run else ''}written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
