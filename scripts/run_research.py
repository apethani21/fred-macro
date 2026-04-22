#!/usr/bin/env python3
"""Run the research phase: detect findings, write knowledge/findings.md + data/stats/.

Common modes:
  python scripts/run_research.py
      Full scan over the curated core universe and relationship pairs.

  python scripts/run_research.py --pairs DGS10 UNRATE
      Scan a single ad-hoc pair (takes two series IDs; label auto-generated).

  python scripts/run_research.py --dry-run
      Print the scan report; do not write findings.md or stats parquet files.

  python scripts/run_research.py --overwrite
      Replace existing findings with the same slug (default: keep prior).

  python scripts/run_research.py --enrich
      Run per-finding web research to populate Sources blocks (requires Anthropic key).

  python scripts/run_research.py --enrich --slug regime_transition__vixcls__2026-04-09
      Enrich only a specific finding by slug substring.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.research import scan as research_scan


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", nargs="+", help="Explicit pair(s): A B [C D ...]. Must be even length.")
    parser.add_argument("--notable", nargs="+", help="Explicit notable-move watchlist.")
    parser.add_argument("--today", help="Override scan date (YYYY-MM-DD).")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing findings with same slug.")
    parser.add_argument("--dry-run", action="store_true", help="Run detectors but don't write outputs.")
    parser.add_argument("--enrich", action="store_true", help="Run web research to populate finding Sources blocks.")
    parser.add_argument("--slug", help="Enrich only the finding whose slug contains this substring.")
    args = parser.parse_args()

    pairs = None
    if args.pairs:
        if len(args.pairs) % 2:
            print("--pairs must be an even number of series IDs (A B [C D ...]).")
            return 2
        pairs = [
            (args.pairs[i], args.pairs[i + 1], f"{args.pairs[i]} vs {args.pairs[i + 1]}")
            for i in range(0, len(args.pairs), 2)
        ]

    today = date.fromisoformat(args.today) if args.today else date.today()

    # Enrich mode: run web research on existing findings and exit.
    if args.enrich:
        from pathlib import Path as _Path
        from src.research.enrich import enrich_all
        findings_path = PROJECT_ROOT / "knowledge" / "findings.md"
        if not findings_path.exists():
            print("knowledge/findings.md not found — run a scan first.")
            return 1
        n = enrich_all(
            findings_path,
            dry_run=args.dry_run,
            slug_filter=args.slug,
            skip_sourced=not args.overwrite,
        )
        print(f"Enriched {n} finding(s).")
        return 0

    if args.dry_run:
        # Import lazily so we can monkeypatch the I/O functions.
        from src.research import findings as F
        F.write_findings_md = lambda *a, **kw: (0, 0)        # type: ignore[assignment]
        F.append_stats = lambda *a, **kw: None                # type: ignore[assignment]
        research_scan.append_stats = F.append_stats           # type: ignore[assignment]
        research_scan.write_findings_md = F.write_findings_md # type: ignore[assignment]

    report = research_scan.run_scan(
        pairs=pairs,
        notable_watchlist=args.notable,
        today=today,
        overwrite_findings=args.overwrite,
    )
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
