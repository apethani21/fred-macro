"""Regenerate knowledge/concepts.md by drafting each concept via Claude + web_fetch.

Run once on EC2 (where ANTHROPIC_API_KEY is available) to rebuild from scratch.
The file is hand-editable thereafter; run this only to re-seed from a clean state.

Usage:
    python src/knowledge/seed_concepts.py
    python src/knowledge/seed_concepts.py --concept "VIX" --append
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Priority concepts in seed order.
CONCEPTS = [
    {
        "name": "VIX / Implied Volatility",
        "fred_ids": ["VIXCLS"],
        "primary_sources": [
            "https://www.cboe.com/micro/vix/vixwhite.pdf",
            "https://fred.stlouisfed.org/series/VIXCLS",
        ],
    },
    {
        "name": "Yield Curve Slope (2s10s and 3m10y)",
        "fred_ids": ["T10Y2Y", "DGS10", "DGS2", "DGS3MO"],
        "primary_sources": [
            "https://www.newyorkfed.org/research/staff_reports/sr194",
            "https://fred.stlouisfed.org/series/T10Y2Y",
        ],
    },
    {
        "name": "Federal Funds Rate and SOFR",
        "fred_ids": ["DFF", "SOFR", "IORB"],
        "primary_sources": [
            "https://www.newyorkfed.org/markets/reference-rates/sofr",
            "https://fred.stlouisfed.org/series/DFF",
        ],
    },
    {
        "name": "Consumer Price Index (CPI)",
        "fred_ids": ["CPIAUCSL", "CPILFESL", "CUSR0000SAH1"],
        "primary_sources": [
            "https://www.bls.gov/opub/hom/cpi/home.htm",
            "https://fred.stlouisfed.org/series/CPIAUCSL",
        ],
    },
    {
        "name": "PCE (Personal Consumption Expenditures Price Index)",
        "fred_ids": ["PCEPI", "PCEPILFE"],
        "primary_sources": [
            "https://www.bea.gov/resources/methodologies/concepts-methods-US-NIPA",
            "https://fred.stlouisfed.org/series/PCEPILFE",
        ],
    },
    {
        "name": "Unemployment Rate",
        "fred_ids": ["UNRATE", "U6RATE", "SAHMREALTIME"],
        "primary_sources": [
            "https://www.bls.gov/opub/hom/ces/home.htm",
            "https://fred.stlouisfed.org/series/UNRATE",
        ],
    },
    {
        "name": "Nonfarm Payrolls",
        "fred_ids": ["PAYEMS", "USPRIV"],
        "primary_sources": [
            "https://www.bls.gov/opub/hom/ces/home.htm",
            "https://fred.stlouisfed.org/series/PAYEMS",
        ],
    },
    {
        "name": "10-Year Treasury Yield and Real Yield Decomposition",
        "fred_ids": ["DGS10", "DFII10", "T10YIE"],
        "primary_sources": [
            "https://www.newyorkfed.org/research/data_indicators/term_premia",
            "https://fred.stlouisfed.org/series/DGS10",
        ],
    },
    {
        "name": "Retail Sales",
        "fred_ids": ["RSXFS", "RRSFS"],
        "primary_sources": [
            "https://www.census.gov/retail/methodology.html",
            "https://fred.stlouisfed.org/series/RSXFS",
        ],
    },
    {
        "name": "Industrial Production",
        "fred_ids": ["INDPRO"],
        "primary_sources": [
            "https://www.federalreserve.gov/releases/g17/about.htm",
            "https://fred.stlouisfed.org/series/INDPRO",
        ],
    },
    {
        "name": "GDP",
        "fred_ids": ["GDPC1"],
        "primary_sources": [
            "https://www.bea.gov/resources/methodologies/concepts-methods-US-NIPA",
            "https://fred.stlouisfed.org/series/GDPC1",
        ],
    },
    {
        "name": "Breakeven Inflation",
        "fred_ids": ["T5YIE", "T10YIE", "T5YIFR"],
        "primary_sources": [
            "https://fred.stlouisfed.org/series/T10YIE",
        ],
    },
    {
        "name": "Fed Balance Sheet",
        "fred_ids": ["WALCL", "RESBALNS", "RRPONTSYD"],
        "primary_sources": [
            "https://www.federalreserve.gov/releases/h41/",
            "https://fred.stlouisfed.org/series/WALCL",
        ],
    },
    {
        "name": "Credit Spreads (HY and IG)",
        "fred_ids": ["BAMLH0A0HYM2", "BAMLC0A0CM"],
        "primary_sources": [
            "https://fred.stlouisfed.org/series/BAMLH0A0HYM2",
        ],
    },
]

_SYSTEM = """\
You are a macro economist writing a technical knowledge-base entry for a quant equity professional
(3 years experience) who is building deep macro knowledge. They understand statistics, time series,
and financial markets. Do not explain equities basics; explain macro-specific mechanics.

For each concept produce a markdown section following this structure exactly:
## {Concept Name}
**FRED series**: list the relevant series IDs

**What it measures**: 1-2 sentences, precise.

**Construction**: How the data is actually built (survey methods, index construction, known quirks).
Focus on what a practitioner needs to know that is NOT in the first paragraph of a Wikipedia article.

**Why it matters**: Mechanism-based explanation. What does it cause or predict, and why? Include
one specific empirical claim with numbers (with source) if you know one well.

**Typical regimes**: Quantitative thresholds with their typical macro associations.

**Common misreadings**: 2-3 specific ways this series is routinely misinterpreted.

**Related FRED series**: comma-separated list.

Be precise. Be non-obvious. Quantify everything you can.
"""


def _draft_concept(name: str, fred_ids: list[str], sources: list[str]) -> str:
    import anthropic
    import os

    key_file = Path.home() / "keys" / "anthropic" / "key.txt"
    if key_file.exists():
        api_key = key_file.read_text().strip()
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("Anthropic API key not found.")

    client = anthropic.Anthropic(api_key=api_key)

    user_msg = (
        f"Concept: {name}\n"
        f"FRED series: {', '.join(fred_ids)}\n"
        f"Primary sources to consult: {', '.join(sources)}\n\n"
        "Write the knowledge-base entry following the structure in the system prompt."
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text.strip()


def seed_all(output_path: Path, append: bool = False, only: str | None = None) -> None:
    concepts = CONCEPTS
    if only:
        concepts = [c for c in CONCEPTS if only.lower() in c["name"].lower()]
        if not concepts:
            raise ValueError(f"No concept matching {only!r}")

    mode = "a" if append else "w"
    with output_path.open(mode, encoding="utf-8") as f:
        if not append:
            f.write("# Macro Concepts Knowledge Base\n\n")
            f.write("Generated by src/knowledge/seed_concepts.py. Hand-editable thereafter.\n\n")
            f.write("---\n\n")
        for concept in concepts:
            logger.info("Seeding: %s", concept["name"])
            try:
                section = _draft_concept(
                    concept["name"],
                    concept["fred_ids"],
                    concept["primary_sources"],
                )
                f.write(section + "\n\n---\n\n")
                f.flush()
                logger.info("Done: %s", concept["name"])
            except Exception as e:
                logger.error("Failed to seed %s: %s", concept["name"], e)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept", help="Seed only a matching concept name substring.")
    parser.add_argument("--append", action="store_true", help="Append to existing file instead of overwriting.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "knowledge" / "concepts.md"))
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_all(output_path, append=args.append, only=args.concept)
    print(f"Written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
