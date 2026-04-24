"""Curated universe for the research phase.

Scanning the full 7k-series universe for correlation shifts and notable moves
is computationally fine but semantically noisy — most pairs are meaningless.
The research phase instead works from a curated "core" set of macro series
and a curated list of relationship pairs worth monitoring.

Series outside this list can still be loaded on demand (e.g., referenced from
concepts.md); this file only defines what gets *scanned* every run.
"""
from __future__ import annotations

# Core macro series grouped by role. Order within each group is cosmetic.
CORE_UNIVERSE: dict[str, tuple[str, ...]] = {
    "policy_rate": ("DFF", "FEDFUNDS", "DFEDTARU", "DFEDTARL", "SOFR", "IORB"),
    "treasury_yields": ("DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30"),
    "curve_slopes": ("T10Y2Y", "T10Y3M"),
    "real_rates": ("DFII5", "DFII10"),
    "breakevens": ("T5YIE", "T10YIE"),
    "labor": ("UNRATE", "PAYEMS", "AHETPI", "CIVPART", "JTSJOL"),
    "inflation": ("CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE", "PPIACO"),
    "growth": ("GDPC1", "INDPRO", "RRSFS", "UMCSENT"),
    "credit_spreads": ("BAMLH0A0HYM2", "AAA10Y", "BAA10Y"),
    "volatility_equity": ("VIXCLS", "SP500"),
    "commodities_fx": ("DCOILWTICO", "DTWEXBGS"),
    "recession_flag": ("USREC",),
}


def core_series() -> list[str]:
    """Flat list of every core series in deterministic order."""
    return [sid for group in CORE_UNIVERSE.values() for sid in group]


# CORE_PAIRS is kept for backward compatibility but RELATIONSHIPS (in relationship_config.py)
# is the single source of truth. run_scan() defaults to RELATIONSHIPS-derived pairs.
# Any new pairs should be added to RELATIONSHIPS, not here.
from .relationship_config import relationships_as_pairs as _rel_pairs

CORE_PAIRS: tuple[tuple[str, str, str], ...] = tuple(
    _rel_pairs(kinds=("correlation", "lead_lag"))
)


# Series for which "notable-move" detection runs (the rare tail event matters
# more than percentile rank alone). Excludes slow-moving quantities where a
# fresh extreme is almost always mechanical (e.g., level of CPI).
NOTABLE_MOVE_WATCHLIST: tuple[str, ...] = (
    "T10Y2Y", "T10Y3M", "DGS10", "DGS2", "DGS30", "DFII10", "T10YIE",
    "VIXCLS", "BAMLH0A0HYM2", "AAA10Y", "BAA10Y",
    "UNRATE", "UMCSENT",
    "DCOILWTICO", "DTWEXBGS", "SP500",
    "SOFR", "DFF",
)
