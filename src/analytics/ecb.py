"""ECB/euro area analytical helpers.

Mirrors the pattern of bonds.py and indicators.py: small, composable functions
that load ECB series from the shared parquet store and return aligned DataFrames
or computed indicators. Raw series are loaded via data.load_series(); derived
series (BTPBUND.SPREAD, BUND.SLOPE) are already computed by ecb_update.py and
stored in parquet — load them directly.

ECB series IDs:
  ECB.DFR              Deposit facility rate (daily)
  ECB.ESTR             €STR overnight rate (daily)
  ECB.HICP.EA.TOTAL    EA HICP all-items YoY % (monthly)
  ECB.HICP.EA.CORE     EA HICP ex food/energy YoY % (monthly)
  ECB.YC.AAA.2Y        EA AAA sovereign yield curve 2Y (daily)
  ECB.YC.AAA.10Y       EA AAA sovereign yield curve 10Y (daily)
  ECB.DE.10Y           Germany 10Y Maastricht yield (monthly)
  ECB.IT.10Y           Italy 10Y Maastricht yield / BTP (monthly)
  ECB.M3.EA            EA M3 money supply annual growth % (monthly)
  ECB.WAGES.NEG        EA negotiated wages YoY % (quarterly)
  ECB.BTPBUND.SPREAD   BTP-Bund spread in pp (derived, monthly)
  ECB.BUND.SLOPE       EA AAA 10Y-2Y slope in pp (derived, daily)

FRED counterpart series used in cross-region comparisons:
  DFF   Effective Federal Funds Rate (daily)
  CPIAUCSL   US CPI all-items YoY (requires computing from levels, monthly)
  CPILFESL   US CPI ex food/energy YoY (requires computing from levels, monthly)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import load_aligned, load_series
from .stats import percentile_rank, zscore_vs_history

# ---------- ECB rate-differential regime labels ----------

_REGIME_THRESHOLDS = {
    "ecb_much_tighter": 100,   # ECB ≥ Fed + 100bp
    "ecb_tighter": 25,         # ECB ≥ Fed + 25bp
    "roughly_equal": -25,      # within ±25bp
    "ecb_looser": -100,        # Fed ≥ ECB + 25bp
    # below -100bp: "ecb_much_looser"
}


@dataclass(frozen=True)
class DivergenceSnapshot:
    """Summary of ECB vs Fed policy rate divergence at the latest available date."""
    date: pd.Timestamp
    ecb_dfr: float          # ECB deposit facility rate, %
    fed_funds: float        # Effective Fed funds rate, %
    spread_bp: float        # ECB minus Fed in basis points
    regime: str             # one of the five regime labels
    spread_z: float         # z-score vs full history of spread
    spread_pct: float       # percentile rank vs full history


# ---------- BTP-Bund spread ----------

def btp_bund_spread(use_stored: bool = True) -> pd.Series:
    """BTP-Bund 10Y spread in percentage points (monthly).

    Uses the pre-computed ECB.BTPBUND.SPREAD stored by the ECB update pipeline
    by default. Pass use_stored=False to recompute from component series.
    """
    if use_stored:
        try:
            return load_series("ECB.BTPBUND.SPREAD")
        except (KeyError, FileNotFoundError):
            pass

    df = load_aligned(
        ["ECB.IT.10Y", "ECB.DE.10Y"],
        freq="M",
        how="coarsest",
        dropna="any",
    )
    return (df["ECB.IT.10Y"] - df["ECB.DE.10Y"]).rename("ECB.BTPBUND.SPREAD")


def btp_bund_snapshot() -> dict:
    """Current reading of the BTP-Bund spread with historical context."""
    s = btp_bund_spread().dropna()
    if s.empty:
        return {}
    latest = float(s.iloc[-1])
    return {
        "date": s.index[-1],
        "spread_pp": latest,
        "spread_bp": latest * 100,
        "z_score": float(zscore_vs_history(s)),
        "percentile": float(percentile_rank(s)),
        "history_n": len(s),
        "regime": _btp_bund_regime(latest * 100),
    }


def _btp_bund_regime(spread_bp: float) -> str:
    """Label the BTP-Bund spread regime by conventional market thresholds.

    ~200bp is the ECB TPI informal trigger; ~150bp signals elevated stress;
    <100bp is considered low/benign.
    """
    if spread_bp >= 300:
        return "crisis"         # GFC/euro crisis territory (2011-12 peak ~550bp)
    if spread_bp >= 200:
        return "elevated"       # TPI informal trigger zone (~230bp in 2022)
    if spread_bp >= 130:
        return "stressed"
    if spread_bp >= 80:
        return "moderate"
    return "benign"


# ---------- Bund yield curve slope ----------

def bund_slope(use_stored: bool = True) -> pd.Series:
    """EA AAA sovereign yield curve slope (10Y - 2Y) in percentage points (daily).

    Uses pre-computed ECB.BUND.SLOPE by default.
    """
    if use_stored:
        try:
            return load_series("ECB.BUND.SLOPE")
        except (KeyError, FileNotFoundError):
            pass

    df = load_aligned(
        ["ECB.YC.AAA.10Y", "ECB.YC.AAA.2Y"],
        freq="D",
        how="finest",
        dropna="any",
    )
    return (df["ECB.YC.AAA.10Y"] - df["ECB.YC.AAA.2Y"]).rename("ECB.BUND.SLOPE")


# ---------- ECB vs Fed policy divergence ----------

def ecb_dfr_vs_fed_funds(freq: str = "M") -> pd.DataFrame:
    """Align ECB deposit facility rate and US effective fed funds on a common index.

    Arguments:
        freq — target frequency: 'D' (daily), 'M' (monthly). Monthly averages
               the daily DFF; ECB.DFR is also daily but changes infrequently.

    Returns DataFrame with columns ['ECB.DFR', 'DFF'], common dates only.
    """
    return load_aligned(
        ["ECB.DFR", "DFF"],
        freq=freq,
        how="coarsest",
        agg="last",
        dropna="any",
    )


def ecb_fed_spread(freq: str = "M") -> pd.Series:
    """ECB DFR minus US fed funds rate, in basis points.

    Positive = ECB is tighter than the Fed. History is short: ECB DFR data in
    the SDW starts 1999-01-01; meaningful divergence began post-GFC.
    """
    df = ecb_dfr_vs_fed_funds(freq=freq)
    return ((df["ECB.DFR"] - df["DFF"]) * 100).rename("ECB_Fed_spread_bp")


def ecb_fed_divergence_regime(freq: str = "M") -> DivergenceSnapshot | None:
    """Snapshot of the current ECB-Fed policy divergence with regime label.

    Returns None if either series has no data.
    """
    df = ecb_dfr_vs_fed_funds(freq=freq).dropna()
    if df.empty:
        return None

    spread = (df["ECB.DFR"] - df["DFF"]) * 100  # basis points
    latest_date = df.index[-1]
    ecb_latest = float(df["ECB.DFR"].iloc[-1])
    fed_latest = float(df["DFF"].iloc[-1])
    spread_latest = float(spread.iloc[-1])

    regime = _divergence_regime(spread_latest)
    z = float(zscore_vs_history(spread))
    pct = float(percentile_rank(spread))

    return DivergenceSnapshot(
        date=latest_date,
        ecb_dfr=ecb_latest,
        fed_funds=fed_latest,
        spread_bp=spread_latest,
        regime=regime,
        spread_z=z,
        spread_pct=pct,
    )


def _divergence_regime(spread_bp: float) -> str:
    if spread_bp >= _REGIME_THRESHOLDS["ecb_much_tighter"]:
        return "ecb_much_tighter"
    if spread_bp >= _REGIME_THRESHOLDS["ecb_tighter"]:
        return "ecb_tighter"
    if spread_bp >= _REGIME_THRESHOLDS["roughly_equal"]:
        return "roughly_equal"
    if spread_bp >= _REGIME_THRESHOLDS["ecb_looser"]:
        return "ecb_looser"
    return "ecb_much_looser"


# ---------- HICP vs CPI cross-region ----------

def hicp_vs_cpi_aligned(
    ea_id: str = "ECB.HICP.EA.TOTAL",
    us_cpi_id: str = "CPIAUCSL",
    us_as_yoy: bool = True,
) -> pd.DataFrame:
    """Align EA HICP YoY% with US CPI on a common monthly index.

    The ECB HICP series is already stored as annual % change (YoY). CPIAUCSL
    and CPILFESL are stored as index levels; set us_as_yoy=True (default) to
    convert to 12-month % change before aligning.

    Returns DataFrame with columns [ea_id, us_cpi_id] on monthly period-start dates.
    """
    ea = load_series(ea_id)
    us = load_series(us_cpi_id)

    if us_as_yoy:
        us = us.pct_change(12) * 100
        us.name = us_cpi_id

    df = pd.concat([ea, us], axis=1)
    df.columns = [ea_id, us_cpi_id]

    # Resample both to monthly period-start, keep last observation in period.
    df = df.resample("MS").last()
    return df.dropna(how="any")


def hicp_cpi_differential() -> pd.Series:
    """EA HICP all-items minus US CPI all-items, in percentage points (monthly).

    Positive = EA inflation higher than US. Useful for framing ECB-Fed policy
    divergence: historically, this differential has predicted relative rate paths
    with a ~6-month lead.
    """
    df = hicp_vs_cpi_aligned()
    if df.empty:
        return pd.Series(dtype=float, name="HICP_minus_CPI_pp")
    col_ea, col_us = df.columns[0], df.columns[1]
    return (df[col_ea] - df[col_us]).rename("HICP_minus_CPI_pp")


# ---------- Negotiated wages vs core HICP ----------

def negotiated_wages_vs_hicp(
    wages_id: str = "ECB.WAGES.NEG",
    hicp_id: str = "ECB.HICP.EA.CORE",
) -> pd.DataFrame:
    """Align EA negotiated wages YoY% with EA core HICP YoY% on quarterly index.

    Wages are quarterly (Q); core HICP is monthly (M). This downsamples HICP to
    quarterly (last observation in period) to match the lower-frequency wages data.

    The lead-lag between wages and core inflation is a key ECB variable: the ECB
    watches whether wage growth is moderating toward HICP target or remaining
    above it ('second-round effects'). Negotiated wages tend to lag spot CPI by
    1-2 quarters due to annual bargaining cycles.

    Returns DataFrame with columns [wages_id, hicp_id] on quarterly dates.
    """
    return load_aligned(
        [wages_id, hicp_id],
        freq="Q",
        how="coarsest",
        agg="last",
        dropna="any",
    )


def wages_hicp_gap() -> pd.Series:
    """EA negotiated wages YoY minus EA core HICP YoY, in percentage points (quarterly).

    Positive = wage growth above core inflation (potential second-round pressure).
    Negative = real wage compression.
    """
    df = negotiated_wages_vs_hicp().dropna()
    if df.empty:
        return pd.Series(dtype=float, name="wages_hicp_gap_pp")
    return (df["ECB.WAGES.NEG"] - df["ECB.HICP.EA.CORE"]).rename("wages_hicp_gap_pp")
