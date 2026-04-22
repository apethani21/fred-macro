"""Canonical macro indicators as composable helpers.

Every function here is a small transformation of one or more FRED series into
a widely-cited derived series. The point is to avoid each caller re-rolling
Sahm rule / curve slopes / Taylor rule from scratch (and getting the
definitions subtly wrong).

Unless noted, inputs are the original FRED series (levels in percent for
rates, month-end timestamps for monthly series). Outputs are pandas Series
with a descriptive `.name`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .data import load_aligned, load_series


# ---------- labor market ----------

def sahm_rule(unrate: pd.Series | None = None) -> pd.Series:
    """Sahm Recession Indicator (Sahm 2019).

    Defined as the 3-month moving average of the unemployment rate minus its
    low over the prior 12 months. A value ≥ 0.5pp has coincided with the
    start of every US recession since 1960.

    Pass a custom UNRATE series or let it load from storage.
    """
    s = unrate if unrate is not None else load_series("UNRATE")
    three_mo = s.rolling(3, min_periods=3).mean()
    prior_12_low = s.rolling(12, min_periods=12).min().shift(1)
    sahm = (three_mo - prior_12_low).rename("sahm_rule")
    return sahm


# ---------- yield curve ----------

@dataclass(frozen=True)
class CurveSlope:
    name: str         # "2s10s", etc.
    long_id: str      # FRED id for the long leg
    short_id: str     # FRED id for the short leg
    description: str


CURVE_SLOPES: tuple[CurveSlope, ...] = (
    CurveSlope("3m10y", "DGS10", "DGS3MO",
               "Classic NY Fed recession-probability slope (Estrella & Mishkin)."),
    CurveSlope("2s10s", "DGS10", "DGS2",
               "Canonical benchmark slope; FRED publishes directly as T10Y2Y."),
    CurveSlope("5s30s", "DGS30", "DGS5",
               "Long-end steepness; less sensitive to near-term policy path."),
    CurveSlope("2s30s", "DGS30", "DGS2",
               "Full curve span; useful when 10y is distorted by supply dynamics."),
)
CURVE_SLOPES_BY_NAME: dict[str, CurveSlope] = {c.name: c for c in CURVE_SLOPES}


def curve_slope(name: str) -> pd.Series:
    """Compute a named slope as `long - short` (in percentage points)."""
    if name not in CURVE_SLOPES_BY_NAME:
        raise KeyError(f"Unknown slope {name!r}. Known: {list(CURVE_SLOPES_BY_NAME)}")
    cs = CURVE_SLOPES_BY_NAME[name]
    df = load_aligned([cs.long_id, cs.short_id], freq="D", how="finest", dropna="any")
    return (df[cs.long_id] - df[cs.short_id]).rename(cs.name)


def all_curve_slopes() -> pd.DataFrame:
    """Return every slope in CURVE_SLOPES as columns on a common daily index.

    Columns are NaN where either leg is missing; handy for a one-chart view of
    the whole curve's shape evolution.
    """
    cols: dict[str, pd.Series] = {}
    for cs in CURVE_SLOPES:
        try:
            cols[cs.name] = curve_slope(cs.name)
        except (FileNotFoundError, KeyError):
            continue
    return pd.concat(cols, axis=1).sort_index()


# ---------- real vs nominal ----------

def real_yield_decomposition(
    tenor: Literal["5y", "10y"] = "10y",
) -> pd.DataFrame:
    """Decompose a nominal Treasury yield into real + breakeven.

    Uses FRED constant-maturity TIPS (`DFII10` / `DFII5`) and breakeven
    (`T10YIE` / `T5YIE`). Columns: `nominal`, `real`, `breakeven` — the
    accounting identity `nominal ≈ real + breakeven` holds to within a few
    basis points (inflation-risk premium and indexation lag).
    """
    if tenor == "10y":
        nom, real, be = "DGS10", "DFII10", "T10YIE"
    elif tenor == "5y":
        nom, real, be = "DGS5", "DFII5", "T5YIE"
    else:
        raise ValueError(f"Unknown tenor {tenor!r}")
    df = load_aligned([nom, real, be], freq="D", how="finest", dropna="any")
    out = df.rename(columns={nom: "nominal", real: "real", be: "breakeven"})
    return out[["nominal", "real", "breakeven"]]


# ---------- policy rule ----------

@dataclass
class TaylorRuleInputs:
    policy_rate: pd.Series    # FEDFUNDS or DFEDTARU
    inflation: pd.Series      # Core PCE YoY (decimal or percent — auto-detected)
    output_gap: pd.Series     # Percent of potential GDP
    neutral_real: float       # r*, percentage points
    inflation_target: float   # π*, percentage points


def taylor_rule(
    inputs: TaylorRuleInputs,
    inflation_weight: float = 0.5,
    output_weight: float = 0.5,
) -> pd.DataFrame:
    """Classic Taylor (1993) rule implied nominal policy rate.

    r* + π + 0.5·(π − π*) + 0.5·gap

    Returns a DataFrame with columns:
      `implied`     — rule-implied nominal rate
      `actual`      — policy_rate aligned to the same index
      `deviation`   — actual − implied (positive = policy more hawkish)

    All numbers in percentage points. Assumes series arrive aligned at the
    same frequency (typically quarterly or monthly).
    """
    pi = inputs.inflation
    gap = inputs.output_gap
    actual = inputs.policy_rate
    df = pd.concat([actual.rename("actual"), pi.rename("pi"), gap.rename("gap")], axis=1).dropna()
    implied = (
        inputs.neutral_real
        + df["pi"]
        + inflation_weight * (df["pi"] - inputs.inflation_target)
        + output_weight * df["gap"]
    )
    out = pd.DataFrame({
        "implied": implied,
        "actual": df["actual"],
        "deviation": df["actual"] - implied,
    })
    return out


def taylor_rule_from_fred(
    neutral_real: float = 0.5,
    inflation_target: float = 2.0,
    inflation_weight: float = 0.5,
    output_weight: float = 0.5,
) -> pd.DataFrame:
    """Convenience wrapper: pull FEDFUNDS + core PCE YoY + output gap from FRED.

    Output gap is computed from GDPC1 and GDPPOT as 100·(actual/potential − 1).
    Inflation is 12-month core PCE (PCEPILFE). Monthly frequency, aligned to
    GDPPOT (quarterly) via upsample-ffill.

    Defaults (r* = 0.5, π* = 2.0) reflect post-2015 Fed thinking; tweak for
    counterfactuals (e.g., r* = 2.0 is the original 1993 value).
    """
    ff = load_series("FEDFUNDS")
    core_pce = load_series("PCEPILFE")
    pi_yoy = core_pce.pct_change(12) * 100  # percent
    gdp = load_series("GDPC1")
    potential = load_series("GDPPOT")
    # Align all to month-end; potential is quarterly — ffill to monthly.
    df = (
        pd.concat([ff.rename("ff"), pi_yoy.rename("pi"), gdp.rename("gdp"), potential.rename("pot")], axis=1)
        .resample("ME").last()
        .ffill()
    )
    df["gap"] = 100 * (df["gdp"] / df["pot"] - 1)
    inp = TaylorRuleInputs(
        policy_rate=df["ff"],
        inflation=df["pi"],
        output_gap=df["gap"],
        neutral_real=neutral_real,
        inflation_target=inflation_target,
    )
    return taylor_rule(inp, inflation_weight=inflation_weight, output_weight=output_weight)
