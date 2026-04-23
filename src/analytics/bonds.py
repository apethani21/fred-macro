"""Bond return predictability: forward rate construction and CP factor (M3).

Implements the Cochrane-Piazzesi (2005) framework:
  - Par yields from FRED constant-maturity series (DGS1–DGS10)
  - Discrete forward rates: f(n-1,n) ≈ n*y(n) - (n-1)*y(n-1)
  - CP factor = first PC of the forward rate curve (the "tent-shaped" single
    factor that prices excess returns at all maturities)
  - 12-month excess return approximation using duration
  - OLS regression with HAC standard errors

The full Cochrane-Piazzesi regressions require ~20y of monthly data to be
meaningful; the CP factor itself is a useful real-time risk premium indicator
regardless of the regression.

Reference: Cochrane & Piazzesi (2005), "Bond Risk Premia," AER 95(1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import load_aligned
from .stats import pca, percentile_rank, zscore_vs_history

# FRED constant-maturity par yields used in the base universe.
# DGS20/DGS30 have shorter histories; included in _ALL_TENORS for optional use.
_TENOR_MAP: dict[int, str] = {
    1: "DGS1",
    2: "DGS2",
    3: "DGS3",
    5: "DGS5",
    7: "DGS7",
    10: "DGS10",
}

_ALL_TENORS: dict[int, str] = {
    **_TENOR_MAP,
    20: "DGS20",
    30: "DGS30",
}


@dataclass(frozen=True)
class CPRegressionResult:
    maturity: int
    r_squared: float
    n_obs: int
    cp_coefficient: float
    cp_tstat: float
    f_pvalue: float


@dataclass(frozen=True)
class CPFactorSnapshot:
    """Current-reading summary for the CP factor risk premium proxy."""
    date: pd.Timestamp
    cp_value: float
    z_score: float
    percentile_rank: float
    history_n: int
    r2_5y: float    # R² from OLS of 5y excess return on CP factor (nan if unavailable)
    r2_10y: float   # R² from OLS of 10y excess return on CP factor


def par_yields(
    tenors: dict[int, str] | None = None,
    freq: str = "M",
) -> pd.DataFrame:
    """Load FRED constant-maturity par yields, aligned to a common frequency.

    Columns are integer tenor labels (years). Columns with insufficient data
    are silently dropped. Returns empty DataFrame if no data is available.
    """
    t = tenors if tenors is not None else _TENOR_MAP
    ids = list(t.values())
    try:
        raw = load_aligned(ids, freq=freq, how="coarsest", agg="last", dropna="none")
    except (FileNotFoundError, KeyError):
        return pd.DataFrame()
    rename = {v: k for k, v in t.items()}
    df = raw.rename(columns={c: rename[c] for c in raw.columns if c in rename})
    int_cols = sorted(c for c in df.columns if isinstance(c, int))
    return df[int_cols] if int_cols else pd.DataFrame(index=df.index)


def forward_rates(yield_df: pd.DataFrame) -> pd.DataFrame:
    """Compute discrete annualised forward rates from par yields.

    f(n-1, n) ≈ n*y(n) - (n-1)*y(n-1) for adjacent integer tenors.
    For non-adjacent tenors the formula is annualised over the span.
    Column label = far-end tenor (e.g. column 2 = 1y×2y forward).
    """
    tenors = sorted(c for c in yield_df.columns if isinstance(c, int))
    fwds: dict[int, pd.Series] = {}
    for i in range(1, len(tenors)):
        n, nm1 = tenors[i], tenors[i - 1]
        span = n - nm1
        fwds[n] = (n * yield_df[n] - nm1 * yield_df[nm1]) / span
    if not fwds:
        return pd.DataFrame(index=yield_df.index)
    return pd.DataFrame(fwds, index=yield_df.index)


def cp_factor(
    yield_df: pd.DataFrame | None = None,
    forward_df: pd.DataFrame | None = None,
) -> pd.Series:
    """Compute the Cochrane-Piazzesi bond risk premium factor.

    Approximated as the first principal component of the forward rate curve.
    CP (2005) show a single tent-shaped factor predicts excess returns at all
    maturities with near-identical coefficients — the first PC of forwards
    captures this.

    Provide either a par yield DataFrame (forwards will be computed) or a
    pre-computed forward rate DataFrame.

    Returns an empty Series if fewer than 24 months or 2 forward tenors are available.
    """
    if forward_df is None:
        if yield_df is None:
            yield_df = par_yields()
        forward_df = forward_rates(yield_df)

    clean = forward_df.dropna(how="any")
    if clean.shape[0] < 24 or clean.shape[1] < 2:
        return pd.Series(dtype=float, name="cp_factor")

    result = pca(clean, n_components=1, standardize=True)
    scores_arr = result.scores.iloc[:, 0].values
    pc1 = pd.Series(scores_arr, index=clean.index, name="cp_factor")
    # Convention: positive CP factor = high risk premium (steep forward curve).
    # If the first PC loads negatively on the longest tenor, flip sign.
    if float(result.components.iloc[0, -1]) < 0:
        pc1 = -pc1
    return pc1


def approximate_excess_returns(
    yield_df: pd.DataFrame,
    risk_free_col: int = 1,
    horizon_months: int = 12,
) -> pd.DataFrame:
    """Approximate 1-year excess bond returns using duration.

    rx(n, t) ≈ -n * [y(n-1, t+H) - y(n, t)]/100 + y(n, t)/100 * H/12
                - y(1, t)/100 * H/12

    where y(n-1, t+H) is the future yield of the (formerly n-year) bond
    after it has seasoned H months. Modified duration approximated as n years
    (exact for zero-coupon; rough for par bonds but directionally correct).

    Results for the last `horizon_months` rows will be NaN (future yields unknown).
    """
    tenors = sorted(c for c in yield_df.columns if isinstance(c, int) and c != risk_free_col)
    rf = yield_df[risk_free_col]
    H = horizon_months
    # Use same-maturity yield shifted H months forward (yield-change approximation).
    # Holds the maturity label fixed rather than rolling down to (n-1): works for
    # all available maturities without requiring adjacent tenors.
    rows: dict[int, pd.Series] = {}
    for n in tenors:
        y_future = yield_df[n].shift(-H)
        capital_gain = -n * (y_future - yield_df[n]) / 100
        carry = yield_df[n] / 100 * H / 12
        rf_cost = rf / 100 * H / 12
        rows[n] = (capital_gain + carry - rf_cost) * 100
    if not rows:
        return pd.DataFrame(index=yield_df.index)
    return pd.DataFrame(rows, index=yield_df.index)


def cp_regression(
    rx_df: pd.DataFrame,
    cp: pd.Series,
    hac_lags: int = 12,
) -> list[CPRegressionResult]:
    """OLS of each column of rx_df on the CP factor with Newey-West HAC SEs.

    Returns one CPRegressionResult per maturity column. Skips columns with
    fewer than 30 overlapping observations.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return []

    results: list[CPRegressionResult] = []
    for col in rx_df.columns:
        y = rx_df[col].dropna()
        x = cp.reindex(y.index).dropna()
        common = y.index.intersection(x.index)
        if len(common) < 30:
            continue
        y_, x_ = y.loc[common], x.loc[common]
        X = sm.add_constant(x_.rename("cp_factor"))
        try:
            ols = sm.OLS(y_, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        except Exception:
            continue
        results.append(CPRegressionResult(
            maturity=int(col),
            r_squared=float(ols.rsquared),
            n_obs=len(common),
            cp_coefficient=float(ols.params.get("cp_factor", float("nan"))),
            cp_tstat=float(ols.tvalues.get("cp_factor", float("nan"))),
            f_pvalue=float(ols.f_pvalue),
        ))
    return results


def cp_factor_snapshot(cp: pd.Series, reg_results: list[CPRegressionResult]) -> CPFactorSnapshot | None:
    """Compute a summary snapshot of the current CP factor reading."""
    clean = cp.dropna()
    if clean.empty:
        return None
    latest_val = float(clean.iloc[-1])
    latest_date = clean.index[-1]
    z = float(zscore_vs_history(clean))
    pct = float(percentile_rank(clean))
    r2_5y = r2_10y = float("nan")
    for r in reg_results:
        if r.maturity == 5:
            r2_5y = r.r_squared
        elif r.maturity == 10:
            r2_10y = r.r_squared
    return CPFactorSnapshot(
        date=latest_date,
        cp_value=latest_val,
        z_score=z,
        percentile_rank=pct,
        history_n=len(clean),
        r2_5y=r2_5y,
        r2_10y=r2_10y,
    )
