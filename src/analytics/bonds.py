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


# ── Nelson-Siegel Yield Curve Factors (M2) ────────────────────────────────────
#
# Diebold-Li (2006) parameterisation with fixed λ = 0.0609 (maximises curvature
# loading at ~30-month maturity). For fixed λ the model is linear in factors:
#
#   y(τ) = β0·L(τ) + β1·S(τ) + β2·C(τ)
#
# where L(τ)=1, S(τ)=(1−e^{−λτ})/(λτ), C(τ)=S(τ)−e^{−λτ}
# β0 = level (long-run yield), β1 = slope (short − long), β2 = curvature (hump)
#
# Reference: Diebold & Li (2006), "Forecasting the Term Structure of Government
# Bond Yields," Journal of Econometrics 130(2):337-364.

_NS_LAMBDA = 0.0609  # Diebold-Li canonical decay parameter


@dataclass(frozen=True)
class NSFactors:
    """Nelson-Siegel factor snapshot for a single date."""
    date: pd.Timestamp
    level: float       # β0 — long-run yield level (%)
    slope: float       # β1 — curve steepness; negative = inverted
    curvature: float   # β2 — medium-term hump; positive = belly rich
    fit_rmse: float    # root mean squared fitting error across maturities (bps)
    n_maturities: int  # maturities used in fit


@dataclass(frozen=True)
class NSFactorSnapshot:
    """Current NS factor readings with historical context."""
    date: pd.Timestamp
    level: float
    slope: float
    curvature: float
    level_pct: float        # percentile rank vs full history
    slope_pct: float
    curvature_pct: float
    level_z: float          # z-score vs full history
    slope_z: float
    curvature_z: float
    history_n: int          # months of factor history


def _ns_loadings(tenors_years: np.ndarray, lam: float = _NS_LAMBDA) -> np.ndarray:
    """Return (n_tenors × 3) loading matrix [L, S, C] for given maturities."""
    lt = lam * tenors_years
    # Avoid division by zero for τ → 0
    lt_safe = np.where(lt < 1e-8, 1e-8, lt)
    S = (1 - np.exp(-lt_safe)) / lt_safe
    C = S - np.exp(-lt_safe)
    L = np.ones_like(S)
    return np.column_stack([L, S, C])


def _fit_ns_row(yields_row: np.ndarray, tenors: np.ndarray) -> tuple[float, float, float, float]:
    """Fit NS to a single cross-section of yields. Returns (β0, β1, β2, rmse_bps)."""
    X = _ns_loadings(tenors)
    mask = np.isfinite(yields_row)
    if mask.sum() < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    X_, y_ = X[mask], yields_row[mask]
    # OLS: β = (X'X)^{-1} X'y
    try:
        betas, _, _, _ = np.linalg.lstsq(X_, y_, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), float("nan"), float("nan")
    fitted = X_ @ betas
    rmse_bps = float(np.sqrt(np.mean((y_ - fitted) ** 2)) * 100)
    return float(betas[0]), float(betas[1]), float(betas[2]), rmse_bps


def nelson_siegel_factors(
    tenors: dict[int, str] | None = None,
    freq: str = "M",
    lam: float = _NS_LAMBDA,
) -> pd.DataFrame:
    """Compute time series of Nelson-Siegel factors (level, slope, curvature).

    Uses _ALL_TENORS (1Y–30Y) by default: the wider maturity range is essential for
    numerical stability — with only 1Y–10Y the loading matrix is near-singular at
    λ=0.0609, producing extreme and unstable factor estimates.

    Returns DataFrame with columns ['level', 'slope', 'curvature', 'fit_rmse'],
    indexed by month-end dates. Rows with fewer than 3 valid maturities are dropped.
    """
    t = tenors if tenors is not None else _ALL_TENORS
    yields = par_yields(tenors=t, freq=freq)
    if yields.empty:
        return pd.DataFrame(columns=["level", "slope", "curvature", "fit_rmse"])

    tenor_arr = np.array(sorted(c for c in yields.columns if isinstance(c, int)), dtype=float)
    results = []
    for dt, row in yields.iterrows():
        b0, b1, b2, rmse = _fit_ns_row(row.values.astype(float), tenor_arr)
        results.append({"date": dt, "level": b0, "slope": b1, "curvature": b2, "fit_rmse": rmse})

    df = pd.DataFrame(results).set_index("date")
    return df.dropna(subset=["level", "slope", "curvature"])


def ns_factor_snapshot(factors: pd.DataFrame | None = None) -> NSFactorSnapshot | None:
    """Compute current NS factor snapshot with historical percentile/z-score context.

    Pass pre-computed factors DataFrame or call with None to compute fresh.
    Returns None if fewer than 24 months of data are available.
    """
    if factors is None:
        factors = nelson_siegel_factors()
    if factors is None or len(factors) < 24:
        return None

    cur = factors.iloc[-1]
    dt = factors.index[-1]

    def _pct(col: str) -> float:
        return float(percentile_rank(factors[col], float(cur[col])))

    def _z(col: str) -> float:
        return float(zscore_vs_history(factors[col]))

    return NSFactorSnapshot(
        date=dt,
        level=float(cur["level"]),
        slope=float(cur["slope"]),
        curvature=float(cur["curvature"]),
        level_pct=_pct("level"),
        slope_pct=_pct("slope"),
        curvature_pct=_pct("curvature"),
        level_z=_z("level"),
        slope_z=_z("slope"),
        curvature_z=_z("curvature"),
        history_n=len(factors),
    )


def ns_macro_var(
    factors: pd.DataFrame | None = None,
    macro_series: dict[str, str] | None = None,
    lags: int = 2,
) -> dict:
    """Fit a VAR on [level, slope, curvature, UNRATE, CPI_YoY] and return summary.

    Returns dict with keys:
      - 'aic': model AIC
      - 'factor_correlations': {factor: {macro_var: correlation}} — contemporaneous correlations
      - 'granger': {target: {cause: p_value}} — Granger causality p-values (macro → NS factors)
      - 'forecast_1m': {factor: forecasted_value} — 1-month-ahead VAR forecast

    Returns {} if statsmodels is not available or data is insufficient.
    """
    try:
        from statsmodels.tsa.vector_ar.var_model import VAR
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return {}

    if factors is None:
        factors = nelson_siegel_factors()
    if factors is None or len(factors) < 36:
        return {}

    macro = macro_series or {"UNRATE": "UNRATE", "CPI_YoY": "CPIAUCSL"}

    macro_data: dict[str, pd.Series] = {}
    for label, sid in macro.items():
        try:
            s = D.load_series(sid, freq="M")
            if label == "CPI_YoY":
                s = s.pct_change(12) * 100
            macro_data[label] = s
        except Exception:
            pass

    panel = factors[["level", "slope", "curvature"]].copy()
    for label, s in macro_data.items():
        panel[label] = s.reindex(panel.index, method="ffill")
    panel = panel.dropna()

    if len(panel) < 36:
        return {}

    result: dict = {}

    # Contemporaneous correlations
    corr = panel.corr()
    factor_corr: dict = {}
    for factor in ["level", "slope", "curvature"]:
        factor_corr[factor] = {
            col: round(float(corr.loc[factor, col]), 3)
            for col in macro_data
            if col in corr.columns
        }
    result["factor_correlations"] = factor_corr

    # VAR fit
    try:
        model = VAR(panel)
        fitted = model.fit(lags, ic=None)
        result["aic"] = round(float(fitted.aic), 2)

        # 1-month-ahead forecast
        forecast = fitted.forecast(panel.values[-lags:], steps=1)[0]
        result["forecast_1m"] = {
            col: round(float(v), 4)
            for col, v in zip(panel.columns, forecast)
        }
    except Exception:
        pass

    # Granger causality: does each macro variable Granger-cause each NS factor?
    granger: dict = {}
    for factor in ["level", "slope", "curvature"]:
        granger[factor] = {}
        for macro_var in macro_data:
            try:
                pair = panel[[factor, macro_var]].dropna()
                gc = grangercausalitytests(pair, maxlag=lags, verbose=False)
                # Report minimum p-value across lags
                min_p = min(gc[lag][0]["ssr_ftest"][1] for lag in range(1, lags + 1))
                granger[factor][macro_var] = round(float(min_p), 4)
            except Exception:
                granger[factor][macro_var] = float("nan")
    result["granger"] = granger

    return result
