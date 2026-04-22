"""Statistics with the discipline from CLAUDE.md baked in.

Wraps pandas/scipy/statsmodels with opinionated defaults (Spearman for
correlations, stability check required, robust stats preferred). Don't
reimplement these elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata, spearmanr
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests

CorrMethod = Literal["pearson", "spearman"]


# ---------- rolling statistics ----------

def rolling_corr(
    a: pd.Series,
    b: pd.Series,
    window: int,
    method: CorrMethod = "spearman",
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling correlation with window-local ranks for Spearman.

    For non-stationary series (levels of rates/prices), pass returns in —
    see `analytics.data.to_returns`.
    """
    min_periods = min_periods or window
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if method == "pearson":
        return df["a"].rolling(window, min_periods=min_periods).corr(df["b"])
    if method != "spearman":
        raise ValueError(f"Unknown method {method!r}")

    # Window-local ranks then Pearson is Spearman by definition.
    def _spearman(arr: np.ndarray) -> float:
        n = len(arr) // 2
        x, y = arr[:n], arr[n:]
        if len(x) < min_periods:
            return np.nan
        return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])

    # Pack both columns into one array per window so we can use `apply(raw=True)`.
    stacked = np.concatenate([df["a"].values, df["b"].values])
    idx = df.index
    out = np.full(len(idx), np.nan)
    for i in range(min_periods - 1, len(idx)):
        start = max(0, i - window + 1)
        x = df["a"].values[start:i + 1]
        y = df["b"].values[start:i + 1]
        if len(x) >= min_periods:
            out[i] = np.corrcoef(rankdata(x), rankdata(y))[0, 1]
    return pd.Series(out, index=idx, name=f"rolling_{method}_{window}")


class CorrelationStability(NamedTuple):
    correlation: float
    method: CorrMethod
    n: int
    subperiod_values: list[float]
    stability_spread: float  # max - min across subperiods


def correlation_with_stability(
    a: pd.Series,
    b: pd.Series,
    method: CorrMethod = "spearman",
    n_subperiods: int = 4,
) -> CorrelationStability:
    """Full-sample correlation plus a subperiod-stability measure.

    CLAUDE.md research quality: when claiming a relationship, report the
    magnitude, window, and a measure of stability. `stability_spread` is the
    max-min of correlations computed over `n_subperiods` equal-length chunks
    of the overlap.
    """
    df = pd.concat([a, b], axis=1).dropna()
    if df.shape[0] < n_subperiods * 2:
        raise ValueError(f"Too few overlapping points ({df.shape[0]}) for {n_subperiods} subperiods")
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    if method == "pearson":
        full = float(np.corrcoef(x, y)[0, 1])
    else:
        full = float(spearmanr(x, y).statistic)

    n = df.shape[0]
    edges = np.linspace(0, n, n_subperiods + 1, dtype=int)
    sub_vals = []
    for i in range(n_subperiods):
        chunk = df.iloc[edges[i]:edges[i + 1]]
        xc = chunk.iloc[:, 0].values
        yc = chunk.iloc[:, 1].values
        if method == "pearson":
            sub_vals.append(float(np.corrcoef(xc, yc)[0, 1]))
        else:
            sub_vals.append(float(spearmanr(xc, yc).statistic))
    return CorrelationStability(
        correlation=full,
        method=method,
        n=df.shape[0],
        subperiod_values=sub_vals,
        stability_spread=max(sub_vals) - min(sub_vals),
    )


def rolling_vol(
    returns: pd.Series,
    window: int,
    annualize_periods: int | None = None,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling standard deviation of returns. Pass `annualize_periods=252` (daily)
    or 12 (monthly) to annualize; omit for raw.
    """
    vol = returns.rolling(window, min_periods=min_periods or window).std()
    if annualize_periods:
        vol = vol * np.sqrt(annualize_periods)
    return vol


# ---------- percentile / z-score ----------

def zscore_vs_history(s: pd.Series, lookback: int | None = None, robust: bool = False) -> float:
    """Z-score of the latest observation vs its history.

    When `robust=True`, use median + MAD instead of mean + stdev. CLAUDE.md
    flags robust statistics as preferred for skewed distributions.
    """
    x = s.dropna()
    if lookback:
        x = x.iloc[-lookback:]
    if x.size < 3:
        return float("nan")
    latest = x.iloc[-1]
    if robust:
        med = float(x.median())
        mad = float((x - med).abs().median()) or float("nan")
        return (latest - med) / (1.4826 * mad) if mad else float("nan")
    mu = float(x.mean())
    sd = float(x.std()) or float("nan")
    return (latest - mu) / sd if sd else float("nan")


def percentile_rank(s: pd.Series, value: float | None = None, lookback: int | None = None) -> float:
    """Percentile of `value` (default: latest) within the series's history.

    Returns a float in [0, 1]. CLAUDE.md: default to full history unless
    a window is specified.
    """
    x = s.dropna()
    if lookback:
        x = x.iloc[-lookback:]
    if x.empty:
        return float("nan")
    v = x.iloc[-1] if value is None else value
    return float((x <= v).mean())


# ---------- lead-lag ----------

def lead_lag_xcorr(
    a: pd.Series,
    b: pd.Series,
    max_lag: int,
    method: CorrMethod = "spearman",
) -> pd.DataFrame:
    """Cross-correlation at lags in [-max_lag, +max_lag].

    Positive lag means `b` leads `a` by `lag` periods (we shift `b` forward).
    Returns columns: lag, correlation, n.
    """
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        aligned = pd.concat([a, b.shift(-lag).rename(b.name)], axis=1).dropna()
        if len(aligned) < 10:
            rows.append({"lag": lag, "correlation": float("nan"), "n": len(aligned)})
            continue
        x = aligned.iloc[:, 0].values
        y = aligned.iloc[:, 1].values
        if method == "pearson":
            c = float(np.corrcoef(x, y)[0, 1])
        else:
            c = float(spearmanr(x, y).statistic)
        rows.append({"lag": lag, "correlation": c, "n": len(aligned)})
    return pd.DataFrame(rows)


# ---------- regime labels ----------

def quantile_regime(
    s: pd.Series, thresholds: tuple[float, ...] = (0.33, 0.67), labels: tuple[str, ...] = ("low", "mid", "high")
) -> pd.Series:
    """Label each point by quantile bucket of the full series history.

    `thresholds` must be sorted ascending in (0,1); `labels` length = thresholds + 1.
    """
    if len(labels) != len(thresholds) + 1:
        raise ValueError("labels must be one longer than thresholds")
    cuts = [s.quantile(q) for q in thresholds]
    bins = [-np.inf, *cuts, np.inf]
    return pd.cut(s, bins=bins, labels=list(labels), include_lowest=True)


# ---------- statistical tests (thin wrappers) ----------

@dataclass
class CointegrationResult:
    statistic: float
    p_value: float
    critical_values: dict[str, float]
    n: int


def cointegration_test(a: pd.Series, b: pd.Series) -> CointegrationResult:
    """Engle-Granger cointegration via statsmodels. Both series must be
    non-stationary in levels; if either is stationary the result is moot.
    """
    df = pd.concat([a, b], axis=1).dropna()
    stat, p, crit = coint(df.iloc[:, 0], df.iloc[:, 1])
    return CointegrationResult(
        statistic=float(stat),
        p_value=float(p),
        critical_values={"1%": float(crit[0]), "5%": float(crit[1]), "10%": float(crit[2])},
        n=int(df.shape[0]),
    )


@dataclass
class StationarityResult:
    statistic: float
    p_value: float
    n: int
    is_stationary_5pct: bool


def adfuller_test(s: pd.Series) -> StationarityResult:
    """Augmented Dickey-Fuller; null = unit root (non-stationary)."""
    x = s.dropna()
    stat, p, *_ = adfuller(x)
    return StationarityResult(
        statistic=float(stat),
        p_value=float(p),
        n=int(x.size),
        is_stationary_5pct=bool(p < 0.05),
    )


# ---------- PCA (yield curve and multi-series decomposition) ----------

@dataclass
class PCAResult:
    components: pd.DataFrame          # index = PC1..PCk, columns = input series
    explained_variance_ratio: pd.Series  # index = PC1..PCk
    scores: pd.DataFrame              # index = time, columns = PC1..PCk (standardized projections)
    means: pd.Series                  # per-column training means
    stds: pd.Series                   # per-column training stds (for standardized PCA)


def pca(
    df: pd.DataFrame,
    n_components: int | None = None,
    standardize: bool = True,
) -> PCAResult:
    """Principal components via SVD. Drops rows with any NaN.

    For yield curves, PC1 ≈ level, PC2 ≈ slope, PC3 ≈ curvature
    (Litterman & Scheinkman 1991). Pass `standardize=False` when inputs share
    units and magnitudes (e.g., the Treasury curve); `True` otherwise.
    """
    x = df.dropna(how="any")
    if x.empty:
        raise ValueError("No rows without NaN — align/ffill before PCA")
    means = x.mean()
    stds = x.std(ddof=0) if standardize else pd.Series(1.0, index=x.columns)
    centered = (x - means) / stds
    # SVD on centered data: centered = U S Vt, components are rows of Vt
    _, s, vt = np.linalg.svd(centered.values, full_matrices=False)
    k = n_components or vt.shape[0]
    comp = pd.DataFrame(
        vt[:k], index=[f"PC{i + 1}" for i in range(k)], columns=x.columns
    )
    var = (s ** 2) / (x.shape[0] - 1)
    evr = pd.Series(var[:k] / var.sum(), index=comp.index, name="explained_variance_ratio")
    scores = pd.DataFrame(
        centered.values @ comp.values.T,
        index=x.index, columns=comp.index,
    )
    return PCAResult(components=comp, explained_variance_ratio=evr, scores=scores, means=means, stds=stds)


# ---------- regime transitions ----------

def transition_matrix(
    regimes: pd.Series,
    as_probabilities: bool = True,
) -> pd.DataFrame:
    """Empirical one-step transition matrix from a categorical regime series.

    Row = state at t; column = state at t+1. With `as_probabilities=True`,
    rows sum to 1. Useful for characterizing persistence of quantile / NBER /
    custom regime labels.
    """
    r = regimes.dropna()
    if r.empty:
        raise ValueError("regimes is empty after dropna")
    pairs = pd.DataFrame({"from": r.shift(1), "to": r}).dropna()
    counts = pairs.groupby(["from", "to"]).size().unstack(fill_value=0)
    # Keep a square matrix over the full observed vocabulary.
    states = sorted(set(r.dropna().unique()))
    counts = counts.reindex(index=states, columns=states, fill_value=0)
    if not as_probabilities:
        return counts
    row_totals = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(row_totals, axis=0).fillna(0.0)


# ---------- local projections (Jordà 2005 IRFs) ----------

@dataclass
class LocalProjectionResult:
    horizons: np.ndarray              # 0..H
    coefficients: pd.Series           # β_h on the shock at each horizon
    std_errors: pd.Series             # HAC (Newey-West) SEs
    ci_lower: pd.Series               # 95% lower bound
    ci_upper: pd.Series               # 95% upper bound
    n_obs: pd.Series                  # obs used at each horizon


def local_projection(
    y: pd.Series,
    shock: pd.Series,
    horizons: int = 12,
    controls: pd.DataFrame | None = None,
    lags: int = 4,
    hac_maxlags: int | None = None,
    ci: float = 0.95,
) -> LocalProjectionResult:
    """Jordà (2005) local-projection impulse response of `y` to `shock`.

    For each horizon h in [0, horizons], regresses y_{t+h} on shock_t plus
    `lags` own-lags of y and optional contemporaneous `controls`. Standard
    errors use Newey-West HAC (default maxlag = horizons) to handle the
    overlapping observations at h > 0.

    Returns β_h and its CI — plot β_h vs h for an impulse-response chart.
    Interpretable when `shock` is an identified innovation (monetary-policy
    surprise, oil shock, etc.), noisy otherwise.
    """
    from scipy import stats as _sstats

    if hac_maxlags is None:
        hac_maxlags = max(horizons, 1)
    z = _sstats.norm.ppf(0.5 + ci / 2)

    df = pd.concat(
        [y.rename("y"), shock.rename("shock"),
         *([controls] if controls is not None else [])],
        axis=1,
    ).dropna()
    # Own lags of y as controls; cuts sample by `lags` at the head.
    for L in range(1, lags + 1):
        df[f"y_lag{L}"] = df["y"].shift(L)
    df = df.dropna()

    betas, ses, lowers, uppers, ns = {}, {}, {}, {}, {}
    for h in range(horizons + 1):
        target = df["y"].shift(-h)
        design_cols = ["shock"] + [f"y_lag{L}" for L in range(1, lags + 1)]
        if controls is not None:
            design_cols += [c for c in controls.columns if c in df.columns]
        design = df[design_cols]
        subset = pd.concat([target.rename("target"), design], axis=1).dropna()
        if subset.shape[0] < len(design_cols) + 5:
            betas[h] = np.nan; ses[h] = np.nan; lowers[h] = np.nan; uppers[h] = np.nan; ns[h] = subset.shape[0]
            continue
        y_h = subset["target"].values
        X = sm.add_constant(subset[design_cols].values)
        model = sm.OLS(y_h, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})
        # Coefficient on `shock` is the 2nd entry (after the constant).
        b = float(model.params[1])
        se = float(model.bse[1])
        betas[h] = b; ses[h] = se
        lowers[h] = b - z * se; uppers[h] = b + z * se
        ns[h] = subset.shape[0]

    idx = np.arange(horizons + 1)
    return LocalProjectionResult(
        horizons=idx,
        coefficients=pd.Series(betas, name="beta"),
        std_errors=pd.Series(ses, name="se"),
        ci_lower=pd.Series(lowers, name="ci_lower"),
        ci_upper=pd.Series(uppers, name="ci_upper"),
        n_obs=pd.Series(ns, name="n"),
    )


def granger_min_p(y: pd.Series, x: pd.Series, max_lag: int = 4) -> dict[int, float]:
    """Granger-causality F-test p-values for x → y at each lag up to `max_lag`.

    Thin wrapper; statsmodels reports multiple test flavors — we take the
    F-test ('ssr_ftest') which is the common reporting choice.
    """
    df = pd.concat([y, x], axis=1).dropna()
    res = grangercausalitytests(df, maxlag=max_lag, verbose=False)
    return {lag: float(v[0]["ssr_ftest"][1]) for lag, v in res.items()}
