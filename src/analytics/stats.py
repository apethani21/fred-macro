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


# ---------- event study ----------

@dataclass
class EventStudyResult:
    horizons: np.ndarray               # lag values, e.g. [-1, 0, 1, 2, 3, 4, 5]
    mean_change: pd.Series             # average 1-day change at each lag across events
    cumulative_change: pd.Series       # average cumulative change from lag=0
    std_change: pd.Series              # cross-event std of 1-day changes
    n_events: pd.Series                # non-null event count at each horizon


def event_study(
    series: pd.Series,
    event_dates: list[pd.Timestamp],
    pre: int = 1,
    post: int = 5,
) -> EventStudyResult:
    """Average cumulative response of `series` around `event_dates`.

    For each lag in [-pre, +post], extracts the 1-day change in `series` on
    the nearest available trading day and averages across events.

    Cumulative change is computed relative to the event day (lag=0): it is the
    sum of mean_change from lag=1 onward (lag=0 is the same-day move).

    Useful for any identified event set: FOMC meetings, macro releases, etc.
    """
    s = series.dropna().sort_index()
    lags = list(range(-pre, post + 1))

    changes_by_lag: dict[int, list[float]] = {lag: [] for lag in lags}

    for ev_date in event_dates:
        for lag in lags:
            target = ev_date + pd.Timedelta(days=lag)
            # Find nearest index within ±1 calendar day.
            candidates = s.index[
                (s.index >= target - pd.Timedelta(days=1)) &
                (s.index <= target + pd.Timedelta(days=1))
            ]
            if len(candidates) == 0:
                continue
            actual = candidates[abs(candidates - target).argmin()]
            pos = s.index.get_loc(actual)
            if pos == 0:
                continue
            chg = float(s.iloc[pos] - s.iloc[pos - 1])
            changes_by_lag[lag].append(chg)

    horizons = np.array(lags)
    mean_chg = pd.Series(
        {lag: float(np.mean(v)) if v else float("nan") for lag, v in changes_by_lag.items()},
        name="mean_change",
    )
    std_chg = pd.Series(
        {lag: float(np.std(v, ddof=1)) if len(v) > 1 else float("nan") for lag, v in changes_by_lag.items()},
        name="std_change",
    )
    n_ev = pd.Series({lag: len(v) for lag, v in changes_by_lag.items()}, name="n_events")

    # Cumulative from lag=0 inclusive.
    cumul_vals: dict[int, float] = {}
    running = 0.0
    for lag in lags:
        v = mean_chg.get(lag, float("nan"))
        if not np.isnan(v):
            running += v
        cumul_vals[lag] = running if lag >= 0 else float("nan")
    cumul = pd.Series(cumul_vals, name="cumulative_change")

    return EventStudyResult(
        horizons=horizons,
        mean_change=mean_chg,
        cumulative_change=cumul,
        std_change=std_chg,
        n_events=n_ev,
    )


# ---------- structural breaks (Bai-Perron via dynamic programming) ----------

@dataclass
class StructuralBreakResult:
    break_dates: list[pd.Timestamp]   # estimated break dates (left edge of new regime)
    n_breaks: int                      # number of breaks in the selected model
    regimes: pd.Series                 # 0-indexed integer label per observation
    regime_stats: pd.DataFrame         # columns: regime, start, end, n, mean, std
    bic_selected: bool                 # True if n_breaks was BIC-chosen
    bic_curve: pd.Series | None        # BIC at k=0..max_breaks when BIC-selected


def structural_breaks(
    s: pd.Series,
    max_breaks: int = 5,
    n_breaks: int | None = None,
    min_segment: int | None = None,
    resample: str | None = None,
    max_obs: int = 1500,
) -> StructuralBreakResult:
    """Bai-Perron structural break detection via dynamic programming (ruptures Dynp, l2 cost).

    Detects mean shifts in `s`. Pass a rolling-correlation series, a level
    series (rates, spreads), or first-differences — the l2 cost tests for
    shifts in the conditional mean.

    Parameters
    ----------
    s            : Series with a DatetimeIndex.
    max_breaks   : Maximum number of breaks to consider for BIC selection.
    n_breaks     : If set, skips BIC and uses this many breaks directly.
    min_segment  : Minimum observations per segment. Rule of thumb: 24 for
                   monthly, 8 for quarterly, 5 for annual. Defaults to
                   max(len(s) // (max_breaks + 2), 3).
    resample     : Pandas offset alias ('ME', 'QE', 'YE') to resample before
                   fitting. Applied before max_obs check.
    max_obs      : If series exceeds this length after any resampling, auto-
                   resample to monthly ('ME'). Dynp is O(n²); daily series
                   of 15k obs will hang. Default 1500.

    Returns
    -------
    StructuralBreakResult with break_dates, regime_stats, and BIC curve.
    """
    import ruptures as rpt

    x = s.dropna()
    if x.size < 10:
        raise ValueError(f"Series too short after dropna ({x.size} obs)")

    if resample:
        x = x.resample(resample).mean().dropna()
    if x.size > max_obs:
        x = x.resample("ME").mean().dropna()

    n = x.size
    if min_segment is None:
        min_segment = max(n // (max_breaks + 2), 3)

    signal = x.values.reshape(-1, 1)
    algo = rpt.Dynp(model="l2", min_size=min_segment, jump=1).fit(signal)

    bic_selected = n_breaks is None
    bic_curve: pd.Series | None = None

    if bic_selected:
        # BIC: n*log(RSS/n) + (k+1)*log(n)  where k+1 = number of segments
        bics: dict[int, float] = {}
        for k in range(0, max_breaks + 1):
            try:
                bkps = algo.predict(n_bkps=k)
                # ruptures returns breakpoint indices (right-exclusive); last is always n
                rss = _rss_from_breakpoints(signal[:, 0], bkps)
                bics[k] = n * np.log(rss / n) + (k + 1) * np.log(n)
            except rpt.exceptions.BadSegmentationParameters:
                break
        if not bics:
            raise ValueError("No valid segmentation found — try reducing max_breaks or min_segment")
        best_k = int(min(bics, key=bics.__getitem__))
        bic_curve = pd.Series(bics, name="bic")
        n_breaks = best_k

    bkps = algo.predict(n_bkps=n_breaks)  # type: ignore[arg-type]
    # bkps is a list of right-exclusive indices, last = n; convert to dates
    break_indices = [b for b in bkps if b < n]  # drop the trailing sentinel
    break_dates = [x.index[i] for i in break_indices]

    # Build regime label series
    boundaries = [0] + break_indices + [n]
    labels = np.empty(n, dtype=int)
    for regime_idx, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        labels[lo:hi] = regime_idx
    regimes = pd.Series(labels, index=x.index, name="regime")

    # Regime summary stats
    rows = []
    for r in range(n_breaks + 1):  # type: ignore[operator]
        seg = x[regimes == r]
        rows.append({
            "regime": r,
            "start": seg.index[0],
            "end": seg.index[-1],
            "n": len(seg),
            "mean": float(seg.mean()),
            "std": float(seg.std()),
        })
    regime_stats = pd.DataFrame(rows)

    return StructuralBreakResult(
        break_dates=break_dates,
        n_breaks=n_breaks,  # type: ignore[arg-type]
        regimes=regimes,
        regime_stats=regime_stats,
        bic_selected=bic_selected,
        bic_curve=bic_curve,
    )


def _rss_from_breakpoints(signal: np.ndarray, bkps: list[int]) -> float:
    """Total residual sum of squares for a piecewise-constant fit."""
    rss = 0.0
    prev = 0
    for b in bkps:
        seg = signal[prev:b]
        rss += float(np.sum((seg - seg.mean()) ** 2))
        prev = b
    return max(rss, 1e-12)  # guard against log(0) when signal is constant


def granger_min_p(y: pd.Series, x: pd.Series, max_lag: int = 4) -> dict[int, float]:
    """Granger-causality F-test p-values for x → y at each lag up to `max_lag`.

    Thin wrapper; statsmodels reports multiple test flavors — we take the
    F-test ('ssr_ftest') which is the common reporting choice.
    """
    df = pd.concat([y, x], axis=1).dropna()
    res = grangercausalitytests(df, maxlag=max_lag, verbose=False)
    return {lag: float(v[0]["ssr_ftest"][1]) for lag, v in res.items()}
