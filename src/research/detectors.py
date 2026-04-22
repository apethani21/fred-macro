"""Finding detectors.

Each detector is a pure function: takes already-loaded series data and
returns a list of `DetectorHit` instances (possibly empty). The orchestrator
is responsible for loading, aligning, filtering by significance, and
persisting.

A `DetectorHit` is intentionally minimal: the kind, the series it concerns,
the window, and a dict of numeric evidence. The finding-composition layer
turns this into prose and a markdown entry; that prose doesn't live here.

CLAUDE.md research-quality rules enforced in these detectors:
  - rolling/recent windows are always explicit (no cherry-picking);
  - robust statistics used where distributions are skewed;
  - non-stationary series are differenced before correlation;
  - every hit carries a stability measure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.analytics import stats as S


@dataclass
class DetectorHit:
    kind: str                             # e.g. "notable_move", "correlation_shift"
    series_ids: tuple[str, ...]
    window: str                           # human-readable: "rolling 252d", "full history"
    evidence: dict[str, Any]              # reproducible numeric payload
    score: float                          # rank-ordering within a scan (higher = more interesting)
    tags: tuple[str, ...] = field(default_factory=tuple)


# -------------------------------------------------------------------------
# Notable moves: latest observation is a tail event vs its own history.
# -------------------------------------------------------------------------

def detect_notable_move(
    series: pd.Series,
    *,
    series_id: str,
    z_threshold: float = 2.5,
    pct_tail: float = 0.01,          # flag top/bottom 1% of history
    diff_for_levels: bool = True,    # z-score the daily change, not the level, for non-stationary series
) -> list[DetectorHit]:
    """Flag when the latest observation is a tail event.

    Two independent checks — a hit can trigger either or both:
      1. Level percentile: where the current reading sits in the full history.
         Only flagged at the top/bottom `pct_tail` tails.
      2. Change z-score: the latest one-period change vs the distribution of
         all one-period changes. Uses robust (median/MAD) z-score so that
         prior tail events don't inflate the denominator.

    Returns 0-2 hits.
    """
    s = series.dropna()
    if s.size < 60:
        return []

    hits: list[DetectorHit] = []
    latest_val = float(s.iloc[-1])
    latest_date = s.index[-1]

    # Level percentile
    pct = S.percentile_rank(s)
    if pct <= pct_tail or pct >= 1 - pct_tail:
        hits.append(DetectorHit(
            kind="notable_move_level",
            series_ids=(series_id,),
            window=f"full history ({s.index[0].date()} → {latest_date.date()}, n={s.size})",
            evidence={
                "latest_value": latest_val,
                "latest_date": str(latest_date.date()),
                "percentile": float(pct),
                "full_min": float(s.min()),
                "full_max": float(s.max()),
                "full_median": float(s.median()),
                "n": int(s.size),
            },
            score=abs(pct - 0.5) * 2,     # 1.0 at the extremes, 0 at the median
            tags=("level-extreme",),
        ))

    # Change z-score (robust)
    if diff_for_levels:
        chg = s.diff().dropna()
        if chg.size >= 60:
            latest_chg = float(chg.iloc[-1])
            z = S.zscore_vs_history(chg, robust=True)
            if np.isfinite(z) and abs(z) >= z_threshold:
                hits.append(DetectorHit(
                    kind="notable_move_change",
                    series_ids=(series_id,),
                    window=f"1-period change, robust z vs {chg.size} prior changes",
                    evidence={
                        "latest_change": latest_chg,
                        "latest_date": str(latest_date.date()),
                        "robust_z": float(z),
                        "median_change": float(chg.median()),
                        "mad_change": float((chg - chg.median()).abs().median()),
                        "n": int(chg.size),
                    },
                    score=float(abs(z)),
                    tags=("change-extreme", "robust-z"),
                ))

    return hits


# -------------------------------------------------------------------------
# Correlation shift: recent rolling correlation differs from the full-sample
# baseline by enough to matter.
# -------------------------------------------------------------------------

def detect_correlation_shift(
    a: pd.Series,
    b: pd.Series,
    *,
    series_ids: tuple[str, str],
    window: int,
    min_shift: float = 0.4,
    max_baseline_instability: float = 0.5,
    min_peak_magnitude: float = 0.25,
    method: S.CorrMethod = "spearman",
    use_returns: bool = True,
) -> list[DetectorHit]:
    """Flag when recent rolling correlation deviates from the full-sample one.

    Two values compared:
      - `baseline`: full-sample correlation with stability spread across 4
        equal subperiods (our CLAUDE.md-mandated discipline check).
      - `recent`: the last observation of the same rolling-window correlation.

    Triggers when |recent − baseline| ≥ `min_shift`. Stability spread is
    always reported so the reader can judge whether `baseline` itself was a
    stable quantity to begin with (a big shift from an unstable baseline is
    less meaningful).

    For non-stationary series, pass `use_returns=True` (default). First
    difference for rates, pct_change for prices — mixing both in one scan
    is fine since the caller is already responsible for picking the pair.
    """
    if use_returns:
        a_use = a.diff()
        b_use = b.diff()
    else:
        a_use = a
        b_use = b

    aligned = pd.concat([a_use, b_use], axis=1).dropna()
    if aligned.shape[0] < window * 2:
        return []

    stab = S.correlation_with_stability(aligned.iloc[:, 0], aligned.iloc[:, 1], method=method, n_subperiods=4)

    rolling = S.rolling_corr(aligned.iloc[:, 0], aligned.iloc[:, 1], window=window, method=method)
    rolling = rolling.dropna()
    if rolling.empty:
        return []
    recent = float(rolling.iloc[-1])
    shift = recent - stab.correlation
    if abs(shift) < min_shift:
        return []
    # Quality gates: don't claim a shift when the baseline itself is unstable
    # (the "baseline" is then a weighted average of regime values, not a real
    # number to deviate from), and don't claim a shift when neither endpoint
    # is meaningfully non-zero (shift of +0.4 from −0.05 to +0.35 is more
    # interesting than from 0 to 0.4 in noise-vs-noise pairs).
    if stab.stability_spread > max_baseline_instability:
        return []
    if max(abs(stab.correlation), abs(recent)) < min_peak_magnitude:
        return []

    # Find the date the rolling correlation crossed the baseline ±min_shift/2
    # threshold most recently — useful for narration.
    crossing = rolling[(rolling - stab.correlation).abs() >= min_shift / 2]
    crossed_at = crossing.index[0] if not crossing.empty else rolling.index[-1]

    return [DetectorHit(
        kind="correlation_shift",
        series_ids=series_ids,
        window=f"rolling {window}, {method}",
        evidence={
            "recent_correlation": recent,
            "recent_date": str(rolling.index[-1].date()),
            "baseline_correlation": float(stab.correlation),
            "baseline_stability_spread": float(stab.stability_spread),
            "baseline_subperiod_values": [round(v, 4) for v in stab.subperiod_values],
            "shift": float(shift),
            "crossed_threshold_at": str(crossed_at.date()),
            "n_overlap": int(stab.n),
            "method": method,
            "on_returns": bool(use_returns),
        },
        score=float(abs(shift)),
        tags=("regime-shift",) if abs(shift) > 0.6 else (),
    )]


# -------------------------------------------------------------------------
# Lead-lag change: whether the lag at which two series correlate most
# strongly has moved between a historical window and a recent window.
# -------------------------------------------------------------------------

def detect_lead_lag_change(
    a: pd.Series,
    b: pd.Series,
    *,
    series_ids: tuple[str, str],
    max_lag: int,
    recent_points: int,
    min_peak_correlation: float = 0.25,
    method: S.CorrMethod = "spearman",
    use_returns: bool = True,
) -> list[DetectorHit]:
    """Compare lead-lag peak between the full history (minus recent) and the recent tail.

    A 'change' is flagged when the peak lag *or* the peak correlation has
    moved substantially between the two windows. Peak lag is a discrete
    quantity, so any integer shift is reported; peak correlation is flagged
    when it moves by ≥ 0.3.

    Requires at least ~4x `recent_points` of history for the baseline.
    """
    if use_returns:
        a = a.diff()
        b = b.diff()

    aligned = pd.concat([a, b], axis=1).dropna()
    n = aligned.shape[0]
    if n < recent_points * 4 or recent_points < max_lag * 4:
        return []

    hist = aligned.iloc[: n - recent_points]
    recent = aligned.iloc[n - recent_points:]

    def _peak(df: pd.DataFrame) -> tuple[int, float]:
        xc = S.lead_lag_xcorr(df.iloc[:, 0], df.iloc[:, 1], max_lag=max_lag, method=method).dropna(subset=["correlation"])
        if xc.empty:
            return (0, float("nan"))
        row = xc.loc[xc["correlation"].abs().idxmax()]
        return int(row["lag"]), float(row["correlation"])

    hist_lag, hist_corr = _peak(hist)
    recent_lag, recent_corr = _peak(recent)
    lag_shift = recent_lag - hist_lag
    corr_shift = recent_corr - hist_corr

    if abs(lag_shift) < 1 and abs(corr_shift) < 0.3:
        return []
    # Quality gate: "peak lag" is only meaningful when the peak correlation
    # is itself non-trivial. If both endpoints are below the threshold the
    # peak lag is noise; don't fabricate a "lag shift" story from that.
    if max(abs(hist_corr), abs(recent_corr)) < min_peak_correlation:
        return []

    score = float(abs(lag_shift)) + float(abs(corr_shift))
    return [DetectorHit(
        kind="lead_lag_change",
        series_ids=series_ids,
        window=f"recent {recent_points} obs vs prior {n - recent_points}, max_lag={max_lag}",
        evidence={
            "hist_peak_lag": hist_lag,
            "hist_peak_correlation": hist_corr,
            "recent_peak_lag": recent_lag,
            "recent_peak_correlation": recent_corr,
            "lag_shift": lag_shift,
            "correlation_shift": corr_shift,
            "method": method,
            "on_returns": bool(use_returns),
            "n_hist": int(hist.shape[0]),
            "n_recent": int(recent.shape[0]),
        },
        score=score,
    )]


# -------------------------------------------------------------------------
# Regime transition: a quantile regime boundary was crossed recently.
# -------------------------------------------------------------------------

def detect_regime_transition(
    series: pd.Series,
    *,
    series_id: str,
    thresholds: tuple[float, ...] = (0.33, 0.67),
    labels: tuple[str, ...] = ("low", "mid", "high"),
    recent_days: int = 90,
) -> list[DetectorHit]:
    """Flag when the series has entered a new quantile regime within `recent_days`."""
    s = series.dropna()
    if s.size < 100:
        return []
    regime = S.quantile_regime(s, thresholds=thresholds, labels=labels).dropna()
    if regime.empty:
        return []

    # Find the most recent regime change.
    changed = regime.ne(regime.shift())
    change_dates = regime.index[changed]
    if len(change_dates) < 2:      # only the boundary at t0; never actually changed
        return []
    last_change_date = change_dates[-1]
    cutoff = s.index[-1] - pd.Timedelta(days=recent_days)
    if last_change_date < cutoff:
        return []

    # How long was the previous regime?
    prev_change_date = change_dates[-2]
    prior_regime_days = int((last_change_date - prev_change_date).days)
    new_regime = str(regime.iloc[-1])
    prior_regime = str(regime.loc[change_dates[-2]: last_change_date - pd.Timedelta(days=1)].iloc[0]) if prev_change_date != last_change_date else "?"
    cuts = [float(s.quantile(q)) for q in thresholds]

    return [DetectorHit(
        kind="regime_transition",
        series_ids=(series_id,),
        window=f"quantile regime (thresholds={thresholds}), recent {recent_days}d",
        evidence={
            "new_regime": new_regime,
            "prior_regime": prior_regime,
            "transition_date": str(last_change_date.date()),
            "prior_regime_duration_days": prior_regime_days,
            "thresholds": list(thresholds),
            "cut_values": cuts,
            "latest_value": float(s.iloc[-1]),
            "latest_date": str(s.index[-1].date()),
        },
        score=float(prior_regime_days) / 365.0,     # longer prior regime → more meaningful flip
        tags=("regime-shift",),
    )]


# -------------------------------------------------------------------------
# Structural: cointegration that has broken down.
# -------------------------------------------------------------------------

def detect_cointegration_break(
    a: pd.Series,
    b: pd.Series,
    *,
    series_ids: tuple[str, str],
    recent_points: int,
) -> list[DetectorHit]:
    """Flag when full-history Engle-Granger cointegration was significant but
    the recent window alone is not. Useful for "the 2s10s / Fed funds link
    has broken down" type findings.

    Both tests use 5% significance. Requires at least 50 recent points.
    """
    aligned = pd.concat([a, b], axis=1).dropna()
    n = aligned.shape[0]
    if n < recent_points * 2 or recent_points < 50:
        return []

    full = S.cointegration_test(aligned.iloc[:, 0], aligned.iloc[:, 1])
    recent = S.cointegration_test(aligned.iloc[n - recent_points:].iloc[:, 0],
                                  aligned.iloc[n - recent_points:].iloc[:, 1])

    # Only interested when full sample was cointegrated and recent is not.
    if full.p_value >= 0.05 or recent.p_value < 0.05:
        return []

    return [DetectorHit(
        kind="cointegration_break",
        series_ids=series_ids,
        window=f"full n={full.n} vs recent n={recent.n}",
        evidence={
            "full_sample_p": full.p_value,
            "full_sample_statistic": full.statistic,
            "recent_p": recent.p_value,
            "recent_statistic": recent.statistic,
            "recent_n": recent.n,
            "full_n": full.n,
        },
        score=float(recent.p_value - full.p_value),
    )]
