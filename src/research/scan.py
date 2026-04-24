"""Research-phase orchestrator.

Walks the curated core universe and curated relationship pairs, calls the
detectors, filters by significance, converts hits into `Finding` records,
and persists to `knowledge/findings.md` + `data/stats/*.parquet`.

The orchestrator is the only place that does I/O and prose synthesis; the
detectors stay pure.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

import pandas as pd

from src.analytics import data as D
from src.analytics import stats as S

from .config import CORE_PAIRS, NOTABLE_MOVE_WATCHLIST
from .questions import generate_questions
from .relationship_config import relationships_as_pairs
from .relationship_monitor import run_relationship_monitor
from .detectors import (
    DetectorHit,
    detect_breakeven_anomaly,
    detect_cointegration_break,
    detect_correlation_shift,
    detect_cp_factor_signal,
    detect_inflation_episode_anomaly,
    detect_lead_lag_change,
    detect_notable_move,
    detect_regime_transition,
    detect_structural_break,
)
from .findings import Finding, append_stats, existing_slugs, make_slug, write_findings_md

log = logging.getLogger(__name__)


@dataclass
class ScanReport:
    hits: list[DetectorHit] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)   # (series or pair, reason)
    stats_rows_written: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        by_kind: dict[str, int] = {}
        for h in self.hits:
            by_kind[h.kind] = by_kind.get(h.kind, 0) + 1
        lines = [f"Total hits: {len(self.hits)}"]
        for k, n in sorted(by_kind.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {k}: {n}")
        if self.skipped:
            lines.append(f"Skipped: {len(self.skipped)} (first 5: {self.skipped[:5]})")
        if self.stats_rows_written:
            lines.append("Stats rows written:")
            for name, n in self.stats_rows_written.items():
                lines.append(f"  {name}: {n}")
        return "\n".join(lines)


# -------------------------------------------------------------------------
# Per-detector runners
# -------------------------------------------------------------------------

def scan_notable_moves(watchlist: Iterable[str]) -> tuple[list[DetectorHit], pd.DataFrame, list[tuple[str, str]]]:
    """Scan a watchlist for level / change tail events. Returns (hits, stats_rows, skipped)."""
    hits: list[DetectorHit] = []
    skipped: list[tuple[str, str]] = []
    rows: list[dict] = []

    for sid in watchlist:
        try:
            s = D.load_series(sid)
        except (FileNotFoundError, KeyError) as e:
            skipped.append((sid, f"load: {e}"))
            continue
        found = detect_notable_move(s, series_id=sid)
        for h in found:
            hits.append(h)
            if h.kind == "notable_move_level":
                rows.append({
                    "series_id": sid,
                    "date": h.evidence["latest_date"],
                    "kind": "level",
                    "value": h.evidence["latest_value"],
                    "z_score": float("nan"),
                    "percentile": h.evidence["percentile"],
                    "description": f"level percentile={h.evidence['percentile']:.3f}",
                })
            elif h.kind == "notable_move_change":
                rows.append({
                    "series_id": sid,
                    "date": h.evidence["latest_date"],
                    "kind": "change",
                    "value": h.evidence["latest_change"],
                    "z_score": h.evidence["robust_z"],
                    "percentile": float("nan"),
                    "description": f"1-period change robust z={h.evidence['robust_z']:.2f}",
                })

    stats_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "series_id", "date", "kind", "value", "z_score", "percentile", "description",
    ])
    return hits, stats_df, skipped


def scan_correlation_shifts(
    pairs: Iterable[tuple[str, str, str]],
    window_daily: int = 252,
    window_monthly: int = 36,
) -> tuple[list[DetectorHit], pd.DataFrame, list[tuple[str, str]]]:
    hits: list[DetectorHit] = []
    skipped: list[tuple[str, str]] = []
    rows: list[dict] = []

    for a_id, b_id, label in pairs:
        try:
            df = D.load_aligned([a_id, b_id], how="coarsest", dropna="any")
        except (FileNotFoundError, KeyError) as e:
            skipped.append((f"{a_id}×{b_id}", f"load: {e}"))
            continue

        # Pick a window appropriate to the aligned frequency.
        try:
            a_freq = D.series_metadata(a_id).get("frequency_short")
            b_freq = D.series_metadata(b_id).get("frequency_short")
        except KeyError:
            a_freq = b_freq = "D"
        coarse = {"D", "W", "M", "Q", "A"}
        # load_aligned defaults to coarsest; infer it back.
        coarser = a_freq if a_freq in coarse and b_freq in coarse and ({a_freq, b_freq} & {"A"}) or a_freq == b_freq else "M"
        window = window_monthly if coarser in ("M", "Q", "A") else window_daily

        if df.shape[0] < window * 2:
            skipped.append((f"{a_id}×{b_id}", f"insufficient overlap ({df.shape[0]} < {window*2})"))
            continue

        found = detect_correlation_shift(df[a_id], df[b_id], series_ids=(a_id, b_id), window=window)
        for h in found:
            h.evidence["label"] = label
            hits.append(h)

        # Always persist the full rolling-correlation series (downsampled) for this pair,
        # regardless of whether a hit fired — it becomes the audit trail for the
        # baseline/stability the detector reported.
        rc = S.rolling_corr(df[a_id].diff(), df[b_id].diff(), window=window, method="spearman").dropna()
        # Thin to roughly 1 point per month to bound file size.
        rc_thin = rc.iloc[::max(1, window // 12)]
        for dt, v in rc_thin.items():
            rows.append({
                "series_a": a_id,
                "series_b": b_id,
                "window": window,
                "date": dt.date().isoformat(),
                "correlation": float(v),
                "method": "spearman",
            })

    stats_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "series_a", "series_b", "window", "date", "correlation", "method",
    ])
    return hits, stats_df, skipped


def scan_lead_lag(
    pairs: Iterable[tuple[str, str, str]],
    recent_points_daily: int = 252,
    recent_points_monthly: int = 24,
    max_lag_daily: int = 60,
    max_lag_monthly: int = 6,
) -> tuple[list[DetectorHit], pd.DataFrame, list[tuple[str, str]]]:
    hits: list[DetectorHit] = []
    skipped: list[tuple[str, str]] = []
    rows: list[dict] = []

    for a_id, b_id, label in pairs:
        try:
            df = D.load_aligned([a_id, b_id], how="coarsest", dropna="any")
        except (FileNotFoundError, KeyError) as e:
            skipped.append((f"{a_id}×{b_id}", f"load: {e}"))
            continue

        # Infer the aligned frequency by the spacing of the index.
        if df.shape[0] < 60:
            skipped.append((f"{a_id}×{b_id}", f"insufficient overlap ({df.shape[0]})"))
            continue
        median_gap = (df.index[1:] - df.index[:-1]).days
        is_daily = median_gap.size and float(pd.Series(median_gap).median()) <= 7
        recent_points = recent_points_daily if is_daily else recent_points_monthly
        max_lag = max_lag_daily if is_daily else max_lag_monthly

        found = detect_lead_lag_change(
            df[a_id], df[b_id],
            series_ids=(a_id, b_id),
            max_lag=max_lag,
            recent_points=recent_points,
        )
        for h in found:
            h.evidence["label"] = label
            hits.append(h)

        # Persist current-window lead-lag curve (full sample) as the audit trail.
        xc = S.lead_lag_xcorr(df[a_id].diff(), df[b_id].diff(), max_lag=max_lag, method="spearman")
        for _, row in xc.iterrows():
            rows.append({
                "series_a": a_id,
                "series_b": b_id,
                "window": f"full:{df.shape[0]}",
                "lag": int(row["lag"]),
                "correlation": float(row["correlation"]) if pd.notna(row["correlation"]) else float("nan"),
                "n": int(row["n"]),
            })

    stats_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "series_a", "series_b", "window", "lag", "correlation", "n",
    ])
    return hits, stats_df, skipped


def scan_regime_transitions(watchlist: Iterable[str]) -> tuple[list[DetectorHit], pd.DataFrame, list[tuple[str, str]]]:
    hits: list[DetectorHit] = []
    skipped: list[tuple[str, str]] = []
    rows: list[dict] = []

    for sid in watchlist:
        try:
            s = D.load_series(sid)
        except (FileNotFoundError, KeyError) as e:
            skipped.append((sid, f"load: {e}"))
            continue
        found = detect_regime_transition(s, series_id=sid)
        for h in found:
            hits.append(h)
        # Persist current regime labels (tail only, to keep size bounded).
        regime = S.quantile_regime(s).dropna()
        for dt, lbl in regime.iloc[-250:].items():
            rows.append({
                "series_id": sid,
                "date": dt.date().isoformat(),
                "regime": str(lbl),
            })

    stats_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["series_id", "date", "regime"])
    return hits, stats_df, skipped


def scan_structural_breaks_series(
    watchlist: Iterable[str],
    min_mean_shift_pct_std: float = 0.3,
) -> tuple[list[DetectorHit], pd.DataFrame, list[tuple[str, str]]]:
    """Run Bai-Perron on each series in the watchlist.

    `min_mean_shift_pct_std` is the minimum shift (as a fraction of the
    series's own std) required to fire — filters noise from series with
    continuous micro-breaks.
    """
    hits: list[DetectorHit] = []
    skipped: list[tuple[str, str]] = []
    rows: list[dict] = []

    for sid in watchlist:
        try:
            s = D.load_series(sid)
        except (FileNotFoundError, KeyError) as e:
            skipped.append((sid, f"load: {e}"))
            continue
        full_std = float(s.dropna().std()) or 1.0
        found = detect_structural_break(
            s,
            series_id=sid,
            min_mean_shift=full_std * min_mean_shift_pct_std,
        )
        for h in found:
            hits.append(h)
            ev = h.evidence
            rows.append({
                "series_id": sid,
                "series_b": None,
                "kind": "series",
                "n_breaks": ev["n_breaks"],
                "break_dates": str(ev["break_dates"]),
                "most_recent_break_date": ev["most_recent_break_date"],
                "mean_shift_last": ev["mean_shift"],
                "run_date": date.today().isoformat(),
            })

    stats_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "series_id", "series_b", "kind", "n_breaks", "break_dates",
        "most_recent_break_date", "mean_shift_last", "run_date",
    ])
    return hits, stats_df, skipped


def scan_structural_breaks_corr(
    pairs: Iterable[tuple[str, str, str]],
    window_daily: int = 252,
    window_monthly: int = 36,
    min_corr_shift: float = 0.2,
) -> tuple[list[DetectorHit], pd.DataFrame, list[tuple[str, str]]]:
    """Run Bai-Perron on the rolling correlation series for each pair.

    Resamples the rolling corr to monthly before fitting so daily series
    don't produce 10k-obs inputs (Dynp is O(n²)).
    """
    hits: list[DetectorHit] = []
    skipped: list[tuple[str, str]] = []
    rows: list[dict] = []

    for a_id, b_id, label in pairs:
        try:
            df = D.load_aligned([a_id, b_id], how="coarsest", dropna="any")
        except (FileNotFoundError, KeyError) as e:
            skipped.append((f"{a_id}×{b_id}", f"load: {e}"))
            continue

        # Infer frequency from index spacing to pick appropriate window
        if df.shape[0] < 60:
            skipped.append((f"{a_id}×{b_id}", "insufficient overlap"))
            continue
        median_gap = float(pd.Series((df.index[1:] - df.index[:-1]).days).median())
        is_daily = median_gap <= 7
        window = window_daily if is_daily else window_monthly

        if df.shape[0] < window * 2:
            skipped.append((f"{a_id}×{b_id}", f"insufficient overlap ({df.shape[0]} < {window*2})"))
            continue

        rc = S.rolling_corr(df[a_id].diff(), df[b_id].diff(), window=window, method="spearman").dropna()
        if rc.size < 24:
            skipped.append((f"{a_id}×{b_id}", "rolling corr too short"))
            continue

        # Resample to monthly — structural breaks in correlations don't need
        # daily resolution and keeps the Dynp runtime manageable.
        rc_monthly = rc.resample("ME").mean().dropna()
        pair_id = f"{a_id}_x_{b_id}"

        found = detect_structural_break(
            rc_monthly,
            series_id=pair_id,
            min_mean_shift=min_corr_shift,
            max_obs=1500,
            is_correlation=True,
        )
        for h in found:
            # Tag with the pair label and both series IDs for the composer.
            h.evidence["series_a"] = a_id
            h.evidence["series_b"] = b_id
            h.evidence["label"] = label
            h = DetectorHit(
                kind=h.kind,
                series_ids=(a_id, b_id),
                window=h.window,
                evidence=h.evidence,
                score=h.score,
                tags=h.tags,
            )
            hits.append(h)
            ev = h.evidence
            rows.append({
                "series_id": a_id,
                "series_b": b_id,
                "kind": "rolling_corr",
                "n_breaks": ev["n_breaks"],
                "break_dates": str(ev["break_dates"]),
                "most_recent_break_date": ev["most_recent_break_date"],
                "mean_shift_last": ev["mean_shift"],
                "run_date": date.today().isoformat(),
            })

    stats_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "series_id", "series_b", "kind", "n_breaks", "break_dates",
        "most_recent_break_date", "mean_shift_last", "run_date",
    ])
    return hits, stats_df, skipped


# -------------------------------------------------------------------------
# M1: Inflation episode cross-sectional analysis
# -------------------------------------------------------------------------

def scan_inflation_episodes(
    threshold: float = 4.0,
    min_duration: int = 3,
) -> tuple[list[DetectorHit], pd.DataFrame, list[tuple[str, str]]]:
    """Identify historical inflation episodes, compare current to distribution."""
    from src.analytics.episodes import (
        identify_inflation_episodes,
        current_inflation_episode,
        inflation_episode_distribution,
    )

    hits: list[DetectorHit] = []
    skipped: list[tuple[str, str]] = []
    rows: list[dict] = []

    try:
        from src.analytics.data import load_series
        raw = load_series("CPIAUCSL").dropna()
        cpi_yoy = (raw.pct_change(12) * 100).rename("cpi_yoy")
    except (FileNotFoundError, KeyError) as e:
        skipped.append(("CPIAUCSL", f"load: {e}"))
        return hits, pd.DataFrame(), skipped

    try:
        episodes = identify_inflation_episodes(cpi_yoy, threshold=threshold, min_duration=min_duration)
        current = current_inflation_episode(cpi_yoy, threshold=threshold, min_duration=min_duration)
    except Exception as e:
        skipped.append(("inflation_episodes", f"compute: {e}"))
        return hits, pd.DataFrame(), skipped

    # Persist one row per episode to parquet.
    for ep in episodes:
        rows.append({
            "episode_idx": ep.idx,
            "start": ep.start.isoformat(),
            "end": ep.end.isoformat(),
            "peak_date": ep.peak_date.isoformat(),
            "peak_value": ep.peak_value,
            "duration_months": ep.duration_months,
            "driver": ep.driver,
            "real_ff_min": ep.real_ff_min,
            "unrate_change": ep.unrate_change,
            "run_date": date.today().isoformat(),
        })

    found = detect_inflation_episode_anomaly(episodes, current)
    hits.extend(found)

    stats_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "episode_idx", "start", "end", "peak_date", "peak_value",
        "duration_months", "driver", "real_ff_min", "unrate_change", "run_date",
    ])
    return hits, stats_df, skipped


# -------------------------------------------------------------------------
# M8: Breakeven decomposition
# -------------------------------------------------------------------------

def scan_breakeven_decomposition(
    tenor: str = "10y",
) -> tuple[list[DetectorHit], pd.DataFrame, list[tuple[str, str]]]:
    """Decompose TIPS breakeven into expected inflation and risk premium proxy."""
    from src.analytics.indicators import breakeven_decomposition
    from src.analytics.stats import zscore_vs_history, percentile_rank

    hits: list[DetectorHit] = []
    skipped: list[tuple[str, str]] = []

    try:
        df = breakeven_decomposition(tenor=tenor)
    except (FileNotFoundError, KeyError) as e:
        skipped.append((f"breakeven_{tenor}", f"load: {e}"))
        return hits, pd.DataFrame(), skipped
    except Exception as e:
        skipped.append((f"breakeven_{tenor}", f"compute: {e}"))
        return hits, pd.DataFrame(), skipped

    if df.empty or len(df) < 24:
        skipped.append((f"breakeven_{tenor}", f"insufficient data ({len(df)} rows)"))
        return hits, pd.DataFrame(), skipped

    found = detect_breakeven_anomaly(df, tenor=tenor)
    hits.extend(found)

    # Persist decomposition with z-scores for the parquet record.
    rp = df["risk_premium_proxy"].dropna()
    z_series = rp.expanding(min_periods=24).apply(
        lambda x: float((x.iloc[-1] - x.median()) / max((x - x.median()).abs().median() * 1.4826, 1e-9)),
        raw=False,
    )
    pct_series = rp.expanding(min_periods=24).apply(
        lambda x: float((x < x.iloc[-1]).mean()), raw=False
    )

    rows_df = df.copy()
    rows_df["risk_premium_z"] = z_series
    rows_df["risk_premium_pct_rank"] = pct_series
    rows_df["tenor"] = tenor
    rows_df["run_date"] = date.today().isoformat()
    rows_df = rows_df.reset_index().rename(columns={"index": "date"})
    # Thin to one row per month (already monthly, just ensure it)
    rows_df = rows_df.tail(120)   # last 10 years is enough for audit trail

    return hits, rows_df, skipped


# -------------------------------------------------------------------------
# M3: Bond return predictability / CP factor
# -------------------------------------------------------------------------

def scan_bond_predictability() -> tuple[list[DetectorHit], pd.DataFrame, list[tuple[str, str]]]:
    """Compute CP factor and excess returns; fire when CP factor is at extreme."""
    from src.analytics.bonds import (
        par_yields,
        forward_rates,
        cp_factor as compute_cp,
        approximate_excess_returns,
        cp_regression,
        cp_factor_snapshot,
    )

    hits: list[DetectorHit] = []
    skipped: list[tuple[str, str]] = []

    try:
        yields = par_yields(freq="M")
    except Exception as e:
        skipped.append(("bond_yields", f"load: {e}"))
        return hits, pd.DataFrame(), skipped

    if yields.empty or yields.shape[0] < 60 or yields.shape[1] < 3:
        skipped.append(("bond_yields", f"insufficient data ({yields.shape})"))
        return hits, pd.DataFrame(), skipped

    try:
        fwds = forward_rates(yields)
        cp = compute_cp(forward_df=fwds)
    except Exception as e:
        skipped.append(("cp_factor", f"compute: {e}"))
        return hits, pd.DataFrame(), skipped

    if cp.empty or len(cp) < 60:
        skipped.append(("cp_factor", f"insufficient history ({len(cp)} months)"))
        return hits, pd.DataFrame(), skipped

    # Excess returns and regression (best-effort; may have NaNs at tail).
    try:
        rx = approximate_excess_returns(yields, risk_free_col=1, horizon_months=12)
        reg_results = cp_regression(rx, cp, hac_lags=12)
    except Exception:
        rx = pd.DataFrame()
        reg_results = []

    found = detect_cp_factor_signal(cp, reg_results)
    hits.extend(found)

    # Persist CP factor time series with z-scores.
    clean = cp.dropna()
    z_expanding = clean.expanding(min_periods=24).apply(
        lambda x: float((x.iloc[-1] - x.median()) / max((x - x.median()).abs().median() * 1.4826, 1e-9)),
        raw=False,
    )
    pct_expanding = clean.expanding(min_periods=24).apply(
        lambda x: float((x < x.iloc[-1]).mean()), raw=False
    )

    rows: list[dict] = []
    snap = cp_factor_snapshot(cp, reg_results)
    for dt, val in clean.tail(120).items():
        rows.append({
            "date": dt.date().isoformat(),
            "cp_factor": round(float(val), 6),
            "cp_factor_z": round(float(z_expanding.get(dt, float("nan"))), 4),
            "cp_factor_pct_rank": round(float(pct_expanding.get(dt, float("nan"))), 4),
            "r2_5y": round(snap.r2_5y, 4) if snap and not (snap.r2_5y != snap.r2_5y) else float("nan"),
            "r2_10y": round(snap.r2_10y, 4) if snap and not (snap.r2_10y != snap.r2_10y) else float("nan"),
            "run_date": date.today().isoformat(),
        })

    stats_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "date", "cp_factor", "cp_factor_z", "cp_factor_pct_rank", "r2_5y", "r2_10y", "run_date",
    ])
    return hits, stats_df, skipped


# -------------------------------------------------------------------------
# BTP-Bund regime scanner
# -------------------------------------------------------------------------

# Thresholds in percentage points (same as the raw spread series units).
_BTP_BUND_THRESHOLDS_PP = {
    "elevated": 2.00,    # TPI informal trigger zone (~230bp in 2022)
    "stressed": 1.30,
    "moderate": 0.80,
}

_BTP_BUND_REGIME_LABELS = {
    "elevated": "elevated (≥200bp; TPI trigger zone)",
    "stressed": "stressed (130-200bp)",
    "moderate": "moderate (80-130bp)",
    "benign": "benign (<80bp)",
    "crisis": "crisis (≥300bp)",
}


def _btp_bund_label(spread_pp: float) -> str:
    if spread_pp >= 3.00:
        return "crisis"
    if spread_pp >= _BTP_BUND_THRESHOLDS_PP["elevated"]:
        return "elevated"
    if spread_pp >= _BTP_BUND_THRESHOLDS_PP["stressed"]:
        return "stressed"
    if spread_pp >= _BTP_BUND_THRESHOLDS_PP["moderate"]:
        return "moderate"
    return "benign"


def scan_btp_bund_regime() -> tuple[list[DetectorHit], list[tuple[str, str]]]:
    """Monitor BTP-Bund spread for regime crossings and tail readings.

    Fires a `btp_bund_regime` hit when:
      - The spread is currently in the 'stressed', 'elevated', or 'crisis' regime, OR
      - The spread has crossed a meaningful level in the last 3 months (regime change).

    Always computes the regime snapshot regardless of whether a hit fires;
    the hit's evidence block carries full context for the composer.
    """
    hits: list[DetectorHit] = []
    skipped: list[tuple[str, str]] = []

    try:
        s = D.load_series("ECB.BTPBUND.SPREAD").dropna()
    except (FileNotFoundError, KeyError) as e:
        skipped.append(("ECB.BTPBUND.SPREAD", f"load: {e}"))
        return hits, skipped

    if len(s) < 24:
        skipped.append(("ECB.BTPBUND.SPREAD", f"insufficient history ({len(s)} obs)"))
        return hits, skipped

    latest_val = float(s.iloc[-1])
    latest_date = s.index[-1]
    current_regime = _btp_bund_label(latest_val)

    # Historical context.
    pct = float(S.percentile_rank(s))
    z = float(S.zscore_vs_history(s))
    hist_median = float(s.median())
    hist_p90 = float(s.quantile(0.90))
    hist_max = float(s.max())
    hist_min = float(s.min())

    # Regime 3 months ago (approximately 3 monthly observations).
    prior_window = min(3, len(s) - 1)
    prior_val = float(s.iloc[-1 - prior_window])
    prior_regime = _btp_bund_label(prior_val)

    regime_crossed = current_regime != prior_regime

    # Fire if in a stressed/elevated/crisis regime OR regime just changed.
    should_fire = (
        current_regime in ("stressed", "elevated", "crisis")
        or regime_crossed
    )

    if should_fire:
        score = {
            "benign": 0.2,
            "moderate": 0.3,
            "stressed": 0.6,
            "elevated": 0.85,
            "crisis": 1.0,
        }.get(current_regime, 0.5)
        # Extra score bump for a fresh regime crossing.
        if regime_crossed:
            score = min(1.0, score + 0.15)

        hits.append(DetectorHit(
            kind="btp_bund_regime",
            series_ids=("ECB.IT.10Y", "ECB.DE.10Y"),
            window=prior_window,
            evidence={
                "latest_value_pp": latest_val,
                "latest_value_bp": round(latest_val * 100, 1),
                "latest_date": latest_date.date().isoformat(),
                "current_regime": current_regime,
                "current_regime_label": _BTP_BUND_REGIME_LABELS[current_regime],
                "prior_regime": prior_regime,
                "regime_crossed": regime_crossed,
                "percentile": pct,
                "z_score": z,
                "hist_median_pp": hist_median,
                "hist_median_bp": round(hist_median * 100, 1),
                "hist_p90_pp": hist_p90,
                "hist_p90_bp": round(hist_p90 * 100, 1),
                "hist_max_bp": round(hist_max * 100, 1),
                "hist_min_bp": round(hist_min * 100, 1),
                "n_obs": len(s),
            },
            score=score,
            tags=("europe", "sovereign", "ecb"),
        ))

    return hits, skipped


# -------------------------------------------------------------------------
# Hit → Finding conversion
# -------------------------------------------------------------------------

def _finding_from_hit(hit: DetectorHit, today: date) -> Finding:
    """Build a Finding (prose + structure) from a detector hit.

    Prose here is intentionally templated and minimal — it documents *what*
    was found in neutral language. The email composer generates the
    reader-facing narrative separately, consulting both the prose and the
    raw evidence.
    """
    ev = hit.evidence
    if hit.kind == "notable_move_level":
        sid = hit.series_ids[0]
        pct = ev["percentile"]
        title = f"{sid} latest reading is at {pct:.1%} of its full history"
        slug = make_slug(hit.kind, hit.series_ids, extra=ev["latest_date"])
        claim = (
            f"{sid} closed at {ev['latest_value']:.4g} on {ev['latest_date']}, "
            f"ranking at the {pct:.1%} of its {ev['n']}-observation history "
            f"(min {ev['full_min']:.4g}, median {ev['full_median']:.4g}, max {ev['full_max']:.4g})."
        )
    elif hit.kind == "notable_move_change":
        sid = hit.series_ids[0]
        z = ev["robust_z"]
        title = f"{sid} one-period change is {z:+.2f} robust-z vs history"
        slug = make_slug(hit.kind, hit.series_ids, extra=ev["latest_date"])
        claim = (
            f"On {ev['latest_date']}, {sid} moved by {ev['latest_change']:+.4g}, a "
            f"robust z-score of {z:+.2f} against {ev['n']} prior one-period changes "
            f"(median change {ev['median_change']:+.4g}, MAD {ev['mad_change']:.4g})."
        )
    elif hit.kind == "correlation_shift":
        a, b = hit.series_ids
        label = ev.get("label", f"{a} vs {b}")
        title = f"{label}: rolling correlation shifted by {ev['shift']:+.2f}"
        slug = make_slug(hit.kind, hit.series_ids, extra=f"w{ev.get('method','')}")
        claim = (
            f"Full-sample {ev['method']} correlation (on {'returns' if ev['on_returns'] else 'levels'}) "
            f"between {a} and {b} is {ev['baseline_correlation']:+.2f} with subperiod stability spread "
            f"{ev['baseline_stability_spread']:.2f}; the most recent rolling value is "
            f"{ev['recent_correlation']:+.2f} as of {ev['recent_date']}, a shift of {ev['shift']:+.2f}. "
            f"The threshold was first crossed at {ev['crossed_threshold_at']}."
        )
    elif hit.kind == "lead_lag_change":
        a, b = hit.series_ids
        label = ev.get("label", f"{a} vs {b}")
        if ev["lag_shift"] == 0:
            title = (
                f"{label}: peak correlation at lag {ev['hist_peak_lag']:+d} moved "
                f"{ev['hist_peak_correlation']:+.2f} → {ev['recent_peak_correlation']:+.2f}"
            )
        else:
            title = (
                f"{label}: lead-lag peak moved from lag {ev['hist_peak_lag']:+d} "
                f"to lag {ev['recent_peak_lag']:+d}"
            )
        slug = make_slug(hit.kind, hit.series_ids)
        claim = (
            f"Historical ({ev['n_hist']} obs) peak cross-correlation for {a} vs {b} is "
            f"{ev['hist_peak_correlation']:+.2f} at lag {ev['hist_peak_lag']:+d}; in the recent "
            f"{ev['n_recent']}-obs tail, the peak is {ev['recent_peak_correlation']:+.2f} at lag "
            f"{ev['recent_peak_lag']:+d} ({ev['method']}, "
            f"{'returns' if ev['on_returns'] else 'levels'})."
        )
    elif hit.kind == "regime_transition":
        sid = hit.series_ids[0]
        title = f"{sid} entered the {ev['new_regime']} regime on {ev['transition_date']}"
        slug = make_slug(hit.kind, hit.series_ids, extra=ev["transition_date"])
        claim = (
            f"{sid} crossed from the {ev['prior_regime']} to the {ev['new_regime']} quantile regime "
            f"on {ev['transition_date']} after {ev['prior_regime_duration_days']} days in the prior "
            f"regime. Latest reading {ev['latest_value']:.4g} (regime cut-points at "
            f"{[round(v, 4) for v in ev['cut_values']]})."
        )
    elif hit.kind == "structural_break":
        ev = hit.evidence
        is_corr = ev.get("is_correlation", False)
        is_spread = ev.get("is_spread", False)
        a_id = ev.get("series_a") or ev["series_id"]
        b_id = ev.get("series_b")
        label = ev.get("name") or ev.get("label", ev["series_id"])
        break_dates_str = ", ".join(ev["break_dates"])
        last_mean = ev["last_regime_mean"]
        prior_mean = ev["prior_regime_mean"]
        shift = ev["mean_shift"]
        most_recent = ev["most_recent_break_date"]
        n_breaks = ev["n_breaks"]

        if is_corr:
            title = (
                f"{label}: rolling correlation structural break at {most_recent} "
                f"(mean shift {prior_mean:+.2f} → {last_mean:+.2f})"
            )
            slug = make_slug("structural_break_corr", (a_id, b_id or ""))
            regime_lines = "; ".join(
                f"regime {r['regime']}: {r['start'][:7]}–{r['end'][:7]}, mean={r['mean']:+.2f}"
                for r in ev["regime_stats"]
            )
            claim = (
                f"Bai-Perron (Dynp, l2 cost, BIC-selected) finds {n_breaks} structural break(s) "
                f"in the monthly rolling Spearman correlation between {a_id} and {b_id}. "
                f"Break date(s): {break_dates_str}. "
                f"Mean correlation shifted from {prior_mean:+.2f} (prior regime) to "
                f"{last_mean:+.2f} (current regime), a change of {shift:+.2f}. "
                f"Regimes: {regime_lines}."
            )
        elif is_spread:
            sid_a_raw = hit.series_ids[0] if len(hit.series_ids) > 0 else a_id
            sid_b_raw = hit.series_ids[1] if len(hit.series_ids) > 1 else "?"
            title = (
                f"{label} ({sid_a_raw}−{sid_b_raw}): spread structural break at {most_recent} "
                f"(mean shift {prior_mean:+.4g} → {last_mean:+.4g})"
            )
            slug = make_slug("structural_break_spread", (sid_a_raw, sid_b_raw))
            regime_lines = "; ".join(
                f"regime {r['regime']}: {r['start'][:7]}–{r['end'][:7]}, mean={r['mean']:.4g}"
                for r in ev["regime_stats"]
            )
            claim = (
                f"Bai-Perron (Dynp, l2 cost, BIC-selected) finds {n_breaks} structural break(s) "
                f"in the derived spread {label} ({sid_a_raw} minus {sid_b_raw}). "
                f"Break date(s): {break_dates_str}. "
                f"Spread mean shifted from {prior_mean:.4g} (prior regime) to "
                f"{last_mean:.4g} (current regime), shift {shift:+.4g}. "
                f"Regimes: {regime_lines}. "
                f"Basis: {ev.get('basis', '')}."
            )
        else:
            sid = ev["series_id"]
            title = (
                f"{sid}: structural break at {most_recent} "
                f"(mean shift {prior_mean:+.4g} → {last_mean:+.4g})"
            )
            slug = make_slug("structural_break", (sid,))
            regime_lines = "; ".join(
                f"regime {r['regime']}: {r['start'][:7]}–{r['end'][:7]}, mean={r['mean']:.4g}"
                for r in ev["regime_stats"]
            )
            claim = (
                f"Bai-Perron (Dynp, l2 cost, BIC-selected) finds {n_breaks} structural break(s) "
                f"in {sid}. Break date(s): {break_dates_str}. "
                f"Mean in the current regime ({last_mean:.4g}) vs prior regime ({prior_mean:.4g}), "
                f"shift {shift:+.4g}. Regimes: {regime_lines}."
            )
    elif hit.kind == "cointegration_break":
        a, b = hit.series_ids
        title = f"{a} vs {b}: cointegration present in full sample but not in recent window"
        slug = make_slug(hit.kind, hit.series_ids)
        claim = (
            f"Engle-Granger p-value over the full sample (n={ev['full_n']}) is {ev['full_sample_p']:.3f} "
            f"(reject no-cointegration at 5%); restricted to the last {ev['recent_n']} observations "
            f"it is {ev['recent_p']:.3f} (cannot reject). The equilibrium relationship appears to "
            f"have weakened or broken."
        )
    elif hit.kind == "inflation_episode_anomaly":
        check = ev["check"]
        ep_start = ev["current_episode_start"][:7]
        driver = ev["current_driver"]
        if check == "duration":
            curr = int(ev["current_value"])
            title = (
                f"Current inflation episode ({ep_start}–, {driver}) duration "
                f"of {curr}m is at {ev['percentile_rank']:.0%} of historical distribution"
            )
            slug = make_slug("inflation_episode_duration", ("CPIAUCSL",))
            claim = (
                f"The current CPI inflation episode beginning {ep_start} has lasted {curr} months "
                f"(driver: {driver}), placing it at the {ev['percentile_rank']:.0%} percentile of "
                f"{ev['n_historical']} historical episodes since 1950 "
                f"(median {ev['hist_median']:.0f}m, 75th pct {ev['hist_p75']:.0f}m, max {ev['hist_max']:.0f}m). "
                f"Real fed funds minimum during episode: {ev['current_real_ff_min']:.2f}pp. "
                f"Unemployment change: {ev['current_unrate_change']:+.2f}pp."
            )
        else:
            curr_peak = ev["current_peak_value"]
            peak_dt = ev["current_peak_date"][:7]
            title = (
                f"Current inflation episode peaked at {curr_peak:.1f}% YoY ({peak_dt}), "
                f"{ev['percentile_rank']:.0%} of historical distribution"
            )
            slug = make_slug("inflation_episode_peak", ("CPIAUCSL",))
            claim = (
                f"CPI YoY peaked at {curr_peak:.2f}% in {peak_dt} in the current episode "
                f"(driver: {driver}), ranking at the {ev['percentile_rank']:.0%} percentile "
                f"of {ev['n_historical']} historical episodes "
                f"(median peak {ev['hist_median']:.1f}%, 75th pct {ev['hist_p75']:.1f}%, "
                f"max {ev['hist_max']:.1f}%). Episode duration: {ev['current_duration_months']} months."
            )
    elif hit.kind == "breakeven_anomaly":
        tenor = ev["tenor"]
        rp = ev["risk_premium_proxy"]
        direction = ev["direction"]
        z = ev["robust_z"]
        title = (
            f"{tenor} inflation risk premium proxy {direction} at {rp:.2f}pp "
            f"(robust z = {z:+.2f})"
        )
        slug = make_slug("breakeven_anomaly", hit.series_ids, extra=tenor)
        claim = (
            f"The {tenor} breakeven inflation risk premium proxy (T10YIE minus Michigan 1y survey "
            f"expectation) stands at {rp:.2f}pp as of {ev['latest_date']}, "
            f"a robust z-score of {z:+.2f} vs {ev['n_months']} months of history "
            f"(historical median {ev['hist_median']:.2f}pp, p10 {ev['hist_p10']:.2f}pp, "
            f"p90 {ev['hist_p90']:.2f}pp). "
            f"The 5y5y forward breakeven stands at {ev['latest_five_y_five_y']:.2f}pp "
            f"(medium-term expectations anchor). "
            f"Note: the proxy conflates the pure inflation risk premium and TIPS liquidity premium."
        )
    elif hit.kind == "cp_factor_signal":
        direction = ev["direction"]
        z = ev["robust_z"]
        cp_val = ev["cp_value"]
        r2_5y = ev["r2_by_maturity"].get(5, float("nan"))
        r2_10y = ev["r2_by_maturity"].get(10, float("nan"))
        title = (
            f"Cochrane-Piazzesi bond risk premium factor {direction} "
            f"(robust z = {z:+.2f}, {ev['percentile_rank']:.0%} pctile)"
        )
        slug = make_slug("cp_factor_signal", hit.series_ids)
        r2_str = ""
        if r2_5y == r2_5y:  # not nan
            r2_str = f" Predictive R² (CP factor on 1y excess returns): 5y {r2_5y:.1%}, 10y {r2_10y:.1%}."
        claim = (
            f"The Cochrane-Piazzesi factor — first principal component of the Treasury forward rate "
            f"curve (DGS1–DGS10) — stands at {cp_val:.3f} as of {ev['latest_date']}, "
            f"a robust z-score of {z:+.2f} vs {ev['n_months']} months of history "
            f"(p10 {ev['hist_p10']:.3f}, median {ev['hist_median']:.3f}, p90 {ev['hist_p90']:.3f}). "
            f"High CP factor historically predicts high 1-year excess Treasury returns "
            f"(Cochrane & Piazzesi 2005, AER 95(1)).{r2_str}"
        )
    elif hit.kind == "spread_extreme":
        sid_a, sid_b = hit.series_ids[:2]
        name = ev.get("name", f"{sid_a}−{sid_b}")
        pct = ev["percentile"]
        val = ev["latest_value"]
        title = f"{name}: spread at {pct:.1%} of full history ({val:+.3g})"
        slug = make_slug("spread_extreme", (sid_a, sid_b), extra=ev["latest_date"])
        claim = (
            f"The derived spread {name} ({sid_a} minus {sid_b}) stands at {val:+.4g} "
            f"as of {ev['latest_date']}, ranking at the {pct:.1%} of its "
            f"{ev['n']}-observation history "
            f"(min {ev['full_min']:+.4g}, median {ev['full_median']:+.4g}, "
            f"max {ev['full_max']:+.4g}). "
            f"Theoretical basis: {ev.get('basis', '')}."
        )
    elif hit.kind == "decomposition_shift":
        total = ev["total_series"]
        comp = ev["component"]
        hist_share = ev["hist_share"]
        recent_share = ev["recent_share"]
        shift = ev["share_shift"]
        direction = ev["direction"]
        name = ev.get("name", f"{total} decomposition")
        title = (
            f"{name}: {comp} contribution {direction} "
            f"({hist_share:.0%} → {recent_share:.0%} of {total} moves)"
        )
        slug = make_slug("decomposition_shift", (total, comp))
        claim = (
            f"In the recent {ev['recent_obs']} observations, {comp} accounted for "
            f"{recent_share:.0%} of {total} moves (signed contribution share), "
            f"vs a historical mean of {hist_share:.0%} over {ev['hist_obs']} prior observations "
            f"(shift: {shift:+.0%}). "
            f"This indicates the recent {total} move has been disproportionately "
            f"{'driven by' if shift > 0 else 'suppressed relative to'} {comp}. "
            f"Theoretical basis: {ev.get('basis', '')}."
        )
    elif hit.kind == "btp_bund_regime":
        val_bp = ev["latest_value_bp"]
        regime = ev["current_regime_label"]
        crossed = ev["regime_crossed"]
        prior = ev["prior_regime"]
        pct = ev["percentile"]
        cross_note = (
            f" Regime change: was '{prior}', now '{ev['current_regime']}'."
            if crossed else ""
        )
        title = (
            f"BTP-Bund spread {val_bp:.0f}bp ({ev['current_regime']}): "
            f"{pct:.1%} of full history"
        )
        slug = make_slug("btp_bund_regime", hit.series_ids, extra=ev["latest_date"])
        claim = (
            f"The BTP-Bund 10Y sovereign spread (ECB.IT.10Y minus ECB.DE.10Y) stands at "
            f"{val_bp:.0f}bp ({ev['latest_value_pp']:.3f}pp) as of {ev['latest_date']}, "
            f"in the '{ev['current_regime']}' regime ({regime}).{cross_note} "
            f"Percentile rank vs {ev['n_obs']}-month history: {pct:.1%} "
            f"(historical median {ev['hist_median_bp']:.0f}bp, 90th pctile {ev['hist_p90_bp']:.0f}bp, "
            f"max {ev['hist_max_bp']:.0f}bp). "
            f"Robust z-score: {ev['z_score']:+.2f}. "
            f"TPI informal trigger: ~200bp; 2022 peak ~230bp."
        )
    else:
        title = f"Unknown hit kind: {hit.kind}"
        slug = make_slug(hit.kind, hit.series_ids)
        claim = f"Evidence: {ev}"

    return Finding(
        slug=slug,
        title=title,
        kind=hit.kind,
        discovered=today,
        series_ids=hit.series_ids,
        window=hit.window,
        claim=claim,
        evidence=ev,
        interpretation="",
        sources=[],
        status="new",
        score=hit.score,
    )


# -------------------------------------------------------------------------
# Top-level entry
# -------------------------------------------------------------------------

def run_scan(
    *,
    pairs: Iterable[tuple[str, str, str]] | None = None,
    notable_watchlist: Iterable[str] | None = None,
    today: date | None = None,
    overwrite_findings: bool = False,
    skip_relationships: bool = False,
) -> ScanReport:
    """Run every detector, persist stats and findings. Returns a report."""
    # Default to RELATIONSHIPS-derived pairs (superset of old CORE_PAIRS).
    pairs = list(pairs) if pairs is not None else relationships_as_pairs(kinds=("correlation", "lead_lag"))
    notable_watchlist = list(notable_watchlist) if notable_watchlist is not None else list(NOTABLE_MOVE_WATCHLIST)
    today = today or date.today()
    report = ScanReport()

    log.info("scan: notable moves on %d series", len(notable_watchlist))
    nh, ndf, nsk = scan_notable_moves(notable_watchlist)
    report.hits.extend(nh); report.skipped.extend(nsk)
    append_stats("notable_moves", ndf); report.stats_rows_written["notable_moves"] = len(ndf)

    log.info("scan: correlation shifts on %d pairs", len(pairs))
    ch, cdf, csk = scan_correlation_shifts(pairs)
    report.hits.extend(ch); report.skipped.extend(csk)
    append_stats("rolling_correlations", cdf); report.stats_rows_written["rolling_correlations"] = len(cdf)

    log.info("scan: lead-lag on %d pairs", len(pairs))
    lh, ldf, lsk = scan_lead_lag(pairs)
    report.hits.extend(lh); report.skipped.extend(lsk)
    append_stats("lead_lag", ldf); report.stats_rows_written["lead_lag"] = len(ldf)

    log.info("scan: regime transitions on %d series", len(notable_watchlist))
    rh, rdf, rsk = scan_regime_transitions(notable_watchlist)
    report.hits.extend(rh); report.skipped.extend(rsk)
    append_stats("regime_labels", rdf); report.stats_rows_written["regime_labels"] = len(rdf)

    log.info("scan: structural breaks (series) on %d series", len(notable_watchlist))
    sbsh, sbsdf, sbssk = scan_structural_breaks_series(notable_watchlist)
    report.hits.extend(sbsh); report.skipped.extend(sbssk)
    append_stats("structural_breaks", sbsdf); report.stats_rows_written["structural_breaks_series"] = len(sbsdf)

    log.info("scan: structural breaks (rolling corr) on %d pairs", len(pairs))
    sbch, sbcdf, sbcsk = scan_structural_breaks_corr(pairs)
    report.hits.extend(sbch); report.skipped.extend(sbcsk)
    if len(sbcdf):
        append_stats("structural_breaks", sbcdf)
    report.stats_rows_written["structural_breaks_corr"] = len(sbcdf)

    log.info("scan: inflation episodes (M1)")
    ieh, iedf, iesk = scan_inflation_episodes()
    report.hits.extend(ieh); report.skipped.extend(iesk)
    append_stats("inflation_episodes", iedf); report.stats_rows_written["inflation_episodes"] = len(iedf)

    log.info("scan: breakeven decomposition (M8)")
    bdh, bddf, bdsk = scan_breakeven_decomposition()
    report.hits.extend(bdh); report.skipped.extend(bdsk)
    append_stats("breakeven_components", bddf); report.stats_rows_written["breakeven_components"] = len(bddf)

    log.info("scan: bond predictability / CP factor (M3)")
    bph, bpdf, bpsk = scan_bond_predictability()
    report.hits.extend(bph); report.skipped.extend(bpsk)
    append_stats("bond_predictability", bpdf); report.stats_rows_written["bond_predictability"] = len(bpdf)

    log.info("scan: BTP-Bund regime monitor")
    bbrh, bbrsk = scan_btp_bund_regime()
    report.hits.extend(bbrh); report.skipped.extend(bbrsk)

    if not skip_relationships:
        log.info("scan: relationship monitor (spread + decomposition)")
        rh, rsk, rstats = run_relationship_monitor(today=today)
        report.hits.extend(rh)
        report.skipped.extend(rsk)
        for stat_name, rdf in rstats.items():
            if len(rdf):
                append_stats(stat_name, rdf)
                report.stats_rows_written[stat_name] = len(rdf)

    log.info("scan: research questions (M4 always; M10/M11 triggered by hits)")
    questions = generate_questions(report.hits, today)
    q_findings: list[Finding] = []
    for q in questions:
        log.info("  running %s: %s", q.method_id, q.dedup_key)
        qf, qstats = q.execute()
        q_findings.extend(qf)
        for stat_name, qdf in qstats.items():
            if len(qdf):
                append_stats(stat_name, qdf)
                report.stats_rows_written[stat_name] = report.stats_rows_written.get(stat_name, 0) + len(qdf)

    # Turn hits into findings and write markdown.
    prior = existing_slugs()
    findings: list[Finding] = []
    for h in report.hits:
        f = _finding_from_hit(h, today)
        findings.append(f)
    # Append question-driven findings (M4, M10, M11) — already fully formed.
    findings.extend(q_findings)
    # Order highest-score first so fresh scans write the most salient ones at the top.
    findings.sort(key=lambda f: -f.score)
    added, kept = write_findings_md(findings, overwrite=overwrite_findings)
    report.findings = findings
    log.info("findings.md: %d new, %d kept (prior count %d)", added, kept, len(prior))
    return report
