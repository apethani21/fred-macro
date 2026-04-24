"""Relationship monitor: spread and decomposition detectors for named macro relationships.

The correlation/lead-lag kinds are handled by the existing scan_correlation_shifts /
scan_lead_lag functions in scan.py (they accept a pairs list derived from RELATIONSHIPS).
This module handles the two new kinds that require computing derived series:

  spread        — series[0] - series[1]; runs notable-move + structural-break on the result
  decomposition — series[0] ≈ series[1] + series[2]; tracks rolling contribution shares

Both return the same (hits, stats_df, skipped) tuple as the other scan functions so they
slot cleanly into run_scan().
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Iterable

import pandas as pd

from src.analytics import data as D
from src.analytics import stats as S
from .detectors import DetectorHit, detect_notable_move, detect_structural_break
from .relationship_config import Relationship, RELATIONSHIPS

log = logging.getLogger(__name__)


# ── Spread relationships ──────────────────────────────────────────────────────

def scan_spread_relationships(
    relationships: Iterable[Relationship],
) -> tuple[list[DetectorHit], pd.DataFrame, list[tuple[str, str]]]:
    """Compute derived spread (series[0] - series[1]) and run notable-move + structural-break.

    Notable-move uses pct_tail=0.03 (top/bottom 3% fires) rather than 1% because
    spreads are derived and noisier than raw series.
    Structural-break threshold is 0.3 standard deviations (same as the series scan).

    Hits carry kind="spread_extreme" (notable-move) or kind="structural_break" (with
    is_spread=True in evidence so the finding prose can render it correctly).
    """
    hits: list[DetectorHit] = []
    skipped: list[tuple[str, str]] = []
    rows: list[dict] = []

    for rel in relationships:
        if rel.kind != "spread" or len(rel.series) < 2:
            continue
        sid_a, sid_b = rel.series[0], rel.series[1]
        spread_id = f"{sid_a}_minus_{sid_b}"

        try:
            s_a = D.load_series(sid_a).dropna()
            s_b = D.load_series(sid_b).dropna()
        except (FileNotFoundError, KeyError) as e:
            skipped.append((spread_id, f"load: {e}"))
            continue

        aligned = pd.concat([s_a, s_b], axis=1, join="inner").dropna()
        if len(aligned) < 60:
            skipped.append((spread_id, f"insufficient overlap ({len(aligned)})"))
            continue

        spread = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        spread.name = spread_id

        # Notable-move on the spread level
        for h in detect_notable_move(spread, series_id=spread_id, pct_tail=0.03):
            hits.append(DetectorHit(
                kind="spread_extreme",
                series_ids=(sid_a, sid_b),
                window=h.window,
                evidence={**h.evidence, "spread_id": spread_id, "name": rel.name, "basis": rel.basis},
                score=h.score + 1.0,
                tags=rel.tags,
            ))

        # Structural break on the spread
        full_std = float(spread.std()) or 1.0
        for h in detect_structural_break(
            spread,
            series_id=spread_id,
            min_mean_shift=full_std * 0.3,
        ):
            hits.append(DetectorHit(
                kind="structural_break",
                series_ids=(sid_a, sid_b),
                window=h.window,
                evidence={
                    **h.evidence,
                    "spread_id": spread_id,
                    "name": rel.name,
                    "basis": rel.basis,
                    "is_spread": True,
                },
                score=h.score,
                tags=rel.tags,
            ))

        # Persist current stats
        pct = S.percentile_rank(spread, float(spread.iloc[-1]))
        z = S.zscore_vs_history(spread)
        rows.append({
            "spread_id": spread_id,
            "series_a": sid_a,
            "series_b": sid_b,
            "name": rel.name,
            "latest_value": float(spread.iloc[-1]),
            "latest_date": str(spread.index[-1].date()),
            "percentile_full": float(pct) if pct is not None else float("nan"),
            "z_score_full": float(z) if z is not None else float("nan"),
            "n_obs": len(spread),
            "run_date": date.today().isoformat(),
        })

    stats_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "spread_id", "series_a", "series_b", "name", "latest_value", "latest_date",
        "percentile_full", "z_score_full", "n_obs", "run_date",
    ])
    return hits, stats_df, skipped


# ── Decomposition relationships ───────────────────────────────────────────────

def _signed_contribution_share(
    window_df: pd.DataFrame,
    comp_col: str,
    total_col: str,
    min_move_quantile: float = 0.33,
) -> float:
    """Mean signed share: comp_change / total_change, filtered to significant total moves.

    Filters to the top (1 - min_move_quantile) of total moves by absolute magnitude
    to avoid division near zero dominating the mean. Clips individual ratios to [-2, 2]
    to handle the few cases where components over-explain the total.
    """
    comp = window_df[comp_col]
    tot = window_df[total_col]
    threshold = float(tot.abs().quantile(min_move_quantile))
    if threshold < 1e-12:
        return float("nan")
    mask = tot.abs() > threshold
    if mask.sum() < 5:
        return float("nan")
    shares = (comp[mask] / tot[mask]).clip(-2.0, 2.0)
    return float(shares.mean())


def scan_decomposition_relationships(
    relationships: Iterable[Relationship],
    recent_days: int = 90,
    min_share_shift: float = 0.20,
) -> tuple[list[DetectorHit], pd.DataFrame, list[tuple[str, str]]]:
    """Monitor rolling contribution shares for additive decompositions.

    For decomposition (total ≈ comp_a + comp_b), computes for each component:
      - Historical mean signed contribution share (excluding recent window)
      - Recent mean signed contribution share (last recent_days)
      - Fires when |recent - historical| >= min_share_shift (default 20pp)

    Uses first-differences (changes) to avoid non-stationarity issues.
    Score scales linearly with the size of the shift above the threshold.
    """
    hits: list[DetectorHit] = []
    skipped: list[tuple[str, str]] = []
    rows: list[dict] = []

    for rel in relationships:
        if rel.kind != "decomposition" or len(rel.series) < 3:
            continue
        sid_total = rel.series[0]
        sid_comps = rel.series[1:]
        decomp_id = f"decomp_{sid_total}"

        try:
            df = D.load_aligned(list(rel.series), how="coarsest", dropna="any")
        except (FileNotFoundError, KeyError) as e:
            skipped.append((decomp_id, f"load: {e}"))
            continue

        if len(df) < 60:
            skipped.append((decomp_id, f"insufficient data ({len(df)})"))
            continue

        diffs = df.diff().dropna()

        # Estimate recent_obs from calendar days
        median_gap = float(pd.Series((df.index[1:] - df.index[:-1]).days).median())
        recent_obs = max(int(recent_days / max(median_gap, 1)), 10)
        if len(diffs) < recent_obs + 20:
            skipped.append((decomp_id, f"insufficient obs for split ({len(diffs)}, need {recent_obs + 20})"))
            continue

        hist_diffs = diffs.iloc[:-recent_obs]
        recent_diffs = diffs.iloc[-recent_obs:]

        for sid_comp in sid_comps:
            if sid_comp not in diffs.columns or sid_total not in diffs.columns:
                continue

            hist_share = _signed_contribution_share(hist_diffs, sid_comp, sid_total)
            recent_share = _signed_contribution_share(recent_diffs, sid_comp, sid_total)

            if hist_share != hist_share or recent_share != recent_share:  # nan
                continue

            share_shift = recent_share - hist_share

            rows.append({
                "decomp_id": decomp_id,
                "total_series": sid_total,
                "component": sid_comp,
                "name": rel.name,
                "hist_share": round(hist_share, 4),
                "recent_share": round(recent_share, 4),
                "share_shift": round(share_shift, 4),
                "recent_obs": recent_obs,
                "hist_obs": len(hist_diffs),
                "run_date": date.today().isoformat(),
            })

            if abs(share_shift) >= min_share_shift:
                direction = "elevated" if share_shift > 0 else "suppressed"
                score = abs(share_shift) / min_share_shift * 3.0
                hits.append(DetectorHit(
                    kind="decomposition_shift",
                    series_ids=(sid_total,) + tuple(sid_comps),
                    window=f"recent {recent_obs} obs vs {len(hist_diffs)} historical",
                    evidence={
                        "total_series": sid_total,
                        "component": sid_comp,
                        "all_components": list(sid_comps),
                        "name": rel.name,
                        "basis": rel.basis,
                        "hist_share": round(hist_share, 4),
                        "recent_share": round(recent_share, 4),
                        "share_shift": round(share_shift, 4),
                        "direction": direction,
                        "recent_obs": recent_obs,
                        "hist_obs": len(hist_diffs),
                    },
                    score=score,
                    tags=rel.tags,
                ))

    stats_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "decomp_id", "total_series", "component", "name",
        "hist_share", "recent_share", "share_shift", "recent_obs", "hist_obs", "run_date",
    ])
    return hits, stats_df, skipped


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_relationship_monitor(
    today: date | None = None,
    relationships: tuple[Relationship, ...] | None = None,
) -> tuple[list[DetectorHit], list[tuple[str, str]], dict[str, pd.DataFrame]]:
    """Run spread and decomposition monitors over all configured relationships.

    Returns (hits, skipped, {stat_name: DataFrame}).
    Correlation and lead-lag kinds are handled by the existing scan functions in scan.py;
    this function only covers spread and decomposition.
    """
    rels = relationships if relationships is not None else RELATIONSHIPS
    today = today or date.today()

    all_hits: list[DetectorHit] = []
    all_skipped: list[tuple[str, str]] = []
    all_stats: dict[str, pd.DataFrame] = {}

    log.info("relationship monitor: spread relationships")
    sh, sdf, ssk = scan_spread_relationships(rels)
    all_hits.extend(sh)
    all_skipped.extend(ssk)
    if not sdf.empty:
        all_stats["relationship_spreads"] = sdf

    log.info("relationship monitor: decomposition relationships")
    dh, ddf, dsk = scan_decomposition_relationships(rels)
    all_hits.extend(dh)
    all_skipped.extend(dsk)
    if not ddf.empty:
        all_stats["relationship_decompositions"] = ddf

    return all_hits, all_skipped, all_stats
