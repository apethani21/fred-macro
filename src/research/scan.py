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
from .detectors import (
    DetectorHit,
    detect_cointegration_break,
    detect_correlation_shift,
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
        a_id = ev.get("series_a") or ev["series_id"]
        b_id = ev.get("series_b")
        label = ev.get("label", ev["series_id"])
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
) -> ScanReport:
    """Run every detector, persist stats and findings. Returns a report."""
    pairs = list(pairs) if pairs is not None else list(CORE_PAIRS)
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

    # Turn hits into findings and write markdown.
    prior = existing_slugs()
    findings: list[Finding] = []
    for h in report.hits:
        f = _finding_from_hit(h, today)
        findings.append(f)
    # Order highest-score first so fresh scans write the most salient ones at the top.
    findings.sort(key=lambda f: -f.score)
    added, kept = write_findings_md(findings, overwrite=overwrite_findings)
    report.findings = findings
    log.info("findings.md: %d new, %d kept (prior count %d)", added, kept, len(prior))
    return report
