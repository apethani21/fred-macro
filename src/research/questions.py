"""Research question generation: detector hits → targeted follow-up analyses.

When the scan detects a pattern, this module maps it to a concrete research
question and an analysis function to run. Closes the loop between the detector
layer and the methods library (methods.md M4, M10, M11).

Design:
- `generate_questions(hits)` returns a deduplicated list of ResearchQuestion objects.
- Each question has an `execute()` method returning (findings, stats_df_dict).
- `run_scan()` in scan.py calls this after all detectors fire.

Already-implemented methods (M1, M3, M6, M7, M8) run unconditionally in scan.py
and don't need to be triggered here. This module handles the conditional ones:
  M4 (recession logit) — always, refreshes the current probability estimate
  M10 (sentiment → bonds) — when VIX or HY spreads show a notable move
  M11 (jump detection) — when a notable_move fires on credit or rate series
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import pandas as pd

from src.analytics.data import load_series
from src.analytics.recession import recession_prediction_logit
from src.analytics.stats import detect_jumps, sentiment_bond_forecast
from src.research.detectors import DetectorHit
from src.research.findings import Finding, make_slug

log = logging.getLogger(__name__)

# Series groups used in trigger matching
_CREDIT_SERIES = frozenset({"BAMLH0A0HYM2", "BAMLC0A0CM", "AAA10Y", "BAA10Y"})
_RATE_SERIES   = frozenset({"DGS10", "DGS2", "DGS30", "T10Y2Y", "T10Y3M", "DFII10", "T10YIE", "SOFR", "DFF"})
_VIX_SENTIMENT = frozenset({"VIXCLS", "UMCSENT", "USEPUINDXD"})


@dataclass
class ResearchQuestion:
    method_id: str              # "M4", "M10", "M11"
    question: str               # human-readable research question
    dedup_key: str              # uniqueness key — prevents running same analysis twice per scan
    priority: int               # lower = run first
    _fn: Callable[[], tuple[list[Finding], dict[str, pd.DataFrame]]] = field(repr=False)

    def execute(self) -> tuple[list[Finding], dict[str, pd.DataFrame]]:
        """Run the analysis and return (findings, {stat_name: df})."""
        try:
            return self._fn()
        except Exception as exc:
            log.warning("Question %s (%s) failed: %s", self.dedup_key, self.method_id, exc)
            return [], {}


def generate_questions(hits: list[DetectorHit], today: date | None = None) -> list[ResearchQuestion]:
    """Map detector hits to follow-up research questions.

    M4 (recession logit) always runs — it gives a current probability reading
    regardless of whether any specific detector fired.

    M10 and M11 are conditional: they only run when a relevant hit is present.
    """
    today = today or date.today()
    questions: list[ResearchQuestion] = []
    seen_keys: set[str] = set()

    def _add(q: ResearchQuestion) -> None:
        if q.dedup_key not in seen_keys:
            seen_keys.add(q.dedup_key)
            questions.append(q)

    # M4: always refresh recession probability estimate
    _add(_make_recession_question(today))

    # Conditional questions triggered by detector hits
    for hit in hits:
        series_set = set(hit.series_ids)

        # M11: jump detection fires when a notable_move hits a credit or rate series
        if hit.kind in ("notable_move_level", "notable_move_change"):
            for sid in series_set:
                if sid in _CREDIT_SERIES | _RATE_SERIES:
                    _add(_make_jump_question(sid, today))

        # M10: sentiment → bond forecast fires when VIX or HY spreads are notable
        if hit.kind in ("notable_move_level", "notable_move_change", "regime_transition"):
            if series_set & (_VIX_SENTIMENT | _CREDIT_SERIES):
                _add(_make_sentiment_bond_question(today))

    return sorted(questions, key=lambda q: q.priority)


# -------------------------------------------------------------------------
# Question factories
# -------------------------------------------------------------------------

def _make_recession_question(today: date) -> ResearchQuestion:
    return ResearchQuestion(
        method_id="M4",
        question=(
            "What is the current model-implied 4-quarter-ahead recession probability "
            "from the yield-curve + HY-spread logit? How has it moved over the past 6 months?"
        ),
        dedup_key="M4_recession_logit",
        priority=10,
        _fn=lambda: _run_recession_logit(today),
    )


def _make_jump_question(series_id: str, today: date) -> ResearchQuestion:
    return ResearchQuestion(
        method_id="M11",
        question=(
            f"Has {series_id} experienced statistically detectable jumps "
            "(non-parametric BNS test, |z| > 4) recently? How do jump dates cluster "
            "relative to FOMC meetings and major macro releases?"
        ),
        dedup_key=f"M11_jump_{series_id}",
        priority=20,
        _fn=lambda sid=series_id: _run_jump_detection(sid, today),
    )


def _make_sentiment_bond_question(today: date) -> ResearchQuestion:
    return ResearchQuestion(
        method_id="M10",
        question=(
            "Given current VIX and HY spread levels, what does the sentiment-to-bond-return "
            "regression imply for 10Y excess returns over the next month? Is the implied "
            "return above or below the historical distribution?"
        ),
        dedup_key="M10_sentiment_bond",
        priority=30,
        _fn=lambda: _run_sentiment_bond(today),
    )


# -------------------------------------------------------------------------
# Analysis runners: return (findings, stats_dfs)
# -------------------------------------------------------------------------

def _run_recession_logit(today: date) -> tuple[list[Finding], dict[str, pd.DataFrame]]:
    result = recession_prediction_logit()
    if not result.fit_ok:
        log.warning("M4 recession logit failed: %s", result.error)
        return [], {}

    prob = result.current_probability
    prior = result.six_month_ago_probability
    direction = "up" if prob > prior else "down"
    prob_pct  = round(prob * 100, 1)
    prior_pct = round(prior * 100, 1)
    delta_pp   = round((prob - prior) * 100, 1)

    # Format top two coefficients by absolute magnitude
    top_coefs = sorted(result.coefficients.items(), key=lambda kv: -abs(kv[1]))[:2]
    coef_str = "; ".join(f"{k}: β={v:+.2f}" for k, v in top_coefs)

    claim = (
        f"Logistic recession model (predictors: {', '.join(result.feature_names)}) "
        f"estimates a {prob_pct:.1f}% probability of NBER recession onset within "
        f"the next 4 quarters as of {today.isoformat()}, {direction} {abs(delta_pp):.1f}pp "
        f"from {prior_pct:.1f}% six months ago. "
        f"In-sample AUROC: {result.auroc:.2f} (n={result.n_obs}, "
        f"{result.n_recessions} recession-onset quarters)."
    )
    interpretation = (
        f"Largest standardised coefficients: {coef_str}. "
        f"Current predictor values: "
        + ", ".join(f"{k}={v:.2f}" for k, v in result.current_predictors.items())
        + ". "
        "Interpretation is conditional on the logit's in-sample fit: AUROC measures "
        "discriminatory power but does not guarantee probability calibration."
    )

    slug = make_slug("recession_logit_m4", ("USREC",), extra=today.isoformat())
    finding = Finding(
        slug=slug,
        title=f"M4 recession logit: {prob_pct:.1f}% 4Q-ahead probability ({direction} {abs(delta_pp):.1f}pp in 6 months)",
        kind="recession_logit",
        discovered=today,
        series_ids=("USREC", "T10Y3M", "T10Y2Y", "BAMLH0A0HYM2", "VIXCLS"),
        window=f"full history to {today.isoformat()}, 4-quarter horizon",
        claim=claim,
        evidence={
            "current_probability": prob_pct,
            "six_month_ago_probability": prior_pct,
            "delta_pp": delta_pp,
            "auroc": round(result.auroc, 3),
            "n_obs": result.n_obs,
            "n_recessions": result.n_recessions,
            "coefficients": {k: round(v, 3) for k, v in result.coefficients.items()},
            "current_predictors": {k: round(v, 3) for k, v in result.current_predictors.items()},
        },
        interpretation=interpretation,
        sources=[],
        status="new",
        score=min(abs(delta_pp) / 10, 1.0),  # larger moves = more interesting
    )

    # Stats parquet: full history time series
    history_rows = [
        {"date": str(dt.date()), "recession_probability": round(float(v), 4), "run_date": today.isoformat()}
        for dt, v in result.history.items()
    ]
    stats_df = pd.DataFrame(history_rows)

    return [finding], {"recession_logit": stats_df}


def _run_jump_detection(series_id: str, today: date) -> tuple[list[Finding], dict[str, pd.DataFrame]]:
    s = load_series(series_id)
    if s is None or len(s.dropna()) < 120:
        return [], {}

    result = detect_jumps(s, series_id, window=60, z_threshold=4.0, lookback_days=365)

    if result.recent_jump is None:
        log.debug("M11 jump detection: no recent jumps in %s", series_id)
        return [], {}

    z = result.recent_jump_z or 0.0
    mag = result.jump_magnitudes[-1] if result.jump_magnitudes else 0.0

    claim = (
        f"{series_id} had a statistically detected jump on {result.recent_jump}: "
        f"daily change of {mag:+.3g}, BNS z-score {z:+.1f} (threshold |z|≥4.0). "
        f"Total detected jumps in full history: {result.n_jumps_total}."
    )
    interpretation = (
        "The BNS test uses bipower variation — the product of adjacent absolute changes "
        "— as a jump-robust variance estimator. A move flagged here cannot be explained "
        "by the diffusive (continuous) component of the series alone. "
        "Jump clustering with known macro events (FOMC, payrolls) is not yet cross-referenced "
        "automatically — review the date manually."
    )

    slug = make_slug("jump_detection_m11", (series_id,), extra=result.recent_jump)
    finding = Finding(
        slug=slug,
        title=f"M11 jump detection: {series_id} jump on {result.recent_jump} (z={z:+.1f})",
        kind="jump_detection",
        discovered=today,
        series_ids=(series_id,),
        window=f"full history, BNS window=60d",
        claim=claim,
        evidence={
            "recent_jump_date": result.recent_jump,
            "recent_jump_z": round(z, 2),
            "recent_jump_magnitude": round(mag, 4),
            "n_jumps_total": result.n_jumps_total,
            "z_threshold": 4.0,
            "window": result.window,
        },
        interpretation=interpretation,
        sources=[],
        status="new",
        score=min(abs(z) / 10, 1.0),
    )

    stats_rows = [
        {
            "series_id": series_id,
            "jump_date": d,
            "magnitude": round(m, 4),
            "z_score": round(z_, 2),
            "run_date": today.isoformat(),
        }
        for d, m, z_ in zip(result.jump_dates, result.jump_magnitudes, result.jump_zscores)
    ]
    stats_df = pd.DataFrame(stats_rows) if stats_rows else pd.DataFrame(
        columns=["series_id", "jump_date", "magnitude", "z_score", "run_date"]
    )

    return [finding], {"jump_detections": stats_df}


def _run_sentiment_bond(today: date) -> tuple[list[Finding], dict[str, pd.DataFrame]]:
    def _safe_load(sid: str) -> pd.Series | None:
        try:
            return load_series(sid)
        except (KeyError, Exception):
            return None

    dgs10 = _safe_load("DGS10")
    dtb3  = _safe_load("DTB3")
    vix   = _safe_load("VIXCLS")
    hy    = _safe_load("BAMLH0A0HYM2")

    if dgs10 is None or dgs10.empty or dtb3 is None or dtb3.empty:
        return [], {}

    MIN_MONTHLY_OBS = 36
    sentiment: dict[str, pd.Series] = {}
    if vix is not None and not vix.empty and len(vix.resample("MS").last().dropna()) >= MIN_MONTHLY_OBS:
        sentiment["VIX"] = vix
    if hy is not None and not hy.empty and len(hy.resample("MS").last().dropna()) >= MIN_MONTHLY_OBS:
        sentiment["HY_OAS"] = hy

    if not sentiment:
        return [], {}

    result = sentiment_bond_forecast(dgs10, dtb3, sentiment_series=sentiment)
    if not result.fit_ok:
        log.warning("M10 sentiment→bond failed: %s", result.error)
        return [], {}

    implied = result.implied_excess_return_bp
    r2 = result.r_squared
    direction = "positive" if implied > 0 else "negative"
    coef_str = "; ".join(
        f"{k}: {v:+.1f}bp per unit (t={result.t_stats.get(k, float('nan')):.1f})"
        for k, v in result.coefficients.items()
    )

    claim = (
        f"OLS of next-month 10Y excess returns on VIX and HY OAS "
        f"(Newey-West SEs, {result.n_obs} monthly obs) yields in-sample R²={r2:.2f}. "
        f"Given current predictor values "
        f"({', '.join(f'{k}={v:.1f}' for k, v in result.current_predictors.items())}), "
        f"the model implies a {direction} implied excess return of {implied:+.0f}bp "
        f"over the next month."
    )
    interpretation = (
        f"Coefficient signs: {coef_str}. "
        "A positive implied return reflects the flight-to-quality premium — when stress "
        "indicators are elevated, subsequent Treasury returns have historically been above "
        "average. Low R² is expected: sentiment predicts direction, not magnitude. "
        "This is a conditional mean forecast, not a trading signal."
    )

    slug = make_slug("sentiment_bond_m10", ("DGS10", "VIXCLS"), extra=today.isoformat())
    finding = Finding(
        slug=slug,
        title=f"M10 sentiment→bond: current stress implies {implied:+.0f}bp 10Y excess return",
        kind="sentiment_bond",
        discovered=today,
        series_ids=("DGS10", "DTB3", "VIXCLS", "BAMLH0A0HYM2"),
        window=f"full history to {today.isoformat()}, 1-month horizon",
        claim=claim,
        evidence={
            "implied_excess_return_bp": implied,
            "r_squared": round(r2, 3),
            "n_obs": result.n_obs,
            "coefficients_bp_per_unit": result.coefficients,
            "t_stats": result.t_stats,
            "current_predictors": result.current_predictors,
        },
        interpretation=interpretation,
        sources=[],
        status="new",
        score=min(abs(implied) / 50, 1.0),
    )

    return [finding], {}
