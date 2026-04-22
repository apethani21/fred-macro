#!/usr/bin/env python3
"""Smoke test for the analytics package.

Loads a couple of daily series, runs through data alignment, correlations
with stability, percentile/z-score, lead-lag, and emits a chart. Verifies
the package composes end-to-end without needing the full universe.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.analytics import charts, data, episodes, format as fmt, stats
from src.ingest.paths import STATE_DIR


def main() -> int:
    # --- data: load + align two daily series ---
    dgs10 = data.load_series("DGS10")
    dgs2 = data.load_series("DGS2")
    print(f"DGS10: {len(dgs10)} points, {dgs10.index.min().date()} → {dgs10.index.max().date()}")
    print(f"DGS2:  {len(dgs2)} points, {dgs2.index.min().date()} → {dgs2.index.max().date()}")

    aligned = data.load_aligned(["DGS10", "DGS2"], freq="D", how="coarsest")
    print(f"aligned shape: {aligned.shape}, head:\n{aligned.tail(3)}")

    # --- stats: differences (levels of rates are non-stationary), rolling corr, stability ---
    diffs = data.to_returns(aligned, kind="diff").dropna()
    rc = stats.rolling_corr(diffs["DGS10"], diffs["DGS2"], window=126, method="pearson")
    print(f"rolling Pearson (126d) of daily changes: latest = {rc.iloc[-1]:.3f}")

    stab = stats.correlation_with_stability(diffs["DGS10"], diffs["DGS2"], method="spearman", n_subperiods=4)
    print(
        f"full-sample Spearman = {stab.correlation:.3f}; "
        f"subperiods = {[round(v, 3) for v in stab.subperiod_values]}; "
        f"spread = {stab.stability_spread:.3f}"
    )

    # --- percentile / z-score on the 2s10s spread level ---
    t10y2y = data.load_series("T10Y2Y")
    pct = stats.percentile_rank(t10y2y)
    z = stats.zscore_vs_history(t10y2y)
    print(f"T10Y2Y today: {fmt.fmt_level(t10y2y.dropna().iloc[-1])} — {fmt.fmt_percentile(pct)}, {fmt.fmt_z(z)}")

    # --- lead-lag: does DGS2 lead DGS10 at short lags? (daily diffs) ---
    ll = stats.lead_lag_xcorr(diffs["DGS10"], diffs["DGS2"], max_lag=5, method="pearson")
    peak = ll.loc[ll["correlation"].idxmax()]
    print(f"lead-lag peak: lag={int(peak['lag'])}, corr={peak['correlation']:.3f} (n={int(peak['n'])})")

    # --- episodes: in-recession check + episode window slice ---
    print(f"is 2008-09-15 in a recession? {episodes.in_recession('2008-09-15')}")
    gfc_window = episodes.slice_to_episode(t10y2y.dropna(), "gfc")
    print(f"T10Y2Y during GFC: min={gfc_window.min():.2f}, max={gfc_window.max():.2f}, mean={gfc_window.mean():.2f}")

    # --- chart: time-series with NBER shading + latest annotation ---
    out_dir = STATE_DIR / "charts" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = charts.time_series(
        t10y2y,
        title="10-Year minus 2-Year Treasury Spread (T10Y2Y)",
        subtitle="Daily, 1976–present. NBER recessions shaded.",
        series_id="T10Y2Y",
        ylabel="Percent",
        shade_nber=True,
        annotate_last=True,
    )
    ax.axhline(0, color="#6B7280", linewidth=0.8, linestyle="--")
    charts.add_source_footer(fig, ["T10Y2Y"], as_of=t10y2y.dropna().index[-1])
    out_path = charts.save_to(fig, out_dir / "t10y2y.png")
    print(f"wrote chart: {out_path}")

    # --- multi-series chart with NBER shading ---
    fig2, _ = charts.multi_series(
        aligned,
        title="Treasury Yields: 2-Year and 10-Year",
        subtitle="Daily constant-maturity yields.",
        ylabel="Percent",
    )
    charts.add_source_footer(fig2, ["DGS10", "DGS2"], as_of=aligned.dropna().index[-1])
    out2 = charts.save_to(fig2, out_dir / "yields_2s_10s.png")
    print(f"wrote chart: {out2}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
