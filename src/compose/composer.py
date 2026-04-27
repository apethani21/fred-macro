"""Email composer: data context → LLM draft → fact-check pass → HTML.

Flow for each email:
  1. build_data_context  — queries parquet for current value, z-score, percentile rank
  2. draft_email         — Claude writes prose around the numbers (numbers come from step 1)
  3. fact_check_draft    — second Claude pass flags banned phrases, wrong numbers, focus drift
  4. generate_charts     — up to 3 charts interwoven with body via {{CHART_N}} placeholders
  5. compose_email       — orchestrates 1–4, returns ComposedEmail

Key constraint from CLAUDE.md: the LLM writes prose; it does not produce numbers.
Every number in the email comes from a parquet query.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from src.analytics.data import load_series, series_metadata
from src.analytics.stats import zscore_vs_history, percentile_rank
from src.ingest.paths import STATE_DIR
from src.research.findings import Finding
from src.select.selector import LessonPick


# ---------- market snapshot table ----------
import math as _math

# kind controls level and change formatting:
#   "rate_pct"  — level "X.XX%",    change in bp (×100), integer
#   "spread_bp" — level "XXX bp",   change in bp (×100), integer
#   "plain_1"   — level "X.X",      change ±X.X
#   "plain_2"   — level "X.XX",     change ±X.XX
#   "fx_4"      — level "X.XXXX",   change ±X.XXX (3dp, saves width)
#   "dollar_0"  — level "$X,XXX",   change ±$X
#   "dollar_1"  — level "$X.X",     change ±$X.X

_SNAPSHOT_SERIES: list = [
    ("__section__", "US RATES"),
    ("DTB3",   "3M T-Bill",   "rate_pct"),
    ("DGS2",   "2Y UST",      "rate_pct"),
    ("DGS5",   "5Y UST",      "rate_pct"),
    ("DGS10",  "10Y UST",     "rate_pct"),
    ("DGS30",  "30Y UST",     "rate_pct"),
    ("__section__", "SPREADS / INFLATION"),
    ("T10Y2Y",  "2s10s",       "spread_bp"),
    ("T10Y3M",  "3M10Y",       "spread_bp"),
    ("DFII10",  "10Y Real",    "rate_pct"),
    ("T10YIE",  "10Y BEI",     "rate_pct"),
    ("T5YIFR",  "5Y5Y fwd",    "rate_pct"),
    ("__section__", "ECB POLICY"),
    ("ECB.DFR",          "ECB DFR",      "rate_pct"),
    ("ECB.ESTR",         "€STR",         "rate_pct"),
    ("__section__", "EUROPEAN RATES"),
    ("ECB.YC.AAA.2Y",    "EA AAA 2Y",   "rate_pct"),
    ("ECB.YC.AAA.10Y",   "EA AAA 10Y",  "rate_pct"),
    ("ECB.BUND.SLOPE",   "Bund slope",  "spread_bp"),
    ("ECB.BTPBUND.SPREAD","BTP-Bund",   "spread_bp"),
    ("IRLTLT01FRM156N",  "France 10Y",  "rate_pct"),
    ("IRLTLT01GBM156N",  "UK 10Y",      "rate_pct"),
    ("__section__", "CREDIT"),
    ("BAMLC0A0CM",   "IG OAS",  "spread_bp"),
    ("BAMLH0A0HYM2", "HY OAS",  "spread_bp"),
    ("__section__", "G10 FX"),
    ("DTWEXBGS",  "USD broad",  "plain_2"),
    ("DEXUSEU",   "EUR/USD",    "fx_4"),
    ("DEXJPUS",   "USD/JPY",    "plain_1"),
    ("DEXUSUK",   "GBP/USD",    "fx_4"),
    ("DEXUSAL",   "AUD/USD",    "fx_4"),
    ("__section__", "COMMODITIES & VOL"),
    ("DCOILWTICO",       "WTI",   "dollar_1"),
    ("GOLDAMGBD228NLBM", "Gold",  "dollar_0"),
    ("VIXCLS",           "VIX",   "plain_1"),
]

_SNAPSHOT_GLOSSARY = (
    "BEI = breakeven inflation · OAS = option-adjusted spread · "
    "2s10s = 10Y−2Y spread · 3M10Y = 10Y−3M spread · "
    "Real = TIPS yield · 5Y5Y fwd = 5Y-5Y forward inflation · "
    "IG/HY = investment/high-yield grade · "
    "DFR = ECB deposit facility rate · €STR = euro short-term rate (ECB overnight) · "
    "EA AAA = euro-area AAA sovereign yield curve · BTP-Bund = Italy−Germany 10Y spread · "
    "Bund slope = EA AAA 10Y−2Y · USD broad = trade-weighted dollar"
)


def _fmt_level(raw: float, kind: str) -> str:
    if kind == "rate_pct":
        return f"{raw:.2f}%"
    elif kind == "spread_bp":
        return f"{raw * 100:.0f} bp"
    elif kind == "plain_2":
        return f"{raw:.2f}"
    elif kind == "fx_4":
        return f"{raw:.4f}"
    elif kind == "dollar_0":
        return f"${raw:,.0f}"
    elif kind == "dollar_1":
        return f"${raw:.1f}"
    else:  # plain_1
        return f"{raw:.1f}"


def _fmt_raw_chg(chg: float, kind: str) -> str:
    sign = "+" if chg >= 0 else ""
    if kind in ("rate_pct", "spread_bp"):
        bp = chg * 100
        sign = "+" if bp >= 0 else ""
        return f"{sign}{bp:.0f}bp"
    elif kind == "plain_2":
        return f"{sign}{chg:.2f}"
    elif kind == "fx_4":
        return f"{chg:+.3f}"
    elif kind == "dollar_0":
        return (f"-${abs(chg):,.0f}" if chg < 0 else f"+${chg:,.0f}")
    elif kind == "dollar_1":
        return (f"-${abs(chg):.1f}" if chg < 0 else f"+${chg:.1f}")
    else:  # plain_1
        return f"{sign}{chg:.1f}"


def _fmt_sigma(sigma: float | None) -> str:
    """Format σ value — no symbol, just number (column header carries the unit)."""
    if sigma is None or not _math.isfinite(sigma):
        return "—"
    sign = "+" if sigma >= 0 else ""
    return f"{sign}{sigma:.1f}"


def _pct_cell_style(pct: float | None) -> str:
    if pct is None:
        return ""
    if pct >= 0.90:
        return "background:#fee2e2; color:#7f1d1d; font-weight:600;"
    elif pct >= 0.75:
        return "background:#fef3c7; color:#78350f; font-weight:600;"
    elif pct <= 0.10:
        return "background:#dbeafe; color:#1e3a5f; font-weight:600;"
    elif pct <= 0.25:
        return "background:#e0f2fe; color:#0c4a6e; font-weight:600;"
    return ""


def _pct_label(pct: float | None) -> str:
    if pct is None:
        return "—"
    return f"{pct * 100:.0f}"


def _sigma_cell_style(sigma: float | None) -> str:
    """Sign-based gradient: positive moves shade green, negative shade red. Intensity by magnitude."""
    if sigma is None or not _math.isfinite(sigma):
        return ""
    if sigma >= 3.0:
        return "background:#86efac; color:#14532d; font-weight:700;"
    elif sigma >= 2.0:
        return "background:#bbf7d0; color:#166534; font-weight:600;"
    elif sigma >= 1.5:
        return "background:#dcfce7; color:#166534; font-weight:600;"
    elif sigma >= 1.0:
        return "background:#f0fdf4; color:#374151;"
    elif sigma <= -3.0:
        return "background:#fca5a5; color:#7f1d1d; font-weight:700;"
    elif sigma <= -2.0:
        return "background:#fecaca; color:#991b1b; font-weight:600;"
    elif sigma <= -1.5:
        return "background:#fee2e2; color:#991b1b; font-weight:600;"
    elif sigma <= -1.0:
        return "background:#fef2f2; color:#374151;"
    return ""


def _compute_changes(s: pd.Series) -> dict:
    """1D/1W/1M raw changes + σ-normalised (1D-equivalent).

    σ formula: chg_N / (daily_std × √N_days)
    All horizons expressed in the same daily-σ unit for comparability.
    """
    result: dict = {}
    if len(s) < 22:
        return result
    cur = float(s.iloc[-1])
    result["chg_1d"] = cur - float(s.iloc[-2]) if len(s) >= 2 else None
    result["chg_1w"] = cur - float(s.iloc[-6]) if len(s) >= 6 else None
    result["chg_1m"] = cur - float(s.iloc[-22]) if len(s) >= 22 else None

    diffs = s.diff().dropna()
    daily_std = float(diffs.tail(252).std()) if len(diffs) >= 21 else None
    if daily_std and daily_std > 0:
        result["sigma_1d"] = result["chg_1d"] / daily_std if result["chg_1d"] is not None else None
        result["sigma_1w"] = result["chg_1w"] / (daily_std * _math.sqrt(5)) if result["chg_1w"] is not None else None
        result["sigma_1m"] = result["chg_1m"] / (daily_std * _math.sqrt(21)) if result["chg_1m"] is not None else None
    return result


_RATE_LEVEL_BOUNDS = (-1.0, 25.0)   # % — flags anything outside as suspicious
_RATE_1D_CHANGE_WARN = 0.50         # 50bp single-day move → log a warning

def _validate_row(fred_id: str, label: str, kind: str, level: float, chgs: dict) -> None:
    """Log warnings for values that look anomalous so issues surface before sending."""
    import logging as _log
    _vlog = _log.getLogger(__name__)
    if kind == "rate_pct":
        lo, hi = _RATE_LEVEL_BOUNDS
        if not (lo <= level <= hi):
            _vlog.warning("Snapshot sanity: %s (%s) level %.4f outside [%.1f, %.1f]%%",
                          label, fred_id, level, lo, hi)
        chg1d = chgs.get("chg_1d")
        if chg1d is not None and abs(chg1d) > _RATE_1D_CHANGE_WARN:
            _vlog.warning("Snapshot sanity: %s (%s) 1D change %.4f (%.0fbp) exceeds ±50bp — verify",
                          label, fred_id, chg1d, chg1d * 100)
    if kind == "spread_bp":
        chg1d = chgs.get("chg_1d")
        if chg1d is not None and abs(chg1d * 100) > 100:
            _vlog.warning("Snapshot sanity: %s (%s) 1D spread change %.0fbp exceeds ±100bp — verify",
                          label, fred_id, chg1d * 100)


_DAILY_FREQS = {"D", "BW", "W"}  # frequencies with meaningful intraday/intraweek changes

def _build_rows(series_list: list) -> list[dict]:
    lb1y = 252
    lb5y = 252 * 5
    rows: list[dict] = []
    for entry in series_list:
        if entry[0] == "__section__":
            rows.append({"__section__": entry[1]})
            continue
        fred_id, label, kind = entry
        try:
            s = load_series(fred_id).dropna()
            if s.empty:
                continue
            cur = float(s.iloc[-1])
            meta = series_metadata(fred_id)
            is_daily = (meta is None) or (str(meta.get("frequency_short", "D")).upper() in _DAILY_FREQS)
            if is_daily:
                chgs = _compute_changes(s)
                _validate_row(fred_id, label, kind, cur, chgs)
            else:
                chgs = {}  # suppress 1D/1W/1M columns for monthly/quarterly series
            rows.append({
                "label": label,
                "kind": kind,
                "level": _fmt_level(cur, kind),
                "chg_1d": _fmt_raw_chg(chgs["chg_1d"], kind) if chgs.get("chg_1d") is not None else "—",
                "sigma_1d": chgs.get("sigma_1d"),
                "chg_1w": _fmt_raw_chg(chgs["chg_1w"], kind) if chgs.get("chg_1w") is not None else "—",
                "sigma_1w": chgs.get("sigma_1w"),
                "chg_1m": _fmt_raw_chg(chgs["chg_1m"], kind) if chgs.get("chg_1m") is not None else "—",
                "sigma_1m": chgs.get("sigma_1m"),
                "pct_1y": percentile_rank(s, cur, lookback=lb1y),
                "pct_5y": percentile_rank(s, cur, lookback=lb5y),
            })
        except Exception:
            continue
    while rows and "__section__" in rows[-1]:
        rows.pop()
    return rows


def build_market_snapshot() -> list[dict]:
    return _build_rows(_SNAPSHOT_SERIES)


# ── Release calendar table ────────────────────────────────────────────────────

# Keyword → (short label, equity watch note).  Matched case-insensitively against release_name.
_RELEASE_EQUITY: list[tuple[str, str, str]] = [
    ("employment situation",    "NFP",              "Wages + payrolls drive labor-cost repricing for rate-sensitive sectors (utilities, REITs) and consumer discretionary."),
    ("consumer price",          "CPI",              "Core surprise is the primary input to short-end rate expectations; moves rate-sensitive and inflation-linked positions."),
    ("personal income",         "PCE / Income",     "PCE is the Fed's preferred inflation gauge; a hot print hardens the hawkish path more than an equivalent CPI beat."),
    ("gross domestic product",  "GDP",              "Growth read: beat → cyclicals and financials; miss → defensives. Watch the consumption and investment split."),
    ("retail sales",            "Retail Sales",     "Direct read on consumer spending; beats move consumer discretionary vs. staples spread."),
    ("industrial production",   "IP",               "Manufacturing output; moves industrials and materials. Watch capacity utilization for capex signals."),
    ("job openings",            "JOLTS",            "Quits rate is a leading wage-growth indicator; high openings/quits = more Fed tightening risk."),
    ("federal open market",     "FOMC",             "Direct rate decision; equity multiples compress on surprise hikes, expand on cuts."),
    ("producer price",          "PPI",              "Upstream cost pressure; leads margin compression in consumer-facing sectors with a 1–2 quarter lag."),
    ("housing starts",          "Housing Starts",   "Construction activity; moves homebuilders (XHB), lumber, and mortgage-sensitive financials."),
    ("durable goods",           "Durable Goods",    "Capex proxy (ex-defense, ex-aircraft nondefense capital goods); leads industrials and tech hardware."),
    ("trade balance",           "Trade Balance",    "Net exports affect GDP revisions and USD; wide deficit can drag on multinationals' earnings translation."),
    ("consumer confidence",     "Conf. Board",      "Forward consumer sentiment; leads discretionary spending by 2–3 months."),
    ("consumer sentiment",      "UMich Sentiment",  "Inflation expectations component directly informs Fed communication and breakeven pricing."),
    ("ism manufacturing",       "ISM Mfg",          "Purchasing managers' activity index; new orders sub-index is the primary equity signal."),
    ("ism services",            "ISM Services",     "Services activity; employment sub-index is a soft leading indicator for NFP."),
    ("purchasing managers",     "PMI",              "Flash PMI gives the earliest monthly read on activity; composite above/below 50 drives risk-on/off."),
    ("beige book",              "Beige Book",       "Qualitative Fed read; usually low equity impact unless it contradicts recent hard data."),
    ("fomc minutes",            "FOMC Minutes",     "Detailed committee debate; watch for dissent counts and language around the balance sheet."),
    ("import price",            "Import Prices",    "Passes through to CPI goods with a ~1-month lag; USD strength mutes the signal."),
    ("s&p/case-shiller",        "Case-Shiller",     "Home price appreciation; feeds OER component of CPI with a 12–18 month lag."),
    ("existing home sales",     "Existing Home",    "Demand side of housing; lock-in effect depresses turnover when mortgage rates are high."),
    ("new home sales",          "New Home Sales",   "Builder activity proxy; more timely than existing sales for construction demand."),
    # ECB / euro area
    ("ecb governing council",   "ECB Decision",     "Rate decision + forward guidance; EUR/USD and European bank equities are the primary movers."),
    ("eurostat flash hicp",     "EA HICP Flash",    "EA inflation print; drives ECB rate path expectations and EUR-denominated asset repricing."),
    ("eurostat hicp",           "EA HICP",          "EA inflation print; drives ECB rate path expectations and EUR-denominated asset repricing."),
    ("flash hicp",              "EA HICP Flash",    "EA inflation print; drives ECB rate path expectations and EUR-denominated asset repricing."),
    # Other G7 + Japan central banks
    ("bank of england mpc",     "BoE Decision",     "Rate decision + MPC vote split (9 members); GBP/USD and gilt yields are the primary movers. Super Thursday meetings (Feb/May/Aug/Nov) also publish the Monetary Policy Report."),
    ("bank of england",         "BoE Decision",     "Rate decision + MPC vote split (9 members); GBP/USD and gilt yields are the primary movers. Super Thursday meetings (Feb/May/Aug/Nov) also publish the Monetary Policy Report."),
    ("bank of japan mpm",       "BoJ Decision",     "Rate and YCC/QQE policy decision; JPY crosses and JGB yields are the primary movers. Hawkish surprises amplify yen carry unwinds across global risk assets."),
    ("bank of japan",           "BoJ Decision",     "Rate and YCC/QQE policy decision; JPY crosses and JGB yields are the primary movers. Hawkish surprises amplify yen carry unwinds across global risk assets."),
    ("bank of canada",          "BoC Decision",     "Rate decision; CAD/USD and Canadian government bonds are the primary movers. MPR meetings (Jan/Apr/Jul/Oct) add growth and inflation projections. Oil price pass-through makes BoC hawkishness more commodity-linked than peers."),
]


def _release_equity_note(release_name: str) -> tuple[str, str]:
    """Return (short_label, equity_watch) for a release name, or generic fallback."""
    name_lower = release_name.lower()
    for keyword, label, note in _RELEASE_EQUITY:
        if keyword in name_lower:
            return label, note
    return release_name, "—"


def build_release_calendar_table(today: date, days_ahead: int = 10) -> str:
    """Return an HTML table of upcoming releases: Date | Release | Equity Watch."""
    from src.ingest.paths import DATA_DIR

    cal_path = DATA_DIR / "release_calendar.parquet"
    if not cal_path.exists():
        return ""
    try:
        cal = pd.read_parquet(cal_path)
    except Exception:
        return ""

    cal["release_date"] = pd.to_datetime(cal["release_date"])
    start_ts = pd.Timestamp(today)
    end_ts = start_ts + pd.Timedelta(days=days_ahead)
    upcoming = cal[(cal["release_date"] >= start_ts) & (cal["release_date"] <= end_ts)].sort_values("release_date").reset_index(drop=True)
    if upcoming.empty:
        return ""

    # Deduplicate by release_name per date; skip unrecognised continuous feeds
    seen: set[tuple] = set()
    rows_html = []
    for _, row in upcoming.iterrows():
        rdate = row["release_date"]
        rname = str(row.get("release_name", ""))
        label, note = _release_equity_note(rname)
        if note == "—":
            continue  # not a recognised discrete economic release
        key = (rdate, label)
        if key in seen:
            continue
        seen.add(key)
        if isinstance(rdate, pd.Timestamp):
            date_str = rdate.strftime("%a %-d %b")
        else:
            date_str = str(rdate)
        td_date  = f'<td style="font-size:11px;padding:4px 8px;white-space:nowrap;border-bottom:1px solid #e5e7eb;color:#374151;">{date_str}</td>'
        td_label = f'<td style="font-size:11px;padding:4px 8px;white-space:nowrap;border-bottom:1px solid #e5e7eb;font-weight:600;">{label}</td>'
        td_note  = f'<td style="font-size:11px;padding:4px 8px;border-bottom:1px solid #e5e7eb;color:#374151;">{note}</td>'
        rows_html.append(f"<tr>{td_date}{td_label}{td_note}</tr>")

    if not rows_html:
        return ""

    header = (
        '<tr>'
        '<th style="background:#111;color:#fff;font-size:10px;font-weight:600;padding:4px 8px;text-align:left;">Date</th>'
        '<th style="background:#111;color:#fff;font-size:10px;font-weight:600;padding:4px 8px;text-align:left;">Release</th>'
        '<th style="background:#111;color:#fff;font-size:10px;font-weight:600;padding:4px 8px;text-align:left;">Equity Watch</th>'
        '</tr>'
    )
    title_bar = (
        '<p style="font-family:Helvetica,Arial,sans-serif;font-size:10px;font-weight:700;'
        'text-transform:uppercase;letter-spacing:1px;color:#111;margin:20px 0 4px 0;">'
        f'Upcoming Releases — next {days_ahead} days</p>'
    )
    table = (
        title_bar
        + '<table style="border-collapse:collapse;width:100%;font-family:Helvetica,Arial,sans-serif;">'
        + header
        + "".join(rows_html)
        + "</table>"
    )
    return table


# ── HTML rendering helpers ────────────────────────────────────────────────────

_TH_BASE = (
    "background:#111; color:#fff; font-weight:600; font-size:10px; "
    "letter-spacing:0.3px; white-space:nowrap; padding:3px 5px;"
)
_TD_BASE = "border-bottom:1px solid #e5e7eb; font-size:11px; white-space:nowrap; padding:2px 5px;"

_SECTION_STYLE = (
    "background:#f3f4f6; font-size:9px; font-weight:700; text-transform:uppercase; "
    "letter-spacing:0.8px; color:#6b7280; padding:4px 5px 2px 5px; border-top:1px solid #d1d5db;"
)


def _th(text: str, align: str = "right", **attrs: str) -> str:
    extra = "".join(f' {k}="{v}"' for k, v in attrs.items())
    return f'<th style="{_TH_BASE}text-align:{align};"{extra}>{text}</th>'


def _td(text: str, bg: str, extra_style: str = "", align: str = "right") -> str:
    return f'<td style="{bg}{_TD_BASE}text-align:{align};{extra_style}">{text}</td>'


_COL_BORDER = "border-left:2px solid #9ca3af;"

def _render_table_html(rows: list[dict], title: str, as_of: date) -> str:
    """Render the market snapshot table with multi-index column headers."""
    if not rows:
        return ""

    n_cols = 10  # Market + Level + (6 change cols) + 2 %ile
    group_th_style = f"{_TH_BASE}text-align:center; border-bottom:1px solid #444; {_COL_BORDER}"
    head = (
        '<thead>'
        '<tr>'
        + _th("Market", align="left", rowspan="2")
        + _th("Level", rowspan="2")
        + _th("Δ 1D", align="center", colspan="2", style=group_th_style)
        + _th("Δ 1W", align="center", colspan="2", style=group_th_style)
        + _th("Δ 1M", align="center", colspan="2", style=group_th_style)
        + _th("%ile", align="center", colspan="2",
              style=f"{_TH_BASE}text-align:center; {_COL_BORDER}")
        + '</tr><tr>'
        + _th("Val", style=f"{_TH_BASE}text-align:right; {_COL_BORDER}") + _th("SD")
        + _th("Val", style=f"{_TH_BASE}text-align:right; {_COL_BORDER}") + _th("SD")
        + _th("Val", style=f"{_TH_BASE}text-align:right; {_COL_BORDER}") + _th("SD")
        + _th("1Y", style=f"{_TH_BASE}text-align:right; {_COL_BORDER}") + _th("5Y")
        + '</tr></thead>'
    )

    _border_td_style = f"font-variant-numeric:tabular-nums; {_COL_BORDER}"

    body_rows: list[str] = []
    data_idx = 0
    for r in rows:
        if "__section__" in r:
            body_rows.append(
                f'<tr><td colspan="{n_cols}" style="{_SECTION_STYLE}">{r["__section__"]}</td></tr>'
            )
            continue
        bg = "background:#f9fafb;" if data_idx % 2 == 1 else ""
        s1d, s1w, s1m = r.get("sigma_1d"), r.get("sigma_1w"), r.get("sigma_1m")
        body_rows.append(
            "<tr>"
            + _td(r["label"], bg, "font-weight:600;", "left")
            + _td(r["level"], bg, "font-variant-numeric:tabular-nums;")
            + _td(r["chg_1d"], bg, f"color:#374151; {_border_td_style}")
            + _td(_fmt_sigma(s1d), bg, _sigma_cell_style(s1d))
            + _td(r["chg_1w"], bg, f"color:#374151; {_border_td_style}")
            + _td(_fmt_sigma(s1w), bg, _sigma_cell_style(s1w))
            + _td(r["chg_1m"], bg, f"color:#374151; {_border_td_style}")
            + _td(_fmt_sigma(s1m), bg, _sigma_cell_style(s1m))
            + _td(_pct_label(r["pct_1y"]), bg, f"{_pct_cell_style(r['pct_1y'])}{_COL_BORDER}")
            + _td(_pct_label(r["pct_5y"]), bg, _pct_cell_style(r["pct_5y"]))
            + "</tr>"
        )
        data_idx += 1

    legend_colors = (
        '<span style="background:#dbeafe; padding:0 3px;">low</span> · '
        '<span style="background:#fef3c7; padding:0 3px;">elevated</span> · '
        '<span style="background:#fee2e2; padding:0 3px;">extreme</span>'
    )
    sd_legend = (
        '<span style="background:#f0fdf4; padding:0 3px;">+≥1</span> · '
        '<span style="background:#86efac; padding:0 3px;">+≥2</span> · '
        '<span style="background:#fef2f2; padding:0 3px;">−≥1</span> · '
        '<span style="background:#fca5a5; padding:0 3px;">−≥2</span>'
    )

    note_parts = [
        f"As of {as_of.isoformat()}",
        "Val = bp for rates/spreads · SD = 1D-equivalent σ",
        f"%ile: {legend_colors}",
        f"SD: {sd_legend}",
    ]

    return (
        f'<p style="font-family:Helvetica,Arial,sans-serif; font-size:10px; font-weight:700; '
        f'color:#374151; margin:24px 0 4px 0; text-transform:uppercase; letter-spacing:0.5px;">'
        f'{title}</p>'
        f'<table style="border-collapse:collapse; width:100%; margin:0 0 4px 0; '
        f'font-family:Helvetica,Arial,sans-serif;">'
        + head
        + "<tbody>" + "".join(body_rows) + "</tbody></table>"
        + f'<p style="font-family:Helvetica,Arial,sans-serif; font-size:9px; color:#9ca3af; margin:2px 0 4px 0;">'
        + " · ".join(note_parts) + "</p>"
        + f'<p style="font-family:Helvetica,Arial,sans-serif; font-size:9px; color:#9ca3af; margin:0 0 8px 0;">'
        + _SNAPSHOT_GLOSSARY + "</p>"
    )


def render_snapshot_table(rows: list[dict], as_of: date) -> str:
    return _render_table_html(rows, "Market Snapshot", as_of)

logger = logging.getLogger(__name__)

CHARTS_DIR = STATE_DIR / "charts"
LAST_EMAIL_PATH = STATE_DIR / "last_email.html"

# ---------- Anthropic client ----------

def _anthropic_key() -> str:
    key_file = Path.home() / "keys" / "anthropic" / "key.txt"
    if key_file.exists():
        k = key_file.read_text().strip()
        if k:
            return k
    env = os.environ.get("ANTHROPIC_API_KEY", "")
    if env:
        return env
    raise RuntimeError(
        "Anthropic API key not found.\n"
        "Either create ~/keys/anthropic/key.txt with your key,\n"
        "or set the ANTHROPIC_API_KEY environment variable."
    )


def _client():
    import anthropic
    return anthropic.Anthropic(api_key=_anthropic_key())


MODEL = "claude-sonnet-4-6"

# ---------- prompts ----------

_DRAFT_SYSTEM = """\
You compose a daily macro education email for a quantitative equity finance professional (3 years, equities focus).
They are fluent with statistics, time series, and financial markets.
Explain macro-specific jargon and institutional mechanics they may not know from equities.
When a macro concept has an equities-market analogue, point it out.

TARGET: 500–700 words total (excluding the Terms & Mechanics section). One email, one organizing idea.

STRUCTURE — divide the email using the <h3> section headers listed below. Use EXACTLY these header strings and no others. Do not create custom headers or use the finding title as a header.
1. (no header) Hook (2–3 sentences): set the scene — introduce the topic and why it is worth understanding today. If anchored to an upcoming release, name it and briefly say what it will test. Do NOT open with the empirical punchline; the hook is the door, not the room. No generic opener.
2. <h3>The Concept</h3> (2–4 short paragraphs): explain what the metric is, how it is constructed, what drives it, and why the reader should care — woven together naturally, not as a list. Start with the intuitive plain-English version ("think of it as X"), add the technical detail, name any direct equities analogue, and in the same flow explain what information it carries that equity prices alone do not. By the end of this section the reader understands both what this thing is and why it matters. Define every fixed-income or macro-specific term the moment it first appears — briefly, inline, in parentheses or a subordinate clause; do not save definitions for Terms & Mechanics. When the equities analogue is the clearest entry point to the concept, lead with it before the formal definition. Do not shy away from technical precision: if the concept has a clean mathematical definition (e.g., modified duration ≈ −ΔP/P ÷ Δy), state it — a quant reader learns more from the formula than from a roundabout description.
3. <h3>The Analysis</h3> (1–2 paragraphs): the specific empirical work — what the data actually shows, with numbers, windows, and citation. This is the research behind today's email. Place {{CHART_1}} immediately after this section.
4. <h3>Where We Are Now</h3> (1 paragraph): current reading vs history. Where in the distribution? How does today compare to prior episodes? Place {{CHART_2}} immediately after. Place {{CHART_3}} immediately after {{CHART_2}} if a third chart exists.
5. <h3>What to Watch</h3> (1–2 sentences): the specific release, data point, or signal that would update the picture. Dated and specific — no generic closers.
6. Terms & Mechanics (optional, 2–3 items): cover only macro-specific institutional mechanics or jargon the reader likely doesn't know cold — data construction nuances (e.g. birth-death model, seasonal adjustment, chained dollars), Fed operational plumbing (IORB, ON RRP, SRF, reserve requirements), or release methodology quirks. Skip financial basics the reader already knows: basis points, yield, standard deviation, what a spread is, price-return relationships. Format: a <ul> with one <li> per term — <strong>Term</strong>: one-sentence definition. No Q: prefix. These may appear inline in the most relevant section rather than grouped at the end. If no genuinely non-obvious mechanics appear in today's email, omit this section entirely. Do NOT use <h3>Terms & Mechanics</h3> as a section header.

CHART PLACEHOLDERS:
- Emit {{CHART_1}}, {{CHART_2}}, and optionally {{CHART_3}} as standalone lines in body_html at the specified positions above.
- What each chart shows depends on the finding kind (provided in FINDING RECORD below):
  • correlation_shift / lead_lag_change: {{CHART_1}} = primary series full history; {{CHART_2}} = rolling Spearman correlation over time; {{CHART_3}} = split cross-correlogram (historical vs. recent lag structure).
  • regime_transition / notable_move_level / notable_move_change: {{CHART_1}} = primary series full history; {{CHART_2}} = rolling percentile rank over time; {{CHART_3}} = full-history distribution histogram with current value marked.
  • fomc_event_study: {{CHART_1}} = primary series full history; {{CHART_2}} = era comparison bar chart (path vs timing surprise share, or OLS coefficients, by Fed era). This bar chart is the analytical heart — place {{CHART_2}} immediately after the empirical paragraph that explains the era breakdown.
  • harvested_source / structural_break: {{CHART_1}} = multi-series time plot of all referenced series (z-score normalised if units differ, otherwise raw); {{CHART_2}} = rolling percentile rank of primary series; {{CHART_3}} = full-history distribution histogram.
- Only emit a placeholder if the chart will genuinely add information not already in the prose. Never emit a placeholder just to fill a slot.
- Place each placeholder immediately after the paragraph it illustrates. Never place two placeholders in the same paragraph block.
- Do not write "the chart above/below" — let the chart speak.

EQUATION PLACEHOLDER:
- If the finding involves a regression model (e.g. OLS, event study), emit {{EQUATION}} as a standalone line in body_html immediately after the sentence that introduces the regression specification — typically in The Analysis section.
- {{EQUATION}} renders as a typeset formula image. It is for the regression model itself, not for scalar numbers — do not use it to show a single coefficient value.
- Only emit {{EQUATION}} when a regression equation is genuinely central to understanding the analysis. Omit it for correlation-only or descriptive findings.
- Whether an equation is available is indicated in the FINDING RECORD below under EQUATION.

TABLES:
- Use an HTML table (<table>, <tr>, <th>, <td>) when presenting 3+ data points that share a common structure (e.g., percentile readings across lookback windows, regime durations, correlation values at multiple lags).
- Tables replace the prose enumeration of numbers — do not repeat table contents in surrounding prose.
- Keep tables narrow: 2–4 columns maximum. Do not add inline styles — the email stylesheet handles table formatting.
- Bold key cells where a value is the point of the row (e.g., the current reading in a historical comparison table).

DIAGRAMS:
- Use a <pre> block with ASCII art when a diagram would clarify a structural or mechanical relationship better than prose. Good candidates: rate floors/ceilings (Fed funds corridor), flow-of-funds chains (reserves → banks → money markets), transmission mechanisms, balance-sheet T-accounts.
- Keep diagrams narrow (≤60 chars wide) and label every node. Use arrows (→, ←, ↑, ↓, ↔) and box corners (+, ─, │) for clarity.
- A diagram replaces prose about structure, not prose about data — don't diagram a time series.

CITATIONS AND HYPERLINKS:
- When citing a specific paper, author, or institutional source that appears in the finding's Sources list or CONCEPT CONTEXT, wrap it in an <a href="URL"> tag using the URL from that source.
- Format: <a href="URL">Author et al. (Year)</a>. If no URL is available in the provided context, cite without a link rather than inventing a URL.
- Never fabricate URLs.

PROSE RULES:
- First sentence of email contains information, not meta-commentary.
- No filler transitions: "Moreover," "Furthermore," "It is worth noting" — cut.
- No hedging-as-authority: "Many economists believe" is banned. Cite a specific economist/paper/institution or drop the claim.
- No platitudes at the close: "Only time will tell," "Markets will be watching" — banned.
- Every number has context (comparison to prior, expectation, historical reference, or threshold).
- Do NOT include FRED or ECB series IDs in prose (e.g. "VIXCLS", "DGS10", "UNRATE", "ECB.DFR", "ECB.BTPBUND.SPREAD"). They are internal identifiers. Refer to series by their full descriptive name only (e.g. "ECB deposit facility rate" not "ECB.DFR"; "BTP-Bund spread" not "ECB.BTPBUND.SPREAD").
- Use '%ile' not 'percentile': e.g., '52nd %ile' not '52nd percentile'.
- In tables, use short lookback labels: 'Full', '5Y', '10Y' — not '5-Year', 'Full History', etc.
- All standard-deviation citations: 1 decimal, bold: '<strong>−0.2σ</strong>'. Never '−0.20σ'.
- Bold ALL key statistics: wrap metric values, z-scores, percentile ranks, correlations, and rate/spread changes in <strong> tags. E.g. <strong>17.94</strong>, <strong>52nd %ile</strong>, <strong>−0.2σ</strong>, <strong>+25 bp</strong>, <strong>−60 → +30</strong>.
- Correlations are expressed on a −100 to +100 scale (already multiplied by 100 in DATA CONTEXT). Always append a % sign: write e.g. <strong>−53%</strong> not <strong>−0.53</strong> or <strong>−53</strong>.
- Never combine a %ile citation with a regime label for the same reading — the number conveys the context.
- Break any sentence over 25 words.
- Bullets are allowed anywhere they aid clarity. Required for any Q&A mechanics callouts.
- Technical depth: do not round off corners. If a mechanism has three steps, explain all three. Do not substitute vague summary language ("it tightens conditions") for the actual chain of causation.
- Causal chains: when explaining how A leads to B leads to C, write the chain explicitly — name the mechanism at each link. Do not jump from premise to conclusion.

FACTUAL ACCURACY RULES:
- Use ONLY the numbers from DATA CONTEXT or CONTEXTUAL BACKGROUND DATA provided. Do not invent or estimate numbers.
- Every FRED-derived value in the email traces to a series in one of those two sections.
- EXCEPTION for harvested_source findings: numbers that appear verbatim in the FINDING RECORD evidence block are from the cited academic or institutional source and may be quoted directly in the email, attributed to that source. Do not invent numbers not present in the evidence block.
- When CONTEXTUAL BACKGROUND DATA is present, use it to ground the "Where We Are Now" section — state where the relevant series sits today and in its historical distribution. Label these readings as background context, not as the finding's direct claim.
- Mechanistic and interpretive claims must be grounded in the finding's Sources list or CONCEPT CONTEXT. If a mechanism is not explicitly supported there, either omit it or soften it to "the mechanism is not established here." Do not assert causal mechanisms as fact on the basis of general knowledge alone.
- When data is ambiguous, surface the uncertainty — do not hide it.

OUTPUT FORMAT:
Return a valid JSON object with exactly three keys: subject, body_html, body_text.
CRITICAL JSON RULES — violations will break parsing:
- No markdown code fences. Start your response with { and end with }.
- No literal newlines inside string values. Use the two-character sequence \n instead.
- No unescaped double-quotes inside string values. Use \" instead, or use HTML &quot; inside the HTML.
- body_html may use <p>, <strong>, <em>, <ul>, <li>, <h3>, <a href="...">, <table>, <tr>, <th>, <td>, <pre> tags only. No divs. No h1 or h2.
- body_html may include {{CHART_1}}, {{CHART_2}}, {{CHART_3}}, and {{EQUATION}} as literal placeholder strings (they will be replaced).
- body_text is plain text with no HTML tags and no chart placeholders.
Example structure (not content): {"subject": "...", "body_html": "<p>...</p>\n<h3>The Concept</h3>\n<p>...</p>\n<h3>The Analysis</h3>\n<p>...</p>", "body_text": "..."}
"""

_FACTCHECK_SYSTEM = """\
You are a fact-checker reviewing a macro education email draft.

Check for:
1. Numbers in the draft that do not match the DATA CONTEXT (factual errors — most important)
2. Hedging-as-authority phrases: "many economists believe", "analysts say", "most experts think", "widely expected"
3. Banned closers: "Only time will tell", "Markets will be watching", "it remains to be seen", "we shall see"
4. Throat-clearing opener: first sentence contains no information, only setup
5. Focus drift: draft covers more than one organizing idea
6. FRED or ECB series IDs appearing in prose (e.g., "VIXCLS", "DGS10", "UNRATE", "ECB.DFR", "ECB.BTPBUND.SPREAD" — series codes must not appear in the email body, including inside h3 headers)
7. "percentile" spelled out in full — must be "%ile"
8. Unsupported historical claims: a claim about a specific historical event with no source in the finding's Sources list or CONCEPT CONTEXT
9. σ citations with more than 1 decimal place (e.g., "−0.20σ" should be "−0.2σ")
10. Key statistics (values, z-scores, %ile ranks, correlations, bp changes) not wrapped in <strong> tags
11. Terms & Mechanics entries that explain true financial basics the reader knows from equities work (what a yield is, what a spread is, what basis points are, what standard deviation is, basic price-return math) — flag as too obvious. Do NOT flag entries for FI/macro-specific mechanics the reader may not know cold from equities: duration, convexity, repo mechanics, haircuts, SOFR construction, seasonal adjustment, CPI basket methodology, BLS birth-death model, chained-dollar deflation, Fed operational details (IORB, ON RRP, SRF) — these are appropriate and useful for this reader
12. Terms & Mechanics items using "Q: ..." format instead of bullet list format

OUTPUT:
Return ONLY a raw JSON object — no preamble, no reasoning, no bullet lists, no markdown. Start your response with { and end with }.
{
  "flags": ["description of issue 1", "description of issue 2"],
  "approved": true
}
Set "approved" to false ONLY if a number in the draft does not match the DATA CONTEXT (wrong value, not a rounding difference).
All style issues — banned phrases, series IDs in prose, spelling of "percentile", unsupported historical context, σ decimal places — are flags only and NEVER block approval.
"""

_CITATION_SYSTEM = """\
You are a citation researcher for a macro finance education email.

Given an email draft, identify all specific mechanistic, causal, or interpretive claims that
go beyond the empirical data (i.e. claims about *why* something happens, historical analogues,
or cross-asset transmission mechanisms). For each claim, search for a primary academic or
institutional source that supports it.

Guidelines:
- Target at most 3 claims — pick the most specific and non-obvious ones.
- Prefer: academic papers (NBER, SSRN), FEDS Notes, Liberty Street Economics, BIS papers.
- Fetch the most promising URL to verify the source actually supports the claim.
- Only return a citation if the source genuinely addresses the mechanism claimed.
- Do not force a citation onto a vague or general claim where no tight source exists.
- If a claim cannot be sourced, include it in "unsourced" — do not fabricate a citation.

OUTPUT FORMAT (raw JSON, no code fence):
{
  "citations": [
    {
      "claim_excerpt": "exact short phrase from the draft identifying the claim",
      "url": "https://...",
      "author": "Surname et al." or "Institution",
      "year": "YYYY",
      "title": "Title of paper or note",
      "relevance": "One sentence: what this source says that supports the claim."
    }
  ],
  "unsourced": ["claim excerpt that could not be sourced"]
}
If no claims need sourcing: { "citations": [], "unsourced": [] }
"""

_REVISION_SYSTEM = """\
You are editing a macro finance email to add inline citations.

You will receive:
1. The current body_html of the email (may contain {{CHART_1}}, {{CHART_2}}, {{CHART_3}},
   {{EQUATION}} placeholders — preserve these exactly).
2. A list of citations, each with: claim_excerpt, url, author, year, title.

Your task:
- For each citation, find the sentence in body_html containing the claim_excerpt.
  Wrap the claim_excerpt text in an anchor tag: <a href="URL" style="color:#1a1a1a;">[N]</a>
  where N is the citation number (1, 2, 3, ...). Place the anchor immediately after the
  claim_excerpt phrase (not wrapping it — insert the superscript after).
  Use: claim_excerpt<a href="URL" style="color:#555; font-size:0.8em; vertical-align:super;">[N]</a>
- Append a References section at the very end of body_html (before any closing tags):
  <h3>References</h3><ol style="font-family:Helvetica,Arial,sans-serif; font-size:13px; color:#444; line-height:1.6;">
  <li><a href="URL">Author (Year), "Title."</a></li>
  ...
  </ol>
- Also append a plain-text references section to body_text.
- Do NOT change any other prose. Do NOT alter numbers, sentences, or structure.
- Preserve all {{CHART_N}} and {{EQUATION}} placeholders exactly as-is.

OUTPUT FORMAT (raw JSON, no code fence):
{"body_html": "...", "body_text": "..."}
"""


# ---------- data context ----------

@dataclass
class SeriesSnapshot:
    series_id: str
    title: str
    units: str
    frequency: str
    current_value: float | None
    current_date: str | None
    # Full-history stats
    z_score: float | None
    percentile_rank: float | None
    # 5-year window stats
    z_score_5y: float | None
    percentile_5y: float | None
    # 10-year window stats
    z_score_10y: float | None
    percentile_10y: float | None
    history_start: str | None
    n_obs: int


# Approximate observations per year by FRED frequency_short.
_OBS_PER_YEAR: dict[str, int] = {
    "D": 252, "W": 52, "BW": 26, "M": 12, "Q": 4, "SA": 2, "A": 1,
}


def _lookback(freq: str, years: int) -> int:
    return _OBS_PER_YEAR.get(freq, 12) * years


def _float_or_none(x) -> float | None:
    try:
        v = float(x)
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return None


def _build_series_snapshots(series_ids: list[str]) -> dict[str, SeriesSnapshot]:
    """Build SeriesSnapshot objects for an explicit list of series IDs."""
    ctx: dict[str, SeriesSnapshot] = {}
    for sid in series_ids:
        try:
            s = load_series(sid)
            meta = series_metadata(sid)
            valid = s.dropna()
            if valid.empty:
                continue
            cur_val = float(valid.iloc[-1])
            cur_date = str(valid.index[-1].date())
            freq = str(meta.get("frequency_short", "M"))
            lb5 = _lookback(freq, 5)
            lb10 = _lookback(freq, 10)
            ctx[sid] = SeriesSnapshot(
                series_id=sid,
                title=str(meta.get("title", sid)),
                units=str(meta.get("units", "")),
                frequency=freq,
                current_value=cur_val,
                current_date=cur_date,
                z_score=_float_or_none(zscore_vs_history(s)),
                percentile_rank=_float_or_none(percentile_rank(s, cur_val)),
                z_score_5y=_float_or_none(zscore_vs_history(s, lookback=lb5)),
                percentile_5y=_float_or_none(percentile_rank(s, cur_val, lookback=lb5)),
                z_score_10y=_float_or_none(zscore_vs_history(s, lookback=lb10)),
                percentile_10y=_float_or_none(percentile_rank(s, cur_val, lookback=lb10)),
                history_start=str(s.first_valid_index().date()) if s.first_valid_index() is not None else None,
                n_obs=int(valid.shape[0]),
            )
        except (KeyError, IndexError, FileNotFoundError) as e:
            logger.warning("Could not build data context for %s: %s", sid, e)
    return ctx


def build_data_context(finding: Finding) -> dict[str, SeriesSnapshot]:
    """Load current value plus full-history and windowed z-scores / percentile ranks."""
    return _build_series_snapshots(list(finding.series_ids))


# Keyword → FRED series mapping for harvested findings with no tagged series.
_TOPIC_SERIES: dict[str, list[str]] = {
    "wage": ["AHETPI"], "wages": ["AHETPI"], "earning": ["AHETPI"], "earnings": ["AHETPI"],
    "labor": ["UNRATE", "AHETPI"], "employment": ["UNRATE", "PAYEMS"],
    "unemployment": ["UNRATE"], "payroll": ["PAYEMS"], "payrolls": ["PAYEMS"],
    "inflation": ["CPIAUCSL", "T10YIE"], "cpi": ["CPIAUCSL"], "pce": ["PCEPI"],
    "tariff": ["CPIAUCSL", "PCEPI"], "price": ["CPIAUCSL"],
    "expectation": ["MICH"], "expectations": ["MICH"],
    "yield": ["DGS10", "T10Y2Y"], "treasury": ["DGS10"], "rate": ["DFF", "DGS10"],
    "credit": ["BAA10Y"], "spread": ["BAA10Y", "T10Y2Y"],
    "vix": ["VIXCLS"], "volatility": ["VIXCLS"],
    "dollar": ["DTWEXBGS"], "fx": ["DTWEXBGS"],
    "oil": ["DCOILWTICO"], "crude": ["DCOILWTICO"], "energy": ["DCOILWTICO"],
    "gdp": ["GDPC1"], "growth": ["GDPC1", "INDPRO"],
    "housing": ["HOUST", "MORTGAGE30US"], "mortgage": ["MORTGAGE30US"],
    "breakeven": ["T10YIE", "DFII10"], "tips": ["DFII10"],
    "fomc": ["DFF", "DGS2"], "monetary": ["DFF"],
    "recession": ["T10Y2Y", "UNRATE"], "cycle": ["T10Y2Y", "UNRATE"],
    "repo": ["SOFR"], "sofr": ["SOFR"],
}


def _infer_context_series(finding: Finding, max_series: int = 3) -> list[str]:
    """For harvested findings with no tagged FRED series, infer relevant series from title/claim."""
    import re as _re
    text = f"{finding.title} {finding.claim}".lower()
    words = _re.findall(r"\b\w+\b", text)
    seen: set[str] = set()
    result: list[str] = []
    for word in words:
        for sid in _TOPIC_SERIES.get(word, []):
            if sid not in seen:
                seen.add(sid)
                result.append(sid)
            if len(result) >= max_series:
                return result
    return result


def _ctx_to_dict(ctx: dict[str, SeriesSnapshot]) -> dict:
    return {
        sid: {
            "title": snap.title,
            "units": snap.units,
            "frequency": snap.frequency,
            "current_value": snap.current_value,
            "current_date": snap.current_date,
            "z_score_full": snap.z_score,
            "percentile_full": snap.percentile_rank,
            "z_score_5y": snap.z_score_5y,
            "percentile_5y": snap.percentile_5y,
            "z_score_10y": snap.z_score_10y,
            "percentile_10y": snap.percentile_10y,
            "history_start": snap.history_start,
            "n_obs": snap.n_obs,
        }
        for sid, snap in ctx.items()
    }


# ---------- LLM calls ----------

def _extract_json(text: str) -> dict:
    """Parse JSON from LLM response, with repair fallback for common LLM formatting issues."""
    from json_repair import repair_json
    text = text.strip()
    # Strip ```json ... ``` fences if present
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = m.group(1) if m else None
    if candidate is None:
        brace = text.find("{")
        if brace != -1:
            candidate = text[brace:]
    if candidate is None:
        raise ValueError(f"No JSON found in LLM response: {text[:200]}")
    # Try strict parse first, fall back to repair
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = repair_json(candidate, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
        raise ValueError(f"JSON repair failed for LLM response: {text[:200]}")


def _load_concept_context(series_ids: list[str], keywords: list[str] | None = None) -> str:
    """Return the most relevant knowledge chunks for the given series (BM25-ranked).

    Chunks all files in knowledge/ by H2 section, ranks by BM25 + series-ID bonus,
    returns the top-4 sections as a single string.  Falls back to empty string if
    knowledge files are missing or rank_bm25 is unavailable.
    """
    try:
        from src.knowledge.retriever import retrieve_context
    except ImportError:
        logger.warning("knowledge.retriever not available — emails will have no concept context")
        return ""
    try:
        chunks = retrieve_context(
            series_ids=series_ids,
            keywords=keywords or [],
            top_n=4,
        )
    except Exception as exc:
        logger.warning("Concept context retrieval failed: %s", exc)
        return ""
    if not chunks:
        logger.warning("No concept context found for series %s — email lacks institutional background", series_ids)
        return ""
    return "\n\n".join(c.text for c in chunks)


def draft_email(
    pick: LessonPick,
    ctx: dict[str, SeriesSnapshot],
    inferred_ctx: "dict[str, SeriesSnapshot] | None" = None,
) -> dict:
    """Call Claude to produce subject + HTML/text body.

    Returns dict with keys: subject, body_html, body_text.
    """
    client = _client()
    finding = pick.finding

    user_content = (
        f"FINDING RECORD:\n{json.dumps(_finding_to_dict(finding), indent=2)}\n\n"
        f"DATA CONTEXT (query results — use these numbers, do not invent):\n"
        f"{json.dumps(_ctx_to_dict(ctx), indent=2)}\n\n"
    )
    if pick.release_context and pick.release_context.get("releases"):
        names = [r.get("release_name", "") for r in pick.release_context["releases"][:3]]
        user_content += (
            f"UPCOMING RELEASES IN NEXT 2 DAYS: {', '.join(names)}\n"
            "Use the most relevant release as the hook anchor. In the hook, briefly explain the "
            "mechanistic link between that release and today's finding — why does it update or "
            "test the thesis? Do not just name the release; explain the connection in one sentence.\n\n"
        )
    # Extract keywords from the finding title and claim for better BM25 retrieval.
    _kw_source = f"{finding.title} {finding.claim}"
    _kw_tokens = [w.lower() for w in _kw_source.split() if len(w) > 4]
    concept_ctx = _load_concept_context(list(finding.series_ids), keywords=_kw_tokens[:10])
    if concept_ctx:
        user_content += f"CONCEPT CONTEXT (background on the relevant series — use for historical regime context and institutional detail):\n{concept_ctx}\n\n"
    if inferred_ctx:
        user_content += (
            f"CONTEXTUAL BACKGROUND DATA (FRED series inferred from finding topic — not direct evidence from the finding, "
            f"but use their current readings to anchor 'Where We Are Now'. Cite values from here as background context, "
            f"not as the finding's core claim):\n{json.dumps(_ctx_to_dict(inferred_ctx), indent=2)}\n\n"
        )
    user_content += "Write the daily macro education email. Return only the JSON object."

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=[{"type": "text", "text": _DRAFT_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_content}],
    )
    raw = response.content[0].text
    return _extract_json(raw)


def fact_check_draft(
    draft: dict,
    pick: LessonPick,
    ctx: dict[str, SeriesSnapshot],
    inferred_ctx: "dict[str, SeriesSnapshot] | None" = None,
) -> dict:
    """Return {"flags": [...], "approved": bool}."""
    client = _client()

    merged_ctx = {**ctx, **(inferred_ctx or {})}
    user_content = (
        f"DRAFT (body text):\n{draft.get('body_text', '')}\n\n"
        f"FINDING RECORD:\n{json.dumps(_finding_to_dict(pick.finding), indent=2)}\n\n"
        f"DATA CONTEXT:\n{json.dumps(_ctx_to_dict(merged_ctx), indent=2)}\n\n"
        "Review the draft and return the JSON result."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=[{"type": "text", "text": _FACTCHECK_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_content}],
    )
    try:
        return _extract_json(response.content[0].text)
    except (ValueError, json.JSONDecodeError) as e:
        logger.warning("Fact-check parse error: %s", e)
        return {"flags": ["Fact-check response could not be parsed"], "approved": True}


# ---------- citation check (web-search pass) ----------

_KIND_GROUPS: dict[str, list[str]] = {
    "fomc_event_study": ["B"],
    "structural_break": ["C", "D", "E"],
    "correlation_shift": ["D", "G"],
    "cross_asset_factor": ["D", "N"],
    "bond_predictability": ["M", "C"],
    "ns_factor_extreme": ["C"],
    "recession_prediction": ["F", "G"],
    "notable_move_level": ["G", "D"],
    "harvested_source": [],
}


def _matching_library_papers(
    finding: "Finding",
    extra_series: "list[str] | None" = None,
    max_papers: int = 5,
) -> list:
    """Return paper library entries relevant to this finding.

    Matching priority: (1) FRED series overlap, (2) finding-kind → group mapping,
    (3) keyword search on title/claim.
    """
    from src.research.paper_library import papers_for_series, search_papers, filter_papers

    all_series = list(finding.series_ids or []) + list(extra_series or [])
    seen: set[str] = set()
    scored: list[tuple[int, object]] = []

    for sid in all_series:
        for paper in papers_for_series(sid):
            if paper.title not in seen:
                seen.add(paper.title)
                scored.append((paper.macro_relevance + paper.fred_implementability, paper))

    relevant_groups = _KIND_GROUPS.get(finding.kind, [])
    if relevant_groups:
        for paper in filter_papers(min_macro_relevance=4, min_fred_implementability=3):
            if paper.group in relevant_groups and paper.title not in seen:
                seen.add(paper.title)
                scored.append((paper.macro_relevance + paper.fred_implementability, paper))

    import re as _re
    text = f"{finding.title or ''} {finding.claim or ''}".lower()
    kw_hits = set(_re.findall(r'\b[a-z]{4,}\b', text)) & {
        "inflation", "yield", "curve", "credit", "recession", "dollar", "equity",
        "volatility", "spread", "fomc", "breakeven", "regime", "momentum", "factor",
        "risk", "commodity", "exchange", "currency", "housing", "wage", "labor",
    }
    for word in kw_hits:
        for paper in search_papers(word):
            if paper.title not in seen:
                seen.add(paper.title)
                scored.append((paper.macro_relevance + paper.fred_implementability - 1, paper))

    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:max_papers]]


def _web_search(query: str, max_results: int = 5) -> str:
    from ddgs import DDGS
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}\n"
                )
    except Exception as exc:
        return f"Search failed: {exc}"
    return "\n---\n".join(results) if results else "No results found."


def _web_fetch(url: str, max_chars: int = 6000) -> str:
    import requests
    from bs4 import BeautifulSoup
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (research bot)"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        lines = [ln for ln in soup.get_text(separator="\n", strip=True).splitlines() if ln.strip()]
        return "\n".join(lines)[:max_chars]
    except Exception as exc:
        return f"Fetch failed: {exc}"


def citation_check_draft(
    draft: dict,
    pick: LessonPick,
    inferred_ctx: "dict[str, SeriesSnapshot] | None" = None,
) -> dict:
    """Agentic web-search pass: find primary sources for mechanistic claims.

    Returns {"citations": [...], "unsourced": [...]}.
    """
    client = _client()
    finding = pick.finding

    extra_series = list(inferred_ctx.keys()) if inferred_ctx else None
    library_papers = _matching_library_papers(finding, extra_series=extra_series)
    if library_papers:
        library_lines = [
            "LIBRARY CANDIDATES (from curated catalog — search for these specific papers "
            "first to find their canonical URLs and verify they address the claims, before "
            "doing free-form web searches):"
        ]
        for i, p in enumerate(library_papers, 1):
            blurb = p.notes[:120].rstrip() if p.notes else p.methodology[:120].rstrip()
            library_lines.append(f"  {i}. \"{p.title}\" — {blurb}")
        library_section = "\n".join(library_lines) + "\n\n"
        logger.info("Citation pass: %d library candidates for %s", len(library_papers), finding.slug)
    else:
        library_section = ""

    user_content = (
        f"EMAIL DRAFT (body text — identify mechanistic/interpretive claims to source):\n"
        f"{draft.get('body_text', '')}\n\n"
        f"FINDING RECORD (already-sourced claims are in finding.sources — do not re-source these):\n"
        f"{json.dumps(_finding_to_dict(finding), indent=2)}\n\n"
        f"{library_section}"
        "Identify up to 3 specific mechanistic or interpretive claims in the draft that lack "
        "primary source support. Search for the library candidates above (or do free-form web "
        "searches if none match) to find academic papers or institutional research notes that "
        "support each claim. Fetch the most promising URL to verify relevance. "
        "Return JSON."
    )

    tools = [
        {
            "name": "web_search",
            "description": "Search the web for academic or institutional sources.",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "web_fetch",
            "description": "Fetch and read a web page to verify source relevance.",
            "input_schema": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    ]

    messages: list[dict] = [{"role": "user", "content": user_content}]

    _CITATION_TIMEOUT_S = 90
    _start = time.monotonic()

    for _ in range(10):
        if time.monotonic() - _start > _CITATION_TIMEOUT_S:
            logger.warning("Citation check timed out after %ds — skipping citations", _CITATION_TIMEOUT_S)
            break
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=[{"type": "text", "text": _CITATION_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    try:
                        result = _extract_json(block.text)
                        n = len(result.get("citations", []))
                        logger.info("Citation check: %d citations found, %d unsourced",
                                    n, len(result.get("unsourced", [])))
                        return result
                    except (ValueError, json.JSONDecodeError) as e:
                        logger.warning("Citation check parse error: %s", e)
            break

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                try:
                    if block.name == "web_search":
                        content = _web_search(block.input.get("query", ""))
                    elif block.name == "web_fetch":
                        content = _web_fetch(block.input.get("url", ""))
                    else:
                        content = f"Unknown tool: {block.name}"
                except Exception as exc:
                    content = f"Tool error: {exc}"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return {"citations": [], "unsourced": []}


def revise_with_citations(draft: dict, citation_result: dict) -> dict:
    """Insert inline citation superscripts and append a References section."""
    citations = citation_result.get("citations", [])
    if not citations:
        return draft

    client = _client()
    user_content = (
        f"BODY HTML:\n{draft['body_html']}\n\n"
        f"BODY TEXT:\n{draft.get('body_text', '')}\n\n"
        f"CITATIONS:\n{json.dumps(citations, indent=2)}\n\n"
        "Add inline citation markers and append the References section. Return JSON."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=[{"type": "text", "text": _REVISION_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_content}],
    )
    try:
        revised = _extract_json(response.content[0].text)
        return {**draft, **revised}
    except (ValueError, json.JSONDecodeError) as e:
        logger.warning("Citation revision parse error: %s — returning un-cited draft", e)
        return draft


# ---------- equation generation ----------

# Latex strings use matplotlib mathtext notation (no \text{}, use \mathrm{}).
# Keys are matched against finding.slug (substring) then finding.kind.
_EQUATION_LATEX: dict[str, str] = {
    # FOMC OLS regression: DGS10 ~ timing + path, per era
    "fomc_dgs10_timing_path_ols": (
        r"\Delta y_{10y,t} = \alpha"
        r" + \beta_1 \cdot \mathrm{timing}_t"
        r" + \beta_2 \cdot \mathrm{path}_t"
        r" + \varepsilon_t"
    ),
    # Cochrane-Piazzesi bond excess return regression
    "cp_factor_signal": (
        r"rx_{t \to t+1}^{(n)} = \alpha + \beta \cdot \mathrm{CP}_t + \varepsilon_{t+1}"
    ),
    # Fama-MacBeth second-pass cross-sectional regression
    "cross_asset_factor": (
        r"\bar{R}_i = \lambda_0 + \sum_{k=1}^{K} \lambda_k \hat{\beta}_{ik} + \varepsilon_i"
    ),
}


def _equation_latex(finding: "Finding") -> str | None:
    """Return the mathtext LaTeX string for this finding's regression, or None."""
    for key, latex in _EQUATION_LATEX.items():
        if key in finding.slug or key == finding.kind:
            return latex
    return None


def generate_equation_image(finding: "Finding", chart_dir: Path, today: date) -> Path | None:
    """Render the regression equation to PNG and save. Returns path or None."""
    latex = _equation_latex(finding)
    if not latex:
        return None
    from src.analytics.charts import render_equation_image
    png_bytes = render_equation_image(latex)
    slug_short = finding.slug[:60]
    out = chart_dir / f"{today.isoformat()}_{slug_short}_equation.png"
    out.write_bytes(png_bytes)
    logger.info("Equation image saved: %s", out)
    return out


# ---------- chart generation ----------

_CHART_KINDS = {"correlation_shift", "regime_transition", "notable_move_level",
                "notable_move_change", "lead_lag_change", "cointegration_break",
                "harvested_source", "fomc_event_study", "structural_break",
                "cp_factor_signal", "ns_factor_extreme", "btp_bund_regime",
                "spread_extreme", "decomposition_shift", "cross_asset_factor"}


def _chart1_primary_series(
    finding: "Finding", chart_dir: Path, slug_short: str, today: date
) -> "Path | None":
    """Full time-series of the primary series with NBER shading and trigger marker."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from src.analytics.charts import (
        time_series, time_series_zoom, save_to, add_source_footer, _span_years, _ZOOM_MIN_SPAN_YEARS,
    )
    sid1 = list(finding.series_ids)[0]
    try:
        s = load_series(sid1)
        meta = series_metadata(sid1)
        span = _span_years(s.dropna().index) if len(s.dropna()) > 1 else 0
        if span >= _ZOOM_MIN_SPAN_YEARS:
            fig, (ax, _) = time_series_zoom(
                s, title=meta.get("title", sid1), ylabel=str(meta.get("units", "")),
                shade_nber=True, zoom_years=5,
            )
        else:
            fig, ax = time_series(
                s, title=meta.get("title", sid1), ylabel=str(meta.get("units", "")),
                shade_nber=True, annotate_last=True,
            )
        if finding.discovered:
            trigger_num = mdates.date2num(finding.discovered)
            xlim = ax.get_xlim()
            if xlim[0] <= trigger_num <= xlim[1]:
                ax.axvline(trigger_num, color="#D55E00", linewidth=1, linestyle="--", alpha=0.7)
        add_source_footer(fig, [sid1], today)
        out = chart_dir / f"{today.isoformat()}_{slug_short}_chart1.png"
        save_to(fig, out)
        plt.close(fig)
        logger.info("Chart 1 saved: %s", out)
        return out
    except Exception as e:
        logger.warning("Chart 1 (time series) failed: %s", e)
        return None


def _chart2_rolling_corr(
    finding: "Finding", chart_dir: Path, slug_short: str, today: date
) -> "Path | None":
    """Rolling Spearman correlation between two series over time."""
    import matplotlib.pyplot as plt
    from src.analytics.charts import time_series, save_to, add_source_footer
    from src.analytics.data import load_aligned
    from src.analytics.stats import rolling_corr
    sids = list(finding.series_ids[:2])
    try:
        df = load_aligned(sids)
        freq = series_metadata(sids[0]).get("frequency_short", "M")
        window = {"D": 126, "W": 52, "M": 48, "Q": 12, "A": 5}.get(freq, 48)
        total_obs = len(df.dropna())
        if total_obs < window * 4:
            raise ValueError(f"Too few observations ({total_obs}) for window {window}")
        corr = rolling_corr(df.iloc[:, 0], df.iloc[:, 1], window=window).dropna() * 100
        if corr.empty:
            raise ValueError("Rolling correlation is empty after dropna")
        m0 = series_metadata(sids[0]).get("title", sids[0])
        m1 = series_metadata(sids[1]).get("title", sids[1])
        fig, ax = time_series(
            corr,
            title=f"Rolling {window}-Obs Spearman Correlation",
            subtitle=f"{m0} vs {m1}",
            ylabel="Spearman rank correlation (×100)",
            shade_nber=True, annotate_last=True,
        )
        ax.axhline(0, color="#6B7280", linewidth=0.8, linestyle="--")
        add_source_footer(fig, sids, today)
        out = chart_dir / f"{today.isoformat()}_{slug_short}_chart2.png"
        save_to(fig, out)
        plt.close(fig)
        logger.info("Chart 2 (rolling corr) saved: %s", out)
        return out
    except Exception as e:
        logger.warning("Chart 2 (rolling corr) skipped: %s", e)
        return None


def _chart2_rolling_percentile(
    finding: "Finding", chart_dir: Path, slug_short: str, today: date
) -> "Path | None":
    """Rolling percentile rank of the primary series over time."""
    import matplotlib.pyplot as plt
    from src.analytics.charts import rolling_percentile, save_to, add_source_footer
    sid1 = list(finding.series_ids)[0]
    try:
        s = load_series(sid1)
        meta = series_metadata(sid1)
        freq = meta.get("frequency_short", "D")
        obs_per_year = {"D": 252, "W": 52, "M": 12, "Q": 4, "A": 1}.get(freq, 252)
        window = obs_per_year * 2
        fig, _ = rolling_percentile(
            s,
            window=window,
            title=f"{meta.get('title', sid1)} — Rolling Percentile Rank",
            subtitle="2-year rolling window; bands at 33rd and 67th %ile",
            shade_nber=True,
        )
        add_source_footer(fig, [sid1], today)
        out = chart_dir / f"{today.isoformat()}_{slug_short}_chart2.png"
        save_to(fig, out)
        plt.close(fig)
        logger.info("Chart 2 (rolling percentile) saved: %s", out)
        return out
    except Exception as e:
        logger.warning("Chart 2 (rolling percentile) skipped: %s", e)
        return None


def _chart3_split_correlogram(
    finding: "Finding", chart_dir: Path, slug_short: str, today: date
) -> "Path | None":
    """Side-by-side cross-correlogram: historical vs. recent."""
    import matplotlib.pyplot as plt
    from src.analytics.charts import lead_lag_bar_split, save_to, add_source_footer
    from src.analytics.data import load_aligned
    from src.analytics.stats import lead_lag_xcorr
    sids = list(finding.series_ids[:2])
    try:
        df = load_aligned(sids)
        ev = finding.evidence or {}
        n_hist = int(ev.get("n_hist", max(len(df) - 24, 10)))
        n_recent = int(ev.get("n_recent", min(24, len(df) - n_hist)))
        max_lag = 12
        if n_recent < max_lag + 3 or n_hist < max_lag * 2:
            raise ValueError(f"Insufficient obs for split correlogram: n_hist={n_hist}, n_recent={n_recent}")
        xcorr_hist = lead_lag_xcorr(df.iloc[:n_hist, 0], df.iloc[:n_hist, 1], max_lag=max_lag)
        xcorr_recent = lead_lag_xcorr(df.iloc[n_hist:, 0], df.iloc[n_hist:, 1], max_lag=max_lag)
        fig, _ = lead_lag_bar_split(
            xcorr_hist, xcorr_recent,
            title=f"Cross-Correlogram: {sids[0]} vs {sids[1]}",
            subtitle="How the lead-lag structure has changed",
            hist_label=f"Historical (n={n_hist})",
            recent_label=f"Recent (n={n_recent})",
        )
        add_source_footer(fig, sids, today)
        out = chart_dir / f"{today.isoformat()}_{slug_short}_chart3.png"
        save_to(fig, out)
        plt.close(fig)
        logger.info("Chart 3 (split correlogram) saved: %s", out)
        return out
    except Exception as e:
        logger.warning("Chart 3 (split correlogram) skipped: %s", e)
        return None


def _chart3_distribution(
    finding: "Finding", ctx: "dict[str, SeriesSnapshot]", chart_dir: Path, slug_short: str, today: date
) -> "Path | None":
    """Full-history distribution histogram with current value marked."""
    import matplotlib.pyplot as plt
    from src.analytics.charts import distribution, save_to, add_source_footer
    sid1 = list(finding.series_ids)[0]
    if sid1 not in ctx:
        return None
    snap = ctx[sid1]
    if snap.current_value is None:
        return None
    try:
        s = load_series(sid1)
        meta = series_metadata(sid1)
        pct_label = f"{snap.percentile_rank * 100:.0f}th %ile" if snap.percentile_rank is not None else ""
        fig, _ = distribution(
            s,
            current_value=snap.current_value,
            title=f"{meta.get('title', sid1)} — Full History Distribution",
            subtitle=f"Current: {snap.current_value:.2f}{' — ' + pct_label if pct_label else ''}",
            xlabel=str(meta.get("units", "")),
        )
        add_source_footer(fig, [sid1], today)
        out = chart_dir / f"{today.isoformat()}_{slug_short}_chart3.png"
        save_to(fig, out)
        plt.close(fig)
        logger.info("Chart 3 (distribution) saved: %s", out)
        return out
    except Exception as e:
        logger.warning("Chart 3 (distribution) failed: %s", e)
        return None


def _chart1_multi_series_harvested(
    finding: "Finding", chart_dir: Path, slug_short: str, today: date
) -> "Path | None":
    """Multi-series chart for harvested_source findings (replaces single-series chart 1)."""
    import matplotlib.pyplot as plt
    from src.analytics.charts import multi_series, multi_series_zoom, save_to, add_source_footer, _span_years, _ZOOM_MIN_SPAN_YEARS
    from src.analytics.data import load_aligned
    sids_all = list(finding.series_ids)
    try:
        df_multi = load_aligned(sids_all).dropna(how="all")
        if df_multi.empty or len(df_multi.columns) < 2:
            raise ValueError("Insufficient aligned data for multi-series chart")
        units_set = {series_metadata(sid).get("units", "") for sid in df_multi.columns}
        normalize = "none" if len(units_set) == 1 else "zscore"
        legend_labels = [series_metadata(sid).get("title", sid) for sid in df_multi.columns]
        primary_title = series_metadata(sids_all[0]).get("title", sids_all[0])
        units_label = list(units_set)[0] if len(units_set) == 1 else "z-score"
        span = _span_years(df_multi.dropna(how="all").index)
        if span >= _ZOOM_MIN_SPAN_YEARS:
            fig, _ = multi_series_zoom(
                df_multi, title=f"{primary_title} and Related Series", ylabel=units_label,
                normalize=normalize, legend_labels=legend_labels, shade_nber=True,
            )
        else:
            fig, _ = multi_series(
                df_multi, title=f"{primary_title} and Related Series", ylabel=units_label,
                normalize=normalize, legend_labels=legend_labels, shade_nber=True,
            )
        add_source_footer(fig, list(df_multi.columns), today)
        out = chart_dir / f"{today.isoformat()}_{slug_short}_chart1m.png"
        save_to(fig, out)
        plt.close(fig)
        logger.info("Chart 1 (multi-series harvested) saved: %s", out)
        return out
    except Exception as e:
        logger.warning("Chart 1 multi-series (harvested) skipped: %s", e)
        return None


def _chart2_fomc_era_comparison(
    finding: "Finding", chart_dir: Path, slug_short: str, today: date
) -> "Path | None":
    """Era comparison bar chart for fomc_event_study findings."""
    import matplotlib.pyplot as plt
    from src.analytics.charts import era_comparison_bar, save_to, add_source_footer
    ev = finding.evidence or {}
    eras = ev.get("eras") if isinstance(ev, dict) else None
    if not eras or len(eras) < 2:
        return None
    try:
        first = eras[0]
        if "path_share" in first:
            for e in eras:
                e["timing_share"] = 1.0 - float(e["path_share"])
            fig, _ = era_comparison_bar(
                eras, metric_a="timing_share", metric_b="path_share",
                label_a="Timing surprise share", label_b="Path surprise share",
                title="FOMC Announcement: Path vs Timing Surprise Share by Era",
                subtitle="Share of total 2-year yield move attributable to each component",
                ylabel="Share of total move (%)", stacked=True, pct_scale=True,
            )
        elif "beta_path" in first:
            fig, _ = era_comparison_bar(
                eras, metric_a="beta_timing", metric_b="beta_path",
                label_a="β timing", label_b="β path",
                title="FOMC Day: 10-Year Yield Sensitivity to Surprise Components by Era",
                subtitle="OLS coefficients — 10y change on timing and path surprises",
                ylabel="OLS coefficient", stacked=False, pct_scale=False,
            )
        else:
            raise ValueError("No known metric found in era evidence")
        add_source_footer(fig, list(finding.series_ids), today)
        out = chart_dir / f"{today.isoformat()}_{slug_short}_chart2.png"
        save_to(fig, out)
        plt.close(fig)
        logger.info("Chart 2 (era comparison fomc) saved: %s", out)
        return out
    except Exception as e:
        logger.warning("Chart 2 (era comparison fomc) skipped: %s", e)
        return None


def _chart1_context_series(
    series_ids: list[str], chart_dir: Path, slug_short: str, today: date
) -> "Path | None":
    """Single contextual time-series chart for harvested findings with no tagged series."""
    import matplotlib.pyplot as plt
    from src.analytics.charts import time_series, time_series_zoom, save_to, add_source_footer, _span_years, _ZOOM_MIN_SPAN_YEARS
    if not series_ids:
        return None
    sid = series_ids[0]
    try:
        s = load_series(sid)
        meta = series_metadata(sid)
        span = _span_years(s.dropna().index) if len(s.dropna()) > 1 else 0
        if span >= _ZOOM_MIN_SPAN_YEARS:
            fig, _ = time_series_zoom(
                s, title=meta.get("title", sid), ylabel=str(meta.get("units", "")),
                shade_nber=True, zoom_years=5,
            )
        else:
            fig, _ = time_series(
                s, title=meta.get("title", sid), ylabel=str(meta.get("units", "")),
                shade_nber=True, annotate_last=True,
            )
        add_source_footer(fig, [sid], today)
        out = chart_dir / f"{today.isoformat()}_{slug_short}_chart1_ctx.png"
        save_to(fig, out)
        plt.close(fig)
        logger.info("Chart 1 (context) saved: %s", out)
        return out
    except Exception as e:
        logger.warning("Chart 1 (context) failed for %s: %s", sid, e)
        return None


def generate_charts(
    pick: LessonPick,
    ctx: dict[str, SeriesSnapshot],
    today: date,
    inferred_ctx: "dict[str, SeriesSnapshot] | None" = None,
) -> list[Path]:
    """Generate up to 3 charts for the finding. Returns list of saved PNG paths."""
    finding = pick.finding
    if finding.kind not in _CHART_KINDS:
        return []

    # Harvested findings with no tagged series: fall back to a single contextual chart.
    if not finding.series_ids:
        if finding.kind == "harvested_source" and inferred_ctx:
            chart_dir = CHARTS_DIR / today.isoformat()
            chart_dir.mkdir(parents=True, exist_ok=True)
            slug_short = finding.slug[:40].replace("/", "-")
            p = _chart1_context_series(list(inferred_ctx.keys()), chart_dir, slug_short, today)
            return [p] if p else []
        return []

    chart_dir = CHARTS_DIR / today.isoformat()
    chart_dir.mkdir(parents=True, exist_ok=True)
    slug_short = finding.slug[:40].replace("/", "-")
    sid1 = list(finding.series_ids)[0]
    paths: list[Path] = []

    # Chart 1: always primary time-series
    p = _chart1_primary_series(finding, chart_dir, slug_short, today)
    if p:
        paths.append(p)

    # Chart 2: kind-specific
    if finding.kind in {"correlation_shift", "lead_lag_change"} and len(finding.series_ids) >= 2:
        p = _chart2_rolling_corr(finding, chart_dir, slug_short, today)
        if p: paths.append(p)
    elif finding.kind in {"notable_move_level", "notable_move_change", "regime_transition"}:
        p = _chart2_rolling_percentile(finding, chart_dir, slug_short, today)
        if p: paths.append(p)

    # Chart 3: kind-specific
    if finding.kind == "lead_lag_change" and len(finding.series_ids) >= 2:
        p = _chart3_split_correlogram(finding, chart_dir, slug_short, today)
        if p: paths.append(p)
    elif finding.kind in {"notable_move_level", "notable_move_change", "regime_transition", "structural_break"}:
        p = _chart3_distribution(finding, ctx, chart_dir, slug_short, today)
        if p: paths.append(p)

    # harvested_source: upgrade chart 1 to multi-series, then fill chart 2 + 3
    if finding.kind == "harvested_source" and len(finding.series_ids) >= 2:
        p = _chart1_multi_series_harvested(finding, chart_dir, slug_short, today)
        if p:
            if paths:
                paths[0] = p
            else:
                paths.append(p)

    if finding.kind in {"harvested_source", "structural_break"}:
        if len(paths) < 2:
            p = _chart2_rolling_percentile(finding, chart_dir, slug_short, today)
            if p: paths.append(p)
        if len(paths) < 3:
            p = _chart3_distribution(finding, ctx, chart_dir, slug_short, today)
            if p: paths.append(p)

    # fomc_event_study: era comparison bar as chart 2
    if finding.kind == "fomc_event_study":
        p = _chart2_fomc_era_comparison(finding, chart_dir, slug_short, today)
        if p: paths.append(p)

    return paths


# ---------- HTML rendering ----------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{subject}</title>
</head>
<style>
  h3 {{ font-family: Helvetica,Arial,sans-serif; font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:1px; color:#444; margin:28px 0 6px 0; padding-bottom:4px; border-bottom:1px solid #e5e7eb; }}
  table {{ border-collapse:collapse; width:100%; margin:16px 0; font-family:Helvetica,Arial,sans-serif; font-size:14px; }}
  th {{ background:#1a1a1a; color:#fff; padding:8px 12px; text-align:left; font-weight:600; font-size:12px; letter-spacing:0.5px; }}
  td {{ padding:7px 12px; border-bottom:1px solid #e5e7eb; vertical-align:top; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:nth-child(even) td {{ background:#f9fafb; }}
  td:not(:first-child), th:not(:first-child) {{ text-align:right; }}
  pre {{ font-family:"Courier New",Courier,monospace; font-size:13px; background:#f3f4f6; border:1px solid #e5e7eb; border-radius:4px; padding:12px 14px; overflow-x:auto; line-height:1.5; white-space:pre; }}
</style>
<body style="font-family: Georgia,'Times New Roman',serif; max-width:640px; margin:0 auto; padding:24px 16px; color:#1a1a1a; line-height:1.7; font-size:16px;">
  <div style="border-bottom:2px solid #1a1a1a; padding-bottom:8px; margin-bottom:24px;">
    <span style="font-family:Helvetica,Arial,sans-serif; font-size:11px; color:#666; text-transform:uppercase; letter-spacing:1.5px;">MACRO EDUCATION — {date_long}</span>
  </div>
  {body_html}
  <div style="border-top:1px solid #ddd; margin-top:32px; padding-top:12px; font-family:Helvetica,Arial,sans-serif; font-size:11px; color:#888;">
    Data: FRED, Federal Reserve Bank of St. Louis. Computed {date_iso}.
  </div>
</body>
</html>
"""

_CHART_DIV = '<div style="margin:24px 0;"><img src="{src}" style="max-width:100%; height:auto;" alt="Chart {n}"></div>'
_EQUATION_DIV = '<div style="margin:16px auto; text-align:center;"><img src="{src}" style="max-height:48px; height:auto;" alt="Regression equation"></div>'


def render_html(
    subject: str,
    body_html: str,
    today: date,
    chart_paths: list[Path] | None = None,
    chart_cids: list[str] | None = None,
    equation_path: Path | None = None,
    equation_cid: str | None = None,
) -> str:
    """Render the email HTML, substituting {{CHART_N}} placeholders.

    For dry-run: pass `chart_paths` (embedded as base64 data URIs).
    For live email: pass `chart_cids` (MIME CID references).
    Any unmatched placeholders are stripped.
    Prepends a market snapshot table above the body content.
    """
    import base64

    chart_paths = chart_paths or []
    chart_cids = chart_cids or []

    # Replace placeholders with inline images (1-indexed: {{CHART_1}}, {{CHART_2}})
    for i, path in enumerate(chart_paths, 1):
        if path and path.exists():
            data = base64.b64encode(path.read_bytes()).decode()
            src = f"data:image/png;base64,{data}"
            body_html = body_html.replace(
                f"{{{{CHART_{i}}}}}",
                _CHART_DIV.format(src=src, n=i),
            )
    for i, cid in enumerate(chart_cids, 1):
        body_html = body_html.replace(
            f"{{{{CHART_{i}}}}}",
            _CHART_DIV.format(src=f"cid:{cid}", n=i),
        )

    # Strip any remaining unreplaced chart placeholders
    body_html = re.sub(r"\{\{CHART_\d+\}\}", "", body_html)

    # Replace {{EQUATION}} placeholder
    if equation_cid:
        body_html = body_html.replace(
            "{{EQUATION}}",
            _EQUATION_DIV.format(src=f"cid:{equation_cid}"),
        )
    elif equation_path and equation_path.exists():
        data = base64.b64encode(equation_path.read_bytes()).decode()
        body_html = body_html.replace(
            "{{EQUATION}}",
            _EQUATION_DIV.format(src=f"data:image/png;base64,{data}"),
        )
    body_html = body_html.replace("{{EQUATION}}", "")  # strip if unused

    # Append market snapshot + release calendar below the body
    tables_html = ""
    try:
        snapshot_rows = build_market_snapshot()
        tables_html += render_snapshot_table(snapshot_rows, today)
    except Exception as e:
        logger.warning("Market snapshot table failed: %s", e)
    try:
        tables_html += build_release_calendar_table(today, days_ahead=10)
    except Exception as e:
        logger.warning("Release calendar table failed: %s", e)

    return _HTML_TEMPLATE.format(
        subject=subject,
        date_long=today.strftime("%B %d, %Y"),
        date_iso=today.isoformat(),
        body_html=body_html + tables_html,
    )


# ---------- composed result ----------

@dataclass
class ComposedEmail:
    subject: str
    html_body: str          # Final HTML with CID references (for live send)
    html_body_template: str # Body HTML with {{CHART_N}}/{{EQUATION}} placeholders (for dry-run re-render)
    text_body: str
    chart_paths: list[Path]
    equation_path: Path | None
    fact_check_flags: list[str]
    approved: bool
    data_context: dict
    citation_count: int = 0


# ---------- main entry point ----------

def compose_email(pick: LessonPick, today: date | None = None) -> ComposedEmail:
    """Build the email for today's lesson pick.

    Raises RuntimeError if the Anthropic key is missing.
    """
    today = today or date.today()

    ctx = build_data_context(pick.finding)
    if not ctx:
        logger.warning("No data context built for finding %s — series may not be tracked.", pick.finding.slug)

    # For harvested findings with no tagged series, infer contextually relevant FRED series.
    inferred_ctx: dict[str, SeriesSnapshot] = {}
    if not pick.finding.series_ids and pick.finding.kind == "harvested_source":
        inferred_ids = _infer_context_series(pick.finding)
        if inferred_ids:
            logger.info("Harvested finding has no series; inferring context from: %s", inferred_ids)
            inferred_ctx = _build_series_snapshots(inferred_ids)

    draft = draft_email(pick, ctx, inferred_ctx=inferred_ctx)
    fc = fact_check_draft(draft, pick, ctx, inferred_ctx=inferred_ctx)

    flags = fc.get("flags", [])
    approved = fc.get("approved", True)

    if flags:
        logger.info("Fact-check flags (%d): %s", len(flags), "; ".join(flags))

    # Web-search citation pass: find primary sources for mechanistic claims
    citation_result = citation_check_draft(draft, pick, inferred_ctx=inferred_ctx)
    citation_count = len(citation_result.get("citations", []))
    if citation_count:
        draft = revise_with_citations(draft, citation_result)
        logger.info("Citations added: %d", citation_count)

    chart_paths = generate_charts(pick, ctx, today, inferred_ctx=inferred_ctx)
    chart_dir = CHARTS_DIR / today.isoformat()
    chart_dir.mkdir(parents=True, exist_ok=True)
    equation_path = generate_equation_image(pick.finding, chart_dir, today)

    cids = [f"chart_{i}" for i in range(len(chart_paths))]
    eq_cid = "equation_0" if equation_path else None
    template_body = draft["body_html"]  # Preserves {{CHART_N}}/{{EQUATION}} placeholders
    html_body = render_html(
        draft["subject"], template_body, today,
        chart_cids=cids, equation_cid=eq_cid,
    )

    return ComposedEmail(
        subject=draft["subject"],
        html_body=html_body,
        html_body_template=template_body,
        text_body=draft.get("body_text", ""),
        chart_paths=chart_paths,
        equation_path=equation_path,
        fact_check_flags=flags,
        approved=approved,
        citation_count=citation_count,
        data_context=_ctx_to_dict(ctx),
    )


# ---------- helpers ----------

_CORR_EVIDENCE_KEYS = {
    "hist_peak_correlation", "recent_peak_correlation", "correlation_shift",
    "baseline_correlation", "recent_correlation",
}


def _scale_evidence_correlations(evidence: dict) -> dict:
    """Scale raw [-1, 1] correlation values to [-100, 100] for display."""
    return {
        k: round(v * 100, 1) if k in _CORR_EVIDENCE_KEYS and isinstance(v, float) else v
        for k, v in evidence.items()
    }


def _finding_to_dict(f: Finding) -> dict:
    evidence = _scale_evidence_correlations(f.evidence) if isinstance(f.evidence, dict) else f.evidence
    d = {
        "slug": f.slug,
        "title": f.title,
        "kind": f.kind,
        "discovered": f.discovered.isoformat(),
        "series_ids": list(f.series_ids),
        "window": f.window,
        "claim": f.claim,
        "evidence": evidence,
        "interpretation": f.interpretation,
        "sources": f.sources,
        "status": f.status,
    }
    latex = _equation_latex(f)
    if latex:
        d["equation"] = latex  # shown to LLM so it knows {{EQUATION}} is available
    return d
