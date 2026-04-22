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
    ("__section__", "EUROPEAN RATES"),
    ("IRLTLT01DEM156N", "Germany 10Y",  "rate_pct"),
    ("IRLTLT01ITM156N", "Italy 10Y",    "rate_pct"),
    ("IRLTLT01FRM156N", "France 10Y",   "rate_pct"),
    ("IRLTLT01GBM156N", "UK 10Y",       "rate_pct"),
    ("IRLTLT01EZM156N", "Euro Area 10Y","rate_pct"),
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
    "IG/HY = investment/high-yield grade · DFR = deposit facility rate · "
    "USD broad = trade-weighted dollar"
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
        # Cap FX changes at 3dp to keep columns narrow
        return f"{sign}{abs(chg):.3f}" if chg < 0 else f"+{chg:.3f}"
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
2. <h3>The Concept</h3> (2–4 short paragraphs): explain what the metric is, how it is constructed, what drives it, and why the reader should care — woven together naturally, not as a list. Start with the intuitive plain-English version ("think of it as X"), add the technical detail, name any direct equities analogue, and in the same flow explain what information it carries that equity prices alone do not. By the end of this section the reader understands both what this thing is and why it matters.
3. <h3>The Analysis</h3> (1–2 paragraphs): the specific empirical work — what the data actually shows, with numbers, windows, and citation. This is the research behind today's email. Place {{CHART_1}} immediately after this section.
4. <h3>Where We Are Now</h3> (1 paragraph): current reading vs history. Where in the distribution? How does today compare to prior episodes? Place {{CHART_2}} immediately after. Place {{CHART_3}} immediately after {{CHART_2}} if a third chart exists.
5. <h3>What to Watch</h3> (1–2 sentences): the specific release, data point, or signal that would update the picture. Dated and specific — no generic closers.
6. Terms & Mechanics (optional, 2–3 items): cover only macro-specific institutional mechanics or jargon the reader likely doesn't know cold — data construction nuances (e.g. birth-death model, seasonal adjustment, chained dollars), Fed operational plumbing (IORB, ON RRP, SRF, reserve requirements), or release methodology quirks. Skip financial basics the reader already knows: basis points, yield, standard deviation, what a spread is, price-return relationships. Format: a <ul> with one <li> per term — <strong>Term</strong>: one-sentence definition. No Q: prefix. These may appear inline in the most relevant section rather than grouped at the end. If no genuinely non-obvious mechanics appear in today's email, omit this section entirely. Do NOT use <h3>Terms & Mechanics</h3> as a section header.

CHART PLACEHOLDERS:
- Emit {{CHART_1}}, {{CHART_2}}, and optionally {{CHART_3}} as standalone lines in body_html at the specified positions above.
- {{CHART_1}}: the primary time-series chart (full history with NBER recession shading).
- {{CHART_2}}: rolling percentile rank over time (for regime/notable-move findings) or rolling correlation (for correlation/lead-lag findings).
- {{CHART_3}}: distribution histogram (for regime/notable-move) or split cross-correlogram comparing historical vs. recent lead-lag structure (for lead-lag findings). Emit only if there is genuinely a third chart — do not force it.
- Place each placeholder immediately after the paragraph it illustrates. Never place two placeholders in the same paragraph block.
- Do not write "the chart above/below" — let the chart speak.

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
- Do NOT include FRED series IDs in prose (e.g. "VIXCLS", "DGS10", "UNRATE"). They are internal identifiers. Refer to series by their full name only.
- Use '%ile' not 'percentile': e.g., '52nd %ile' not '52nd percentile'.
- In tables, use short lookback labels: 'Full', '5Y', '10Y' — not '5-Year', 'Full History', etc.
- All standard-deviation citations: 1 decimal, bold: '<strong>−0.2σ</strong>'. Never '−0.20σ'.
- Bold ALL key statistics: wrap metric values, z-scores, percentile ranks, correlations, and rate/spread changes in <strong> tags. E.g. <strong>17.94</strong>, <strong>52nd %ile</strong>, <strong>−0.2σ</strong>, <strong>+25 bp</strong>, <strong>−60 → +30</strong>.
- Correlations are expressed on a −100 to +100 scale (already multiplied by 100 in DATA CONTEXT). Always append a % sign: write e.g. <strong>−53%</strong> not <strong>−0.53</strong> or <strong>−53</strong>.
- Never combine a %ile citation with a regime label for the same reading — the number conveys the context.
- Break any sentence over 25 words.
- Bullets are allowed anywhere they aid clarity. Required for any Q&A mechanics callouts.

FACTUAL ACCURACY RULES:
- Use ONLY the numbers from the DATA CONTEXT provided. Do not invent or estimate numbers.
- Every value in the email traces to a FRED series in the data context.
- Historical claims must come from the finding's Sources list or CONCEPT CONTEXT, or be explicitly framed as "commonly argued."
- When data is ambiguous, surface the uncertainty — do not hide it.

OUTPUT FORMAT:
Return a valid JSON object with exactly three keys: subject, body_html, body_text.
CRITICAL JSON RULES — violations will break parsing:
- No markdown code fences. Start your response with { and end with }.
- No literal newlines inside string values. Use the two-character sequence \n instead.
- No unescaped double-quotes inside string values. Use \" instead, or use HTML &quot; inside the HTML.
- body_html may use <p>, <strong>, <em>, <ul>, <li>, <h3>, <a href="...">, <table>, <tr>, <th>, <td>, <pre> tags only. No divs. No h1 or h2.
- body_html may include {{CHART_1}}, {{CHART_2}}, {{CHART_3}} as literal placeholder strings (they will be replaced).
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
6. FRED series IDs appearing in prose (e.g., "VIXCLS", "DGS10", "UNRATE" — series codes must not appear in the email body, including inside h3 headers)
7. "percentile" spelled out in full — must be "%ile"
8. Unsupported historical claims: a claim about a specific historical event with no source in the finding's Sources list or CONCEPT CONTEXT
9. σ citations with more than 1 decimal place (e.g., "−0.20σ" should be "−0.2σ")
10. Key statistics (values, z-scores, %ile ranks, correlations, bp changes) not wrapped in <strong> tags
11. Terms & Mechanics entries that explain financial basics the reader already knows (basis points, yield, spread, standard deviation, price-return) — flag as too obvious
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


def build_data_context(finding: Finding) -> dict[str, SeriesSnapshot]:
    """Load current value plus full-history and windowed z-scores / percentile ranks."""
    ctx: dict[str, SeriesSnapshot] = {}
    for sid in finding.series_ids:
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
        return ""
    chunks = retrieve_context(
        series_ids=series_ids,
        keywords=keywords or [],
        top_n=4,
    )
    if not chunks:
        return ""
    return "\n\n".join(c.text for c in chunks)


def draft_email(pick: LessonPick, ctx: dict[str, SeriesSnapshot]) -> dict:
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
    user_content += "Write the daily macro education email. Return only the JSON object."

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=[{"type": "text", "text": _DRAFT_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_content}],
    )
    raw = response.content[0].text
    return _extract_json(raw)


def fact_check_draft(draft: dict, pick: LessonPick, ctx: dict[str, SeriesSnapshot]) -> dict:
    """Return {"flags": [...], "approved": bool}."""
    client = _client()

    user_content = (
        f"DRAFT (body text):\n{draft.get('body_text', '')}\n\n"
        f"FINDING RECORD:\n{json.dumps(_finding_to_dict(pick.finding), indent=2)}\n\n"
        f"DATA CONTEXT:\n{json.dumps(_ctx_to_dict(ctx), indent=2)}\n\n"
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


# ---------- chart generation ----------

_CHART_KINDS = {"correlation_shift", "regime_transition", "notable_move_level",
                "notable_move_change", "lead_lag_change", "cointegration_break"}


def generate_charts(pick: LessonPick, ctx: dict[str, SeriesSnapshot], today: date) -> list[Path]:
    """Generate up to 3 charts for the finding.

    Chart 1 — primary time series with NBER shading and trigger-date marker.
    Chart 2 — rolling percentile rank over time (regime/notable-move) or
               rolling correlation (correlation/lead-lag findings).
    Chart 3 — distribution histogram (regime/notable-move) or
               cross-correlogram / multi-series (correlation/lead-lag).

    Returns list of saved PNG paths in order (may be shorter than 3).
    """
    from src.analytics.charts import (
        time_series, time_series_zoom, distribution, rolling_percentile,
        lead_lag_bar, save_to, add_source_footer, _ZOOM_MIN_SPAN_YEARS,
    )
    from src.analytics.data import load_aligned
    from src.analytics.stats import rolling_corr, lead_lag_xcorr
    import matplotlib.pyplot as plt

    finding = pick.finding
    if finding.kind not in _CHART_KINDS or not finding.series_ids:
        return []

    chart_dir = CHARTS_DIR / today.isoformat()
    chart_dir.mkdir(parents=True, exist_ok=True)
    slug_short = finding.slug[:40].replace("/", "-")
    paths: list[Path] = []

    sid1 = list(finding.series_ids)[0]

    # ── Chart 1: full time-series of primary series ──────────────────────────
    try:
        s = load_series(sid1)
        meta = series_metadata(sid1)
        series_title = meta.get("title", sid1)
        import matplotlib.dates as mdates
        from src.analytics.charts import _span_years

        span = _span_years(s.dropna().index) if len(s.dropna()) > 1 else 0
        if span >= _ZOOM_MIN_SPAN_YEARS:
            fig, (ax, _ax_zoom) = time_series_zoom(
                s,
                title=series_title,
                ylabel=str(meta.get("units", "")),
                shade_nber=True,
                zoom_years=5,
            )
        else:
            fig, ax = time_series(
                s,
                title=series_title,
                ylabel=str(meta.get("units", "")),
                shade_nber=True,
                annotate_last=True,
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
        paths.append(out)
        logger.info("Chart 1 saved: %s", out)
    except Exception as e:
        logger.warning("Chart 1 (time series) failed: %s", e)

    # ── Chart 2 ───────────────────────────────────────────────────────────────
    if finding.kind in {"correlation_shift", "lead_lag_change"} and len(finding.series_ids) >= 2:
        # Rolling correlation over time — use a frequency-appropriate window, not n_hist
        sids = list(finding.series_ids[:2])
        try:
            df = load_aligned(sids)
            freq = series_metadata(sids[0]).get("frequency_short", "M")
            default_windows = {"D": 126, "W": 52, "M": 48, "Q": 12, "A": 5}
            window = default_windows.get(freq, 48)
            total_obs = len(df.dropna())
            # Quality gate: need at least 4 full rolling windows visible to be readable
            if total_obs < window * 4:
                raise ValueError(f"Too few observations ({total_obs}) for window {window} — skipping chart 2")
            corr = rolling_corr(df.iloc[:, 0], df.iloc[:, 1], window=window)
            corr = corr.dropna() * 100
            if not corr.empty:
                m0 = series_metadata(sids[0]).get("title", sids[0])
                m1 = series_metadata(sids[1]).get("title", sids[1])
                fig2, ax2 = time_series(
                    corr,
                    title=f"Rolling {window}-Obs Spearman Correlation",
                    subtitle=f"{m0} vs {m1}",
                    ylabel="Spearman rank correlation (×100)",
                    shade_nber=True,
                    annotate_last=True,
                )
                ax2.axhline(0, color="#6B7280", linewidth=0.8, linestyle="--")
                add_source_footer(fig2, sids, today)
                out2 = chart_dir / f"{today.isoformat()}_{slug_short}_chart2.png"
                save_to(fig2, out2)
                plt.close(fig2)
                paths.append(out2)
                logger.info("Chart 2 (rolling corr) saved: %s", out2)
        except Exception as e:
            logger.warning("Chart 2 (rolling corr) skipped: %s", e)

    elif finding.kind in {"notable_move_level", "notable_move_change", "regime_transition"}:
        # Rolling percentile rank over time — more informative than static histogram
        try:
            s = load_series(sid1)
            meta = series_metadata(sid1)
            # Use ~2y window for daily series, ~5y for monthly
            freq = meta.get("frequency_short", "D")
            obs_per_year = {"D": 252, "W": 52, "M": 12, "Q": 4, "A": 1}.get(freq, 252)
            window = obs_per_year * 2
            fig2, _ = rolling_percentile(
                s,
                window=window,
                title=f"{meta.get('title', sid1)} — Rolling Percentile Rank",
                subtitle=f"2-year rolling window; bands at 33rd and 67th %ile",
                shade_nber=True,
            )
            add_source_footer(fig2, [sid1], today)
            out2 = chart_dir / f"{today.isoformat()}_{slug_short}_chart2.png"
            save_to(fig2, out2)
            plt.close(fig2)
            paths.append(out2)
            logger.info("Chart 2 (rolling percentile) saved: %s", out2)
        except Exception as e:
            logger.warning("Chart 2 (rolling percentile) failed: %s", e)

    # ── Chart 3 ───────────────────────────────────────────────────────────────
    if finding.kind == "lead_lag_change" and len(finding.series_ids) >= 2:
        # Split cross-correlogram: historical vs. recent periods side by side
        sids = list(finding.series_ids[:2])
        try:
            from src.analytics.charts import lead_lag_bar_split
            df = load_aligned(sids)
            ev = finding.evidence or {}
            n_hist = int(ev.get("n_hist", max(len(df) - 24, 10)))
            n_recent = int(ev.get("n_recent", min(24, len(df) - n_hist)))
            max_lag = 12
            # Quality gate: both periods need enough obs to compute 12-lag correlogram reliably
            if n_recent < max_lag + 3 or n_hist < max_lag * 2:
                raise ValueError(
                    f"Insufficient obs for split correlogram: n_hist={n_hist}, n_recent={n_recent}"
                )
            hist_df = df.iloc[:n_hist]
            recent_df = df.iloc[n_hist:]
            xcorr_hist = lead_lag_xcorr(hist_df.iloc[:, 0], hist_df.iloc[:, 1], max_lag=max_lag)
            xcorr_recent = lead_lag_xcorr(recent_df.iloc[:, 0], recent_df.iloc[:, 1], max_lag=max_lag)
            fig3, _ = lead_lag_bar_split(
                xcorr_hist,
                xcorr_recent,
                title=f"Cross-Correlogram: {sids[0]} vs {sids[1]}",
                subtitle="How the lead-lag structure has changed",
                hist_label=f"Historical (n={n_hist})",
                recent_label=f"Recent (n={n_recent})",
            )
            add_source_footer(fig3, sids, today)
            out3 = chart_dir / f"{today.isoformat()}_{slug_short}_chart3.png"
            save_to(fig3, out3)
            plt.close(fig3)
            paths.append(out3)
            logger.info("Chart 3 (split correlogram) saved: %s", out3)
        except Exception as e:
            logger.warning("Chart 3 (split correlogram) skipped: %s", e)

    elif finding.kind in {"notable_move_level", "notable_move_change", "regime_transition"} and sid1 in ctx:
        # Distribution histogram as chart 3 (percentile rank over time was chart 2)
        snap = ctx[sid1]
        if snap.current_value is not None:
            try:
                s = load_series(sid1)
                meta = series_metadata(sid1)
                pct_label = f"{snap.percentile_rank * 100:.0f}th %ile" if snap.percentile_rank is not None else ""
                fig3, _ = distribution(
                    s,
                    current_value=snap.current_value,
                    title=f"{meta.get('title', sid1)} — Full History Distribution",
                    subtitle=f"Current: {snap.current_value:.2f}{' — ' + pct_label if pct_label else ''}",
                    xlabel=str(meta.get("units", "")),
                )
                add_source_footer(fig3, [sid1], today)
                out3 = chart_dir / f"{today.isoformat()}_{slug_short}_chart3.png"
                save_to(fig3, out3)
                plt.close(fig3)
                paths.append(out3)
                logger.info("Chart 3 (distribution) saved: %s", out3)
            except Exception as e:
                logger.warning("Chart 3 (distribution) failed: %s", e)

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


def render_html(
    subject: str,
    body_html: str,
    today: date,
    chart_paths: list[Path] | None = None,
    chart_cids: list[str] | None = None,
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

    # Strip any remaining unreplaced placeholders
    body_html = re.sub(r"\{\{CHART_\d+\}\}", "", body_html)

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
    html_body_template: str # Body HTML with {{CHART_N}} placeholders (for dry-run re-render)
    text_body: str
    chart_paths: list[Path]
    fact_check_flags: list[str]
    approved: bool
    data_context: dict


# ---------- main entry point ----------

def compose_email(pick: LessonPick, today: date | None = None) -> ComposedEmail:
    """Build the email for today's lesson pick.

    Raises RuntimeError if the Anthropic key is missing.
    """
    today = today or date.today()

    ctx = build_data_context(pick.finding)
    if not ctx:
        logger.warning("No data context built for finding %s — series may not be tracked.", pick.finding.slug)

    draft = draft_email(pick, ctx)
    fc = fact_check_draft(draft, pick, ctx)

    flags = fc.get("flags", [])
    approved = fc.get("approved", True)

    if flags:
        logger.info("Fact-check flags (%d): %s", len(flags), "; ".join(flags))

    chart_paths = generate_charts(pick, ctx, today)
    cids = [f"chart_{i}" for i in range(len(chart_paths))]
    template_body = draft["body_html"]  # Preserves {{CHART_N}} placeholders
    html_body = render_html(draft["subject"], template_body, today, chart_cids=cids)

    return ComposedEmail(
        subject=draft["subject"],
        html_body=html_body,
        html_body_template=template_body,
        text_body=draft.get("body_text", ""),
        chart_paths=chart_paths,
        fact_check_flags=flags,
        approved=approved,
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
    evidence = _scale_evidence_correlations(f.evidence) if f.evidence else f.evidence
    return {
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
