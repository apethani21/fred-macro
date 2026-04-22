"""Number/date formatters for the email composer.

Opinionated defaults for macro numbers: percentages and basis points where
appropriate, thousand separators on levels, sensible precision by magnitude.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Callable, Mapping, Sequence

import pandas as pd


def fmt_pct(x: float | None, digits: int = 2) -> str:
    """Percent of a decimal (0.0325 -> '3.25%'). Returns '—' for None/NaN."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x * 100:.{digits}f}%"


def fmt_pct_already(x: float | None, digits: int = 2) -> str:
    """For values already expressed in percent (FRED yields, rates). '4.32' -> '4.32%'."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:.{digits}f}%"


def fmt_bp(x: float | None, from_pct: bool = True, digits: int = 0) -> str:
    """Basis points. `from_pct=True` means input is in percent (0.25 pct -> 25 bp)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    bp = x * 100 if from_pct else x * 10000
    return f"{bp:+.{digits}f} bp"


def fmt_level(x: float | None, digits: int | None = None) -> str:
    """Generic level formatter with auto-precision by magnitude if digits is None."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    if digits is None:
        ax = abs(x)
        digits = 0 if ax >= 100 else 1 if ax >= 10 else 2 if ax >= 1 else 3
    return f"{x:,.{digits}f}"


def fmt_z(z: float | None, digits: int = 1) -> str:
    """Z-score with sign. ±3.0σ or '—'."""
    if z is None or (isinstance(z, float) and pd.isna(z)):
        return "—"
    return f"{z:+.{digits}f}σ"


def fmt_corr(r: float | None, digits: int = 0, signed: bool = True) -> str:
    """Correlation as a signed percent. 0.814 -> '+81%'.

    Readers parse '+81%' faster than '0.81'. `signed=False` drops the explicit
    '+' (useful when rendering a matrix where sign is carried by color).
    """
    if r is None or (isinstance(r, float) and pd.isna(r)):
        return "—"
    pct = r * 100
    sign = "+" if signed else ""
    return f"{pct:{sign}.{digits}f}%"


def fmt_pp(x: float | None, digits: int = 2) -> str:
    """Percentage-point change, signed. For yield/rate changes in level (not bp).

    '1.25' -> '+1.25 pp'. Use `fmt_bp` when the change is sub-percent.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:+.{digits}f} pp"


def fmt_date(d: date | datetime | pd.Timestamp | str | None, style: str = "iso") -> str:
    """Date formatter. style: 'iso' (2026-04-20), 'long' (Apr 20, 2026), 'ym' (Apr 2026)."""
    if d is None or (isinstance(d, float) and pd.isna(d)):
        return "—"
    if isinstance(d, str):
        d = pd.Timestamp(d)
    elif isinstance(d, (date, datetime)):
        d = pd.Timestamp(d)
    if style == "iso":
        return d.strftime("%Y-%m-%d")
    if style == "long":
        return d.strftime("%b %d, %Y")
    if style == "ym":
        return d.strftime("%b %Y")
    raise ValueError(f"Unknown date style: {style}")


def fmt_range(lo: Any, hi: Any, formatter=fmt_level) -> str:
    return f"{formatter(lo)} – {formatter(hi)}"


def fmt_percentile(p: float | None, digits: int = 0) -> str:
    """Percentile rank in history. p in [0, 1]."""
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return "—"
    return f"{p * 100:.{digits}f}th pct"


def series_with_id(series_id: str, title: str | None = None) -> str:
    """Standard inline citation: 'Title (SERIES_ID)'. Falls back to just the id."""
    if title:
        return f"{title} ({series_id})"
    return series_id


# ---------- tables ----------

Formatter = Callable[[Any], str]


def format_table(
    df: pd.DataFrame,
    formatters: Mapping[str, Formatter] | None = None,
    style: str = "markdown",
    index: bool = True,
    align: Mapping[str, str] | None = None,
) -> str:
    """Render a DataFrame with per-column formatters to markdown or plaintext.

    `formatters` maps column name → function that formats one cell; any column
    not listed is rendered with `str`. Integer columns default to `fmt_level`.
    `align` maps column name → 'l' | 'c' | 'r' (markdown only; plaintext is
    right-aligned for numeric-looking cells).
    """
    if style not in ("markdown", "plain"):
        raise ValueError(f"Unknown style {style!r}")
    formatters = dict(formatters or {})
    for col, dtype in df.dtypes.items():
        if col in formatters:
            continue
        if pd.api.types.is_numeric_dtype(dtype):
            formatters[col] = fmt_level
        else:
            formatters[col] = lambda v: "—" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)

    idx_name = df.index.name or ""
    cols: list[str] = ([idx_name] if index else []) + [str(c) for c in df.columns]
    body: list[list[str]] = []
    for ix, row in df.iterrows():
        cells = [str(ix)] if index else []
        for c in df.columns:
            cells.append(formatters[c](row[c]))
        body.append(cells)

    if style == "markdown":
        align = align or {}
        alignments: list[str] = []
        if index:
            alignments.append("l")
        for c in df.columns:
            a = align.get(c, "r" if pd.api.types.is_numeric_dtype(df.dtypes[c]) else "l")
            alignments.append(a)
        sep = {"l": ":---", "c": ":---:", "r": "---:"}
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(sep[a] for a in alignments) + " |",
        ]
        for row in body:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    # plaintext — fixed-width, right-aligned columns
    widths = [max(len(cols[i]), *(len(r[i]) for r in body)) for i in range(len(cols))]

    def _line(cells: Sequence[str]) -> str:
        return "  ".join(c.rjust(w) for c, w in zip(cells, widths))

    return "\n".join([_line(cols), _line(["-" * w for w in widths]), *[_line(r) for r in body]])
