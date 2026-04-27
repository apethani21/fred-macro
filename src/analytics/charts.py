"""Chart builders enforcing CLAUDE.md's chart standards.

Every chart in the project should go through here so palette, fonts, sizing,
date axes, NBER shading, and source attribution are consistent.

Usage:
    from src.analytics import charts
    charts.apply_style()
    fig, ax = charts.time_series(series, title="...", subtitle="...", series_id="DGS10")
    charts.add_source_footer(fig, ["DGS10"])
    fig.savefig("state/charts/2026-04-20/example.png")
"""
from __future__ import annotations

import textwrap
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from .episodes import nber_recession_ranges


# Okabe-Ito colorblind-safe palette, reordered so the first color reads as
# the "primary" series in a multi-line chart.
PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky
    "#F0E442",  # yellow
    "#000000",  # black
)

RECESSION_SHADING = {"color": "#9CA3AF", "alpha": 0.15}
LATEST_MARKER = {"color": "#111827", "s": 35, "zorder": 5}

# Figure sizes (CLAUDE.md)
FIGSIZE_TIME_SERIES = (8, 4)
FIGSIZE_DUAL = (14, 4)  # two side-by-side panels (full history + zoom)
FIGSIZE_DISTRIBUTION = (6.5, 3.8)
FIGSIZE_HEATMAP = (7, 5.5)
FIGSIZE_MULTI_PANEL = (10, 5)
FIGSIZE_DOT_PLOT = (10, 5)

# Minimum series span (years) before showing a zoom panel is useful.
_ZOOM_MIN_SPAN_YEARS = 8

_STYLE_APPLIED = False


def apply_style() -> None:
    """Apply the project-wide matplotlib style. Idempotent."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#4B5563",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.6,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.prop_cycle": plt.cycler("color", PALETTE),
    })
    _STYLE_APPLIED = True


# ---------- primitives ----------

def _span_years(index: pd.DatetimeIndex) -> float:
    return (index.max() - index.min()).days / 365.25


def set_date_axis(ax: Axes, index: pd.DatetimeIndex) -> None:
    """Pick date tick format from the series's span (CLAUDE.md convention)."""
    span = _span_years(index)
    if span >= 15:
        base = max(2, int(span / 8))
        locator = mdates.YearLocator(base=base)
        fmt = "%Y"
    elif span >= 3:
        locator = mdates.YearLocator()
        fmt = "%Y"
    elif span >= 1:
        locator = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
        fmt = "%Y-%m"
    else:
        locator = mdates.MonthLocator()
        fmt = "%b %Y"
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    # Rotate labels to prevent overlap; autofmt_xdate applies at save time.
    ax.figure.autofmt_xdate(rotation=45, ha="right")


def shade_recessions(ax: Axes, index: pd.DatetimeIndex | None = None) -> None:
    """Shade NBER recessions on a time-series axis. Restrict to the plot's
    current xlim so shading doesn't leak past the data range."""
    xmin, xmax = ax.get_xlim()
    for start, end in nber_recession_ranges():
        s_num = mdates.date2num(start)
        e_num = mdates.date2num(end)
        if e_num < xmin or s_num > xmax:
            continue
        ax.axvspan(max(s_num, xmin), min(e_num, xmax), **RECESSION_SHADING)


def annotate_latest(ax: Axes, s: pd.Series, fmt: str = "{:.2f}") -> None:
    """Marker + label on the most recent non-NaN point."""
    x = s.dropna()
    if x.empty:
        return
    t = x.index[-1]
    v = x.iloc[-1]
    ax.scatter([t], [v], **LATEST_MARKER)
    ax.annotate(
        fmt.format(v),
        xy=(t, v),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
        va="center",
    )


def add_source_footer(
    fig: Figure,
    series_ids: Iterable[str],
    as_of: date | datetime | pd.Timestamp | None = None,
    source_label: str = "FRED",
) -> None:
    as_of = as_of or datetime.utcnow().date()
    if isinstance(as_of, (datetime, pd.Timestamp)):
        as_of = as_of.date() if hasattr(as_of, "date") else as_of
    ids = ", ".join(series_ids)
    fig.text(
        0.01, 0.01,
        f"Source: {source_label} ({ids}). As of {as_of.isoformat()}.",
        fontsize=8, color="#6B7280", ha="left", va="bottom",
    )


def _wrap_title(title: str, width: int = 52) -> str:
    """Wrap long titles at word boundaries to avoid single-line overflow."""
    if len(title) <= width:
        return title
    return "\n".join(textwrap.wrap(title, width=width))


def _set_titles(ax: Axes, title: str, subtitle: str | None) -> None:
    wrapped = _wrap_title(title)
    if subtitle:
        ax.set_title(wrapped, loc="left", pad=18)
        ax.text(
            0, 1.02, subtitle,
            transform=ax.transAxes, fontsize=11, color="#374151",
            ha="left", va="bottom",
        )
    else:
        ax.set_title(wrapped, loc="left")


# ---------- standard builders ----------

def time_series(
    s: pd.Series,
    title: str,
    series_id: str | None = None,
    subtitle: str | None = None,
    ylabel: str = "",
    shade_nber: bool = True,
    annotate_last: bool = True,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_TIME_SERIES)
    else:
        fig = ax.figure

    ax.plot(s.index, s.values, color=PALETTE[0], linewidth=1.5, label=series_id or s.name or "")
    _set_titles(ax, title, subtitle)
    ax.set_ylabel(ylabel)
    set_date_axis(ax, s.dropna().index)
    if shade_nber:
        shade_recessions(ax)
    if annotate_last:
        annotate_latest(ax, s)
    return fig, ax


def multi_series(
    df: pd.DataFrame,
    title: str,
    subtitle: str | None = None,
    ylabel: str = "",
    shade_nber: bool = True,
    normalize: str = "none",  # 'none' | 'zscore' | 'index100'
    legend_labels: Sequence[str] | None = None,
) -> tuple[Figure, Axes]:
    apply_style()
    if normalize == "zscore":
        plot_df = (df - df.mean()) / df.std()
        y_label_suffix = " (z-score)"
    elif normalize == "index100":
        # Start each series at 100 from its first non-NaN value.
        first = df.bfill().iloc[0]
        plot_df = df.divide(first) * 100.0
        y_label_suffix = " (indexed, start=100)"
    elif normalize == "none":
        plot_df = df
        y_label_suffix = ""
    else:
        raise ValueError(f"Unknown normalize: {normalize!r}")

    fig, ax = plt.subplots(figsize=FIGSIZE_TIME_SERIES)
    labels = legend_labels or plot_df.columns
    for i, col in enumerate(plot_df.columns):
        ax.plot(
            plot_df.index, plot_df[col].values,
            color=PALETTE[i % len(PALETTE)], linewidth=1.4, label=labels[i],
        )
    _set_titles(ax, title, subtitle)
    ax.set_ylabel(ylabel + y_label_suffix)
    set_date_axis(ax, plot_df.dropna(how="all").index)
    if shade_nber:
        shade_recessions(ax)
    ax.legend(loc="best", frameon=False)
    return fig, ax


def distribution(
    s: pd.Series,
    title: str,
    subtitle: str | None = None,
    xlabel: str = "",
    current_value: float | None = None,
    bins: int = 50,
) -> tuple[Figure, Axes]:
    """Histogram of the full history with optional marker at `current_value`.

    Use for 'current reading is Xth percentile of history' framings.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_DISTRIBUTION)
    x = s.dropna().values
    ax.hist(x, bins=bins, color=PALETTE[0], alpha=0.85, edgecolor="white")
    _set_titles(ax, title, subtitle)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("frequency")
    ax.grid(axis="y")

    cv = current_value if current_value is not None else (float(s.dropna().iloc[-1]) if not s.dropna().empty else None)
    if cv is not None:
        pct = float((x <= cv).mean())
        ax.axvline(cv, color=PALETTE[1], linewidth=2.0)
        ax.annotate(
            f"current: {cv:.2f}  ({pct * 100:.0f}th pct)",
            xy=(cv, ax.get_ylim()[1] * 0.9),
            xytext=(6, 0), textcoords="offset points",
            fontsize=9, color=PALETTE[1],
        )
    return fig, ax


def correlation_heatmap(
    corr: pd.DataFrame,
    title: str,
    subtitle: str | None = None,
    fmt: str = "{:+.2f}",
) -> tuple[Figure, Axes]:
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)
    # Diverging colormap, symmetric around 0.
    vmax = float(np.nanmax(np.abs(corr.values)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(corr.shape[1]))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(corr.shape[0]))
    ax.set_yticklabels(corr.index)
    ax.grid(False)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            v = corr.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9, color="#111827")
    _set_titles(ax, title, subtitle)
    fig.colorbar(im, ax=ax, shrink=0.7, label="correlation")
    return fig, ax


def event_study(
    series: pd.Series,
    event_date: str | pd.Timestamp,
    window_days: tuple[int, int] = (-30, 60),
    title: str = "",
    subtitle: str | None = None,
    ylabel: str = "",
    rebase: bool = True,
) -> tuple[Figure, Axes]:
    """Plot `series` around `event_date`, with x-axis in days-from-event.
    If `rebase`, subtract the value on `event_date` so y shows change.
    """
    apply_style()
    ts = pd.Timestamp(event_date)
    lo, hi = window_days
    start = ts + pd.Timedelta(days=lo)
    end = ts + pd.Timedelta(days=hi)
    s = series.loc[start:end].dropna()
    if s.empty:
        raise ValueError(f"No data for {event_date} ± window")
    baseline = s.asof(ts) if rebase else 0.0
    if baseline is None or (isinstance(baseline, float) and np.isnan(baseline)):
        baseline = 0.0
    day_offset = (s.index - ts).days
    fig, ax = plt.subplots(figsize=FIGSIZE_TIME_SERIES)
    ax.axvline(0, color="#6B7280", linewidth=1, linestyle="--")
    ax.plot(day_offset, s.values - baseline, color=PALETTE[0], linewidth=1.6)
    _set_titles(ax, title or f"Event study around {ts.date().isoformat()}", subtitle)
    ax.set_xlabel("days from event")
    ax.set_ylabel(ylabel + (" (change from event)" if rebase else ""))
    return fig, ax


def rolling_percentile(
    s: pd.Series,
    window: int,
    title: str,
    subtitle: str | None = None,
    ylabel: str = "percentile rank (0–1)",
    shade_nber: bool = True,
) -> tuple[Figure, Axes]:
    """Plot the rolling percentile rank of `s` over time.

    At each point t, rank = fraction of the prior `window` observations that
    are <= s[t].  Shows the series moving through its distribution over time;
    horizontal bands at 0.33 and 0.67 mark regime thirds.
    """
    apply_style()
    x = s.dropna()
    # Rolling percentile: for each window of length `window`, rank last value.
    rank = x.rolling(window, min_periods=max(window // 2, 10)).apply(
        lambda w: float((w[:-1] <= w[-1]).mean()), raw=True
    )
    fig, ax = plt.subplots(figsize=FIGSIZE_TIME_SERIES)
    ax.plot(rank.index, rank.values, color=PALETTE[0], linewidth=1.4)
    ax.axhline(0.33, color="#9CA3AF", linewidth=0.8, linestyle="--")
    ax.axhline(0.67, color="#9CA3AF", linewidth=0.8, linestyle="--")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.33, 0.5, 0.67, 1.0])
    ax.set_yticklabels(["0", "33rd", "50th", "67th", "100th"])
    _set_titles(ax, title, subtitle)
    ax.set_ylabel(ylabel)
    set_date_axis(ax, rank.dropna().index)
    if shade_nber:
        shade_recessions(ax)
    annotate_latest(ax, rank, fmt="{:.2f}")
    return fig, ax


def lead_lag_bar(
    xcorr: pd.DataFrame,
    title: str,
    subtitle: str | None = None,
    highlight_lag: int | None = None,
) -> tuple[Figure, Axes]:
    """Bar chart of cross-correlation at multiple lags (cross-correlogram).

    `xcorr` must have columns 'lag' and 'correlation' (output of
    stats.lead_lag_xcorr).  Positive lag = second series leads first.
    Correlations scaled ×100 (range −100 to +100).
    """
    apply_style()
    df = xcorr.dropna(subset=["correlation"]).sort_values("lag")
    lags = df["lag"].values
    corrs = df["correlation"].values * 100

    colors = [PALETTE[0] if c >= 0 else PALETTE[1] for c in corrs]
    if highlight_lag is not None:
        colors = [
            PALETTE[2] if int(lag) == highlight_lag else c
            for lag, c in zip(lags, colors)
        ]

    fig, ax = plt.subplots(figsize=FIGSIZE_TIME_SERIES)
    ax.bar(lags, corrs, color=colors, width=0.7, edgecolor="white")
    ax.axhline(0, color="#4B5563", linewidth=0.8)
    ax.set_xlabel("lag (periods; positive = second series leads)")
    ax.set_ylabel("Spearman rank correlation (×100)")
    ax.set_ylim(-105, 105)
    _set_titles(ax, title, subtitle)
    return fig, ax


def lead_lag_bar_split(
    xcorr_hist: pd.DataFrame,
    xcorr_recent: pd.DataFrame,
    title: str,
    hist_label: str = "Historical",
    recent_label: str = "Recent",
    subtitle: str | None = None,
) -> tuple[Figure, Axes]:
    """Grouped bar cross-correlogram comparing two periods side by side.

    Shows how the lead-lag structure has changed between periods.
    Correlations scaled ×100 (range −100 to +100).
    """
    apply_style()
    h = xcorr_hist.dropna(subset=["correlation"]).set_index("lag")["correlation"] * 100
    r = xcorr_recent.dropna(subset=["correlation"]).set_index("lag")["correlation"] * 100
    lags = sorted(set(h.index) | set(r.index))

    width = 0.38
    x = np.arange(len(lags))
    hist_vals = [float(h.get(lag, np.nan)) for lag in lags]
    recent_vals = [float(r.get(lag, np.nan)) for lag in lags]

    fig, ax = plt.subplots(figsize=FIGSIZE_TIME_SERIES)
    ax.bar(x - width / 2, hist_vals, width, label=hist_label, color=PALETTE[0], alpha=0.8, edgecolor="white")
    ax.bar(x + width / 2, recent_vals, width, label=recent_label, color=PALETTE[3], alpha=0.9, edgecolor="white")

    ax.axhline(0, color="#4B5563", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(lag) for lag in lags], fontsize=9)
    ax.set_xlabel("lag (periods; positive = second series leads)")
    ax.set_ylabel("Spearman rank correlation (×100)")
    ax.set_ylim(-105, 105)
    ax.legend(fontsize=9, framealpha=0.9)
    _set_titles(ax, title, subtitle)
    return fig, ax


def time_series_zoom(
    s: pd.Series,
    title: str,
    series_id: str | None = None,
    subtitle: str | None = None,
    ylabel: str = "",
    shade_nber: bool = True,
    zoom_years: int = 5,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Two-panel time series: full history (left) + recent zoom (right).

    Only call this when the series spans > _ZOOM_MIN_SPAN_YEARS; for shorter
    series use time_series() directly.
    """
    apply_style()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIGSIZE_DUAL)

    s_clean = s.dropna()

    ax_left.plot(s_clean.index, s_clean.values, color=PALETTE[0], linewidth=1.5)
    _set_titles(ax_left, title, subtitle)
    ax_left.set_ylabel(ylabel)
    set_date_axis(ax_left, s_clean.index)
    if shade_nber:
        shade_recessions(ax_left)
    annotate_latest(ax_left, s_clean)

    cutoff = s_clean.index.max() - pd.DateOffset(years=zoom_years)
    s_zoom = s_clean[s_clean.index >= cutoff]
    ax_right.plot(s_zoom.index, s_zoom.values, color=PALETTE[0], linewidth=1.5)
    _set_titles(ax_right, f"Last {zoom_years}Y", None)
    ax_right.set_ylabel(ylabel)
    set_date_axis(ax_right, s_zoom.index)
    if shade_nber:
        shade_recessions(ax_right)
    annotate_latest(ax_right, s_zoom)

    plt.tight_layout()
    return fig, (ax_left, ax_right)


def multi_series_zoom(
    df: pd.DataFrame,
    title: str,
    subtitle: str | None = None,
    ylabel: str = "",
    shade_nber: bool = True,
    normalize: str = "none",
    legend_labels: Sequence[str] | None = None,
    zoom_years: int = 5,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Two-panel multi-series chart: full history (left) + recent zoom (right)."""
    apply_style()
    if normalize == "zscore":
        plot_df = (df - df.mean()) / df.std()
        y_label_suffix = " (z-score)"
    elif normalize == "index100":
        first = df.bfill().iloc[0]
        plot_df = df.divide(first) * 100.0
        y_label_suffix = " (indexed, start=100)"
    elif normalize == "none":
        plot_df = df
        y_label_suffix = ""
    else:
        raise ValueError(f"Unknown normalize: {normalize!r}")

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIGSIZE_DUAL)
    labels = legend_labels or list(plot_df.columns)

    for i, col in enumerate(plot_df.columns):
        color = PALETTE[i % len(PALETTE)]
        ax_left.plot(plot_df.index, plot_df[col].values, color=color, linewidth=1.4, label=labels[i])
    _set_titles(ax_left, title, subtitle)
    ax_left.set_ylabel(ylabel + y_label_suffix)
    set_date_axis(ax_left, plot_df.dropna(how="all").index)
    if shade_nber:
        shade_recessions(ax_left)
    ax_left.legend(loc="best", frameon=False)

    cutoff = plot_df.index.max() - pd.DateOffset(years=zoom_years)
    zoom_df = plot_df[plot_df.index >= cutoff]
    for i, col in enumerate(zoom_df.columns):
        color = PALETTE[i % len(PALETTE)]
        ax_right.plot(zoom_df.index, zoom_df[col].values, color=color, linewidth=1.4, label=labels[i])
    _set_titles(ax_right, f"Last {zoom_years}Y", None)
    ax_right.set_ylabel(ylabel + y_label_suffix)
    set_date_axis(ax_right, zoom_df.dropna(how="all").index)
    if shade_nber:
        shade_recessions(ax_right)
    ax_right.legend(loc="best", frameon=False)

    plt.tight_layout()
    return fig, (ax_left, ax_right)


def era_comparison_bar(
    eras: list[dict],
    metric_a: str,
    metric_b: str,
    label_a: str,
    label_b: str,
    title: str,
    subtitle: str | None = None,
    ylabel: str = "",
    stacked: bool = False,
    pct_scale: bool = False,
) -> tuple[Figure, Axes]:
    """Grouped or stacked bar chart comparing two metrics across named eras.

    `eras` is a list of dicts with at least 'era', metric_a, and metric_b keys.
    `pct_scale` multiplies values by 100 (for fractions displayed as %).
    """
    apply_style()

    def _fmt_era(e: dict) -> str:
        label = e.get("era", "?")
        start = e.get("era_start", "")
        end = e.get("era_end", "")
        y0 = start[:4] if start else ""
        y1 = end[:4] if end else ""
        readable = {
            "pre_gfc": "Pre-GFC",
            "gfc_zlb": "GFC/ZLB",
            "post_taper": "Post-Taper",
            "post_taper_pre_covid": "Post-Taper",
            "covid_zlb": "COVID ZLB",
            "hiking_2022": "2022 Hike",
            "current": "Current",
        }.get(label, label.replace("_", " ").title())
        if y0 and y1:
            return f"{readable}\n{y0}–{y1}"
        return readable

    era_labels = [_fmt_era(e) for e in eras]
    vals_a = [float(e.get(metric_a, 0)) * (100 if pct_scale else 1) for e in eras]
    vals_b = [float(e.get(metric_b, 0)) * (100 if pct_scale else 1) for e in eras]
    x = np.arange(len(era_labels))

    fig, ax = plt.subplots(figsize=FIGSIZE_TIME_SERIES)

    if stacked:
        ax.bar(x, vals_a, label=label_a, color=PALETTE[0], alpha=0.85, edgecolor="white")
        ax.bar(x, vals_b, bottom=vals_a, label=label_b, color=PALETTE[1], alpha=0.85, edgecolor="white")
        for i, (a, b) in enumerate(zip(vals_a, vals_b)):
            total = a + b
            if total > 0:
                ax.text(x[i], total + 0.5, f"{b / total * 100:.0f}%", ha="center", fontsize=8, color="#374151")
    else:
        width = 0.38
        ax.bar(x - width / 2, vals_a, width, label=label_a, color=PALETTE[0], alpha=0.85, edgecolor="white")
        ax.bar(x + width / 2, vals_b, width, label=label_b, color=PALETTE[3], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(era_labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=False, fontsize=9)
    _set_titles(ax, title, subtitle)
    plt.tight_layout()
    return fig, ax


# ---------- FOMC dot plot ----------

def _dot_jitter(n: int, max_half: float = 0.28) -> np.ndarray:
    """Evenly-spaced horizontal offsets for n dots within one year column."""
    if n <= 1:
        return np.array([0.0])
    return np.linspace(-max_half, max_half, n)


def _expand_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Expand participant_count to one row per participant (forecast_year, rate)."""
    rows: list[dict] = []
    for _, row in df.iterrows():
        base = {"forecast_year": row["forecast_year"], "rate": float(row["rate"])}
        rows.extend([base.copy() for _ in range(int(row["participant_count"]))])
    return pd.DataFrame(rows)


def dot_plot(
    df: pd.DataFrame,
    title: str,
    subtitle: str | None = None,
    compare_df: pd.DataFrame | None = None,
    current_rate: float | None = None,
    meeting_label: str = "Latest SEP",
    compare_label: str = "Previous SEP",
) -> tuple[Figure, Axes]:
    """FOMC dot plot: participant fed-funds-rate projections by forecast year.

    Parameters
    ----------
    df:
        Columns forecast_year, rate, participant_count for the latest meeting.
    compare_df:
        Optional second meeting (shown in a lighter shade for before/after).
    current_rate:
        If given, draws a dashed horizontal line at the current effective FFR.
    meeting_label / compare_label:
        Legend labels for the two meetings.

    Each participant is one dot; dots at the same (year, rate) are spread
    symmetrically within the column.  The median per year is marked with a
    diamond.
    """
    apply_style()

    # Canonical year order: numeric years ascending, then non-numeric ("Longer run").
    all_years = list(df["forecast_year"].unique())
    if compare_df is not None and not compare_df.empty:
        for y in compare_df["forecast_year"].unique():
            if y not in all_years:
                all_years.append(y)

    def _year_key(y: str) -> tuple:
        try:
            return (0, int(y))
        except ValueError:
            return (1, y)

    ordered_years = sorted(all_years, key=_year_key)
    year_pos = {y: i for i, y in enumerate(ordered_years)}

    fig, ax = plt.subplots(figsize=FIGSIZE_DOT_PLOT)

    def _plot(
        meeting_df: pd.DataFrame,
        color: str,
        alpha: float,
        label: str,
        x_shift: float = 0.0,
    ) -> None:
        expanded = _expand_counts(meeting_df)
        medians = {
            y: float(np.median(grp["rate"].values))
            for y, grp in expanded.groupby("forecast_year")
        }
        first_year = True
        for year in ordered_years:
            subset = sorted(expanded[expanded["forecast_year"] == year]["rate"].values)
            if not subset:
                continue
            xc = year_pos[year] + x_shift
            for rate, dx in zip(subset, _dot_jitter(len(subset))):
                ax.scatter(
                    xc + dx, rate,
                    color=color, alpha=alpha, s=38, zorder=3,
                    linewidths=0,
                )
            if year in medians:
                ax.scatter(
                    xc, medians[year],
                    color=color, alpha=min(alpha + 0.2, 1.0),
                    marker="D", s=65, zorder=5,
                    edgecolors="white", linewidths=0.8,
                    label=label if first_year else "_nolegend_",
                )
                first_year = False

    if compare_df is not None and not compare_df.empty:
        _plot(compare_df, PALETTE[5], 0.40, compare_label, x_shift=-0.09)
        _plot(df, PALETTE[0], 0.88, meeting_label, x_shift=0.09)
    else:
        _plot(df, PALETTE[0], 0.88, meeting_label)

    if current_rate is not None:
        ax.axhline(
            current_rate, color=PALETTE[1], linewidth=1.2, linestyle="--", alpha=0.85,
            label=f"Current FFR: {current_rate:.2f}%",
        )

    ax.set_xticks(range(len(ordered_years)))
    ax.set_xticklabels(ordered_years, fontsize=10)
    ax.set_xlim(-0.6, len(ordered_years) - 0.4)
    ax.set_xlabel("End of year", fontsize=11)
    ax.set_ylabel("Percent", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.yaxis.set_minor_locator(MultipleLocator(0.125))
    ax.grid(which="minor", axis="y", color="#F3F4F6", linewidth=0.4)

    if compare_df is not None and not compare_df.empty or current_rate is not None:
        ax.legend(loc="upper right", frameon=False, fontsize=9)

    _set_titles(ax, title, subtitle)
    return fig, ax


# ---------- equation rendering ----------

def render_equation_image(latex_str: str, *, fontsize: int = 16, dpi: int = 150) -> bytes:
    """Render a LaTeX math expression to PNG bytes using matplotlib mathtext.

    No system LaTeX required — uses matplotlib's built-in mathtext parser.
    Wrap the caller's string in $...$ internally; do NOT include $ in latex_str.
    """
    import io
    fig = plt.figure(figsize=(8, 0.7))
    fig.patch.set_facecolor("white")
    fig.text(
        0.5, 0.5,
        f"${latex_str}$",
        fontsize=fontsize,
        ha="center",
        va="center",
        color="#1a1a1a",
        fontfamily="DejaVu Sans",
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------- convenience ----------

def save_to(fig: Figure, path: str | Path, *, create_dirs: bool = True) -> Path:
    p = Path(path)
    if create_dirs:
        p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    return p
