"""Static registry of ECB SDW series tracked by this project.

All ECB series IDs are prefixed "ECB." to avoid collision with FRED IDs.
Derived series (BTPBUND.SPREAD, BUND.SLOPE) are computed from raw series
at update time — they have no ECB flow/key.

All flow/key combinations were verified against the live ECB data API
(data-api.ecb.europa.eu) on 2026-04-24.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EcbSeriesSpec:
    series_id: str          # e.g. "ECB.DFR"
    flow: str               # ECB dataflow code, e.g. "FM"; empty string for derived
    key: str                # SDMX series key; empty string for derived
    title: str
    units: str
    units_short: str
    frequency_short: str    # "D", "M", or "Q" — maps to FREQ_TO_PARTITION
    partition: str          # "daily", "monthly", "quarterly"
    notes: str = ""
    # For derived series: list of (series_id, sign) tuples that sum to this series.
    # e.g. [("ECB.IT.10Y", 1), ("ECB.DE.10Y", -1)] means IT - DE.
    derived_from: list[tuple[str, int]] = field(default_factory=list)

    @property
    def is_derived(self) -> bool:
        return bool(self.derived_from)


# fmt: off
ECB_SERIES_REGISTRY: list[EcbSeriesSpec] = [

    # ── Policy rates ──────────────────────────────────────────────────────────

    EcbSeriesSpec(
        series_id="ECB.DFR",
        flow="FM",
        key="B.U2.EUR.4F.KR.DFR.LEV",
        title="ECB Deposit Facility Rate",
        units="Percent per annum",
        units_short="% p.a.",
        frequency_short="D",
        partition="daily",
        notes=(
            "ECB deposit facility rate — the operative monetary policy rate "
            "post-2022 floor system. Step-change series: only records on dates "
            "of rate changes."
        ),
    ),

    # ── Money market reference rates ──────────────────────────────────────────

    EcbSeriesSpec(
        series_id="ECB.ESTR",
        flow="EST",
        key="B.EU000A2X2A25.WT",
        title="Euro Short-Term Rate (€STR)",
        units="Percent",
        units_short="%",
        frequency_short="D",
        partition="daily",
        notes=(
            "Volume-weighted trimmed mean of overnight unsecured euro lending "
            "transactions. Replaced EONIA as the euro RFR in 2022."
        ),
    ),

    # ── Inflation ─────────────────────────────────────────────────────────────

    EcbSeriesSpec(
        series_id="ECB.HICP.EA.TOTAL",
        flow="ICP",
        key="M.U2.N.000000.4.ANR",
        title="EA HICP All-Items, Annual Rate of Change",
        units="Percent change",
        units_short="%",
        frequency_short="M",
        partition="monthly",
        notes=(
            "Harmonised Index of Consumer Prices for the euro area "
            "(changing composition), year-on-year percent change."
        ),
    ),

    EcbSeriesSpec(
        series_id="ECB.HICP.EA.CORE",
        flow="ICP",
        key="M.U2.N.XEF000.4.ANR",
        title="EA HICP Ex Food and Energy, Annual Rate of Change",
        units="Percent change",
        units_short="%",
        frequency_short="M",
        partition="monthly",
        notes="EA HICP excluding energy and food (all-items ex food/energy), YoY.",
    ),

    # ── Yield curve (ECB AAA) ─────────────────────────────────────────────────

    EcbSeriesSpec(
        series_id="ECB.YC.AAA.2Y",
        flow="YC",
        key="B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y",
        title="EA AAA Yield Curve, 2-Year Spot Rate",
        units="Percent per annum",
        units_short="% p.a.",
        frequency_short="D",
        partition="daily",
        notes=(
            "ECB AAA-rated euro area government bond yield curve, Svensson model, "
            "continuous compounding, 2-year spot rate. Dominated by Germany in practice."
        ),
    ),

    EcbSeriesSpec(
        series_id="ECB.YC.AAA.10Y",
        flow="YC",
        key="B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
        title="EA AAA Yield Curve, 10-Year Spot Rate",
        units="Percent per annum",
        units_short="% p.a.",
        frequency_short="D",
        partition="daily",
        notes=(
            "ECB AAA-rated euro area government bond yield curve, Svensson model, "
            "continuous compounding, 10-year spot rate."
        ),
    ),

    # ── Country bond yields (Maastricht convergence criterion) ────────────────

    EcbSeriesSpec(
        series_id="ECB.DE.10Y",
        flow="IRS",
        key="M.DE.L.L40.CI.0000.EUR.N.Z",
        title="Germany 10-Year Government Bond Yield (Maastricht)",
        units="Percent",
        units_short="%",
        frequency_short="M",
        partition="monthly",
        notes=(
            "Long-term interest rate for convergence purposes — Germany, "
            "10-year debt securities, monthly average. ECB/Eurostat Maastricht criterion series."
        ),
    ),

    EcbSeriesSpec(
        series_id="ECB.IT.10Y",
        flow="IRS",
        key="M.IT.L.L40.CI.0000.EUR.N.Z",
        title="Italy 10-Year Government Bond Yield (Maastricht)",
        units="Percent",
        units_short="%",
        frequency_short="M",
        partition="monthly",
        notes=(
            "Long-term interest rate for convergence purposes — Italy (BTP benchmark), "
            "10-year debt securities, monthly average."
        ),
    ),

    # ── Money supply ──────────────────────────────────────────────────────────

    EcbSeriesSpec(
        series_id="ECB.M3.EA",
        flow="BSI",
        key="M.U2.Y.V.M30.X.I.U2.2300.Z01.A",
        title="EA M3 Monetary Aggregate, Annual Growth Rate",
        units="Percent",
        units_short="%",
        frequency_short="M",
        partition="monthly",
        notes=(
            "Euro area M3 annual growth rate, seasonally adjusted. "
            "ECB's broadest monetary aggregate: M1 + short-term deposits + "
            "marketable instruments."
        ),
    ),

    # ── Labour costs ──────────────────────────────────────────────────────────

    EcbSeriesSpec(
        series_id="ECB.WAGES.NEG",
        flow="INW",
        key="Q.I9.N.INWR.000000.4F0.GY.IX",
        title="EA Negotiated Wage Rates, Annual Growth Rate",
        units="Percent",
        units_short="%",
        frequency_short="Q",
        partition="quarterly",
        notes=(
            "ECB indicator of negotiated wage rates for the euro area 20 "
            "(fixed composition as of 2023-01-01), annual percentage change. "
            "Key leading indicator for core services HICP inflation; typically "
            "leads actual compensation by 1-2 quarters due to sector-level "
            "bargaining cycles."
        ),
    ),

    # ── Derived series ────────────────────────────────────────────────────────

    EcbSeriesSpec(
        series_id="ECB.BTPBUND.SPREAD",
        flow="",
        key="",
        title="BTP-Bund 10Y Spread",
        units="Percentage points",
        units_short="pp",
        frequency_short="M",
        partition="monthly",
        derived_from=[("ECB.IT.10Y", 1), ("ECB.DE.10Y", -1)],
        notes=(
            "Italian BTP minus German Bund 10-year yield spread (percentage points). "
            "Computed from ECB.IT.10Y and ECB.DE.10Y at update time. "
            "Key fragmentation risk indicator: peaked ~550bp (2011-12), "
            "triggered TPI announcement at ~230bp (Jun 2022)."
        ),
    ),

    EcbSeriesSpec(
        series_id="ECB.BUND.SLOPE",
        flow="",
        key="",
        title="EA AAA Yield Curve Slope (10Y - 2Y)",
        units="Percentage points",
        units_short="pp",
        frequency_short="D",
        partition="daily",
        derived_from=[("ECB.YC.AAA.10Y", 1), ("ECB.YC.AAA.2Y", -1)],
        notes=(
            "ECB AAA yield curve slope: 10-year minus 2-year spot rates "
            "(percentage points). Euro area analogue to the US 2s10s."
        ),
    ),
]
# fmt: on

_BY_ID: dict[str, EcbSeriesSpec] = {s.series_id: s for s in ECB_SERIES_REGISTRY}


def get_registry() -> list[EcbSeriesSpec]:
    return ECB_SERIES_REGISTRY


def get_spec(series_id: str) -> EcbSeriesSpec:
    try:
        return _BY_ID[series_id]
    except KeyError:
        raise KeyError(f"Unknown ECB series: {series_id!r}")


def get_raw_series() -> list[EcbSeriesSpec]:
    """Return non-derived specs (those fetched directly from ECB SDW)."""
    return [s for s in ECB_SERIES_REGISTRY if not s.is_derived]


def get_derived_series() -> list[EcbSeriesSpec]:
    """Return derived specs (computed from raw series)."""
    return [s for s in ECB_SERIES_REGISTRY if s.is_derived]
