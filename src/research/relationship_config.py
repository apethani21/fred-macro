"""Named macro relationships: the single source of truth for what the research phase monitors.

Each Relationship carries:
  - the series involved (first element = primary / "total" for decompositions)
  - the relationship kind (drives which detectors run)
  - a basis string that feeds finding interpretation (required — no basis, no finding)
  - optional tags for grouping

Supported kinds
---------------
  "correlation"   — rolling Spearman correlation; structural break on the corr series
  "lead_lag"      — cross-correlation at lags; detect shift in peak-lag
  "spread"        — derived series (series[0] - series[1]); notable-move + structural break
  "decomposition" — additive (series[0] ≈ series[1] + series[2]); monitor contribution shares

Helper: relationships_as_pairs() extracts (a, b, label) tuples for the existing scan functions.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Relationship:
    name: str
    series: tuple[str, ...]
    kind: str
    basis: str
    tags: tuple[str, ...] = ()
    window: int | None = None    # rolling window override (uses frequency-appropriate default if None)
    max_lag: int | None = None   # lead_lag only


RELATIONSHIPS: tuple[Relationship, ...] = (

    # ── Nominal / real / inflation decomposition ─────────────────────────────

    Relationship(
        name="10Y nominal yield: real vs breakeven decomposition",
        series=("DGS10", "DFII10", "T10YIE"),
        kind="decomposition",
        basis=(
            "Fisher decomposition: nominal yield ≈ real yield + inflation expectations + liquidity premium. "
            "Decomposing recent moves reveals whether the driver is monetary tightening (real yield) "
            "or inflation re-pricing (breakeven). A breakeven-driven move implies market expectations "
            "are doing the tightening; a real-yield-driven move implies the Fed is."
        ),
        tags=("rates", "inflation"),
    ),
    Relationship(
        name="10Y nominal vs real yield",
        series=("DGS10", "DFII10"),
        kind="correlation",
        basis=(
            "When nominal and real yields decouple, the implied breakeven (their difference) is moving — "
            "usually driven by inflation expectations or the TIPS liquidity premium. "
            "Correlation near 1 = market pricing pure real moves (growth/policy). "
            "Lower correlation = inflation uncertainty is the driver."
        ),
        tags=("rates", "inflation"),
    ),
    Relationship(
        name="10Y breakeven vs realized core CPI",
        series=("T10YIE", "CPILFESL"),
        kind="correlation",
        basis=(
            "Market inflation expectations vs realized core CPI. "
            "Persistent breakeven overshoot of realized inflation signals either an "
            "inflation risk premium or supply-shock effects not expected to persist. "
            "Persistent undershoot signals anchoring drift or below-trend demand."
        ),
        tags=("inflation",),
    ),
    Relationship(
        name="5Y5Y forward breakeven vs core PCE",
        series=("T5YIFR", "PCEPILFE"),
        kind="correlation",
        basis=(
            "The 5Y5Y forward breakeven is the medium-term inflation anchor the Fed watches most closely. "
            "When it persistently diverges from realized core PCE, it signals either anchoring failure "
            "or risk-premium distortion rather than genuine inflation expectations."
        ),
        tags=("inflation",),
    ),

    # ── Yield curve slopes ────────────────────────────────────────────────────

    Relationship(
        name="2s10s vs 3m10y",
        series=("T10Y2Y", "T10Y3M"),
        kind="correlation",
        basis=(
            "Both measure curve slope but capture different policy horizons: "
            "2s10s reflects medium-term rate expectations (2 years ≈ 8 FOMC meetings forward), "
            "while 3m10y is the Estrella-Mishkin preferred recession predictor. "
            "Divergence signals the market is differentiating near-term policy (3m) "
            "from the medium-term rate path (2y)."
        ),
        tags=("rates", "curve"),
    ),

    # ── Credit ────────────────────────────────────────────────────────────────

    Relationship(
        name="HY-IG quality spread",
        series=("BAMLH0A0HYM2", "BAMLC0A0CM"),
        kind="spread",
        basis=(
            "Quality spread = HY OAS minus IG OAS. Strips out the rate and general credit-risk "
            "component common to both, isolating pure credit-quality discrimination. "
            "Widens as investors differentiate by quality (risk-off). "
            "Compresses in late-cycle credit-risk-appetite regimes and post-crisis recoveries."
        ),
        tags=("credit",),
    ),
    Relationship(
        name="VIX vs HY spreads",
        series=("VIXCLS", "BAMLH0A0HYM2"),
        kind="correlation",
        basis=(
            "Both measure risk aversion — equity implied vol and credit risk premium. "
            "High correlation is the norm; breakdown signals intra-asset divergence: "
            "credit complacency with equity fear, or equities pricing recovery before credit does. "
            "Historically, credit leads equities in deterioration; equities lead in recovery."
        ),
        tags=("credit", "volatility", "cross-asset"),
    ),
    Relationship(
        name="IG spreads vs 2s10s slope",
        series=("BAMLC0A0CM", "T10Y2Y"),
        kind="correlation",
        basis=(
            "IG spreads widen as the curve flattens and inverts in pre-recession environments; "
            "curve steepening in recovery compresses spreads as growth expectations revive. "
            "The sign can flip in bear-steepener regimes where rates sell off with credit."
        ),
        tags=("credit", "rates"),
    ),

    # ── Labor / inflation ─────────────────────────────────────────────────────

    Relationship(
        name="Phillips curve: unemployment vs core CPI",
        series=("UNRATE", "CPILFESL"),
        kind="correlation",
        basis=(
            "Classic Phillips curve: inverse relationship between labor market slack and inflation. "
            "Flattened significantly post-2010 (Hooper, Mishkin & Sufi, 2019, Brookings Papers). "
            "Re-steepened in 2021-22. Sign and magnitude are the defining empirical question "
            "for each Fed cycle."
        ),
        tags=("labor", "inflation"),
    ),
    Relationship(
        name="Wages lead core CPI",
        series=("AHETPI", "CPILFESL"),
        kind="lead_lag",
        basis=(
            "Wage growth (average hourly earnings) leads core CPI inflation by roughly 3-6 months "
            "as labor cost pressures pass through to prices. "
            "Central to the Fed's 'wage-price spiral' risk assessment. "
            "The lead time has compressed since 2021 as firms front-loaded price hikes."
        ),
        tags=("labor", "inflation"),
        max_lag=12,
    ),
    Relationship(
        name="Beveridge curve: job openings vs unemployment",
        series=("JTSJOL", "UNRATE"),
        kind="correlation",
        basis=(
            "Beveridge curve: outward shifts signal structural mismatch between "
            "job seekers and vacancies (skills gap, geographic mismatch). "
            "Inward shifts indicate cyclical improvement in matching efficiency. "
            "The 2021-22 outward shift was among the largest on record. "
            "The quits rate (embedded in JOLTS) is a leading wage growth indicator."
        ),
        tags=("labor",),
    ),
    Relationship(
        name="Consumer sentiment leads retail sales",
        series=("UMCSENT", "RRSFS"),
        kind="lead_lag",
        basis=(
            "UMich consumer sentiment leads real retail sales by 1-3 months. "
            "The inflation expectations component of UMich directly feeds Fed communication "
            "and breakeven pricing. Sentiment collapses that don't precede spending collapses "
            "signal 'talk-act' divergence — informative for the durability of consumption."
        ),
        tags=("growth", "consumer"),
        max_lag=6,
    ),

    # ── Activity ──────────────────────────────────────────────────────────────

    Relationship(
        name="Industrial production vs unemployment",
        series=("INDPRO", "UNRATE"),
        kind="correlation",
        basis=(
            "Industrial production is a coincident activity measure; unemployment is a lagging indicator. "
            "IP declines typically precede unemployment rises by 2-4 months, reflecting "
            "the lag between output cuts and layoffs. "
            "Breakdown: service-dominated economy means IP now captures a smaller share of employment."
        ),
        tags=("growth", "labor"),
    ),
    Relationship(
        name="10Y yield vs unemployment",
        series=("DGS10", "UNRATE"),
        kind="correlation",
        basis=(
            "Sign depends on regime: in expansion, tighter labor markets imply growth and rate-hike "
            "expectations → positive correlation. In recessions, loose labor → Fed cuts → negative. "
            "The sign flip is a useful regime marker and one of the CORE_PAIRS for this reason."
        ),
        tags=("rates", "labor"),
    ),
    Relationship(
        name="2s10s slope vs unemployment",
        series=("T10Y2Y", "UNRATE"),
        kind="lead_lag",
        basis=(
            "Yield curve inversion predicts rising unemployment; Estrella & Mishkin (1996, FRBNY) "
            "is the canonical reference. The lead time from inversion to unemployment rise "
            "has historically been 12-22 months and has lengthened post-2000."
        ),
        tags=("rates", "labor", "recession"),
        max_lag=24,
    ),
    Relationship(
        name="3m10y vs unemployment",
        series=("T10Y3M", "UNRATE"),
        kind="lead_lag",
        basis=(
            "Preferred recession predictor (Estrella & Mishkin 1996). "
            "The 3m maturity is the cleanest short-rate proxy (near-zero duration risk). "
            "Inversion has preceded every recession since 1968 with no false positives "
            "through 2019; the 2023 inversion is being monitored for the same signal."
        ),
        tags=("rates", "labor", "recession"),
        max_lag=24,
    ),

    # ── Cross-asset ───────────────────────────────────────────────────────────

    Relationship(
        name="VIX vs 10Y Treasury yield",
        series=("VIXCLS", "DGS10"),
        kind="correlation",
        basis=(
            "Pre-2022: negative correlation — safe-haven flows drove rates down alongside vol spikes. "
            "Post-2022: positive — inflation regime means both rose together (vol reflects "
            "rate uncertainty, not just equity fear). "
            "The sign flip in 2022 is the defining cross-asset regime shift of the cycle."
        ),
        tags=("volatility", "rates", "cross-asset"),
    ),
    Relationship(
        name="WTI crude vs 10Y breakeven inflation",
        series=("DCOILWTICO", "T10YIE"),
        kind="correlation",
        basis=(
            "Energy prices are a direct input to headline CPI and proxy for global demand. "
            "Breakevens lag crude by roughly 1 month on average. "
            "The relationship weakens when oil moves are supply-driven (OPEC cuts) "
            "vs demand-driven (growth shock). Supply-driven: inflation expectation channel. "
            "Demand-driven: growth channel competes with the inflation channel."
        ),
        tags=("commodities", "inflation"),
    ),
    Relationship(
        name="Broad dollar vs 10Y breakeven",
        series=("DTWEXBGS", "T10YIE"),
        kind="correlation",
        basis=(
            "Stronger dollar is disinflationary via import prices; tends to compress breakevens. "
            "The relationship can invert when dollar strength reflects a US rate premium "
            "(hawkish Fed hiking faster than peers) — in this case, high rates drive both "
            "the dollar and higher breakevens simultaneously."
        ),
        tags=("fx", "inflation"),
    ),
    Relationship(
        name="10Y yield vs broad dollar",
        series=("DGS10", "DTWEXBGS"),
        kind="correlation",
        basis=(
            "Interest rate differentials drive the dollar via uncovered interest parity. "
            "Wide US rate premium vs G10 attracts capital inflows, strengthening the dollar. "
            "Breakdown (high rates, weak dollar) typically signals foreign official selling "
            "of Treasuries or carry-unwind episodes."
        ),
        tags=("rates", "fx"),
    ),

    # ── European sovereign ────────────────────────────────────────────────────

    Relationship(
        name="BTP-Bund sovereign spread",
        series=("IRLTLT01ITM156N", "IRLTLT01DEM156N"),
        kind="spread",
        basis=(
            "Italian-German 10Y sovereign spread — the canonical eurozone stress indicator. "
            "Widens on ECB credibility concerns, Italian fiscal slippage, or political risk. "
            "The ECB's Transmission Protection Instrument (TPI), announced July 2022, "
            "created an implicit backstop; the mere announcement compressed spreads from ~250bp."
        ),
        tags=("rates", "europe", "sovereign"),
    ),
    Relationship(
        name="US 10Y vs Germany 10Y",
        series=("DGS10", "IRLTLT01DEM156N"),
        kind="correlation",
        basis=(
            "Global factor in long rates: US and Bund yields share a common global term premium "
            "and are highly correlated. Decoupling signals diverging monetary policy regimes "
            "(Fed vs ECB on different parts of the cycle), or diverging growth outlooks "
            "(US exceptionalism vs Europe stagnation)."
        ),
        tags=("rates", "europe", "global"),
    ),

    # ── ECB / euro area ───────────────────────────────────────────────────────

    Relationship(
        name="BTP-Bund spread: fragmentation risk monitor",
        series=("ECB.IT.10Y", "ECB.DE.10Y"),
        kind="spread",
        basis=(
            "The BTP-Bund 10Y spread is the canonical EA sovereign stress indicator. "
            "Widening reflects Italian fiscal risk, ECB credibility concerns, or political uncertainty. "
            "Key thresholds: ~130bp = stressed, ~200bp = elevated (TPI informal trigger zone; "
            "ECB announced TPI in July 2022 when spread reached ~230bp), ~300bp = crisis "
            "(2011-12 peak ~550bp). The announcement effect alone compressed spreads 50-80bp in 2022, "
            "confirming the backstop is priced-in before activation. "
            "Source: ECB SDW IRS flow (Maastricht convergence criterion rates)."
        ),
        tags=("rates", "europe", "sovereign", "ecb"),
    ),
    Relationship(
        name="ECB DFR vs US Fed funds: policy divergence",
        series=("ECB.DFR", "DFF"),
        kind="spread",
        basis=(
            "ECB deposit facility rate minus US effective fed funds, in percentage points. "
            "Positive = ECB tighter than Fed (rare; occurred briefly 1999-2000, never since 2009). "
            "Negative = Fed tighter (typical post-2022 as Fed front-loaded and ECB lagged). "
            "Divergence drives EUR/USD via uncovered interest parity and cross-border capital flows. "
            "A narrowing spread (ECB cutting faster or Fed holding) is USD-negative and supports "
            "EUR-denominated assets. The 2022-24 episode produced the widest Fed-over-ECB spread "
            "on record at ~250bp, reflecting the US supply-shock-driven inflation front-loading."
        ),
        tags=("policy", "europe", "ecb", "fx"),
    ),
    Relationship(
        name="ECB negotiated wages lead EA core HICP",
        series=("ECB.WAGES.NEG", "ECB.HICP.EA.CORE"),
        kind="lead_lag",
        basis=(
            "EA negotiated wages (quarterly) vs EA core HICP ex food/energy (monthly). "
            "Sector-level wage agreements (Austria, Germany, Netherlands) are set annually and "
            "index to past inflation. This creates a mechanical 2-4 quarter lead from wages to "
            "core services inflation — the 'second-round effects' channel the ECB watches most closely. "
            "When wage growth exceeds core inflation, it signals sticky services inflation ahead; "
            "when wages decelerate before core HICP, it signals disinflation is broadening. "
            "Source: ECB SDW INW flow (negotiated wages index, annual growth rate)."
        ),
        tags=("labor", "inflation", "europe", "ecb"),
        max_lag=6,
    ),
    Relationship(
        name="EA M3 money supply leads EA HICP",
        series=("ECB.M3.EA", "ECB.HICP.EA.TOTAL"),
        kind="lead_lag",
        basis=(
            "EA M3 annual growth rate vs EA HICP all-items YoY. "
            "Quantity theory (MV=PQ) predicts excess money growth leads inflation by 12-18 months. "
            "ECB targeted M3 growth (reference value 4.5% YoY) as its primary nominal anchor "
            "pre-2003 before shifting to two-pillar framework. The relationship weakened post-GFC "
            "as velocity declined with QE-driven reserve accumulation. "
            "The 2020-21 M3 surge (12%+ YoY) preceded the 2022 inflation spike — "
            "the lead-lag revived briefly. "
            "Source: ECB SDW BSI flow (seasonally adjusted M3, annual growth rate)."
        ),
        tags=("money_supply", "inflation", "europe", "ecb"),
        max_lag=18,
    ),

    # ── Fed funds / short end ─────────────────────────────────────────────────

    Relationship(
        name="3M T-bill vs Fed funds spread",
        series=("DTB3", "DFF"),
        kind="spread",
        basis=(
            "Under normal conditions, the 3M T-bill tracks the Fed funds rate closely "
            "(near-zero spread). Widening signals Treasury supply/demand imbalance, "
            "money market stress, or flight-to-safety into bills over reserves. "
            "During the 2008 and 2020 crises, T-bills traded well below Fed funds."
        ),
        tags=("rates", "money_market"),
    ),
    Relationship(
        name="Fed funds vs unemployment",
        series=("FEDFUNDS", "UNRATE"),
        kind="correlation",
        basis=(
            "Taylor rule core: the Fed tightens when unemployment falls below the natural rate. "
            "The elasticity varies by regime: pre-2008, the Fed responded to unemployment "
            "more aggressively; post-2008, forward guidance and QE flattened the reaction function."
        ),
        tags=("policy", "labor"),
    ),
)


def relationships_as_pairs(
    kinds: tuple[str, ...] | None = None,
) -> list[tuple[str, str, str]]:
    """Extract (series_a, series_b, label) for use with existing scan functions.

    Filters to the given kinds; if None, returns all bivariate relationships.
    Decompositions use the first two series only.
    """
    result: list[tuple[str, str, str]] = []
    for rel in RELATIONSHIPS:
        if kinds is not None and rel.kind not in kinds:
            continue
        if len(rel.series) >= 2:
            result.append((rel.series[0], rel.series[1], rel.name))
    return result
