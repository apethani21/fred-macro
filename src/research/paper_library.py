"""
Curated catalog of macro-relevant academic papers extracted from Justina's Quant Newsletter.
Papers are annotated for FRED implementability, methodology type, and macro relevance.

Groups:
  A - Inflation episode analysis & hedging
  B - Central bank / FOMC event studies
  C - Yield curve & term structure
  D - Cross-asset correlations & regime shifts
  E - Credit markets & fixed income
  F - Nowcasting & macro forecasting
  G - Risk appetite & financial conditions
  H - Currency & FX
  I - Commodity markets
  J - Macro × equity interactions
  K - Rate plumbing / SOFR
  L - European macro: ECB policy, EA financial stability & sovereign-bank nexus
  M - Systematic fixed income factors: Richardson / AQR research program
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Paper:
    title: str
    group: str                       # A–K
    methodology: str                 # key technique(s) used
    difficulty: int                  # 1 (simple) – 5 (very complex)
    fred_implementability: int       # 1 (not feasible) – 5 (directly implementable)
    macro_relevance: int             # 1 (marginal) – 5 (core macro)
    key_series: list[str] = field(default_factory=list)   # FRED series IDs
    notes: str = ""                  # non-obvious implementation or content notes
    newsletter_date: Optional[str] = None


# ---------------------------------------------------------------------------
# Group A: Inflation episode analysis & hedging
# ---------------------------------------------------------------------------
GROUP_A: list[Paper] = [
    Paper(
        title="100 Inflation Shocks: Seven Stylized Facts",
        group="A",
        methodology="Cross-country event study; identifies inflation episodes via threshold rules, computes distributional properties of post-shock paths",
        difficulty=3,
        fred_implementability=5,
        macro_relevance=5,
        key_series=["CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE", "MICH", "T10YIE"],
        notes="Seven empirical facts about inflation dynamics: persistence, wage pass-through, real-rate behavior. Directly replicable on FRED CPI/PCE history.",
        newsletter_date="10/13/2023",
    ),
    Paper(
        title="Inflation Hedging: A Dynamic Approach Using Online Prices",
        group="A",
        methodology="Rolling hedge ratio estimation; compares realized CPI vs. online price indices as inflation proxies for asset allocation",
        difficulty=3,
        fred_implementability=4,
        macro_relevance=4,
        key_series=["CPIAUCSL", "CPILFESL", "GOLDAMGBD228NLBM", "DCOILWTICO", "DGS10"],
        notes="TIPS and commodity exposures as inflation hedges; time-varying effectiveness. FRED covers TIPS breakevens (T5YIE, T10YIE) as proxy for expected inflation.",
        newsletter_date="9/8/2023",
    ),
    Paper(
        title="Inflation Hedging Tools—What Works and What Doesn't",
        group="A",
        methodology="Regression of asset returns on realized CPI; rolling correlation analysis across asset classes",
        difficulty=2,
        fred_implementability=5,
        macro_relevance=5,
        key_series=["CPIAUCSL", "T10YIE", "T5YIE", "GOLDAMGBD228NLBM", "DCOILWTICO", "DFII10"],
        notes="Directly implementable survey of hedge ratios. Breakeven inflation (T10YIE) as forward-looking inflation proxy. TIPS real yield (DFII10) separable from nominal.",
        newsletter_date="10/13/2023",
    ),
    Paper(
        title="The Swaps Strike Back: Evaluating Expectations of One-Year Inflation (Fed)",
        group="A",
        methodology="Decomposition of inflation swap rates into expected inflation and risk premium; comparison against SPF and Michigan Survey",
        difficulty=3,
        fred_implementability=4,
        macro_relevance=5,
        key_series=["T1YIE", "MICH", "EXPINF1YR", "CPIAUCSL"],
        notes="Fed staff note comparing inflation swaps vs. surveys as predictors. Inflation swap rates partially on FRED (T1YIE, T5YIE). SPF forecasts on Philadelphia Fed.",
        newsletter_date="10/13/2023",
    ),
    Paper(
        title="Financial Market Inflation Perceptions",
        group="A",
        methodology="Factor model extracting latent inflation expectations from market prices; compares survey vs. market-implied measures",
        difficulty=4,
        fred_implementability=4,
        macro_relevance=5,
        key_series=["T5YIE", "T10YIE", "MICH", "EXPINF1YR", "DFII10", "DGS10"],
        notes="Disaggregates market vs. survey inflation expectations. Breakeven decomposition using FRED TIPS data directly implementable.",
        newsletter_date="9/8/2023",
    ),
    Paper(
        title="Do Commodity Factors Work as Inflation Hedges and Safe Havens?",
        group="A",
        methodology="Regression and quantile analysis of commodity factor returns conditional on inflation regimes and crisis episodes",
        difficulty=3,
        fred_implementability=4,
        macro_relevance=4,
        key_series=["CPIAUCSL", "DCOILWTICO", "DCOILBRENTEU", "GOLDAMGBD228NLBM", "PPIACO"],
        notes="Tests commodity factors (momentum, basis, hedging pressure) as inflation hedges. FRED has spot prices; futures curves require external data.",
        newsletter_date="10/31/2023",
    ),
    Paper(
        title="Macroeconomic Momentum and Cross-Sectional Equity Market Indices",
        group="A",
        methodology="Constructs macro momentum signal from lagged macro data; tests predictive power for cross-country equity returns",
        difficulty=3,
        fred_implementability=4,
        macro_relevance=4,
        key_series=["CPIAUCSL", "INDPRO", "UNRATE", "PAYEMS", "GDPC1", "ICSA"],
        notes="Macro data revisions matter; ALFRED (FRED real-time database) useful here. Signal constructed from releases available at forecast time.",
        newsletter_date="10/13/2023",
    ),
    Paper(
        title="The Effectiveness of Ex Ante Real Earnings Yields in Forecasting Stock Market Returns",
        group="A",
        methodology="Predictive regression of equity returns on real earnings yield; controls for inflation regime",
        difficulty=2,
        fred_implementability=4,
        macro_relevance=4,
        key_series=["CPIAUCSL", "DGS10", "DFII10", "CAPE"],
        notes="CAPE (Shiller P/E) not on FRED but available from Shiller's website. Real rate (DFII10) on FRED. Inflation regime classification implementable from CPIAUCSL.",
        newsletter_date="9/8/2023",
    ),
    Paper(
        title="Real-Time Uncertainty in Estimating Bias in Macroeconomic Forecasts",
        group="A",
        methodology="Bootstrap confidence intervals for forecast bias using real-time vintage data; tests whether bias is statistically distinguishable from zero",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=4,
        key_series=["GDPC1", "CPIAUCSL", "UNRATE", "PAYEMS"],
        notes="Uses ALFRED real-time vintages. Highlights that apparent forecast bias often within sampling error—relevant for interpreting SPF errors.",
        newsletter_date="10/13/2023",
    ),
    Paper(
        title="Intuit QuickBooks Small Business Index: A New Employment Series for the US, Canada, and the UK",
        group="A",
        methodology="Index construction from administrative payroll data; validation against BLS establishment survey",
        difficulty=2,
        fred_implementability=3,
        macro_relevance=3,
        key_series=["PAYEMS", "USPRIV", "CEU0500000001"],
        notes="High-frequency employment signal from payroll data. FRED covers BLS establishment survey; QuickBooks index available separately. Useful for release-week comparisons.",
        newsletter_date="6/30/2023",
    ),
    Paper(
        title="Deviations from Rational Expectations and the Uncovered Interest Rate Parity Puzzle",
        group="A",
        methodology="GMM estimation decomposing UIP failure into systematic expectation errors vs. risk premium; time-varying component analysis",
        difficulty=4,
        fred_implementability=3,
        macro_relevance=4,
        key_series=["DTWEXBGS", "DFF", "FEDFUNDS"],
        notes="UIP failure is a classic macro puzzle. Time-varying risk premium vs. expectation error decomposition. Requires survey exchange rate expectations (not on FRED).",
        newsletter_date="6/30/2023",
    ),
]

# ---------------------------------------------------------------------------
# Group B: Central bank / FOMC event studies
# ---------------------------------------------------------------------------
GROUP_B: list[Paper] = [
    Paper(
        title="Fed Communication, News, Twitter, and Echo Chambers",
        group="B",
        methodology="Text analysis of FOMC communications + Twitter; measures dispersion of macro beliefs and its asset price impact",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=5,
        key_series=["FEDFUNDS", "DFF", "DGS2", "DGS10", "T10Y2Y"],
        notes="Communication clarity → reduced policy uncertainty → compressed term premium. FOMC calendars on Fed website; asset price impact via FRED rate data.",
        newsletter_date="6/20/2023",
    ),
    Paper(
        title="Mind Your Language: Market Responses to Central Bank Speeches",
        group="B",
        methodology="NLP event study; regresses intraday rate/equity moves on linguistic features of central bank speeches",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=5,
        key_series=["DFF", "DGS2", "DGS10", "VIXCLS"],
        notes="Hawks vs. doves measurable via Fed speech text (available on Fed website). Asset price responses computable from FRED daily data around speech dates.",
        newsletter_date="6/20/2023",
    ),
    Paper(
        title="US Interest Rate Surprises and Currency Returns",
        group="B",
        methodology="Event study around FOMC meetings; decomposes rate surprises using OIS curve; regresses G10 currency returns on surprise component",
        difficulty=3,
        fred_implementability=4,
        macro_relevance=5,
        key_series=["DFF", "DTWEXBGS", "DGS2", "DGS10", "SOFR"],
        notes="FOMC meeting dates from Fed website; rate surprises from OIS (partially on FRED via SOFR). FX returns from FRED trade-weighted indices.",
        newsletter_date="10/13/2023",
    ),
    Paper(
        title="Volatility Dynamics, Return Predictability, and Leverage in the Campbell-Cochrane Model",
        group="B",
        methodology="Structural estimation of Campbell-Cochrane habit model; tests whether surplus consumption ratio predicts equity returns",
        difficulty=5,
        fred_implementability=3,
        macro_relevance=4,
        key_series=["PCEC96", "SP500", "DGS10", "CPIAUCSL"],
        notes="Consumption data (PCEC96) on FRED; equity returns external. Surplus consumption ratio can be approximated from FRED consumption + equity data.",
        newsletter_date="8/14/2023",
    ),
    Paper(
        title="The Implied Views of Bond Traders on the Spot Equity Market",
        group="B",
        methodology="Extracts implicit equity return expectations from Treasury option prices via no-arbitrage; compares to realized equity returns",
        difficulty=5,
        fred_implementability=2,
        macro_relevance=4,
        key_series=["DGS10", "DGS2", "SP500", "VIXCLS"],
        notes="Requires Treasury option data not on FRED. FRED useful for validation: expected vs. realized bond-equity correlation, term structure data.",
        newsletter_date="6/30/2023",
    ),
    Paper(
        title="PEAD in Bond Markets based on Risk Information in Earnings Announcements",
        group="B",
        methodology="Event study of bond price drift following earnings surprises; decomposes into duration and credit spread channels",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=3,
        key_series=["BAMLC0A0CM", "BAMLH0A0HYM2", "DGS10", "DGS2"],
        notes="Bond market PEAD at aggregate level testable with FRED credit spread indices (BAMLC0A0CM). Firm-level requires Compustat/TRACE.",
        newsletter_date="8/14/2023",
    ),
    Paper(
        title="Investor Sentiment and Futures Market Mispricing",
        group="B",
        methodology="Baker-Wurgler sentiment index extended to futures markets; tests basis anomalies conditional on sentiment regime",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=3,
        key_series=["VIXCLS", "DCOILWTICO", "DGS10", "BAMLH0A0HYM2"],
        notes="Sentiment as regime variable for macro-asset relationships. FRED has VIX and credit spreads as fear/sentiment proxies.",
        newsletter_date="8/25/2023",
    ),
]

# ---------------------------------------------------------------------------
# Group C: Yield curve & term structure
# ---------------------------------------------------------------------------
GROUP_C: list[Paper] = [
    Paper(
        title="Forecasting the Yield Curve: The Role of Additional and Time-Varying Decay Parameters, Conditional Heteroscedasticity, and Macro-Economic Factors",
        group="C",
        methodology="Nelson-Siegel-Svensson with time-varying parameters; GARCH errors; augmented with macro factors (unemployment, inflation)",
        difficulty=4,
        fred_implementability=5,
        macro_relevance=5,
        key_series=["DGS1", "DGS2", "DGS5", "DGS10", "DGS30", "UNRATE", "CPIAUCSL", "T10Y2Y"],
        notes="Most FRED-implementable yield curve paper in the list. Full Treasury curve on FRED; macro augmentation with FRED data is direct. Statsmodels has Nelson-Siegel.",
        newsletter_date="8/14/2023",
    ),
    Paper(
        title="On the Predictability of Bonds",
        group="C",
        methodology="Return forecasting regressions for Treasury bonds; forward rate factors, macro variables, and volatility predictors",
        difficulty=3,
        fred_implementability=5,
        macro_relevance=5,
        key_series=["DGS1", "DGS2", "DGS5", "DGS10", "DGS30", "T10Y2Y", "T10Y3M", "INDPRO", "CPIAUCSL"],
        notes="Cochrane-Piazzesi-style return predictability. FRED Treasury data covers full history. Tests whether forward rates predict excess bond returns.",
        newsletter_date="5/12/2023",
    ),
    Paper(
        title="Bonds, Currencies and Expectational Errors",
        group="C",
        methodology="Decompose bond and FX risk premia using survey forecast errors; test whether deviations from rational expectations explain carry returns",
        difficulty=4,
        fred_implementability=3,
        macro_relevance=5,
        key_series=["DGS2", "DGS10", "T10Y2Y", "DTWEXBGS"],
        notes="Requires SPF or Consensus Economics forecasts for expectational error construction. FRED bond data for realized component.",
        newsletter_date="8/14/2023",
    ),
    Paper(
        title="Swap Rates and Term Structure Modelling",
        group="C",
        methodology="Affine term structure model fitted to swap rates; decomposition into OIS and credit/liquidity spread components",
        difficulty=5,
        fred_implementability=3,
        macro_relevance=4,
        key_series=["DGS2", "DGS5", "DGS10", "SOFR", "SOFR90DAYAVG"],
        notes="SOFR swap rates partially on FRED. Pre-LIBOR-to-SOFR transition data requires external sources. Model estimation requires specialized libraries.",
        newsletter_date="5/23/2023",
    ),
    Paper(
        title="Treasury Return Predictability and Investor Sentiment",
        group="C",
        methodology="Predictive regressions of Treasury excess returns on sentiment indices; tests sentiment as a complementary predictor to forward rates",
        difficulty=3,
        fred_implementability=4,
        macro_relevance=4,
        key_series=["DGS2", "DGS10", "DGS30", "T10Y2Y", "VIXCLS", "BAMLH0A0HYM2"],
        notes="VIX and credit spreads as sentiment proxies available on FRED. Forward rate predictors from FRED Treasury curve. Clean replication path.",
        newsletter_date="8/25/2023",
    ),
    Paper(
        title="Bond futures: Delivery Option with Term Structure Modelling",
        group="C",
        methodology="Prices Treasury futures delivery option using affine term structure model; quantifies cheapest-to-deliver optionality",
        difficulty=5,
        fred_implementability=2,
        macro_relevance=3,
        key_series=["DGS2", "DGS5", "DGS10", "DGS30"],
        notes="Futures pricing requires CTD basket data not on FRED. FRED Treasury yields as inputs to term structure calibration.",
        newsletter_date="9/8/2023",
    ),
    Paper(
        title="Discount Models",
        group="C",
        methodology="Reviews cash flow vs. discount rate decomposition of asset price variation; Shiller-style variance decomposition",
        difficulty=3,
        fred_implementability=4,
        macro_relevance=4,
        key_series=["DGS10", "DGS2", "SP500", "GDPC1", "CPIAUCSL"],
        notes="Foundational decomposition paper. FRED has all required data series. Discount rate variation dominates over short horizons; cash flow over long horizons.",
        newsletter_date="6/30/2023",
    ),
    Paper(
        title="Interest rate convexity in a Gaussian framework",
        group="C",
        methodology="Analytical convexity correction in Gaussian short-rate models; quantifies convexity bias in forward-swap rate relationships",
        difficulty=5,
        fred_implementability=2,
        macro_relevance=3,
        key_series=["DGS10", "DGS30", "SOFR"],
        notes="Highly technical rates pricing paper. Limited direct FRED implementation beyond validating model outputs against observed rates.",
        newsletter_date="7/28/2023",
    ),
    Paper(
        title="Test for Jumps in Yield Spreads",
        group="C",
        methodology="Non-parametric jump test applied to credit spread time series; identifies dates and magnitudes of yield spread discontinuities",
        difficulty=4,
        fred_implementability=5,
        macro_relevance=4,
        key_series=["BAMLH0A0HYM2", "BAMLC0A0CM", "T10Y2Y", "DGS10", "BAMLHYH0A0HYM2EY"],
        notes="Directly implementable on FRED credit spread series. Jump dates align with known macro events—useful for episodes.md. Statsmodels has BNS jump tests.",
        newsletter_date="9/29/2023",
    ),
    Paper(
        title="Interest Rate Volatility Risk and the Cross-Section of Expected Corporate Bond Returns",
        group="C",
        methodology="Risk factor model where interest rate volatility (MOVE index) is a priced factor for corporate bond returns",
        difficulty=4,
        fred_implementability=4,
        macro_relevance=4,
        key_series=["BAMLC0A0CM", "BAMLH0A0HYM2", "DGS10", "DGS2"],
        notes="MOVE index (Treasury vol) not on FRED but computable from Treasury option-implied vols. FRED credit spread indices as dependent variable.",
        newsletter_date="9/29/2023",
    ),
    Paper(
        title="Incorporating Short Data into Large Mixed-Frequency VARs for Regional Nowcasting",
        group="C",
        methodology="Mixed-frequency VAR (MF-VAR) with short-sample series via Bayesian shrinkage; applied to regional GDP nowcasting",
        difficulty=4,
        fred_implementability=4,
        macro_relevance=4,
        key_series=["GDPC1", "PAYEMS", "INDPRO", "RSAFS", "JTSJOL"],
        notes="MF-VAR approach applicable to any mix of daily/monthly/quarterly FRED series. Python statsmodels has VAR; MIDAS packages available separately.",
        newsletter_date="5/12/2023",
    ),
    Paper(
        title="PDSim: A Shiny App for Polynomial Diffusion Model Simulation and Estimation",
        group="C",
        methodology="Polynomial diffusion term structure model; moment-matching estimation; R Shiny interface for simulation",
        difficulty=5,
        fred_implementability=2,
        macro_relevance=2,
        key_series=["DGS2", "DGS5", "DGS10"],
        notes="More of a software/methodology paper. Low direct FRED implementability; interest is in the polynomial diffusion methodology for term structure.",
        newsletter_date="7/14/2023",
    ),
]

# ---------------------------------------------------------------------------
# Group D: Cross-asset correlations & regime shifts
# ---------------------------------------------------------------------------
GROUP_D: list[Paper] = [
    Paper(
        title="A Century of Global Equity Market Correlations",
        group="D",
        methodology="DCC-GARCH on long-run equity return series; regime classification via Markov switching; tests whether correlations are mean-reverting",
        difficulty=3,
        fred_implementability=4,
        macro_relevance=5,
        key_series=["SP500", "VIXCLS", "DGS10", "BAMLH0A0HYM2", "T10Y2Y"],
        notes="Long-run regime analysis with NBER recessions as regime labels. FRED has US series back to 1960s+; international equity via external sources. DCC-GARCH in arch (Python).",
        newsletter_date="7/14/2023",
    ),
    Paper(
        title="Network Momentum across Asset Classes",
        group="D",
        methodology="Builds return correlation network; constructs momentum signal using network neighbors' returns as predictors",
        difficulty=4,
        fred_implementability=3,
        macro_relevance=3,
        key_series=["SP500", "DCOILWTICO", "GOLDAMGBD228NLBM", "DGS10", "DTWEXBGS"],
        notes="Network structure reveals macro transmission channels. FRED has daily data for key nodes; full multi-asset implementation requires external futures data.",
        newsletter_date="8/25/2023",
    ),
    Paper(
        title="A Hidden Markov Model for Statistical Arbitrage in International Crude Oil Futures Markets",
        group="D",
        methodology="HMM regime identification on crude oil spread; pair-trading strategy triggered by regime transitions",
        difficulty=4,
        fred_implementability=4,
        macro_relevance=3,
        key_series=["DCOILWTICO", "DCOILBRENTEU", "DHHNGSP"],
        notes="HMM regime detection applicable to any FRED spread series. WTI-Brent spread is a macro-relevant signal (refinery capacity, geopolitics). hmmlearn Python package.",
        newsletter_date="9/29/2023",
    ),
    Paper(
        title="Forecasting Market Portfolio Returns: Sub-portfolios Approach",
        group="D",
        methodology="Disaggregates market returns into sector sub-portfolios; tests whether sector-level predictors improve aggregate forecasts",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=3,
        key_series=["SP500", "INDPRO", "PAYEMS", "GDPC1"],
        notes="Sector predictors require external sector data. Macro factor predictors (FRED) tested as drivers of sector vs. aggregate predictability.",
        newsletter_date="8/14/2023",
    ),
    Paper(
        title="Converting a Covariance Matrix From Local Currencies to a Common Currency",
        group="D",
        methodology="Exact analytical formula for FX-adjusted covariance matrix; avoids numerical integration",
        difficulty=4,
        fred_implementability=3,
        macro_relevance=3,
        key_series=["DTWEXBGS", "DTWEXEMEGS", "DTWEXAFEGS"],
        notes="Technical portfolio construction paper. Relevant for cross-border macro analysis where USD-denominated correlations differ from local-currency ones.",
        newsletter_date="6/9/2023",
    ),
    Paper(
        title="Permutation invariant Gaussian matrix models for financial correlation matrices",
        group="D",
        methodology="Random matrix theory approach to covariance estimation; permutation-invariant prior on correlation structure",
        difficulty=5,
        fred_implementability=2,
        macro_relevance=2,
        key_series=[],
        notes="Highly technical statistics paper. Low direct FRED implementability. Methodology could improve multi-series correlation estimates for research phase.",
        newsletter_date="6/20/2023",
    ),
    Paper(
        title="Default Clustering Risk Premium and its Cross-Market Asset Pricing Implications",
        group="D",
        methodology="Measures co-default risk premium; tests whether default clustering factor is priced in equities, bonds, and CDS",
        difficulty=4,
        fred_implementability=3,
        macro_relevance=4,
        key_series=["BAMLH0A0HYM2", "BAMLC0A0CM", "BAMLHYH0A0HYM2EY", "DRTSCILM"],
        notes="Systemic risk signal priced across asset classes. FRED credit spread indices as proxies; exact replication needs CDS microdata.",
        newsletter_date="9/8/2023",
    ),
]

# ---------------------------------------------------------------------------
# Group E: Credit markets & fixed income
# ---------------------------------------------------------------------------
GROUP_E: list[Paper] = [
    Paper(
        title="Investigating Divergence Measures with Credit Risk Models",
        group="E",
        methodology="KL-divergence and Hellinger distance applied to credit model calibration; comparison of structural vs. reduced-form models",
        difficulty=4,
        fred_implementability=3,
        macro_relevance=3,
        key_series=["BAMLH0A0HYM2", "BAMLC0A0CM"],
        notes="Model calibration paper. Divergence measures applicable to comparing FRED credit spread distributions across regimes.",
        newsletter_date="5/12/2023",
    ),
    Paper(
        title="CDS Tranches Priced Effectively with Novel Model (Danske)",
        group="E",
        methodology="Novel copula model for CDO/CDS tranche pricing; fitted to CDX index tranche quotes",
        difficulty=5,
        fred_implementability=1,
        macro_relevance=2,
        key_series=[],
        notes="Highly technical derivatives pricing paper. No direct FRED implementation. Background context for understanding credit market structure.",
        newsletter_date="5/23/2023",
    ),
    Paper(
        title="Media and Corporate Bond Market Momentum",
        group="E",
        methodology="NLP sentiment from financial news applied to corporate bonds; constructs news-based momentum factor",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=3,
        key_series=["BAMLC0A0CM", "BAMLH0A0HYM2", "BAMLHYH0A0HYM2EY"],
        notes="News sentiment not on FRED; bond spread indices are. Tests whether aggregate news tone predicts IG vs. HY spread changes.",
        newsletter_date="5/23/2023",
    ),
    Paper(
        title="Do Municipal Bond Investors Pay a Convenience Premium to Avoid Taxes?",
        group="E",
        methodology="Tax-adjusted yield comparison; decomposition of muni-Treasury spread into tax premium, liquidity, and credit components",
        difficulty=3,
        fred_implementability=5,
        macro_relevance=3,
        key_series=["WSLB10", "WSLB2", "DGS10", "DGS2"],
        notes="Muni yields on FRED (WSLB series). Tax premium measurement directly implementable. Relevant for understanding risk-free rate benchmarks.",
        newsletter_date="6/30/2023",
    ),
    Paper(
        title="Return-Based Anomalies in Corporate Bonds: Are They There?",
        group="E",
        methodology="Tests momentum, reversal, and value anomalies in IG and HY corporate bond indices; controls for duration and credit risk",
        difficulty=3,
        fred_implementability=4,
        macro_relevance=3,
        key_series=["BAMLC0A0CM", "BAMLH0A0HYM2", "BAMLHYH0A0HYM2EY", "BAMLCC0A0CMEY"],
        notes="ICE BofA indices on FRED. Momentum/reversal tests implementable at index level; firm-level requires TRACE.",
        newsletter_date="9/29/2023",
    ),
    Paper(
        title="Transaction Costs and Capacity of Systematic Corporate Bond Strategies",
        group="E",
        methodology="Estimates bid-ask spreads from transaction data; tests whether systematic bond strategies survive realistic execution costs",
        difficulty=3,
        fred_implementability=2,
        macro_relevance=2,
        key_series=["BAMLC0A0CM", "BAMLH0A0HYM2"],
        notes="Requires TRACE transaction-level data. FRED credit indices show gross returns; transaction cost adjustment methodology is the contribution.",
        newsletter_date="9/29/2023",
    ),
    Paper(
        title="The Corporate Bond Factor Zoo",
        group="E",
        methodology="Multiple testing correction applied to 150+ corporate bond factors; identifies which survive Bonferroni and FDR adjustments",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=3,
        key_series=["BAMLC0A0CM", "BAMLH0A0HYM2", "DGS10", "T10Y2Y"],
        notes="Harvey-Liu-Zhu style multiple testing paper for bonds. Useful reference for which macro factors are robust predictors of credit spreads.",
        newsletter_date="10/13/2023",
    ),
    Paper(
        title="The Relative-Relative Factor: A Multi-Dimensional Liquidity Pricing Approach",
        group="E",
        methodology="Constructs relative liquidity factor from bid-ask and volume data; tests pricing in corporate bonds",
        difficulty=4,
        fred_implementability=2,
        macro_relevance=2,
        key_series=["BAMLH0A0HYM2", "BAMLC0A0CM"],
        notes="Microstructure paper requiring transaction data. Limited FRED implementability beyond using aggregate credit spread levels.",
        newsletter_date="10/13/2023",
    ),
    Paper(
        title="Modeling liquidity in corporate bond markets: applications to price adjustments",
        group="E",
        methodology="Structural liquidity model calibrated to intraday corporate bond data; tests whether liquidity explains cross-sectional spread variation",
        difficulty=4,
        fred_implementability=2,
        macro_relevance=3,
        key_series=["BAMLC0A0CM", "BAMLH0A0HYM2"],
        notes="Microstructure-heavy. FRED aggregate spread indices useful as macro baseline; firm-level replication needs TRACE.",
        newsletter_date="9/8/2023",
    ),
    Paper(
        title="The Expected Returns of Agency MBS (DFA)",
        group="E",
        methodology="Decomposition of agency MBS returns into duration, convexity, and prepayment option components; predictive regression",
        difficulty=4,
        fred_implementability=3,
        macro_relevance=4,
        key_series=["MORTGAGE30US", "DGS10", "DGS5", "DRCCLACBS"],
        notes="Agency MBS are ~30% of Fed SOMA holdings. Prepayment risk driven by mortgage rate (MORTGAGE30US on FRED). Key for understanding Fed balance sheet dynamics.",
        newsletter_date="10/31/2023",
    ),
    Paper(
        title="Direct Lending Returns",
        group="E",
        methodology="Analysis of private credit fund returns; compares to public leveraged loan and HY bond benchmarks; adjusts for illiquidity and smoothing",
        difficulty=3,
        fred_implementability=2,
        macro_relevance=3,
        key_series=["BAMLH0A0HYM2", "DRTSCILM"],
        notes="Private credit data not on FRED. FRED HY spreads as public credit benchmark. Relevant as private credit has become a major part of credit markets.",
        newsletter_date="10/13/2023",
    ),
    Paper(
        title="Short of Cash? Convex Corporate Bond Selling By Mutual Funds and Price Fragility",
        group="E",
        methodology="Tests fire-sale externalities from mutual fund redemptions; flow-induced selling creates predictable price impact and subsequent reversal",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=4,
        key_series=["BAMLC0A0CM", "BAMLH0A0HYM2", "DRTSCILM", "VIXCLS"],
        notes="Mutual fund flows from ICI (not FRED). Credit spread response to flow-induced selling measurable on FRED indices. Relevant for spread widening episodes.",
        newsletter_date="10/31/2023",
    ),
    Paper(
        title="Intermediary Balance Sheet Constraints, Bond Mutual Funds' Strategies, and Bond Returns",
        group="E",
        methodology="Dealers' VaR constraints interact with fund flows; tests whether dealer balance sheet capacity predicts bond returns",
        difficulty=4,
        fred_implementability=3,
        macro_relevance=4,
        key_series=["BAMLC0A0CM", "BAMLH0A0HYM2", "H41RESPPALDKNWW", "WRMFSL"],
        notes="Fed H.4.1 (dealer repo) on FRED as intermediary constraint proxy. Links dealer constraints to credit spread movements—key for liquidity crisis analysis.",
        newsletter_date="10/31/2023",
    ),
    Paper(
        title="The Past, Present, and Future of Low-Risk Corporate Bonds",
        group="E",
        methodology="Risk-adjusted returns for investment-grade bonds; Fama-MacBeth cross-sectional regressions; beta-sorted portfolios",
        difficulty=3,
        fred_implementability=4,
        macro_relevance=3,
        key_series=["BAMLC0A0CM", "BAMLCC0A0CMEY", "DGS10", "T10Y2Y"],
        notes="Low-risk anomaly in corporate bonds. IG spread indices on FRED. Duration-adjusted return analysis implementable from FRED Treasury + credit data.",
        newsletter_date="10/13/2023",
    ),
    Paper(
        title="A Research-based Approach to Fixed Income Factor Portfolio Implementation",
        group="E",
        methodology="Survey of fixed income factor premia (carry, value, momentum, quality); practical implementation with transaction cost constraints",
        difficulty=3,
        fred_implementability=4,
        macro_relevance=4,
        key_series=["BAMLC0A0CM", "BAMLH0A0HYM2", "DGS10", "T10Y2Y", "T10Y3M"],
        notes="Carry = credit spread; value = OAS vs. history; momentum = spread change. All computable from FRED IG/HY indices. Practical reference for factor construction.",
        newsletter_date="9/8/2023",
    ),
]

# ---------------------------------------------------------------------------
# Group F: Nowcasting & macro forecasting
# ---------------------------------------------------------------------------
GROUP_F: list[Paper] = [
    Paper(
        title="Reintroducing the New York Fed Staff Nowcast",
        group="F",
        methodology="Dynamic factor model (DFM) with mixed-frequency data; real-time GDP nowcast from ~40 monthly/weekly indicators",
        difficulty=4,
        fred_implementability=5,
        macro_relevance=5,
        key_series=["GDPC1", "PAYEMS", "INDPRO", "RSAFS", "JTSJOL", "CPIAUCSL", "UNRATE", "ISM"],
        notes="Reference model for nowcasting. All input indicators on FRED. Python statsmodels has DFM; dfm package also available. Crucial for release-week emails.",
        newsletter_date="9/8/2023",
    ),
    Paper(
        title="The 'Hairy' Premium",
        group="F",
        methodology="Documents that equity risk premium estimates have wide confidence intervals ('hairy'); Monte Carlo simulation of estimation uncertainty",
        difficulty=2,
        fred_implementability=4,
        macro_relevance=4,
        key_series=["SP500", "DGS10", "DFII10", "GDPC1"],
        notes="ERP estimation uncertainty using FRED data directly. Relevant for calibrating how much weight to put on valuation-based signals. Simple but underappreciated.",
        newsletter_date="6/30/2023",
    ),
    Paper(
        title="Forecasting Half-Hourly Electricity Prices using a Mixed-Frequency Structural VAR Framework",
        group="F",
        methodology="Structural VAR with mixed-frequency (daily/intraday) inputs; identification via sign restrictions",
        difficulty=4,
        fred_implementability=2,
        macro_relevance=2,
        key_series=["DCOILWTICO", "NATURALGAS"],
        notes="MF-SVAR methodology transferable to macro context (e.g., daily financial data + monthly macro). Direct electricity application low macro relevance.",
        newsletter_date="6/20/2023",
    ),
    Paper(
        title="DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting",
        group="F",
        methodology="Meta-learning framework for updating stock trend models as market regimes shift; few-shot adaptation",
        difficulty=5,
        fred_implementability=1,
        macro_relevance=1,
        key_series=[],
        notes="ML paper with minimal macro content. Included in original Macro/FICC section but marginal relevance. Skip for implementation.",
        newsletter_date="6/30/2023",
    ),
    Paper(
        title="Replication of BBW Factors with WRDS Data",
        group="F",
        methodology="Replication of Bai-Bali-Wen factor model using WRDS; validates out-of-sample predictability of macro factors for equities",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=3,
        key_series=["INDPRO", "CPIAUCSL", "T10Y2Y", "BAMLH0A0HYM2", "UNRATE"],
        notes="BBW factors include macro variables available on FRED. Replication confirms which macro series are robust equity return predictors.",
        newsletter_date="6/30/2023",
    ),
]

# ---------------------------------------------------------------------------
# Group G: Risk appetite & financial conditions
# ---------------------------------------------------------------------------
GROUP_G: list[Paper] = [
    Paper(
        title="Predicting Financial Crises: The Role of Asset Prices",
        group="G",
        methodology="Logit/probit crisis prediction model using equity, bond, and credit variables as predictors; AUROC evaluation against pure credit models",
        difficulty=3,
        fred_implementability=5,
        macro_relevance=5,
        key_series=["SP500", "VIXCLS", "BAMLH0A0HYM2", "T10Y2Y", "T10Y3M", "BAMLC0A0CM", "DRTSCILM"],
        notes="Most FRED-implementable crisis-prediction paper. All predictors on FRED. NBER recession dates as binary outcome. Recession probability directly computable with sklearn.",
        newsletter_date="8/25/2023",
    ),
    Paper(
        title="More than Words: Twitter Chatter and Financial Market Sentiment",
        group="G",
        methodology="High-frequency sentiment index from Twitter; Granger causality tests against VIX and bond yields",
        difficulty=3,
        fred_implementability=2,
        macro_relevance=3,
        key_series=["VIXCLS", "DGS10", "BAMLH0A0HYM2"],
        notes="Twitter data not on FRED; FRED provides the asset price dependent variables. Granger test methodology directly applicable to FRED series.",
        newsletter_date="6/20/2023",
    ),
    Paper(
        title="Uncertainty Sentiment on Twitter and Financial Markets",
        group="G",
        methodology="Uncertainty index from Twitter text; correlates with VIX, EPU, and cross-asset returns",
        difficulty=3,
        fred_implementability=2,
        macro_relevance=3,
        key_series=["VIXCLS", "USEPUINDXD", "DGS10", "BAMLH0A0HYM2"],
        notes="Baker-Bloom-Davis EPU index (USEPUINDXD) on FRED as alternative uncertainty measure. Cross-asset correlation methodology implementable.",
        newsletter_date="7/14/2023",
    ),
    Paper(
        title="Investor Sentiment and Futures Market Mispricing",
        group="G",
        methodology="Conditional regression of basis on sentiment regime; tests whether sentiment predicts futures mispricing across asset classes",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=3,
        key_series=["VIXCLS", "USEPUINDXD", "BAMLH0A0HYM2"],
        notes="EPU index (USEPUINDXD) and VIX as sentiment proxies on FRED. Regime-conditional analysis implementable.",
        newsletter_date="8/25/2023",
    ),
    Paper(
        title="Latent News Factors: Understanding Stock Market Fluctuations",
        group="G",
        methodology="Latent factor model extracting news-driven components of daily equity returns; PCA on news-sorted return series",
        difficulty=4,
        fred_implementability=3,
        macro_relevance=3,
        key_series=["SP500", "VIXCLS", "DGS10", "BAMLH0A0HYM2"],
        notes="FRED has daily VIX and yield data as factor candidates. Full news text required for replication; FRED data useful for validation of macro factor loadings.",
        newsletter_date="9/29/2023",
    ),
    Paper(
        title="Weather Variance Risk Premia",
        group="G",
        methodology="Option-implied vs. realized variance decomposition for weather derivatives; estimates risk premium embedded in weather futures",
        difficulty=4,
        fred_implementability=1,
        macro_relevance=2,
        key_series=["DCOILWTICO"],
        notes="Methodology (variance risk premium decomposition) transferable to FRED-available assets (VIX vs. realized equity vol). Direct weather application low FRED relevance.",
        newsletter_date="10/13/2023",
    ),
    Paper(
        title="The 'Hairy' Premium",
        group="G",
        methodology="Bootstrapped confidence intervals for ERP; shows that 95% CIs on ERP estimates span several hundred basis points",
        difficulty=2,
        fred_implementability=4,
        macro_relevance=4,
        key_series=["SP500", "DGS10", "DFII10"],
        notes="Risk premium estimation uncertainty—relevant for calibrating confidence in valuation signals. ERP from FRED data directly. Underappreciated practical insight.",
        newsletter_date="6/30/2023",
    ),
]

# ---------------------------------------------------------------------------
# Group H: Currency & FX
# ---------------------------------------------------------------------------
GROUP_H: list[Paper] = [
    Paper(
        title="Macro-Based Factors for the Cross-Section of Currency Returns",
        group="H",
        methodology="Fama-MacBeth regressions of G10 currency returns on macro variables; identifies robust macro predictors of FX cross-section",
        difficulty=3,
        fred_implementability=4,
        macro_relevance=4,
        key_series=["DTWEXBGS", "DTWEXEMEGS", "DFF", "CPIAUCSL", "GDPC1"],
        notes="Interest rate differential (Fed funds vs. foreign) on FRED. Trade-weighted dollar indices (DTWEXBGS) on FRED. Cross-country macro data requires IMF/OECD.",
        newsletter_date="5/23/2023",
    ),
    Paper(
        title="Currency Risk Premiums: A Multi-horizon Perspective",
        group="H",
        methodology="Decompose FX risk premiums across horizons (1m, 3m, 1y, 5y); tests whether carry trade profitability varies by horizon",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=4,
        key_series=["DTWEXBGS", "DFF", "DGS2", "DGS10"],
        notes="Interest rate curve shape (FRED) predicts currency returns at different horizons—linked to Fed policy expectations. US rate differential key driver.",
        newsletter_date="7/14/2023",
    ),
    Paper(
        title="Deviations from Rational Expectations and the Uncovered Interest Rate Parity Puzzle",
        group="H",
        methodology="GMM decomposition of UIP failure into risk premium vs. expectational error; time-varying component model",
        difficulty=4,
        fred_implementability=3,
        macro_relevance=4,
        key_series=["DFF", "DTWEXBGS", "DTWEXEMEGS"],
        notes="UIP failure = rates don't predict FX moves as theory says. Time-varying risk premium measurable from FRED rate spreads. Survey expectations from external sources.",
        newsletter_date="6/30/2023",
    ),
    Paper(
        title="US Interest Rate Surprises and Currency Returns",
        group="H",
        methodology="FOMC event study; OIS-identified rate surprises regressed on G10 currency returns in narrow windows",
        difficulty=3,
        fred_implementability=4,
        macro_relevance=5,
        key_series=["DFF", "SOFR", "DTWEXBGS"],
        notes="Rate surprises (Fed funds vs. OIS expectation) with FRED data; FOMC dates from Fed website. FX response measurable from DTWEXBGS daily.",
        newsletter_date="10/13/2023",
    ),
    Paper(
        title="Skewness Risk Premia and the Cross-Section of Currency Returns",
        group="H",
        methodology="Option-implied skewness as predictor of currency excess returns; risk premium decomposition using FX options",
        difficulty=4,
        fred_implementability=2,
        macro_relevance=3,
        key_series=["DTWEXBGS"],
        notes="FX option data required (not on FRED). Methodology for skewness-as-risk-premium transferable to equity options (FRED has VIX term structure).",
        newsletter_date="10/13/2023",
    ),
    Paper(
        title="Price Formation in the Foreign Exchange Market",
        group="H",
        methodology="Structural model of FX price discovery; identifies whether price-formation is driven by order flow, inventory, or information",
        difficulty=4,
        fred_implementability=2,
        macro_relevance=3,
        key_series=["DTWEXBGS", "DTWEXEMEGS"],
        notes="Microstructure paper requiring tick-level FX data. FRED provides daily FX for macro context. Useful background on FX market mechanics.",
        newsletter_date="9/8/2023",
    ),
    Paper(
        title="The Trade Imbalance Network and Currency Returns",
        group="H",
        methodology="Global trade flow network; constructs centrality measure predicting currency returns; tests against carry and momentum",
        difficulty=4,
        fred_implementability=3,
        macro_relevance=4,
        key_series=["BOPGSTB", "DTWEXBGS", "IEABC"],
        notes="US trade balance (BOPGSTB) on FRED. International trade network requires UN Comtrade. Macro interpretation: trade-deficit currencies underperform.",
        newsletter_date="8/14/2023",
    ),
    Paper(
        title="Sovereign Momentum Currency Returns",
        group="H",
        methodology="Time-series momentum applied to sovereign currency returns; performance vs. carry and value factors",
        difficulty=2,
        fred_implementability=3,
        macro_relevance=3,
        key_series=["DTWEXBGS", "DTWEXEMEGS", "DTWEXAFEGS"],
        notes="FRED trade-weighted indices proximate; individual currency time series require external FX data. Momentum methodology directly implementable.",
        newsletter_date="5/12/2023",
    ),
    Paper(
        title="Sovereign CDS and Currency Carry Trades",
        group="H",
        methodology="Carry trade returns explained by sovereign CDS spreads (default risk); tests whether credit risk pricing subsumes currency risk premium",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=4,
        key_series=["DTWEXBGS", "BAMLH0A0HYM2"],
        notes="EM sovereign CDS not on FRED. US HY spreads as credit regime indicator. Links currency carry to default risk—key macro mechanism.",
        newsletter_date="5/12/2023",
    ),
    Paper(
        title="An Integrated Approach to Currency Factor Investing (Robeco, Invesco, BlackRock)",
        group="H",
        methodology="Combines carry, momentum, value, and quality factors for G10 currencies; dynamic factor weighting",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=4,
        key_series=["DFF", "DGS2", "DGS10", "DTWEXBGS"],
        notes="Carry factor = interest rate differential (FRED). Value = PPP deviation. Momentum from FX return series. Industry collaboration paper—strong practitioner orientation.",
        newsletter_date="5/12/2023",
    ),
]

# ---------------------------------------------------------------------------
# Group I: Commodity markets
# ---------------------------------------------------------------------------
GROUP_I: list[Paper] = [
    Paper(
        title="Exploiting the dynamics of commodity futures curves",
        group="I",
        methodology="Kalman-filtered Nelson-Siegel model for commodity futures curves; level/slope/curvature factor trading strategies",
        difficulty=4,
        fred_implementability=3,
        macro_relevance=3,
        key_series=["DCOILWTICO", "DCOILBRENTEU", "DHHNGSP", "GOLDAMGBD228NLBM"],
        notes="Spot prices on FRED; futures curve requires CME/ICE data. Curve shape (backwardation vs. contango) as macro signal implementable from spot + near-futures.",
        newsletter_date="8/14/2023",
    ),
    Paper(
        title="Forecasting Oil Prices With Penalized Regressions, Variance Risk Premia and Google Data",
        group="I",
        methodology="LASSO/ridge regression combining financial (VRP), search, and macro predictors for oil price forecasting",
        difficulty=3,
        fred_implementability=4,
        macro_relevance=4,
        key_series=["DCOILWTICO", "DCOILBRENTEU", "VIXCLS", "INDPRO", "PAYEMS"],
        notes="Oil VRP = implied vol (OVX, not on FRED) minus realized vol (computable from FRED daily prices). Google Trends requires separate API. Macro predictors on FRED.",
        newsletter_date="8/14/2023",
    ),
    Paper(
        title="Commodity ETF Arbitrage: Futures-backed versus Physical-backed ETFs",
        group="I",
        methodology="Pairs trading between physical and futures-backed commodity ETFs; quantifies rolling cost and arbitrage spread dynamics",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=3,
        key_series=["GOLDAMGBD228NLBM", "DCOILWTICO"],
        notes="ETF data not on FRED. FRED has gold and oil spot prices. Contango/backwardation dynamic (roll cost) measureable from FRED spot vs. theoretical futures.",
        newsletter_date="7/28/2023",
    ),
    Paper(
        title="Generic Forward Curve Dynamics for Commodity Derivatives",
        group="I",
        methodology="HJM-style model for commodity forward curves; stochastic volatility and seasonal adjustment",
        difficulty=5,
        fred_implementability=2,
        macro_relevance=2,
        key_series=["DCOILWTICO", "DHHNGSP"],
        notes="Technical derivatives pricing paper. Limited direct FRED implementation. Natural gas seasonality (DHHNGSP) observable from FRED spot data.",
        newsletter_date="6/30/2023",
    ),
    Paper(
        title="Predicting Crude Oil Returns and Trading Position: Evidence from News Sentiment",
        group="I",
        methodology="NLP sentiment from oil-related news as predictor; directional accuracy and Sharpe ratio evaluation",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=3,
        key_series=["DCOILWTICO", "DCOILBRENTEU", "INDPRO"],
        notes="News data external; oil price dependent variable on FRED. Macro predictors (INDPRO, ISM) as alternative signals implementable from FRED.",
        newsletter_date="6/30/2023",
    ),
    Paper(
        title="An Evaluation of the Skewness Model on 22 Commodities Futures (Quantpedia)",
        group="I",
        methodology="Tests option-implied skewness as return predictor across 22 commodity futures; cross-sectional regressions",
        difficulty=3,
        fred_implementability=2,
        macro_relevance=2,
        key_series=["DCOILWTICO", "GOLDAMGBD228NLBM"],
        notes="Options data required. FRED covers spot prices. Methodology (skewness as risk premium) transferable to equity options.",
        newsletter_date="7/14/2023",
    ),
    Paper(
        title="Commodity Futures Trading at the 52-Week High and Low",
        group="I",
        methodology="52-week high/low as anchor for commodity futures trend signals; tests returns after reaching new highs/lows",
        difficulty=2,
        fred_implementability=4,
        macro_relevance=3,
        key_series=["DCOILWTICO", "GOLDAMGBD228NLBM", "DCOILBRENTEU", "DHHNGSP", "PPIACO"],
        notes="Directly implementable from FRED daily price series. 52-week high/low signals computable from any FRED daily series—energy, metals, PPI.",
        newsletter_date="8/25/2023",
    ),
    Paper(
        title="Newswire Tone-Overlay Commodity Portfolios",
        group="I",
        methodology="News tone overlay applied to commodity momentum; adjusts position sizing based on sentiment",
        difficulty=3,
        fred_implementability=2,
        macro_relevance=2,
        key_series=["DCOILWTICO", "GOLDAMGBD228NLBM"],
        notes="News data external. FRED commodity prices as base series. Momentum signal without news overlay fully implementable from FRED.",
        newsletter_date="8/25/2023",
    ),
    Paper(
        title="Ensembling Arimax Model in Algorithmic Investment Strategies on Commodities Market",
        group="I",
        methodology="Ensemble of ARIMAX models for commodity price forecasting; feature selection from macro indicators",
        difficulty=3,
        fred_implementability=4,
        macro_relevance=3,
        key_series=["DCOILWTICO", "DCOILBRENTEU", "INDPRO", "PAYEMS", "CPIAUCSL"],
        notes="ARIMAX with FRED macro regressors directly implementable. Validates which macro releases predict commodity price movements.",
        newsletter_date="10/31/2023",
    ),
    Paper(
        title="A Hidden Markov Model for Statistical Arbitrage in International Crude Oil Futures Markets",
        group="I",
        methodology="HMM identifies crude oil market regimes; spread trading triggered by regime transitions in WTI-Brent relationship",
        difficulty=4,
        fred_implementability=4,
        macro_relevance=3,
        key_series=["DCOILWTICO", "DCOILBRENTEU"],
        notes="WTI and Brent both on FRED. HMM regime detection with hmmlearn directly implementable. WTI-Brent spread as macro signal (logistics, geopolitics).",
        newsletter_date="9/29/2023",
    ),
]

# ---------------------------------------------------------------------------
# Group J: Macro × equity interactions
# ---------------------------------------------------------------------------
GROUP_J: list[Paper] = [
    Paper(
        title="Macroeconomic Momentum and Cross-Sectional Equity Market Indices",
        group="J",
        methodology="Macro momentum signal (trend in key macro releases) predicts next-month country equity index returns; cross-sectional horse race",
        difficulty=3,
        fred_implementability=4,
        macro_relevance=5,
        key_series=["CPIAUCSL", "INDPRO", "UNRATE", "PAYEMS", "GDPC1", "ICSA"],
        notes="FRED monthly releases as momentum signals. Cross-country requires international data; US-only version directly implementable. Tests whether macro trend predicts equity.",
        newsletter_date="10/13/2023",
    ),
    Paper(
        title="The Effectiveness of Ex Ante Real Earnings Yields in Forecasting Stock Market Returns",
        group="J",
        methodology="Real earnings yield (EY - inflation) in predictive regressions; tests whether real vs. nominal yields matter for equity return forecasting",
        difficulty=2,
        fred_implementability=4,
        macro_relevance=4,
        key_series=["CPIAUCSL", "DFII10", "DGS10"],
        notes="Real rate (DFII10) on FRED. Earnings yield requires Shiller CAPE. Inflation regime from FRED CPI. Tests inflation's role in equity valuation.",
        newsletter_date="9/8/2023",
    ),
    Paper(
        title="PEAD in Bond Markets based on Risk Information in Earnings Announcements",
        group="J",
        methodology="Post-earnings announcement drift in credit spreads; risk channel (earnings surprise signals credit quality change) vs. attention channel",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=3,
        key_series=["BAMLC0A0CM", "BAMLH0A0HYM2", "DGS10"],
        notes="Firm-level earnings requires Compustat. Aggregate PEAD in FRED credit indices testable. Tests equity-credit information linkage.",
        newsletter_date="8/14/2023",
    ),
    Paper(
        title="Equity Fragility (JoPM)",
        group="J",
        methodology="Measures equity market fragility from order flow imbalance; tests whether fragility predicts subsequent returns and volatility",
        difficulty=4,
        fred_implementability=2,
        macro_relevance=3,
        key_series=["SP500", "VIXCLS"],
        notes="Microstructure measure; TAQ data required. VIX as macro-available fragility proxy. Useful context for equity-macro linkages.",
        newsletter_date="5/12/2023",
    ),
    Paper(
        title="Volatility Dynamics, Return Predictability, and Leverage in the Campbell-Cochrane Model",
        group="J",
        methodology="Calibrated habit model; tests whether surplus consumption predicts equity returns and explains leverage effect",
        difficulty=5,
        fred_implementability=3,
        macro_relevance=5,
        key_series=["PCEC96", "SP500", "DGS10", "CPIAUCSL"],
        notes="Habit model links macro (consumption) to equity risk premium. Real consumption growth (PCEC96) on FRED. Surplus consumption proxy computable from FRED data.",
        newsletter_date="8/14/2023",
    ),
    Paper(
        title="The Implied Views of Bond Traders on the Spot Equity Market",
        group="J",
        methodology="Extracts equity-market return expectations implied by Treasury bond option pricing; no-arbitrage cross-market model",
        difficulty=5,
        fred_implementability=2,
        macro_relevance=4,
        key_series=["DGS10", "DGS2", "SP500"],
        notes="Treasury options required (not on FRED). Bond-equity correlation (DGS10 vs. SP500 daily) as simplified FRED-based version of cross-asset view extraction.",
        newsletter_date="6/30/2023",
    ),
]

# ---------------------------------------------------------------------------
# Group K: Rate plumbing / SOFR
# ---------------------------------------------------------------------------
GROUP_K: list[Paper] = [
    Paper(
        title="Term Structure Modeling of SOFR: Evaluating the Importance of Scheduled Jumps",
        group="K",
        methodology="Affine term structure model with scheduled jump components at FOMC dates; calibrated to SOFR futures curve",
        difficulty=5,
        fred_implementability=3,
        macro_relevance=3,
        key_series=["SOFR", "SOFR90DAYAVG", "DFF"],
        notes="SOFR on FRED. Jump-at-FOMC component captures rate decision timing. Key for SOFR derivatives pricing; relevant for understanding rate expectations.",
        newsletter_date="5/12/2023",
    ),
    Paper(
        title="Hedging Term SOFR Fixing via SOFR Futures",
        group="K",
        methodology="Convexity adjustment between Term SOFR and compounded SOFR; optimal futures hedge ratio derivation",
        difficulty=4,
        fred_implementability=3,
        macro_relevance=3,
        key_series=["SOFR", "SOFR90DAYAVG", "DFF", "FEDFUNDS"],
        notes="SOFR and 90-day compounded SOFR on FRED. Term SOFR (CME) requires separate data. Key for understanding LIBOR-to-SOFR transition mechanics. (Also listed as 10/31/2023 — same paper.)",
        newsletter_date="9/29/2023",
    ),
]

# ---------------------------------------------------------------------------
# Group L: European macro — ECB policy, EA financial stability & sovereign-bank nexus
# ---------------------------------------------------------------------------
GROUP_L: list[Paper] = [
    Paper(
        title="Asset Purchase Programmes and Financial Markets: Lessons from the Euro Area (Altavilla, Carboni & Motto, ECB WP 1864)",
        group="L",
        methodology="High-frequency event study around ECB APP announcement dates; panel regression of yield changes on purchase amounts; decomposes impact by tenor and country",
        difficulty=3,
        fred_implementability=4,
        macro_relevance=5,
        key_series=["ECB.DFR", "ECB.YC.AAA.10Y", "ECB.BTPBUND.SPREAD", "ECB.DE.10Y", "ECB.IT.10Y"],
        notes="Reference paper for ECB QE transmission. BTP-Bund compression was the dominant channel for each APP announcement, not Bund yields per se. Replicable event-study on ECB.* daily series around GC meeting dates.",
        newsletter_date=None,
    ),
    Paper(
        title="Self-Fulfilling Crises in the Eurozone: An Empirical Test (De Grauwe & Ji, JIMF 2013)",
        group="L",
        methodology="Panel OLS decomposing sovereign spreads into fundamentals (deficit, debt/GDP, current account) vs. residual panic component; shows residual dominates in 2010-12 crisis window",
        difficulty=2,
        fred_implementability=4,
        macro_relevance=5,
        key_series=["ECB.BTPBUND.SPREAD", "ECB.IT.10Y", "ECB.DE.10Y"],
        notes="Foundational paper establishing that EA spreads have a self-fulfilling panic component separable from fiscal fundamentals. OMT/TPI effectiveness depends entirely on this mechanism — if spreads were driven by fundamentals alone, backstop tools wouldn't work. Spread decomposition replicable on ECB.* series.",
        newsletter_date=None,
    ),
    Paper(
        title="The Sovereign-Bank Diabolic Loop and ESBies (Brunnermeier et al., AER P&P 2016)",
        group="L",
        methodology="Stylized model of doom loop equilibria; Monte Carlo simulation of European Safe Bond tranching to break the loop; calibrated to EA sovereign-bank data",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=5,
        key_series=["ECB.BTPBUND.SPREAD", "ECB.IT.10Y", "ECB.DE.10Y"],
        notes="Canonical formal treatment of the doom loop. Key empirical implication: European bank equity in stressed-sovereign countries amplifies BTP-Bund moves, and the elasticity rises non-linearly as spreads cross ~200bp. ESBies proposed as structural fix. Feeds directly into doom loop concept in knowledge/concepts.md.",
        newsletter_date=None,
    ),
    Paper(
        title="A Pyrrhic Victory? Bank Bailouts and Sovereign Credit Risk (Acharya, Drechsler & Schnabl, JF 2014)",
        group="L",
        methodology="Difference-in-differences around EA bank bailout announcements 2008-09; CDS spreads of sovereign widen while bailed-out bank CDS tightens; IV using pre-crisis bank fragility",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=5,
        key_series=["ECB.BTPBUND.SPREAD", "ECB.IT.10Y"],
        notes="Cleanest empirical evidence that bank bailouts transfer risk to sovereign balance sheet — the mechanism underlying the doom loop. CDS microdata external; BTP-Bund spread as macro proxy captures the aggregate signal. Strong reference for any email covering EA financial stability.",
        newsletter_date=None,
    ),
    Paper(
        title="Evaluating the Impact of Unconventional Monetary Policy Measures: The ECB's Securities Markets Programme (Eser & Schwaab, JFE 2016)",
        group="L",
        methodology="Panel regression of daily yield changes on SMP purchase volumes (2010-12); IV strategy using ECB operational constraints; counterfactual spread estimates",
        difficulty=4,
        fred_implementability=3,
        macro_relevance=4,
        key_series=["ECB.BTPBUND.SPREAD", "ECB.IT.10Y", "ECB.DE.10Y"],
        notes="Most credible causal estimate of ECB bond-buying impact. SMP (pre-OMT) compressed IT-DE spread by ~2bp per €1bn purchased; announcement effect >> stock effect. Directly informs interpretation of OMT and TPI: credibility of backstop matters more than volume. Replication requires SMP purchase data (ECB statistical warehouse).",
        newsletter_date=None,
    ),
    Paper(
        title="Bank Exposures and Sovereign Stress Transmission (Altavilla, Pagano & Simonelli, Review of Finance 2017)",
        group="L",
        methodology="Supervisory bank balance-sheet data matched to stock returns; cross-sectional regression of bank equity returns on domestic sovereign holding × spread change; placebo tests using foreign sovereign holdings",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=5,
        key_series=["ECB.BTPBUND.SPREAD", "ECB.IT.10Y"],
        notes="Definitive paper on home-bias doom loop mechanism. Italian banks holding BTP lose ~2-3% equity value for each 100bp widening in BTP-Bund. European bank equity is therefore a real-time doom loop monitor. The home-bias channel is distinct from the NIM channel; both run through ECB.BTPBUND.SPREAD.",
        newsletter_date=None,
    ),
    Paper(
        title="Measuring Euro Area Monetary Policy (Altavilla, Brugnolini, Gürkaynak, Motto & Ragusa, JME 2022 / ECB WP 2250)",
        group="L",
        methodology="High-frequency identification of ECB policy shocks from intraday OIS moves around GC meetings; principal components decompose into target-rate surprise, forward-guidance shock, and QE shock (three orthogonal factors)",
        difficulty=4,
        fred_implementability=4,
        macro_relevance=5,
        key_series=["ECB.DFR", "ECB.ESTR", "ECB.YC.AAA.2Y", "ECB.YC.AAA.10Y"],
        notes="ECB analogue of Gürkaynak-Sack-Swanson (2005). Three-factor decomposition: (1) target rate surprise, (2) forward guidance, (3) QE. Daily ECB.* series allow coarser replication of forward-guidance and QE shocks. EA-MPD dataset publicly available from ECB website. Essential for any ECB event study.",
        newsletter_date=None,
    ),
    Paper(
        title="ECB Unconventional Monetary Policy: Market Impact and International Spillovers (Fratzscher, Lo Duca & Straub, IMF Economic Review 2016)",
        group="L",
        methodology="Event study of ECB non-standard measures 2008-12; GVAR model for international transmission; tests EUR/USD, EM capital flows, and US rate spillbacks",
        difficulty=3,
        fred_implementability=4,
        macro_relevance=5,
        key_series=["ECB.DFR", "ECB.BTPBUND.SPREAD", "DTWEXBGS"],
        notes="ECB QE → EUR depreciation → US financial conditions tightening via exchange rate channel; but also positive EA demand spillover. EUR/USD and DTWEXBGS are FRED-available outputs. Key for framing ECB/Fed divergence emails: ECB easing is not a zero-sum game for US assets.",
        newsletter_date=None,
    ),
    Paper(
        title="The Interdependence of Wages and Prices in the Euro Area (Bobeica, Ciccarelli & Vansteenkiste, ECB WP 2292, 2019)",
        group="L",
        methodology="Structural VAR with sign restrictions; decomposes EA wage-price dynamics into demand, supply, and autonomous wage shocks; impulse responses at 1-8 quarter horizons",
        difficulty=4,
        fred_implementability=4,
        macro_relevance=5,
        key_series=["ECB.WAGES.NEG", "ECB.HICP.EA.CORE", "ECB.HICP.EA.TOTAL"],
        notes="Key paper for EA wage-price spiral risk. Lead time from negotiated wage growth to core HICP is 2-3 quarters; pass-through is asymmetric — stronger in demand-driven upswings than in supply-shock episodes. Directly implementable as lead-lag detector on ECB.WAGES.NEG → ECB.HICP.EA.CORE. ECB WP freely available.",
        newsletter_date=None,
    ),
    Paper(
        title="Monetary Policy and Bank Profitability in a Low Interest Rate Environment (Altavilla, Boucinha & Peydró, Economic Policy 2018)",
        group="L",
        methodology="Panel of European listed banks 1999-2016; OLS and IV regressions of NIM, fee income, and total profitability on short-term rates; tests tiering threshold non-linearity",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=4,
        key_series=["ECB.DFR", "ECB.ESTR"],
        notes="NIRP compresses NIM but higher credit volume and fee income partially offset; net effect on bank profitability is negative but smaller than feared. Tiering (2019 ECB decision) partially neutralizes NIRP drag. European bank equity sensitivity to ECB rate cycle is asymmetric: cuts hurt less than hikes help. Useful for framing bank equity × ECB policy emails.",
        newsletter_date=None,
    ),
    Paper(
        title="The Mystery of the Printing Press: Monetary Policy and Self-Fulfilling Debt Crises (Corsetti & Dedola, JEEA 2016)",
        group="L",
        methodology="Two-country DSGE model; derives conditions under which central bank asset purchases can eliminate self-fulfilling equilibria without fiscal dominance; analytical characterization of credibility threshold",
        difficulty=4,
        fred_implementability=2,
        macro_relevance=4,
        key_series=["ECB.BTPBUND.SPREAD", "ECB.DFR"],
        notes="Theoretical foundation for why OMT/TPI credibility — not volume — is the operative mechanism. Key insight: a sufficiently committed CB can eliminate the bad equilibrium without ever purchasing a single bond, provided markets believe the commitment. Empirical test: BTP-Bund snap-back after Draghi's 'whatever it takes' (2012-07-26) required no actual SMP purchases.",
        newsletter_date=None,
    ),
    Paper(
        title="Why Bank Capital Matters for Monetary Policy (Gambacorta & Shin, JFI 2018)",
        group="L",
        methodology="Panel VAR on international bank balance sheets; tests whether bank equity capital ratio amplifies or attenuates monetary transmission to lending; structural identification via bank-level heterogeneity",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=4,
        key_series=["ECB.DFR", "ECB.M3.EA"],
        notes="Better-capitalized banks transmit ECB rate cuts more fully through the lending channel; undercapitalized banks hoard liquidity. Relevant for EA bank equity analysis: bank equity ratio → M3 credit multiplier → HICP. Directly applicable to post-2022 context where ECB hikes interact with varying EA bank capital buffers.",
        newsletter_date=None,
    ),
    Paper(
        title="The European Sovereign Debt Crisis (Lane, JEP 2012)",
        group="L",
        methodology="Empirical synthesis; documents current account imbalance accumulation 1999-2008, sovereign spread dynamics 2008-12, and bank-sovereign linkages across periphery; regression of spread levels on fiscal and external fundamentals",
        difficulty=2,
        fred_implementability=3,
        macro_relevance=5,
        key_series=["ECB.BTPBUND.SPREAD", "ECB.IT.10Y", "ECB.DE.10Y"],
        notes="Best single readable reference for the 2010-12 EA crisis. Documents the full causal chain: EMU → current account imbalances → sovereign fragility → bank stress → doom loop. Essential framing for any email covering BTP-Bund, TARGET2, or fragmentation risk. JEP articles are freely available and citable as primary source.",
        newsletter_date=None,
    ),
]

# ---------------------------------------------------------------------------
# Group M: Systematic fixed income factors — Richardson / AQR research program
# ---------------------------------------------------------------------------
GROUP_M: list[Paper] = [
    Paper(
        title="The Credit Risk Premium (Asvanunt & Richardson, JFI Winter 2017)",
        group="M",
        methodology="Long-run regression of corporate bond excess returns over Treasuries 1936-2014; decomposes total credit spread into expected default loss and risk premium component; compares to equity risk premium over same history",
        difficulty=2,
        fred_implementability=4,
        macro_relevance=5,
        key_series=["BAMLC0A0CM", "BAMLH0A0HYM2", "BAMLHYH0A0HYM2EY", "BAMLCC0A0CMEY", "DGS10"],
        notes="137bp average credit risk premium over full history; additive to equity and term premia. The credit-market ERP equivalent — the empirical anchor for understanding whether credit spread compensation is 'real.' ICE BofA FRED indices cover 1996+; pre-1996 requires Ibbotson/Lehman data. Key finding directly quotable in emails as a specific, non-obvious empirical claim.",
        newsletter_date=None,
    ),
    Paper(
        title="Common Factors in Corporate Bond Returns (Israel, Palhares & Richardson, JOIM 2018)",
        group="M",
        methodology="Fama-MacBeth cross-sectional regressions of monthly corporate bond excess returns on four characteristics: carry (OAS level), defensive (low duration-times-spread beta), momentum (trailing 6-month return), value (OAS vs. model-implied fair-value spread); robust to transaction costs and subsamples",
        difficulty=3,
        fred_implementability=4,
        macro_relevance=5,
        key_series=["BAMLC0A0CM", "BAMLH0A0HYM2", "BAMLHYH0A0HYM2EY", "BAMLCC0A0CMEY", "T10Y2Y"],
        notes="Four-factor model for corporate bond cross-section. Not subsumed by equity factor equivalents or traditional credit/term premia. Key macro-actionable finding: OAS level (carry) is the strongest single predictor — high-spread bonds persistently outperform after default adjustment. ICE BofA rating-tier indices on FRED allow rough carry-factor replication (IG vs. HY OAS spread as carry proxy). Core reference for any email on credit factor premia.",
        newsletter_date=None,
    ),
    Paper(
        title="Style Investing in Fixed Income (Brooks, Palhares & Richardson, JPM 2018)",
        group="M",
        methodology="Applies four-factor framework (value, momentum, carry, defensive) to both government bonds and corporate bonds; tests diversification benefit of multi-asset cross-fixed-income factor portfolio; low correlation to term and credit risk premia confirmed",
        difficulty=2,
        fred_implementability=5,
        macro_relevance=5,
        key_series=["DGS1", "DGS2", "DGS5", "DGS10", "DGS30", "T10Y2Y", "T10Y3M", "BAMLC0A0CM", "BAMLH0A0HYM2"],
        notes="Best single entry point into Richardson's program — covers govies and credit together. Style factors are uncorrelated to term and credit risk premia; genuinely alpha-generating rather than risk-premium harvesting. For government bonds: carry = yield level, defensive = shorter duration / lower vol, momentum = trailing yield-adjusted return. All four government bond factors directly computable from FRED Treasury curve. Highest FRED implementability of all Richardson papers.",
        newsletter_date=None,
    ),
    Paper(
        title="(Il)liquidity Premium in Credit Markets: A Myth? (Palhares & Richardson, JFI Winter 2019)",
        group="M",
        methodology="Sort IG corporate bonds by liquidity proxies (amount outstanding, bond age, coupon frequency); compute gross and net-of-estimated-transaction-cost excess returns; test whether liquidity premium survives duration and credit risk adjustment",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=4,
        key_series=["BAMLC0A0CM", "BAMLH0A0HYM2", "BAMLCC0A0CMEY"],
        notes="Counterintuitive finding: illiquid bonds have modestly higher spreads but materially higher realized volatility; after risk adjustment the liquidity premium largely disappears. Practical implication: 'liquidity premium' in credit is mostly compensation for hidden risk, not a free lunch. FRED ICE BofA indices represent the liquid segment; full replication requires TRACE. Non-obvious result worth a dedicated email.",
        newsletter_date=None,
    ),
    Paper(
        title="Looking under the Hood of Active Credit Managers (Palhares & Richardson, FAJ 2020)",
        group="M",
        methodology="Regression of 219 credit hedge fund and 96 credit mutual fund returns on four systematic factors (carry, defensive, momentum, value); R² decomposition; compares alpha between fund types",
        difficulty=2,
        fred_implementability=3,
        macro_relevance=4,
        key_series=["BAMLC0A0CM", "BAMLH0A0HYM2", "BAMLHYH0A0HYM2EY"],
        notes="Credit hedge funds: 7-12% of return variation explained by systematic factors. Credit mutual funds: only 2%. Most 'active' credit alpha is unattributed credit beta or hidden systematic tilts. Equities analogy: the credit-market version of showing that most active equity managers are closet indexers. FRED credit indices serve as factor proxies for the four characteristics. Relevant for understanding what active credit managers actually do vs. claim.",
        newsletter_date=None,
    ),
    Paper(
        title="(Systematic) Investing in Emerging Market Debt (Brooks, Richardson & Xu, Working Paper 2020)",
        group="M",
        methodology="Applies four-factor framework (carry, defensive, momentum, value) to hard-currency EM sovereign and quasi-sovereign bonds; transaction-cost and liquidity adjusted long-only implementation; information ratio evaluation",
        difficulty=3,
        fred_implementability=3,
        macro_relevance=4,
        key_series=["BAMLH0A0HYM2", "DTWEXEMEGS", "DTWEXBGS"],
        notes="Same four factors work in EM debt; carry is strongest. Long-only factor portfolio achieved IR > 1. FRED has limited EM bond data; full replication requires EMBIG or Bloomberg EM indices. Cross-asset macro relevance: EM debt carry is tightly correlated with DXY strength and US HY risk appetite (BAMLH0A0HYM2 as proxy) — a relationship directly testable on FRED series.",
        newsletter_date=None,
    ),
    Paper(
        title="Value Investing in Credit Markets (Correia, Richardson & Tuna, RAS 2012)",
        group="M",
        methodology="Cross-sectional regression of CDS and corporate bond returns on fundamental accounting signals: book leverage, earnings quality (accruals), Altman Z-score, distance-to-default; tests whether accounting-based value predicts returns independently of market-based signals",
        difficulty=3,
        fred_implementability=2,
        macro_relevance=3,
        key_series=["BAMLC0A0CM", "BAMLH0A0HYM2"],
        notes="Foundational paper predating the AQR factor program. Accounting signals (Z-score, earnings quality) predict bond returns; fair-value spread deviation used in later papers as the 'value' factor is rooted here. Low FRED implementability — requires Compustat firm-level accounting data. Important conceptual anchor: 'value' in credit is not just mean-reversion of spread level but deviation from fundamentals-implied fair value.",
        newsletter_date=None,
    ),
    Paper(
        title="Computing Corporate Bond Returns: A Word (or Two) of Caution (Andreani, Palhares & Richardson, RAS 2023)",
        group="M",
        methodology="Documents return computation bias from failing to separate rate and spread components; rate component is negatively correlated to spread component; provides WRDS-based correction; shows contamination is worst for IG bonds and time-series regressions",
        difficulty=2,
        fred_implementability=3,
        macro_relevance=3,
        key_series=["BAMLC0A0CM", "BAMLCC0A0CMEY", "DGS10", "DGS2"],
        notes="Methodological paper critical for any credit return research. Rate and spread components of total bond return are negatively correlated — total return conflates duration risk with credit risk. In rising-rate environments this systematically overstates credit spread return. Practical fix for FRED users: use OAS change (not total return) as the credit-specific signal. FRED provides OAS separately from yield-to-maturity for ICE BofA series.",
        newsletter_date=None,
    ),
]

# ---------------------------------------------------------------------------
# Full catalog
# ---------------------------------------------------------------------------
ALL_PAPERS: list[Paper] = (
    GROUP_A + GROUP_B + GROUP_C + GROUP_D + GROUP_E +
    GROUP_F + GROUP_G + GROUP_H + GROUP_I + GROUP_J + GROUP_K + GROUP_L + GROUP_M
)

_GROUP_MAP: dict[str, list[Paper]] = {
    "A": GROUP_A, "B": GROUP_B, "C": GROUP_C, "D": GROUP_D,
    "E": GROUP_E, "F": GROUP_F, "G": GROUP_G, "H": GROUP_H,
    "I": GROUP_I, "J": GROUP_J, "K": GROUP_K, "L": GROUP_L,
    "M": GROUP_M,
}

GROUP_LABELS: dict[str, str] = {
    "A": "Inflation episode analysis & hedging",
    "B": "Central bank / FOMC event studies",
    "C": "Yield curve & term structure",
    "D": "Cross-asset correlations & regime shifts",
    "E": "Credit markets & fixed income",
    "F": "Nowcasting & macro forecasting",
    "G": "Risk appetite & financial conditions",
    "H": "Currency & FX",
    "I": "Commodity markets",
    "J": "Macro × equity interactions",
    "K": "Rate plumbing / SOFR",
    "L": "European macro: ECB policy, EA financial stability & sovereign-bank nexus",
    "M": "Systematic fixed income factors: Richardson / AQR research program",
}


# ---------------------------------------------------------------------------
# Filter / rank helpers
# ---------------------------------------------------------------------------

def filter_papers(
    group: str | None = None,
    min_fred_implementability: int = 1,
    min_macro_relevance: int = 1,
    max_difficulty: int = 5,
) -> list[Paper]:
    """Return papers matching the given criteria, sorted by (fred_implementability + macro_relevance) desc."""
    results = ALL_PAPERS
    if group is not None:
        results = [p for p in results if p.group == group]
    results = [
        p for p in results
        if p.fred_implementability >= min_fred_implementability
        and p.macro_relevance >= min_macro_relevance
        and p.difficulty <= max_difficulty
    ]
    return sorted(results, key=lambda p: -(p.fred_implementability + p.macro_relevance))


def top_implementable(n: int = 10) -> list[Paper]:
    """Top-N papers ranked by combined FRED implementability + macro relevance."""
    return filter_papers(min_fred_implementability=4, min_macro_relevance=4)[:n]


def papers_for_series(series_id: str) -> list[Paper]:
    """Return papers whose key_series list includes the given FRED series ID."""
    sid = series_id.upper()
    return [p for p in ALL_PAPERS if sid in [s.upper() for s in p.key_series]]


def papers_by_group() -> dict[str, list[Paper]]:
    """Return all papers organized by group letter."""
    return {g: list(papers) for g, papers in _GROUP_MAP.items()}


def search_papers(query: str) -> list[Paper]:
    """Case-insensitive search across title, methodology, and notes."""
    q = query.lower()
    return [
        p for p in ALL_PAPERS
        if q in p.title.lower() or q in p.methodology.lower() or q in p.notes.lower()
    ]


def search_ssrn(title: str) -> str:
    """
    Stub: return SSRN search URL for the given paper title.
    Full-text SSRN lookup should be done via web_fetch in the research phase.
    """
    import urllib.parse
    query = urllib.parse.quote_plus(title)
    return f"https://papers.ssrn.com/sol3/results.cfm?RequestTimeout=50000&txtKey_Words={query}&form_name=journalBrowse&journal_id=&nxtpage=1"


def print_summary() -> None:
    """Print a summary of the catalog."""
    print(f"Total papers: {len(ALL_PAPERS)}\n")
    for g, label in GROUP_LABELS.items():
        papers = _GROUP_MAP[g]
        avg_fi = sum(p.fred_implementability for p in papers) / len(papers)
        avg_mr = sum(p.macro_relevance for p in papers) / len(papers)
        print(f"  {g}: {label} ({len(papers)} papers) — avg implementability={avg_fi:.1f}, avg relevance={avg_mr:.1f}")
    print()
    print("Top 10 by implementability + relevance:")
    for p in top_implementable(10):
        print(f"  [{p.group}] {p.title[:70]} (FI={p.fred_implementability}, MR={p.macro_relevance})")


if __name__ == "__main__":
    print_summary()
