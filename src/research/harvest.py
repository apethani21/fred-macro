"""Proactive knowledge harvesting from Fed research publications.

Scrapes index pages for FEDS Notes, Liberty Street Economics, and SF Fed Economic
Letter, filters for macro-relevant articles, and uses Claude to extract empirical
findings for the knowledge base.

Extracted findings are written to knowledge/findings.md with kind='harvested_source'.
A harvest log at state/harvest_log.jsonl tracks processed URLs to avoid re-fetching.

Usage:
    from src.research.harvest import harvest_all
    n = harvest_all(dry_run=False, max_per_source=10)
"""
from __future__ import annotations

import json
import logging
import re
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from src.ingest.paths import STATE_DIR, KNOWLEDGE_DIR
from src.research.findings import Finding, write_findings_md, existing_slugs

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = logging.getLogger(__name__)

HARVEST_LOG = STATE_DIR / "harvest_log.jsonl"

# ── Keywords for relevance filtering ──────────────────────────────────────────
# Article must mention at least one of these (case-insensitive) to be processed.
_RELEVANCE_KEYWORDS = [
    "inflation", "cpi", "pce", "deflation", "price level", "price index",
    "yield curve", "treasury yield", "interest rate", "fed funds", "sofr",
    "term premium", "term spread", "inversion",
    "unemployment", "labor market", "employment", "payroll", "nonfarm",
    "wage", "earnings",
    "gdp", "output gap", "recession", "expansion", "growth",
    "monetary policy", "fomc", "federal reserve", "rate hike", "rate cut",
    "quantitative easing", "balance sheet", "reserves",
    "credit spread", "high yield", "investment grade", "corporate bond",
    "vix", "volatility", "financial conditions",
    "breakeven", "tips", "inflation expectations",
    "retail sales", "consumer spending", "consumption",
    "industrial production", "manufacturing",
    "housing", "mortgage",
    "tariff", "trade policy",  # currently very relevant
]

_FETCH_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) research-bot"}


# ── Article metadata ───────────────────────────────────────────────────────────

@dataclass
class ArticleMeta:
    title: str
    url: str
    pub_date: str   # YYYY-MM-DD or YYYY-MM
    summary: str
    source: str     # "feds_notes" | "liberty_street" | "sf_fed"


# ── Index scrapers ─────────────────────────────────────────────────────────────

def index_feds_notes(max_articles: int = 20) -> list[ArticleMeta]:
    """Scrape the FEDS Notes index for recent articles."""
    base = "https://www.federalreserve.gov"
    url = f"{base}/econres/notes/feds-notes/default.htm"
    try:
        resp = requests.get(url, timeout=15, headers=_FETCH_HEADERS)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("FEDS Notes index fetch failed: %s", exc)
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    seen: set[str] = set()
    articles: list[ArticleMeta] = []

    # Find all links ending in .html that are FEDS Notes article pages
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not re.search(r"/feds-notes/.+-\d{8}\.html?$", href):
            continue
        full_url = base + href if href.startswith("/") else href
        if full_url in seen:
            continue
        seen.add(full_url)

        title = a.get_text(strip=True)
        if not title or title.lower() in ("read more", ""):
            continue

        # Try to extract date from URL (YYYYMMDD suffix)
        m = re.search(r"(\d{8})\.html?$", href)
        pub_date = ""
        if m:
            ds = m.group(1)
            pub_date = f"{ds[:4]}-{ds[4:6]}-{ds[6:]}"

        # Summary: look for next <p> sibling in parent container
        parent = a.find_parent(["div", "li", "article", "section"])
        summary = ""
        if parent:
            p = parent.find("p")
            if p:
                summary = p.get_text(strip=True)[:300]

        articles.append(ArticleMeta(title=title, url=full_url, pub_date=pub_date,
                                    summary=summary, source="feds_notes"))
        if len(articles) >= max_articles:
            break

    logger.info("FEDS Notes index: %d articles found", len(articles))
    return articles


def index_liberty_street(max_articles: int = 15) -> list[ArticleMeta]:
    """Parse the Liberty Street Economics RSS feed for recent articles."""
    rss_url = "https://libertystreeteconomics.newyorkfed.org/feed/"
    try:
        resp = requests.get(rss_url, timeout=15, headers=_FETCH_HEADERS)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Liberty Street RSS fetch failed: %s", exc)
        return []

    soup = BeautifulSoup(resp.text, "xml")
    articles: list[ArticleMeta] = []
    for item in soup.find_all("item"):
        title = item.find("title")
        link = item.find("link")
        pub_date_tag = item.find("pubDate")
        description = item.find("description")
        if not (title and link):
            continue

        raw_date = pub_date_tag.get_text(strip=True) if pub_date_tag else ""
        pub_date = ""
        try:
            pub_date = datetime.strptime(raw_date, "%a, %d %b %Y %H:%M:%S %z").strftime("%Y-%m-%d")
        except ValueError:
            pass

        summary_raw = description.get_text(strip=True) if description else ""
        # Strip HTML from summary
        summary = BeautifulSoup(summary_raw, "lxml").get_text(strip=True)[:300]

        articles.append(ArticleMeta(
            title=title.get_text(strip=True),
            url=link.get_text(strip=True),
            pub_date=pub_date,
            summary=summary,
            source="liberty_street",
        ))
        if len(articles) >= max_articles:
            break

    logger.info("Liberty Street RSS: %d articles found", len(articles))
    return articles


def index_sf_fed(max_articles: int = 15) -> list[ArticleMeta]:
    """Scrape the SF Fed Economic Letter index."""
    index_url = "https://www.frbsf.org/economic-research/publications/economic-letter/"
    try:
        resp = requests.get(index_url, timeout=15, headers=_FETCH_HEADERS)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("SF Fed index fetch failed: %s", exc)
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    seen: set[str] = set()
    articles: list[ArticleMeta] = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/economic-letter/20" not in href and "/publications/economic-letter/20" not in href:
            continue
        if href in seen:
            continue
        title = a.get_text(strip=True)
        if not title or "read" in title.lower() or len(title) < 10:
            continue
        seen.add(href)

        # Extract date from URL path: .../2026/04/slug/
        m = re.search(r"/(\d{4})/(\d{2})/", href)
        pub_date = f"{m.group(1)}-{m.group(2)}" if m else ""

        articles.append(ArticleMeta(title=title, url=href, pub_date=pub_date,
                                    summary="", source="sf_fed"))
        if len(articles) >= max_articles:
            break

    logger.info("SF Fed index: %d articles found", len(articles))
    return articles


# ── Relevance filter ───────────────────────────────────────────────────────────

def is_relevant(article: ArticleMeta) -> bool:
    """True if the article title or summary contains at least one relevance keyword."""
    haystack = (article.title + " " + article.summary).lower()
    return any(kw in haystack for kw in _RELEVANCE_KEYWORDS)


# ── Harvest log (dedup by URL) ─────────────────────────────────────────────────

def _load_harvested_urls() -> set[str]:
    if not HARVEST_LOG.exists():
        return set()
    urls = set()
    for line in HARVEST_LOG.read_text().splitlines():
        try:
            urls.add(json.loads(line)["url"])
        except (json.JSONDecodeError, KeyError):
            pass
    return urls


def _log_harvested(url: str, n_findings: int) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    entry = {"url": url, "harvested_at": date.today().isoformat(), "n_findings": n_findings}
    with HARVEST_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Article extraction (Claude) ────────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
You are an expert macro economist reading Fed research publications to extract empirical findings
for a macro education knowledge base.

Given the full text of a research article, extract any findings that meet ALL of these criteria:
1. SPECIFIC: a precise empirical claim with numbers, dates, and methodology.
2. NON-OBVIOUS: something a well-informed generalist would not already know.
3. RELEVANT: relates to macro series, monetary policy mechanics, labor markets, inflation dynamics,
   credit markets, yield curve behavior, or financial conditions.
4. AUDITABLE: cites a data source (BLS, BEA, FRED, Fed survey) or a specific calculation.

For each qualifying finding, output a JSON object. If an article has no qualifying findings
(e.g., it's purely methodological, it only restates textbook results, or it's a speech with
no quantitative claims), return an empty list.

OUTPUT FORMAT (raw JSON array, no code fence):
[
  {
    "title": "One-sentence specific title for the finding",
    "claim": "2-3 sentences. The specific empirical claim with numbers.",
    "evidence": "The statistic(s) from the article: values, windows, series referenced.",
    "interpretation": "What this might mean. Explicitly marked as interpretive.",
    "fred_series": ["SERIES_ID1", "SERIES_ID2"],
    "relevance": "brief (non-obvious)" | "brief (contextual)" | "foundational (methodology)"
  }
]

Return [] if no qualifying findings.
"""


def _fetch_article_text(url: str) -> str:
    """Fetch and return cleaned article text (max 12000 chars)."""
    from src.research.enrich import _execute_web_fetch
    return _execute_web_fetch(url, max_chars=12000)


def extract_findings_from_article(article: ArticleMeta) -> list[Finding]:
    """Use Claude to extract empirical findings from one article."""
    import anthropic
    import os

    key_file = Path.home() / "keys" / "anthropic" / "key.txt"
    api_key = key_file.read_text().strip() if key_file.exists() else os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("Anthropic API key not found.")

    client = anthropic.Anthropic(api_key=api_key)

    text = _fetch_article_text(article.url)
    if not text or len(text) < 200:
        logger.info("Article too short or fetch failed: %s", article.url)
        return []

    user_msg = (
        f"SOURCE: {article.source}\n"
        f"TITLE: {article.title}\n"
        f"DATE: {article.pub_date}\n"
        f"URL: {article.url}\n\n"
        f"ARTICLE TEXT:\n{text}"
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=[{"type": "text", "text": _EXTRACT_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as exc:
        logger.warning("Claude extraction failed for %s: %s", article.url, exc)
        return []

    raw = ""
    for block in response.content:
        if block.type == "text":
            raw = block.text
            break

    # Parse JSON array
    try:
        raw = raw.strip()
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        items = json.loads(raw[start:end])
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
            items = json.loads(repair_json(raw))
            if not isinstance(items, list):
                return []
        except Exception:
            logger.warning("Could not parse extraction JSON for %s", article.url)
            return []

    findings: list[Finding] = []
    discovered_date = date.today()
    discovered_str = discovered_date.isoformat().replace("-", "")
    for item in items:
        if not isinstance(item, dict):
            continue
        title = item.get("title", "").strip()
        if not title:
            continue

        slug_base = re.sub(r"[^a-z0-9]+", "_", title.lower())[:60].strip("_")
        slug = f"harvested__{slug_base}__{discovered_str}"

        source_line = f"[{_source_label(article.source)}] {article.title} — {article.url} — {article.pub_date}"

        f = Finding(
            slug=slug,
            title=title,
            kind="harvested_source",
            discovered=discovered_date,
            series_ids=frozenset(item.get("fred_series") or []),
            claim=item.get("claim", ""),
            evidence=item.get("evidence", ""),
            interpretation=item.get("interpretation", ""),
            sources=[source_line],
            window="",
            status="new",
        )
        findings.append(f)

    return findings


def _source_label(source: str) -> str:
    return {
        "feds_notes": "FEDS Notes, Federal Reserve Board",
        "liberty_street": "Liberty Street Economics, NY Fed",
        "sf_fed": "SF Fed Economic Letter",
    }.get(source, source)


# ── Orchestrator ───────────────────────────────────────────────────────────────

def harvest_all(
    dry_run: bool = False,
    max_per_source: int = 10,
    sources: list[str] | None = None,
) -> int:
    """Harvest findings from all sources. Returns count of new findings written.

    Args:
        dry_run: If True, print findings but don't write to findings.md.
        max_per_source: Max articles to process per source (after relevance filter).
        sources: Limit to specific sources: "feds_notes", "liberty_street", "sf_fed".
                 None = all three.
    """
    active_sources = sources or ["feds_notes", "liberty_street", "sf_fed"]
    harvested_urls = _load_harvested_urls()
    existing = existing_slugs()

    all_new: list[Finding] = []

    # Collect articles from each source
    index_fns = {
        "feds_notes": lambda: index_feds_notes(max_articles=max_per_source * 3),
        "liberty_street": lambda: index_liberty_street(max_articles=max_per_source * 3),
        "sf_fed": lambda: index_sf_fed(max_articles=max_per_source * 3),
    }

    for src in active_sources:
        articles = index_fns[src]()
        # Filter relevance and already-harvested
        candidates = [
            a for a in articles
            if is_relevant(a) and a.url not in harvested_urls
        ][:max_per_source]

        logger.info("[%s] %d relevant unprocessed articles", src, len(candidates))

        for article in candidates:
            logger.info("  Processing: %s", article.title[:70])
            try:
                findings = extract_findings_from_article(article)
            except Exception as exc:
                logger.warning("  Extraction error: %s", exc)
                findings = []

            new = [f for f in findings if f.slug not in existing]
            logger.info("  → %d finding(s) extracted (%d new)", len(findings), len(new))

            if not dry_run:
                _log_harvested(article.url, len(new))

            all_new.extend(new)
            time.sleep(1.0)  # be polite between fetches

    if not all_new:
        logger.info("Harvest complete — no new findings.")
        return 0

    if dry_run:
        for f in all_new:
            print(f"\n--- {f.slug} ---")
            print(f"Title: {f.title}")
            print(f"Claim: {f.claim[:200]}")
        logger.info("[DRY RUN] Would write %d findings", len(all_new))
        return len(all_new)

    added, kept = write_findings_md(all_new)
    logger.info("Harvest complete — %d new findings written (%d existing kept)", added, kept)
    return added
