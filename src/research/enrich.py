"""Per-finding web research: populate Sources blocks from Fed research publications.

For each finding in knowledge/findings.md that has no sources, searches:
  - FEDS Notes (federalreserve.gov/econres/notes/feds-notes)
  - Liberty Street Economics (libertystreeteconomics.newyorkfed.org)
  - SF Fed Economic Letter (frbsf.org/economic-research/publications/economic-letter)

Extracts the most relevant result, verifies it via web_fetch, and populates the
finding's Sources block. Updates the finding in-place in findings.md.

Usage (via run_research.py):
    python scripts/run_research.py --enrich
    python scripts/run_research.py --enrich --slug regime_transition__vixcls__2026-04-09
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.research.findings import Finding, read_findings_md, write_findings_md

logger = logging.getLogger(__name__)

_ENRICH_SYSTEM = """\
You are a research assistant helping populate the Sources block of an empirical macro finding.

Given a finding record and a set of search results, your job is to:
1. Identify the most relevant published research from the Fed, BIS, or academic sources.
2. Fetch and verify the content of the most relevant URL.
3. Extract the specific claim from that source that corroborates, contradicts, or contextualises
   the finding.
4. Return a structured source record.

Guidelines:
- Only use sources that are genuinely relevant to the specific finding, not just the topic area.
- Prefer FEDS Notes > Liberty Street Economics > SF Fed Economic Letter > NBER papers.
- If no primary source directly addresses the finding, say so — do not return a tangentially
  related source.
- Report the URL, author/institution, publication date (YYYY-MM-DD or YYYY), title, and a
  one-sentence summary of what the source says that is relevant to this finding.

OUTPUT FORMAT (raw JSON, no code fence):
{
  "sources": [
    {
      "url": "https://...",
      "author": "Author or Institution",
      "date": "2023-06-01",
      "title": "Title of piece",
      "relevance": "One sentence: what this source says relevant to the finding."
    }
  ],
  "interpretation_addition": "Optional: one sentence to add to the Interpretation block (or null)."
}
If no relevant source found:
{ "sources": [], "interpretation_addition": null }
"""

_SEARCH_QUERIES: dict[str, list[str]] = {
    "correlation_shift": [
        "{series_a} {series_b} correlation regime change Federal Reserve",
        "{series_a} {series_b} relationship FEDS Notes",
    ],
    "lead_lag_change": [
        "{series_a} {series_b} lead lag relationship macro",
        "{series_a} leading indicator {series_b}",
    ],
    "cointegration_break": [
        "{series_a} {series_b} cointegration structural break",
    ],
    "notable_move_level": [
        "{series_a} historical extreme percentile Federal Reserve research",
        "{series_a} regime shift Liberty Street Economics",
    ],
    "notable_move_change": [
        "{series_a} unusual move macro context FEDS Notes",
    ],
    "regime_transition": [
        "{series_a} regime transition macro Federal Reserve research",
    ],
}

_SEARCH_SITES = [
    "site:federalreserve.gov/econres/notes",
    "site:libertystreeteconomics.newyorkfed.org",
    "site:frbsf.org/economic-research",
]


def _make_queries(finding: Finding) -> list[str]:
    templates = _SEARCH_QUERIES.get(finding.kind, ["{series_a} macro Federal Reserve research"])
    sids = list(finding.series_ids)
    series_a = sids[0] if sids else finding.title[:40]
    series_b = sids[1] if len(sids) > 1 else ""
    queries = []
    for template in templates:
        q = template.format(series_a=series_a, series_b=series_b).strip()
        for site in _SEARCH_SITES[:2]:  # Top 2 sites per query
            queries.append(f"{q} {site}")
    return queries[:4]  # Limit total queries


def _finding_to_dict(f: Finding) -> dict:
    return {
        "slug": f.slug,
        "title": f.title,
        "kind": f.kind,
        "series_ids": list(f.series_ids),
        "claim": f.claim,
        "evidence": f.evidence,
        "interpretation": f.interpretation,
        "sources": f.sources,
        "window": f.window,
    }


def _execute_web_search(query: str, max_results: int = 5) -> str:
    """Run a DuckDuckGo search and return formatted results."""
    from ddgs import DDGS
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}\n")
    except Exception as exc:
        return f"Search failed: {exc}"
    return "\n---\n".join(results) if results else "No results found."


def _execute_web_fetch(url: str, max_chars: int = 8000) -> str:
    """Fetch a URL and return cleaned text content."""
    import requests
    from bs4 import BeautifulSoup
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (research bot)"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Collapse blank lines
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)[:max_chars]
    except Exception as exc:
        return f"Fetch failed: {exc}"


def enrich_finding(finding: Finding) -> Finding:
    """Research a finding and populate its Sources block.

    Requires Anthropic API key at ~/keys/anthropic/key.txt or ANTHROPIC_API_KEY.
    Uses the web_search and web_fetch tools via Claude.
    Returns an updated Finding (the original is not mutated).
    """
    import anthropic
    import os

    key_file = Path.home() / "keys" / "anthropic" / "key.txt"
    if key_file.exists():
        api_key = key_file.read_text().strip()
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("Anthropic API key not found.")

    client = anthropic.Anthropic(api_key=api_key)

    queries = _make_queries(finding)
    query_block = "\n".join(f"- {q}" for q in queries)

    user_content = (
        f"FINDING RECORD:\n{json.dumps(_finding_to_dict(finding), indent=2)}\n\n"
        f"SUGGESTED SEARCH QUERIES (search these to find primary sources):\n{query_block}\n\n"
        "Search for primary sources related to this finding. For each promising result, "
        "fetch the page to verify the content. Return sources that genuinely corroborate, "
        "contradict, or contextualise the specific empirical claim. Return JSON."
    )

    tools = [
        {
            "name": "web_search",
            "description": "Search the web for relevant sources.",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "web_fetch",
            "description": "Fetch and read a web page.",
            "input_schema": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    ]

    messages: list[dict] = [{"role": "user", "content": user_content}]
    max_turns = 8

    for _ in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=[{"type": "text", "text": _ENRICH_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            # Extract JSON from text block
            for block in response.content:
                if block.type == "text":
                    try:
                        result = _extract_json(block.text)
                        return _apply_enrichment(finding, result)
                    except (ValueError, json.JSONDecodeError):
                        logger.warning("Could not parse enrichment JSON for %s", finding.slug)
            break

        if response.stop_reason != "tool_use":
            break

        # Execute tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                try:
                    if block.name == "web_search":
                        content = _execute_web_search(block.input.get("query", ""))
                    elif block.name == "web_fetch":
                        content = _execute_web_fetch(block.input.get("url", ""))
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

    logger.info("Enrichment complete for %s — %d sources found", finding.slug, len(finding.sources))
    return finding


def _extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    brace = text.find("{")
    if brace != -1:
        return json.loads(text[brace : text.rfind("}") + 1])
    raise ValueError("No JSON found")


def _apply_enrichment(finding: Finding, result: dict) -> Finding:
    """Return a new Finding with updated sources and optionally updated interpretation."""
    new_sources = list(finding.sources)
    for src in result.get("sources", []):
        line = (
            f"[{src.get('author', 'Unknown')} ({src.get('date', '')})] "
            f"{src.get('title', '')} — {src.get('url', '')} — "
            f"{src.get('relevance', '')}"
        )
        if line not in new_sources:
            new_sources.append(line)

    new_interpretation = finding.interpretation
    addition = result.get("interpretation_addition")
    if addition and addition not in new_interpretation:
        new_interpretation = finding.interpretation.rstrip() + " " + addition

    from dataclasses import replace
    return replace(finding, sources=new_sources, interpretation=new_interpretation)


def enrich_all(
    findings_path: Path,
    dry_run: bool = False,
    slug_filter: str | None = None,
    skip_sourced: bool = True,
) -> int:
    """Enrich all (or filtered) findings in findings.md that lack sources.

    Returns count of findings enriched.
    """
    findings = read_findings_md(findings_path)
    enriched = 0

    for f in findings:
        if slug_filter and slug_filter not in f.slug:
            continue
        if skip_sourced and f.sources:
            logger.debug("Skipping %s (already has %d sources)", f.slug, len(f.sources))
            continue
        logger.info("Enriching: %s", f.slug)
        try:
            updated = enrich_finding(f)
            if updated.sources != f.sources:
                findings[findings.index(f)] = updated
                enriched += 1
                logger.info("  → %d sources added", len(updated.sources) - len(f.sources))
        except Exception as e:
            logger.warning("Enrichment failed for %s: %s", f.slug, e)

    if enriched > 0 and not dry_run:
        new_n, kept_n = write_findings_md(findings_path, findings, overwrite=True)
        logger.info("Wrote findings.md: %d new, %d kept", new_n, kept_n)

    return enriched
