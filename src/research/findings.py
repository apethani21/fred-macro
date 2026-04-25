"""Finding records: dataclass, markdown render, parquet persistence.

Findings are the atomic unit of research output. A `Finding` is what gets
written to `knowledge/findings.md`; the underlying numeric evidence lives in
`data/stats/*.parquet` so every claim is auditable from structured data.

The markdown file is treated as the source of truth for which findings
exist. Write-back is idempotent: re-running the scan with the same data
produces the same set of findings, deduplicated on slug.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.ingest.paths import KNOWLEDGE_DIR, STATS_DIR
from src.ingest.storage import load_parquet, save_parquet_atomic

FINDINGS_PATH = KNOWLEDGE_DIR / "findings.md"

STATS_FILES = {
    "notable_moves": STATS_DIR / "notable_moves.parquet",
    "rolling_correlations": STATS_DIR / "rolling_correlations.parquet",
    "lead_lag": STATS_DIR / "lead_lag.parquet",
    "regime_labels": STATS_DIR / "regime_labels.parquet",
    "structural_breaks": STATS_DIR / "structural_breaks.parquet",
    # M1: inflation episode cross-sectional analysis
    "inflation_episodes": STATS_DIR / "inflation_episodes.parquet",
    # M8: breakeven decomposition
    "breakeven_components": STATS_DIR / "breakeven_components.parquet",
    # M3: bond return predictability / CP factor
    "bond_predictability": STATS_DIR / "bond_predictability.parquet",
    # M2: Nelson-Siegel yield curve factors
    "ns_factors": STATS_DIR / "ns_factors.parquet",
    # M12: cross-asset factor model (Fama-MacBeth two-pass CSR)
    "cross_asset_factors": STATS_DIR / "cross_asset_factors.parquet",
    # Relationship monitor: named spread and decomposition relationships
    "relationship_spreads": STATS_DIR / "relationship_spreads.parquet",
    "relationship_decompositions": STATS_DIR / "relationship_decompositions.parquet",
}


@dataclass
class Finding:
    """A single finding in the form CLAUDE.md mandates.

    `slug` is the dedup key; same slug on re-scan means same finding (and the
    existing record is left alone unless explicitly overwritten).
    """
    slug: str
    title: str
    kind: str                               # mirrors DetectorHit.kind
    discovered: date
    series_ids: tuple[str, ...]
    window: str
    claim: str                              # 2-3 sentences of prose
    evidence: dict                          # reproducible numbers from the detector
    interpretation: str = ""                # 1-3 sentences, marked as interpretive
    sources: list[str] = field(default_factory=list)
    status: str = "new"                     # new | taught-on-{date} | superseded | stale
    score: float = 0.0

    # ---------- rendering ----------

    def to_markdown(self) -> str:
        """Render as a `##`-headed markdown section.

        Structure matches CLAUDE.md: Title, Discovered, Claim, Evidence,
        Interpretation, Sources, Status.
        """
        out: list[str] = []
        out.append(f"## {self.title}")
        out.append("")
        out.append(f"- **Slug**: `{self.slug}`")
        out.append(f"- **Kind**: {self.kind}")
        out.append(f"- **Discovered**: {self.discovered.isoformat()}")
        out.append(f"- **Series**: {', '.join(self.series_ids)}")
        out.append(f"- **Window**: {self.window}")
        out.append(f"- **Status**: {self.status}")
        out.append("")
        out.append("**Claim.** " + self.claim.strip())
        out.append("")
        out.append("**Evidence.**")
        out.append("")
        out.append("```json")
        out.append(json.dumps(self.evidence, indent=2, default=_json_default))
        out.append("```")
        if self.interpretation:
            out.append("")
            out.append("**Interpretation (interpretive — not a data claim).** "
                       + self.interpretation.strip())
        if self.sources:
            out.append("")
            out.append("**Sources.**")
            for src in self.sources:
                out.append(f"- {src}")
        out.append("")
        return "\n".join(out)


def _json_default(o):  # pragma: no cover — trivial
    if isinstance(o, (date, pd.Timestamp)):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


# -------------------------------------------------------------------------
# Markdown reader
# -------------------------------------------------------------------------

def read_findings_md(path: Path | None = None) -> list[Finding]:
    """Parse knowledge/findings.md back into Finding objects.

    Only sections with a Slug line are returned. Best-effort parsing: fields
    that can't be extracted get sensible defaults. Score is not stored in
    markdown — callers that need ranking should compute it themselves.
    """
    p = path or FINDINGS_PATH
    if not p.exists():
        return []
    sections = _split_sections(p.read_text())
    findings = []
    for slug, section in sections.items():
        if slug.startswith("_hand__"):
            continue
        f = _parse_section(slug, section)
        if f is not None:
            findings.append(f)
    return findings


def _parse_section(slug: str, section: str) -> Finding | None:
    title_m = re.match(r"## (.+)", section.strip())
    title = title_m.group(1).strip() if title_m else slug

    def meta(name: str) -> str:
        m = re.search(rf"^- \*\*{re.escape(name)}\*\*: (.+)$", section, re.MULTILINE)
        return m.group(1).strip() if m else ""

    kind = meta("Kind")
    series_s = meta("Series")
    window = meta("Window")
    status = meta("Status") or "new"

    try:
        discovered_date = date.fromisoformat(meta("Discovered"))
    except (ValueError, AttributeError):
        discovered_date = date.today()

    series_ids = tuple(s.strip() for s in series_s.split(",") if s.strip())

    claim_m = re.search(r"\*\*Claim\.\*\*\s+(.+?)(?=\n\n\*\*|\Z)", section, re.DOTALL)
    claim = claim_m.group(1).strip() if claim_m else ""

    json_m = re.search(r"```json\n(.+?)```", section, re.DOTALL)
    evidence: dict = {}
    if json_m:
        try:
            evidence = json.loads(json_m.group(1))
        except json.JSONDecodeError:
            pass

    interp_m = re.search(r"\*\*Interpretation[^*]*\.\*\*\s+(.+?)(?=\n\n\*\*|\Z)", section, re.DOTALL)
    interpretation = interp_m.group(1).strip() if interp_m else ""

    sources_m = re.search(r"\*\*Sources\.\*\*\n((?:- .+\n?)+)", section)
    sources: list[str] = []
    if sources_m:
        sources = [ln.lstrip("- ").strip() for ln in sources_m.group(1).strip().splitlines() if ln.strip()]

    return Finding(
        slug=slug,
        title=title,
        kind=kind or "unknown",
        discovered=discovered_date,
        series_ids=series_ids,
        window=window,
        claim=claim,
        evidence=evidence,
        interpretation=interpretation,
        sources=sources,
        status=status,
        score=0.0,
    )


# -------------------------------------------------------------------------
# Slug construction
# -------------------------------------------------------------------------

def make_slug(kind: str, series_ids: Iterable[str], extra: str = "") -> str:
    """Deterministic dedup key. Same kind+series (and optional extra) → same slug."""
    base = kind.lower() + "__" + "_".join(series_ids)
    if extra:
        base += "__" + extra
    return re.sub(r"[^a-z0-9_]+", "-", base.lower()).strip("-")


# -------------------------------------------------------------------------
# Markdown read / merge / write
# -------------------------------------------------------------------------

_SECTION_HEADER_RE = re.compile(r"^## (?P<title>.+?)\s*$", re.MULTILINE)
_SLUG_LINE_RE = re.compile(r"^- \*\*Slug\*\*: `(?P<slug>[^`]+)`\s*$", re.MULTILINE)

_PREAMBLE = """# Findings

Ongoing log of empirical findings from the research phase. Each section is
one auditable finding with its series IDs, window, and raw numbers. Status
is updated as findings are used in daily emails.

"""


def _split_sections(md: str) -> dict[str, str]:
    """Split an existing findings.md into {slug: raw_section_markdown}.

    Sections without a Slug line are preserved under a synthesized key so
    hand-edited entries aren't silently dropped.
    """
    if not md.strip():
        return {}
    matches = list(_SECTION_HEADER_RE.finditer(md))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        chunk = md[start:end].rstrip() + "\n"
        slug_match = _SLUG_LINE_RE.search(chunk)
        if slug_match:
            sections[slug_match.group("slug")] = chunk
        else:
            title = m.group("title").strip().lower()
            handle = "_hand__" + re.sub(r"[^a-z0-9_]+", "-", title)
            sections[handle] = chunk
    return sections


def write_findings_md(
    path_or_findings: "Path | list[Finding]",
    findings: "list[Finding] | None" = None,
    *,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Merge `findings` into `knowledge/findings.md`.

    Accepts either:
      write_findings_md(findings)           — uses default FINDINGS_PATH
      write_findings_md(path, findings)     — writes to explicit path

    Returns `(added, kept)`.
    """
    if findings is None:
        target_path = FINDINGS_PATH
        findings = path_or_findings  # type: ignore[assignment]
    else:
        target_path = path_or_findings  # type: ignore[assignment]

    target_path.parent.mkdir(parents=True, exist_ok=True)
    existing_md = target_path.read_text() if target_path.exists() else ""
    sections = _split_sections(existing_md)

    added = 0
    for f in findings:
        if f.slug in sections and not overwrite:
            continue
        if f.slug in sections and overwrite:
            sections[f.slug] = f.to_markdown()
        else:
            sections[f.slug] = f.to_markdown()
            added += 1

    body = _PREAMBLE + "\n".join(sections[k] for k in sections)
    tmp = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp.write_text(body)
    tmp.replace(target_path)
    return added, len(sections) - added


def existing_slugs() -> set[str]:
    """Slugs currently in findings.md — for detectors that want to avoid duplicates pre-scan."""
    if not FINDINGS_PATH.exists():
        return set()
    return {m.group("slug") for m in _SLUG_LINE_RE.finditer(FINDINGS_PATH.read_text())}


# -------------------------------------------------------------------------
# Supporting stats persistence
# -------------------------------------------------------------------------

def append_stats(name: str, rows: pd.DataFrame) -> Path:
    """Append rows to `data/stats/{name}.parquet`, deduplicating by its natural key.

    Natural keys (match CLAUDE.md schema):
      notable_moves        → (series_id, date, kind)
      rolling_correlations → (series_a, series_b, window, date, method)
      lead_lag             → (series_a, series_b, window, lag)
      regime_labels        → (series_id, date)

    Returns the path written.
    """
    if name not in STATS_FILES:
        raise KeyError(f"Unknown stats file {name!r}. Known: {list(STATS_FILES)}")
    if rows.empty:
        return STATS_FILES[name]

    path = STATS_FILES[name]
    existing = load_parquet(path)
    combined = rows if existing is None else pd.concat([existing, rows], ignore_index=True)

    dedup_keys = {
        "notable_moves": ["series_id", "date", "kind"],
        "rolling_correlations": ["series_a", "series_b", "window", "date", "method"],
        "lead_lag": ["series_a", "series_b", "window", "lag"],
        "regime_labels": ["series_id", "date"],
        "structural_breaks": ["series_id", "series_b", "kind", "run_date"],
        "inflation_episodes": ["episode_idx", "run_date"],
        "breakeven_components": ["date", "tenor"],
        "bond_predictability": ["date", "run_date"],
        "ns_factors": ["date", "run_date"],
        "cross_asset_factors": ["regime", "factor_name", "run_date"],
        "relationship_spreads": ["spread_id", "run_date"],
        "relationship_decompositions": ["decomp_id", "component", "run_date"],
    }[name]
    combined = combined.drop_duplicates(subset=dedup_keys, keep="last").reset_index(drop=True)
    save_parquet_atomic(combined, path)
    return path
