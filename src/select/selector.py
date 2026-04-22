"""Lesson selector: picks the best finding to teach today.

Priority order (CLAUDE.md §Teaching phase):
1. Upcoming release/Fed event within 2 days: find a related finding.
2. Highest-priority new finding by kind × recency.
3. Any finding not recently taught (when backlog of 'new' findings is exhausted).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from src.ingest.paths import RELEASE_CALENDAR_PATH, RELEASE_SERIES_PATH, STATE_DIR
from src.ingest.storage import load_parquet
from src.research.findings import Finding, read_findings_md

logger = logging.getLogger(__name__)

LESSON_HISTORY_PATH = STATE_DIR / "lesson_history.jsonl"

# Lower number = higher priority.
_KIND_PRIORITY: dict[str, int] = {
    "notable_move_level": 0,
    "correlation_shift": 1,
    "cointegration_break": 2,
    "regime_transition": 3,
    "lead_lag_change": 4,
    "notable_move_change": 5,
}


@dataclass
class LessonPick:
    kind: str            # "upcoming_release" | "finding" | "continuity"
    finding: Finding
    release_context: dict | None
    reason: str


def pick_lesson(today: date | None = None, lookback_days: int = 14) -> LessonPick:
    """Select today's lesson finding.

    Raises ValueError if findings.md is empty.
    """
    today = today or date.today()
    all_findings = read_findings_md()
    if not all_findings:
        raise ValueError("findings.md is empty. Run scripts/run_research.py first.")

    recent_slugs = _load_recent_slugs(lookback_days)

    eligible = [f for f in all_findings if f.status == "new" and f.slug not in recent_slugs]
    if not eligible:
        eligible = [f for f in all_findings if f.slug not in recent_slugs]
    if not eligible:
        logger.warning("All findings recently taught; ignoring lookback filter.")
        eligible = all_findings

    ranked = _rank(eligible)
    upcoming = _upcoming_releases(today, days_ahead=2)

    if upcoming:
        release_series = _release_series_for(upcoming)
        release_picks = [f for f in ranked if any(s in release_series for s in f.series_ids)]
        if release_picks:
            best = release_picks[0]
            return LessonPick(
                kind="upcoming_release",
                finding=best,
                release_context={"releases": upcoming},
                reason=f"Upcoming: {', '.join(r['release_name'] for r in upcoming[:3])}",
            )

    best = ranked[0]
    return LessonPick(
        kind="finding",
        finding=best,
        release_context={"releases": upcoming} if upcoming else None,
        reason=f"Top new finding: {best.kind} ({', '.join(best.series_ids[:2])})",
    )


def record_lesson_sent(finding: Finding, today: date | None = None) -> None:
    """Append an entry to state/lesson_history.jsonl."""
    today = today or date.today()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "date": today.isoformat(),
        "slug": finding.slug,
        "title": finding.title,
        "kind": finding.kind,
    }
    with LESSON_HISTORY_PATH.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def _load_recent_slugs(days: int) -> set[str]:
    if not LESSON_HISTORY_PATH.exists():
        return set()
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    slugs: set[str] = set()
    for line in LESSON_HISTORY_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if entry.get("date", "") >= cutoff:
                slugs.add(entry["slug"])
        except (json.JSONDecodeError, KeyError):
            pass
    return slugs


def _rank(findings: list[Finding]) -> list[Finding]:
    def key(f: Finding) -> tuple:
        return (_KIND_PRIORITY.get(f.kind, 99), -f.discovered.toordinal())
    return sorted(findings, key=key)


def _upcoming_releases(today: date, days_ahead: int = 2) -> list[dict]:
    cal = load_parquet(RELEASE_CALENDAR_PATH)
    if cal is None:
        return []
    cal["release_date"] = pd.to_datetime(cal["release_date"])
    lo = pd.Timestamp(today)
    hi = pd.Timestamp(today + timedelta(days=days_ahead))
    mask = (cal["release_date"] >= lo) & (cal["release_date"] <= hi)
    subset = cal[mask].drop_duplicates("release_name")
    rows = subset[["release_id", "release_name", "release_date"]].copy()
    rows["release_date"] = rows["release_date"].dt.strftime("%Y-%m-%d")
    return rows.to_dict("records")


def _release_series_for(upcoming: list[dict]) -> set[str]:
    rel_series = load_parquet(RELEASE_SERIES_PATH)
    if rel_series is None or not upcoming:
        return set()
    ids = {r["release_id"] for r in upcoming}
    return set(rel_series[rel_series["release_id"].isin(ids)]["series_id"].tolist())
