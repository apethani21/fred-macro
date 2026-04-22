"""Build a system health snapshot and write state/system_health.json.

Called at the end of each entry script so the snapshot stays fresh.
status.py reads the pre-built file; pass --rebuild to force a fresh compute.

Snapshot structure:
  generated_at        ISO timestamp
  overall_status      ok | warn | error
  processes           per-script last-run summary (from logs/runs.jsonl)
  data_freshness      series staleness (from data/update_log.parquet)
  findings            backlog counts (from knowledge/findings.md)
  emails              sent history (from state/lesson_history.jsonl)
  harvest             article harvest summary (from state/harvest_log.jsonl)
  upcoming_releases   next N releases (from data/release_calendar.parquet)
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"
STATE_DIR = PROJECT_ROOT / "state"
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

RUNS_LOG = LOGS_DIR / "runs.jsonl"
HEALTH_FILE = STATE_DIR / "system_health.json"
LESSON_HISTORY = STATE_DIR / "lesson_history.jsonl"
HARVEST_LOG = STATE_DIR / "harvest_log.jsonl"
UPDATE_LOG = DATA_DIR / "update_log.parquet"
RELEASE_CALENDAR = DATA_DIR / "release_calendar.parquet"
FINDINGS_MD = KNOWLEDGE_DIR / "findings.md"

# Warn if a script hasn't succeeded in this many hours.
_STALE_HOURS: dict[str, int] = {
    "refresh_data": 30,    # daily job; warn after 30h
    "run_research": 30,
    "run_harvest": 50,     # weekday-only; allow a day gap
    "send_daily": 30,      # weekday-only
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Per-section builders
# ---------------------------------------------------------------------------

def _process_health(runs: list[dict[str, Any]]) -> dict[str, Any]:
    scripts = ["refresh_data", "run_research", "run_harvest", "send_daily"]
    now = datetime.now(timezone.utc)
    result: dict[str, Any] = {}

    for script in scripts:
        script_runs = [r for r in runs if r.get("script") == script]
        if not script_runs:
            result[script] = {
                "status": "never_run",
                "last_run": None,
                "last_success": None,
                "consecutive_failures": 0,
                "last_duration_s": None,
                "last_counts": {},
                "last_error": None,
                "dry_run": None,
            }
            continue

        last = script_runs[-1]

        consec = 0
        for r in reversed(script_runs):
            if not r.get("success"):
                consec += 1
            else:
                break

        last_success_rec = next((r for r in reversed(script_runs) if r.get("success")), None)
        last_success_dt = _parse_dt(last_success_rec["finished_at"]) if last_success_rec else None

        status = "ok"
        stale_h = _STALE_HOURS.get(script, 30)
        if consec >= 3:
            status = "error"
        elif consec >= 1:
            status = "warn"
        elif last_success_dt and (now - last_success_dt).total_seconds() > stale_h * 3600:
            status = "warn"

        result[script] = {
            "status": status,
            "last_run": last.get("finished_at"),
            "last_success": last_success_rec["finished_at"] if last_success_rec else None,
            "consecutive_failures": consec,
            "last_duration_s": last.get("duration_s"),
            "last_counts": {k: v for k, v in last.get("counts", {}).items() if k != "traceback"},
            "last_error": last.get("error"),
            "dry_run": last.get("dry_run", False),
        }

    return result


def _data_freshness_health() -> dict[str, Any]:
    if not UPDATE_LOG.exists():
        return {"status": "no_data", "total_series": 0}

    try:
        import pandas as pd
        df = pd.read_parquet(UPDATE_LOG)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    if df.empty:
        return {"status": "no_data", "total_series": 0}

    now = datetime.now(timezone.utc)
    total = len(df)

    newest_obs = oldest_obs = None
    if "last_observation_date" in df.columns:
        obs = pd.to_datetime(df["last_observation_date"], errors="coerce").dropna()
        if not obs.empty:
            newest_obs = str(obs.max().date())
            oldest_obs = str(obs.min().date())

    stale_1d = stale_7d = 0
    last_fetch: str | None = None
    if "last_fetched_at" in df.columns:
        fetched = pd.to_datetime(df["last_fetched_at"], utc=True, errors="coerce")
        stale_1d = int((fetched < now - timedelta(days=1)).sum())
        stale_7d = int((fetched < now - timedelta(days=7)).sum())
        max_fetch = fetched.max()
        last_fetch = max_fetch.isoformat() if pd.notna(max_fetch) else None

    status = "ok"
    if stale_7d > total * 0.3:
        status = "error"
    elif stale_7d > total * 0.1:
        status = "warn"

    return {
        "status": status,
        "total_series": total,
        "last_fetch": last_fetch,
        "newest_observation": newest_obs,
        "oldest_observation": oldest_obs,
        "series_stale_1d": stale_1d,
        "series_stale_7d": stale_7d,
    }


def _findings_health() -> dict[str, Any]:
    if not FINDINGS_MD.exists():
        return {"status": "no_findings", "total": 0, "new_unsent": 0, "sourced": 0}

    text = FINDINGS_MD.read_text(encoding="utf-8")
    # Each finding section starts with "## " and contains a **Slug** line.
    sections = [s for s in text.split("\n## ") if s.strip() and "**Slug**" in s]

    total = len(sections)
    new_count = sum(1 for s in sections if "**Status**: new" in s)
    taught_count = sum(1 for s in sections if "**Status**: taught" in s or "taught-on" in s)
    sourced = sum(1 for s in sections if "**Sources**" in s and "http" in s)

    cutoff = (date.today() - timedelta(days=7)).isoformat()
    recent = 0
    for s in sections:
        disc_idx = s.find("**Discovered**:")
        if disc_idx >= 0:
            disc_val = s[disc_idx + 15: disc_idx + 30].strip()[:10]
            if disc_val >= cutoff:
                recent += 1

    return {
        "status": "ok" if total > 0 else "warn",
        "total": total,
        "new_unsent": new_count,
        "taught": taught_count,
        "sourced": sourced,
        "added_last_7d": recent,
    }


def _email_health(lessons: list[dict[str, Any]]) -> dict[str, Any]:
    last_sent = lessons[-1].get("date") if lessons else None
    cutoff = (date.today() - timedelta(days=7)).isoformat()
    recent = [l for l in lessons if l.get("date", "") >= cutoff]

    status = "ok"
    # Warn if no email in 5 calendar days (covers weekends).
    if last_sent is None or last_sent < (date.today() - timedelta(days=5)).isoformat():
        status = "warn"

    return {
        "status": status,
        "total_sent": len(lessons),
        "last_sent": last_sent,
        "sent_last_7d": [
            {"date": l["date"], "slug": l.get("slug"), "kind": l.get("kind")}
            for l in recent
        ],
    }


def _harvest_health(harvests: list[dict[str, Any]]) -> dict[str, Any]:
    if not harvests:
        return {"status": "never_run", "last_run": None, "total_articles": 0, "added_last_7d": 0}

    last = harvests[-1]
    cutoff = (date.today() - timedelta(days=7)).isoformat()
    added_7d = sum(h.get("n_findings", 0) for h in harvests if h.get("harvested_at", "") >= cutoff)

    return {
        "status": "ok",
        "last_run": last.get("harvested_at"),
        "total_articles": len(harvests),
        "added_last_7d": added_7d,
    }


def _upcoming_releases(n: int = 7) -> list[dict[str, Any]]:
    if not RELEASE_CALENDAR.exists():
        return []
    try:
        import pandas as pd
        df = pd.read_parquet(RELEASE_CALENDAR)
    except Exception:
        return []
    if df.empty:
        return []

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    today_dt = pd.Timestamp(date.today())
    upcoming = (
        df[df["release_date"] >= today_dt]
        .sort_values("release_date")
        .drop_duplicates(subset=["release_date", "release_name"])
        .head(n)
    )
    return [
        {"date": str(r["release_date"].date()), "release_name": str(r.get("release_name", ""))}
        for _, r in upcoming.iterrows()
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_health_snapshot(write: bool = True) -> dict[str, Any]:
    """Aggregate all state files into a health snapshot.

    Returns the snapshot dict.  Writes to state/system_health.json if write=True.
    """
    runs = _read_jsonl(RUNS_LOG)
    lessons = _read_jsonl(LESSON_HISTORY)
    harvests = _read_jsonl(HARVEST_LOG)

    processes = _process_health(runs)
    data_freshness = _data_freshness_health()
    findings = _findings_health()
    emails = _email_health(lessons)
    harvest = _harvest_health(harvests)
    upcoming = _upcoming_releases()

    # Overall status: worst of individual statuses.
    all_statuses = (
        [p.get("status", "ok") for p in processes.values()]
        + [data_freshness.get("status", "ok")]
        + [findings.get("status", "ok")]
        + [emails.get("status", "ok")]
    )
    if "error" in all_statuses:
        overall = "error"
    elif "warn" in all_statuses:
        overall = "warn"
    else:
        overall = "ok"

    snapshot: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "processes": processes,
        "data_freshness": data_freshness,
        "findings": findings,
        "emails": emails,
        "harvest": harvest,
        "upcoming_releases": upcoming,
    }

    if write:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        HEALTH_FILE.write_text(
            json.dumps(snapshot, indent=2, default=str),
            encoding="utf-8",
        )

    return snapshot
