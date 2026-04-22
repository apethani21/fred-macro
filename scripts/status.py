#!/usr/bin/env python3
"""Print a human-readable system health report.

Usage:
  python scripts/status.py            # Read cached state/system_health.json
  python scripts/status.py --rebuild  # Recompute from raw logs/state files first
  python scripts/status.py --json     # Dump the raw JSON snapshot
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.monitor.health import HEALTH_FILE, build_health_snapshot


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_STATUS_ICON = {"ok": "✓", "warn": "⚠", "error": "✗", "never_run": "–", "no_data": "–", "no_findings": "–"}
_STATUS_LABEL = {"ok": "OK", "warn": "WARN", "error": "ERROR", "never_run": "NEVER RUN", "no_data": "NO DATA", "no_findings": "NO FINDINGS"}


def _icon(s: str) -> str:
    return _STATUS_ICON.get(s, "?")


def _label(s: str) -> str:
    return _STATUS_LABEL.get(s, s.upper())


def _ago(iso: str | None) -> str:
    if not iso:
        return "never"
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        total_s = delta.total_seconds()
        if total_s < 3600:
            return f"{int(total_s // 60)}m ago"
        if total_s < 86400:
            return f"{int(total_s // 3600)}h ago"
        return f"{int(total_s // 86400)}d ago"
    except Exception:
        return iso[:16]


def _hr(char: str = "─", width: int = 64) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Section printers
# ---------------------------------------------------------------------------

def _print_header(snap: dict) -> None:
    overall = snap.get("overall_status", "?")
    gen = snap.get("generated_at", "")[:16]
    print()
    print(_hr("═"))
    print(f"  fred-macro  │  {_icon(overall)} {_label(overall)}  │  {gen} UTC")
    print(_hr("═"))


def _print_processes(processes: dict) -> None:
    print(f"\n  PROCESSES")
    print(f"  {_hr()}")
    fmt = "  {icon} {script:<20} {status:<10} {last_run:<14} {dur:<10} {extra}"
    print(fmt.format(icon=" ", script="Script", status="Status", last_run="Last Run", dur="Duration", extra="Key Metrics"))
    print(f"  {_hr('-')}")
    for script, p in processes.items():
        status = p.get("status", "?")
        last_run = _ago(p.get("last_run"))
        dur_s = p.get("last_duration_s")
        dur = f"{dur_s}s" if dur_s is not None else "—"
        counts = {k: v for k, v in p.get("last_counts", {}).items() if k != "traceback"}
        counts_str = "  ".join(f"{k}={v}" for k, v in list(counts.items())[:4])
        consec = p.get("consecutive_failures", 0)
        fail_note = f"  [{consec} consecutive failures]" if consec else ""
        dry_note = "  [dry-run]" if p.get("dry_run") else ""
        print(fmt.format(
            icon=_icon(status),
            script=script,
            status=_label(status),
            last_run=last_run,
            dur=dur,
            extra=counts_str + fail_note + dry_note,
        ))
        if p.get("last_error"):
            print(f"      └─ {p['last_error'][:90]}")


def _print_data_freshness(df: dict) -> None:
    status = df.get("status", "?")
    print(f"\n  DATA FRESHNESS  {_icon(status)} {_label(status)}")
    print(f"  {_hr('-')}")
    print(f"  Series tracked    : {df.get('total_series', '—')}")
    print(f"  Last fetch        : {_ago(df.get('last_fetch'))}")
    print(f"  Newest obs date   : {df.get('newest_observation', '—')}")
    stale_1d = df.get('series_stale_1d', '—')
    stale_7d = df.get('series_stale_7d', '—')
    total = df.get('total_series', 1) or 1
    pct_7d = f"  ({stale_7d / total * 100:.1f}%)" if isinstance(stale_7d, int) else ""
    print(f"  Stale > 1 day     : {stale_1d}")
    print(f"  Stale > 7 days    : {stale_7d}{pct_7d}")


def _print_findings(fi: dict) -> None:
    status = fi.get("status", "?")
    print(f"\n  FINDINGS BACKLOG  {_icon(status)} {_label(status)}")
    print(f"  {_hr('-')}")
    print(f"  Total             : {fi.get('total', '—')}")
    print(f"  New (unsent)      : {fi.get('new_unsent', '—')}")
    print(f"  Taught            : {fi.get('taught', '—')}")
    print(f"  Sourced (w/ URLs) : {fi.get('sourced', '—')}")
    print(f"  Added last 7d     : {fi.get('added_last_7d', '—')}")


def _print_emails(em: dict) -> None:
    status = em.get("status", "?")
    print(f"\n  EMAILS  {_icon(status)} {_label(status)}")
    print(f"  {_hr('-')}")
    print(f"  Total sent        : {em.get('total_sent', '—')}")
    print(f"  Last sent         : {em.get('last_sent', 'never')}")
    recent = em.get("sent_last_7d", [])
    if recent:
        print(f"  Last 7 days:")
        for r in recent[-7:]:
            print(f"    {r['date']}  {(r.get('kind') or '?'):<28} {(r.get('slug') or '')[:48]}")


def _print_harvest(ha: dict) -> None:
    status = ha.get("status", "?")
    print(f"\n  HARVEST  {_icon(status)} {_label(status)}")
    print(f"  {_hr('-')}")
    print(f"  Last run          : {ha.get('last_run', 'never')}")
    print(f"  Total articles    : {ha.get('total_articles', '—')}")
    print(f"  Added last 7d     : {ha.get('added_last_7d', '—')}")


def _print_releases(releases: list) -> None:
    if not releases:
        return
    print(f"\n  UPCOMING RELEASES")
    print(f"  {_hr('-')}")
    for r in releases:
        print(f"  {r['date']}  {r.get('release_name', '')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rebuild = "--rebuild" in sys.argv
    dump_json = "--json" in sys.argv

    if rebuild or not HEALTH_FILE.exists():
        snap = build_health_snapshot(write=True)
        if not dump_json:
            print("(health snapshot rebuilt)")
    else:
        snap = json.loads(HEALTH_FILE.read_text(encoding="utf-8"))

    if dump_json:
        print(json.dumps(snap, indent=2))
        return

    _print_header(snap)
    _print_processes(snap.get("processes", {}))
    _print_data_freshness(snap.get("data_freshness", {}))
    _print_findings(snap.get("findings", {}))
    _print_emails(snap.get("emails", {}))
    _print_harvest(snap.get("harvest", {}))
    _print_releases(snap.get("upcoming_releases", []))
    print()


if __name__ == "__main__":
    main()
