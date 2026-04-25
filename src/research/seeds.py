"""Topic seeds: lightweight nightly snapshots for agentic morning composition.

The nightly scan writes seeds (raw stats, no prose). The morning composer picks
a seed, decides the organizing idea in context of what's happening today, and
composes the email fresh.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path

from src.ingest.paths import STATE_DIR

SEEDS_PATH = STATE_DIR / "topic_seeds.jsonl"
SEED_TTL_DAYS = 7


@dataclass
class TopicSeed:
    id: str
    detector: str
    series_ids: tuple[str, ...]
    key_stats: dict[str, dict]      # {series_id: {value, date, percentile_10y, z_score, ...}}
    concept_anchors: list[str]
    sources: list[dict]             # [{url, title, institution, date}]
    priority_score: float
    created: str                    # ISO datetime
    expires: str                    # ISO date (created + SEED_TTL_DAYS)
    used: bool = False


def make_seed_id(detector: str, series_ids: tuple[str, ...], today: date) -> str:
    primary = series_ids[0] if series_ids else "unknown"
    return f"{detector}-{primary}-{today.isoformat()}"


def write_seed(seed: TopicSeed, path: Path | None = None) -> None:
    """Append seed to JSONL file. Idempotent — skips if id already present."""
    path = path or SEEDS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if seed.id in existing_seed_ids(path):
        return
    with path.open("a") as fh:
        fh.write(json.dumps(_to_dict(seed)) + "\n")


def read_seeds(path: Path | None = None) -> list[TopicSeed]:
    """Read all seeds. Missing file → empty list."""
    path = path or SEEDS_PATH
    if not path.exists():
        return []
    seeds: list[TopicSeed] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            seeds.append(_from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return seeds


def mark_seed_used(seed_id: str, path: Path | None = None) -> None:
    """Mark seed as used. Atomic rewrite via tmp → rename."""
    path = path or SEEDS_PATH
    if not path.exists():
        return
    seeds = read_seeds(path)
    updated = False
    for s in seeds:
        if s.id == seed_id and not s.used:
            s.used = True
            updated = True
    if updated:
        _rewrite(seeds, path)


def expire_old_seeds(path: Path | None = None) -> int:
    """Remove expired seeds. Returns count removed."""
    path = path or SEEDS_PATH
    if not path.exists():
        return 0
    today = date.today().isoformat()
    seeds = read_seeds(path)
    kept = [s for s in seeds if s.expires >= today]
    removed = len(seeds) - len(kept)
    if removed:
        _rewrite(kept, path)
    return removed


def existing_seed_ids(path: Path | None = None) -> set[str]:
    """Return set of seed IDs already present. Missing file → empty set."""
    return {s.id for s in read_seeds(path)}


def update_seed_sources(seed_id: str, sources: list[dict], path: Path | None = None) -> bool:
    """Overwrite seed.sources for the given id. Returns True if seed was found."""
    path = path or SEEDS_PATH
    if not path.exists():
        return False
    seeds = read_seeds(path)
    updated = False
    for i, s in enumerate(seeds):
        if s.id == seed_id:
            from dataclasses import replace
            seeds[i] = replace(s, sources=sources)
            updated = True
            break
    if updated:
        _rewrite(seeds, path)
    return updated


# ---------- private helpers ----------

def _to_dict(seed: TopicSeed) -> dict:
    d = asdict(seed)
    d["series_ids"] = list(seed.series_ids)
    return d


def _from_dict(d: dict) -> TopicSeed:
    d = dict(d)
    d["series_ids"] = tuple(d.get("series_ids", []))
    return TopicSeed(**d)


def _rewrite(seeds: list[TopicSeed], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            for s in seeds:
                fh.write(json.dumps(_to_dict(s)) + "\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
