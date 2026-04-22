"""RunLogger: structured per-run audit log written to logs/runs.jsonl.

Each entry script wraps its main body in a RunLogger context.  On exit (success
or exception) a single JSON record is appended to logs/runs.jsonl with:

  run_id, script, started_at, finished_at, duration_s,
  success, error, dry_run, counts (script-specific metrics)

health.py reads runs.jsonl to build the system health snapshot.
"""
from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"
RUNS_LOG = LOGS_DIR / "runs.jsonl"


class RunLogger:
    """Context manager that appends one structured record to logs/runs.jsonl."""

    def __init__(
        self,
        script: str,
        dry_run: bool = False,
        log_path: Path = RUNS_LOG,
    ) -> None:
        self.script = script
        self.dry_run = dry_run
        self.log_path = log_path
        self._counts: dict[str, Any] = {}
        self._started_at: datetime = datetime.now(timezone.utc)

    # -- public API ----------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Record a named metric or label."""
        self._counts[key] = value

    def add(self, key: str, n: int = 1) -> None:
        """Increment a named counter."""
        self._counts[key] = self._counts.get(key, 0) + n

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "RunLogger":
        self._started_at = datetime.now(timezone.utc)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        tb: Any,
    ) -> bool:
        finished_at = datetime.now(timezone.utc)
        error: str | None = None
        if exc_type is not None:
            error = f"{exc_type.__name__}: {exc_val}"
            tb_lines = traceback.format_tb(tb)
            tb_str = "".join(tb_lines)
            # Keep last 2000 chars so the record stays readable.
            self._counts["traceback"] = tb_str[-2000:] if len(tb_str) > 2000 else tb_str

        record: dict[str, Any] = {
            "run_id": str(uuid4())[:8],
            "script": self.script,
            "started_at": self._started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_s": round((finished_at - self._started_at).total_seconds(), 1),
            "success": exc_type is None,
            "error": error,
            "dry_run": self.dry_run,
            "counts": self._counts,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return False  # never suppress exceptions
