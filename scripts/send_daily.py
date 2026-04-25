#!/usr/bin/env python3
"""Entry point for the daily email: select → compose → deliver.

Usage:
  python scripts/send_daily.py [--dry-run] [--force-finding SLUG]

Flags:
  --dry-run          Write email to state/last_email.html instead of sending.
                     Overrides the DRY_RUN environment variable.
  --force-finding    Use a specific finding slug instead of the selector.
  --today DATE       Override today's date (YYYY-MM-DD). For testing/backfill.

Credentials:
  FRED key    — ~/keys/fred/key.txt  (not needed by this script; data already pulled)
  Anthropic   — ~/keys/anthropic/key.txt or ANTHROPIC_API_KEY env var
  AWS SES     — standard boto3 chain (env vars or ~/.aws/credentials or ~/keys/aws/credentials)
  Email addrs — EMAIL_FROM and EMAIL_TO in .env
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

# Load .env before any src imports so os.environ is populated.
_env_path = Path(__file__).resolve().parents[1] / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# Project root on sys.path so `src` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compose.composer import compose_email
from src.compose.seed_composer import compose_email_from_seed
from src.deliver.sender import DeliveryConfig, send_email, config_from_env
from src.monitor.health import build_health_snapshot
from src.monitor.run_log import RunLogger
from src.research.findings import read_findings_md
from src.research.seeds import read_seeds
from src.select.selector import LessonPick, SeedPick, pick_lesson, pick_seed, record_lesson_sent, record_seed_sent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Send the daily macro education email.")
    p.add_argument("--dry-run", action="store_true", help="Write to state/last_email.html, do not send.")
    p.add_argument("--force-finding", metavar="SLUG", help="Use a specific finding slug.")
    p.add_argument("--force-seed", metavar="SEED_ID", help="Use a specific seed ID.")
    p.add_argument("--today", metavar="DATE", help="Override today's date (YYYY-MM-DD).")
    return p.parse_args()


def _pick(args: argparse.Namespace, today: date) -> tuple[LessonPick | None, SeedPick | None]:
    if args.force_finding:
        findings = {f.slug: f for f in read_findings_md()}
        slug = args.force_finding
        if slug not in findings:
            available = ", ".join(list(findings)[:10])
            raise ValueError(f"Slug {slug!r} not found. Known (first 10): {available}")
        f = findings[slug]
        return LessonPick(kind="finding", finding=f, release_context=None, reason=f"Forced: {slug}"), None

    if args.force_seed:
        seeds = {s.id: s for s in read_seeds()}
        sid = args.force_seed
        if sid not in seeds:
            available = ", ".join(list(seeds)[:10])
            raise ValueError(f"Seed {sid!r} not found. Known (first 10): {available}")
        s = seeds[sid]
        return None, SeedPick(kind="seed", seed=s, release_context=None, reason=f"Forced seed: {sid}")

    seed = pick_seed(today=today)
    if seed is not None:
        return None, SeedPick(kind="seed", seed=seed, release_context=None, reason=f"Seed: {seed.id}")

    return pick_lesson(today=today), None


def main() -> int:
    args = _parse_args()
    today = date.fromisoformat(args.today) if args.today else date.today()

    cfg = config_from_env()
    if args.dry_run:
        cfg.dry_run = True

    logger.info("Today: %s | dry_run: %s", today.isoformat(), cfg.dry_run)

    with RunLogger("send_daily", dry_run=cfg.dry_run) as run:
        # 1. Select
        logger.info("Selecting lesson...")
        lesson_pick, seed_pick = _pick(args, today)

        if seed_pick is not None:
            logger.info("Selected seed: %s — %s", seed_pick.seed.id, seed_pick.reason)
            run.set("slug", seed_pick.seed.id)
            run.set("kind", f"seed:{seed_pick.seed.detector}")
            run.set("reason", seed_pick.reason)
        else:
            logger.info("Selected finding: %s [%s] — %s", lesson_pick.finding.slug, lesson_pick.finding.kind, lesson_pick.reason)
            run.set("slug", lesson_pick.finding.slug)
            run.set("kind", lesson_pick.finding.kind)
            run.set("reason", lesson_pick.reason)

        # 2. Compose
        logger.info("Composing email via Claude (%s)...", "claude-sonnet-4-6")
        try:
            if seed_pick is not None:
                composed = compose_email_from_seed(
                    seed_pick.seed,
                    release_context=seed_pick.release_context,
                    today=today,
                )
            else:
                composed = compose_email(lesson_pick, today=today)
        except Exception as e:
            logger.error("Compose failed (%s), falling back to finding path.", e)
            lesson_pick = pick_lesson(today=today)
            composed = compose_email(lesson_pick, today=today)
            seed_pick = None

        approved = composed.approved
        flags = len(composed.fact_check_flags)
        logger.info("Draft ready. Approved: %s. Flags: %d", approved, flags)
        run.set("approved", approved)
        run.set("fact_check_flags", flags)

        if not approved:
            logger.error("Draft failed fact-check (approved=False). Flags:")
            for fl in composed.fact_check_flags:
                logger.error("  • %s", fl)
            logger.error("Email not sent. Fix finding or re-run with --force-finding after investigation.")
            run.set("exit_code", 1)
            build_health_snapshot()
            return 1

        # 3. Deliver
        logger.info("Delivering email...")
        send_email(composed, cfg, today=today)

        # 4. Record (only on actual send or explicit dry-run that completed cleanly)
        if not args.force_finding and not args.force_seed:
            if seed_pick is not None:
                record_seed_sent(seed_pick.seed, today=today)
                logger.info("Seed recorded: %s", seed_pick.seed.id)
            else:
                record_lesson_sent(lesson_pick.finding, today=today)
                logger.info("Lesson recorded: %s", lesson_pick.finding.slug)

        run.set("exit_code", 0)

    build_health_snapshot()
    return 0


if __name__ == "__main__":
    sys.exit(main())
