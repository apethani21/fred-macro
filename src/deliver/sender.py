"""Email delivery via AWS SES (or dry-run to state/last_email.html).

SES credentials use the standard boto3 credential chain:
  1. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
  2. ~/.aws/credentials (standard INI)
  3. ~/keys/aws/credentials (INI, same format) — set explicitly if present

Charts are embedded as inline MIME attachments (Content-ID references) so
they render inline in most email clients without a separate download.

Dry-run mode (DRY_RUN=true in .env, or --dry-run flag): writes rendered HTML
to state/last_email.html and chart PNGs to state/charts/{date}/ instead of
sending. Always use dry-run for local development.
"""
from __future__ import annotations

import email.utils
import logging
import mimetypes
import os
from dataclasses import dataclass
from datetime import date
from email import encoders
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from src.ingest.paths import STATE_DIR
from src.compose.composer import ComposedEmail

logger = logging.getLogger(__name__)

LAST_EMAIL_PATH = STATE_DIR / "last_email.html"


@dataclass
class DeliveryConfig:
    email_from: str
    email_to: str
    ses_region: str = "eu-west-1"
    dry_run: bool = True


def config_from_env() -> DeliveryConfig:
    """Read delivery config from environment / .env variables."""
    return DeliveryConfig(
        email_from=os.environ.get("EMAIL_FROM", ""),
        email_to=os.environ.get("EMAIL_TO", ""),
        ses_region=os.environ.get("SES_REGION", "eu-west-1"),
        dry_run=os.environ.get("DRY_RUN", "true").lower() in ("1", "true", "yes"),
    )


def send_email(
    composed: ComposedEmail,
    cfg: DeliveryConfig,
    today: date | None = None,
) -> None:
    """Send (or dry-run) the composed email.

    In dry-run mode: writes HTML to state/last_email.html.
    In live mode: sends via SES.
    """
    today = today or date.today()

    if cfg.dry_run:
        _dry_run(composed, today)
    else:
        if not cfg.email_from or not cfg.email_to:
            raise ValueError("EMAIL_FROM and EMAIL_TO must be set in .env for live send.")
        _send_ses(composed, cfg)


# ---------- dry-run ----------

def _dry_run(composed: ComposedEmail, today: date) -> None:
    from src.compose.composer import render_html

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    # Re-render the template body with base64 data URIs so the browser preview shows charts inline.
    html_preview = render_html(
        composed.subject,
        composed.html_body_template,
        today,
        chart_paths=composed.chart_paths,
    )
    LAST_EMAIL_PATH.write_text(html_preview)
    logger.info("[DRY RUN] Email written to %s", LAST_EMAIL_PATH)
    for p in composed.chart_paths:
        if p.exists():
            logger.info("[DRY RUN] Chart at %s", p)
    if composed.fact_check_flags:
        logger.info("[DRY RUN] Fact-check flags: %s", composed.fact_check_flags)
    _print_summary(composed)


def _print_summary(composed: ComposedEmail) -> None:
    print(f"\n{'='*60}")
    print(f"SUBJECT: {composed.subject}")
    print(f"APPROVED: {composed.approved}")
    if composed.fact_check_flags:
        print(f"FLAGS ({len(composed.fact_check_flags)}):")
        for fl in composed.fact_check_flags:
            print(f"  • {fl}")
    if composed.chart_paths:
        for i, p in enumerate(composed.chart_paths, 1):
            print(f"CHART {i}: {p}")
    else:
        print("CHARTS: none")
    print(f"{'='*60}\n")
    # Print plain text preview
    lines = composed.text_body.strip().splitlines()
    preview = "\n".join(lines[:10])
    print(preview)
    if len(lines) > 10:
        print(f"... ({len(lines) - 10} more lines)")


# ---------- live SES send ----------

def _ses_client(region: str):
    import boto3

    # If ~/keys/aws/credentials exists as an INI file, configure explicitly.
    cred_file = Path.home() / "keys" / "aws" / "credentials"
    if cred_file.exists():
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(cred_file)
        section = "default" if "default" in cfg else (list(cfg.sections())[0] if cfg.sections() else None)
        if section:
            return boto3.client(
                "ses",
                region_name=region,
                aws_access_key_id=cfg[section].get("aws_access_key_id"),
                aws_secret_access_key=cfg[section].get("aws_secret_access_key"),
            )

    # Fall back to standard boto3 chain (env vars, ~/.aws/credentials, instance metadata).
    return boto3.client("ses", region_name=region)


def _build_mime(composed: ComposedEmail, from_addr: str, to_addr: str) -> MIMEMultipart:
    """Build a MIME multipart/mixed email with optional inline charts."""
    outer = MIMEMultipart("mixed")
    outer["Subject"] = composed.subject
    outer["From"] = from_addr
    outer["To"] = to_addr
    outer["Date"] = email.utils.formatdate(localtime=True)

    # multipart/alternative for plain + HTML
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(composed.text_body, "plain", "utf-8"))
    alt.attach(MIMEText(composed.html_body, "html", "utf-8"))
    outer.attach(alt)

    # Inline chart attachments — one MIME part per chart, CID = chart_0, chart_1, ...
    for i, path in enumerate(composed.chart_paths or []):
        if path and path.exists():
            with path.open("rb") as f:
                img_data = f.read()
            img = MIMEImage(img_data, "png")
            img.add_header("Content-ID", f"<chart_{i}>")
            img.add_header("Content-Disposition", "inline", filename=path.name)
            outer.attach(img)

    return outer


def _send_ses(composed: ComposedEmail, cfg: DeliveryConfig) -> None:
    ses = _ses_client(cfg.ses_region)
    msg = _build_mime(composed, cfg.email_from, cfg.email_to)
    response = ses.send_raw_email(
        Source=cfg.email_from,
        Destinations=[cfg.email_to],
        RawMessage={"Data": msg.as_bytes()},
    )
    logger.info("Email sent. MessageId: %s", response["MessageId"])
