"""Telegram bot for answering follow-up questions about the daily macro email.

Uses the Telegram Bot API directly via `requests` (no extra library).

Credential / config:
  - Bot token:          ~/keys/telegram/bot_token.txt
  - Anthropic key:      ~/keys/anthropic/key.txt  (or ANTHROPIC_API_KEY env var)
  - Allowed chat ID:    TELEGRAM_ALLOWED_CHAT_ID in .env

The bot only responds to the single chat ID configured in .env. All other
senders are silently ignored.

Run as a daemon on EC2:
  nohup python scripts/telegram_bot.py >> logs/telegram_bot.log 2>&1 &
Or via @reboot cron — see scripts/crontab.example.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import requests

from src.ingest.paths import STATE_DIR

logger = logging.getLogger(__name__)

TELEGRAM_BASE = "https://api.telegram.org/bot{token}/{method}"
OFFSET_FILE = STATE_DIR / "telegram_offset.json"
MAX_MESSAGE_LEN = 4096  # Telegram hard limit


# ---------- credentials ----------

def _load_bot_token() -> str:
    path = Path.home() / "keys" / "telegram" / "bot_token.txt"
    if path.exists():
        return path.read_text().strip()
    raise FileNotFoundError(
        f"Telegram bot token not found at {path}.\n"
        "Create it with: mkdir -p ~/keys/telegram && echo 'YOUR_TOKEN' > ~/keys/telegram/bot_token.txt"
    )


def _load_anthropic_key() -> str:
    import os
    if key := os.environ.get("ANTHROPIC_API_KEY"):
        return key
    path = Path.home() / "keys" / "anthropic" / "key.txt"
    if path.exists():
        return path.read_text().strip()
    raise FileNotFoundError(
        "Anthropic API key not found. Set ANTHROPIC_API_KEY or create ~/keys/anthropic/key.txt"
    )


# ---------- email context ----------

def _strip_html(html_text: str) -> str:
    from bs4 import BeautifulSoup
    text = BeautifulSoup(html_text, "html.parser").get_text(separator="\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _load_email_context(max_chars: int = 4000) -> str | None:
    path = STATE_DIR / "last_email.html"
    if not path.exists():
        return None
    return _strip_html(path.read_text())[:max_chars]


# ---------- Telegram API ----------

def _api(token: str, method: str, **kwargs) -> dict:
    url = TELEGRAM_BASE.format(token=token, method=method)
    resp = requests.post(url, json=kwargs, timeout=40)
    resp.raise_for_status()
    return resp.json()


def _get_updates(token: str, offset: int, timeout: int = 30) -> list[dict]:
    data = _api(token, "getUpdates", offset=offset, timeout=timeout, allowed_updates=["message"])
    return data.get("result", [])


def _send_message(token: str, chat_id: int, text: str) -> None:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    for chunk in _chunk(text):
        _api(token, "sendMessage", chat_id=chat_id, text=chunk)


def _chunk(text: str, limit: int = MAX_MESSAGE_LEN) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        chunks.append(text[:limit])
        text = text[limit:]
    return chunks


# ---------- Claude ----------

def _answer_question(question: str, anthropic_key: str, email_context: str | None) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=anthropic_key)

    system = (
        "You are a macro economics tutor replying to a quantitative finance professional "
        "with three years of equities experience. They read a daily macro education email "
        "and are asking a follow-up question.\n\n"
        "Rules:\n"
        "- Be precise and specific. Use numbers, dates, and FRED series IDs where relevant.\n"
        "- No hedging-as-authority ('many economists believe'). Cite specific sources or be direct.\n"
        "- Explain macro-specific jargon and institutional mechanics. Don't over-explain "
        "statistics or market structure — they know those.\n"
        "- When a macro concept has an equities analogue, point it out.\n"
        "- Keep answers under 400 words unless the question genuinely requires more.\n"
        "- Plain text only. No HTML. Bold (**word**) and bullet lists are fine."
    )

    parts: list[str] = []
    if email_context:
        parts.append(f"Context — today's email:\n\n{email_context}")
    parts.append(f"Question: {question}")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": "\n\n---\n\n".join(parts)}],
    )
    return response.content[0].text


# ---------- offset persistence ----------

def _load_offset() -> int:
    if OFFSET_FILE.exists():
        return json.loads(OFFSET_FILE.read_text()).get("offset", 0)
    return 0


def _save_offset(offset: int) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    OFFSET_FILE.write_text(json.dumps({"offset": offset}))


# ---------- public interface ----------

def get_incoming_chat_ids() -> list[tuple[int, str]]:
    """One-shot poll — returns (chat_id, username_or_name) for setup."""
    token = _load_bot_token()
    updates = _get_updates(token, offset=0, timeout=5)
    results: list[tuple[int, str]] = []
    for u in updates:
        msg = u.get("message", {})
        chat = msg.get("chat", {})
        chat_id = chat.get("id")
        name = chat.get("username") or chat.get("first_name") or "unknown"
        if chat_id:
            results.append((chat_id, name))
    return results


def run_bot(allowed_chat_id: int) -> None:
    """Blocking polling loop. Runs until interrupted."""
    token = _load_bot_token()
    anthropic_key = _load_anthropic_key()
    offset = _load_offset()

    logger.info("Telegram bot started. Allowed chat ID: %d", allowed_chat_id)

    while True:
        try:
            updates = _get_updates(token, offset=offset, timeout=30)
        except requests.exceptions.Timeout:
            continue
        except requests.exceptions.RequestException as exc:
            logger.error("getUpdates failed: %s — retrying in 10s", exc)
            time.sleep(10)
            continue

        for update in updates:
            offset = max(offset, update["update_id"] + 1)
            _save_offset(offset)

            msg = update.get("message", {})
            chat_id: int | None = msg.get("chat", {}).get("id")
            text: str = (msg.get("text") or "").strip()

            if not text or chat_id is None:
                continue

            if chat_id != allowed_chat_id:
                logger.warning("Message from unknown chat_id %d — ignored.", chat_id)
                continue

            logger.info("Received: %s", text[:120])

            if text.lower() in ("/start", "/help"):
                _send_message(
                    token, chat_id,
                    "Ask me anything about macro or today's email. "
                    "I have the text of your most recent daily email as context."
                )
                continue

            try:
                context = _load_email_context()
                answer = _answer_question(text, anthropic_key, context)
                _send_message(token, chat_id, answer)
                logger.info("Replied (%d chars).", len(answer))
            except Exception as exc:
                logger.error("Answer failed: %s", exc, exc_info=True)
                _send_message(token, chat_id, "Hit an error processing that — try again.")
