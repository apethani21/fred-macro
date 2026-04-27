#!/usr/bin/env python3
"""Telegram bot entry point — answer follow-up questions about the daily email.

First-time setup (do this once):
  1. Open Telegram and message @BotFather:
       /newbot
     Follow the prompts, copy the token you receive.

  2. Store the token:
       mkdir -p ~/keys/telegram
       echo 'YOUR_TOKEN_HERE' > ~/keys/telegram/bot_token.txt
       chmod 600 ~/keys/telegram/bot_token.txt

  3. Message your new bot on Telegram (say anything — just to create the chat).

  4. Find your chat ID:
       python scripts/telegram_bot.py --get-chat-id

  5. Add your chat ID to .env:
       TELEGRAM_ALLOWED_CHAT_ID=123456789

  6. Run the bot:
       python scripts/telegram_bot.py
     Or as a background daemon on EC2:
       nohup python scripts/telegram_bot.py >> logs/telegram_bot.log 2>&1 &

Usage:
  python scripts/telegram_bot.py               # run the polling loop (blocking)
  python scripts/telegram_bot.py --get-chat-id # print incoming chat IDs for setup
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Load .env before src imports.
_env_path = Path(__file__).resolve().parents[1] / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deliver.telegram_bot import get_incoming_chat_ids, run_bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Telegram Q&A bot for the daily macro email.")
    p.add_argument(
        "--get-chat-id",
        action="store_true",
        help="Print incoming chat IDs (run this once after messaging your bot to find your ID).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.get_chat_id:
        results = get_incoming_chat_ids()
        if not results:
            print("No messages found. Make sure you've sent at least one message to your bot on Telegram.")
            return 1
        print("Incoming chat IDs:")
        for chat_id, name in results:
            print(f"  {chat_id}  ({name})")
        print("\nAdd the correct one to .env as: TELEGRAM_ALLOWED_CHAT_ID=<id>")
        return 0

    raw = os.environ.get("TELEGRAM_ALLOWED_CHAT_ID", "")
    if not raw:
        print("TELEGRAM_ALLOWED_CHAT_ID not set in .env. Run with --get-chat-id first.")
        return 1
    try:
        allowed_chat_id = int(raw)
    except ValueError:
        print(f"TELEGRAM_ALLOWED_CHAT_ID must be an integer, got: {raw!r}")
        return 1

    run_bot(allowed_chat_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
