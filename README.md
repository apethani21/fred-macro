# fred-macro

Personal system for daily macro education. Ingests Federal Reserve economic data (FRED), builds a knowledge base, and emails a short daily lesson.

Private project. See `CLAUDE.md` for architecture and conventions (gitignored).

## Setup

```bash
conda activate fred-macro
cp .env.example .env  # fill in email config
```

FRED API key expected at `~/keys/fred/key.txt`.

## Entry points

- `scripts/refresh_data.py` — pull/refresh series data
- `scripts/run_research.py` — recompute findings
- `scripts/send_daily.py` — compose and send the daily email
