"""Seed-based email composition.

Morning path: pick a TopicSeed → build fresh data context → Claude decides the
organizing idea → fact-check → charts → render HTML.

The seed carries the statistical backbone (what was detected overnight). Claude
has full latitude to choose the hook and organizing idea based on what's
happening today.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from src.research.seeds import TopicSeed
from src.research.findings import Finding
from src.select.selector import LessonPick

logger = logging.getLogger(__name__)


# ---------- data context from seed ----------

def build_data_context_from_seed(seed: TopicSeed) -> dict:
    """Load fresh parquet values for seed's series. Falls back to seed.key_stats on error."""
    from src.compose.composer import build_data_context, SeriesSnapshot

    fake_finding = _seed_to_finding(seed)
    ctx = build_data_context(fake_finding)

    # For any series that failed to load, fall back to seed.key_stats snapshot
    for sid in seed.series_ids:
        if sid not in ctx and sid in seed.key_stats:
            ks = seed.key_stats[sid]
            ctx[sid] = SeriesSnapshot(
                series_id=sid,
                title=ks.get("title", sid),
                units=ks.get("units", ""),
                frequency="",
                current_value=ks.get("value"),
                current_date=ks.get("date"),
                z_score=ks.get("z_score"),
                percentile_rank=ks.get("percentile_10y"),
                z_score_5y=None,
                percentile_5y=None,
                z_score_10y=None,
                percentile_10y=ks.get("percentile_10y"),
                history_start=None,
                n_obs=0,
            )
            logger.warning("seed %s: using key_stats fallback for %s", seed.id, sid)

    return ctx


# ---------- LLM draft ----------

def draft_email_from_seed(
    seed: TopicSeed,
    ctx: dict,
    release_context: dict | None = None,
) -> dict:
    """Call Claude to produce subject + HTML/text body from a seed.

    Claude decides the organizing idea; numbers come from ctx, not seed.key_stats.
    """
    from src.compose.composer import _client, _DRAFT_SYSTEM, _ctx_to_dict, _extract_json, _load_concept_context

    client = _client()

    release_names: list[str] = []
    if release_context and release_context.get("releases"):
        release_names = [r.get("release_name", "") for r in release_context["releases"][:3]]

    concept_ctx = _load_concept_context(
        list(seed.series_ids),
        keywords=seed.concept_anchors,
    )

    sources_text = ""
    if seed.sources:
        lines = [
            f"- {s.get('institution', '')} ({s.get('date', '')}): {s.get('title', '')} — {s.get('url', '')}"
            for s in seed.sources
        ]
        sources_text = "SOURCES (pre-fetched):\n" + "\n".join(lines)

    ctx_dict = _ctx_to_dict(ctx)

    user_content = (
        f"STATISTICAL BACKBONE (what was detected overnight — raw stats only, do not quote these numbers in prose):\n"
        f"{json.dumps(seed.key_stats, indent=2)}\n\n"
        f"DETECTOR: {seed.detector}\n\n"
        f"DATA CONTEXT (fresh morning values — use THESE numbers in the email, not the backbone):\n"
        f"{json.dumps(ctx_dict, indent=2)}\n\n"
        f"UPCOMING RELEASES IN NEXT 2 DAYS: {', '.join(release_names) if release_names else 'None'}\n"
        "Use the most relevant release as the hook anchor if applicable. "
        "Explain the mechanistic link between the release and today's finding in one sentence.\n\n"
    )
    if concept_ctx:
        user_content += (
            f"CONCEPT CONTEXT (background — use for historical regime context and institutional detail):\n"
            f"{concept_ctx}\n\n"
        )
    if sources_text:
        user_content += sources_text + "\n\n"

    user_content += (
        "TASK: The backbone shows what was detected overnight. You have full latitude to decide:\n"
        "- The organizing idea (what is the educational point?)\n"
        "- The hook (what makes this interesting today?)\n"
        "Draw on the concept context and current DATA CONTEXT values. "
        "Every number in the email must come from DATA CONTEXT, not the backbone. "
        "Return only the JSON object."
    )

    from src.compose.composer import MODEL
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=[{"type": "text", "text": _DRAFT_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_content}],
    )
    return _extract_json(response.content[0].text)


# ---------- fact-check ----------

def fact_check_seed_draft(
    draft: dict,
    seed: TopicSeed,
    ctx: dict,
) -> dict:
    """Fact-check a seed-based draft. Delegates to standard fact_check_draft via shim."""
    from src.compose.composer import fact_check_draft
    shim_pick = _seed_to_lesson_pick(seed, release_context=None)
    return fact_check_draft(draft, shim_pick, ctx)


# ---------- full pipeline ----------

def compose_email_from_seed(
    seed: TopicSeed,
    release_context: dict | None = None,
    today: date | None = None,
) -> "ComposedEmail":  # noqa: F821
    """Full pipeline: fresh data → draft → fact-check → charts → render HTML."""
    from src.compose.composer import (
        ComposedEmail,
        generate_charts,
        generate_equation_image,
        render_html,
        CHARTS_DIR,
        _ctx_to_dict,
    )

    today = today or date.today()

    ctx = build_data_context_from_seed(seed)
    if not ctx:
        logger.warning("No data context for seed %s — series may not be tracked.", seed.id)

    draft = draft_email_from_seed(seed, ctx, release_context=release_context)
    fc = fact_check_seed_draft(draft, seed, ctx)

    flags = fc.get("flags", [])
    approved = fc.get("approved", True)

    if flags:
        logger.info("Seed fact-check flags (%d): %s", len(flags), "; ".join(flags))

    shim_pick = _seed_to_lesson_pick(seed, release_context=release_context)
    chart_paths = generate_charts(shim_pick, ctx, today)

    chart_dir = CHARTS_DIR / today.isoformat()
    chart_dir.mkdir(parents=True, exist_ok=True)
    equation_path = generate_equation_image(shim_pick.finding, chart_dir, today)

    cids = [f"chart_{i}" for i in range(len(chart_paths))]
    eq_cid = "equation_0" if equation_path else None
    template_body = draft["body_html"]
    html_body = render_html(
        draft["subject"], template_body, today,
        chart_cids=cids, equation_cid=eq_cid,
    )

    return ComposedEmail(
        subject=draft["subject"],
        html_body=html_body,
        html_body_template=template_body,
        text_body=draft.get("body_text", ""),
        chart_paths=chart_paths,
        equation_path=equation_path,
        fact_check_flags=flags,
        approved=approved,
        data_context=_ctx_to_dict(ctx),
    )


# ---------- shim helpers ----------

def _seed_to_finding(seed: TopicSeed) -> Finding:
    """Minimal Finding from a seed — used for data context loading."""
    from datetime import date as _date
    return Finding(
        slug=seed.id,
        title=seed.detector,
        kind=seed.detector,
        discovered=_date.today(),
        series_ids=seed.series_ids,
        window=None,
        claim="",
        evidence={},
        interpretation="",
        sources=[],
        status="new",
        score=seed.priority_score,
    )


def _seed_to_lesson_pick(
    seed: TopicSeed,
    release_context: dict | None = None,
) -> LessonPick:
    """LessonPick shim for passing to generate_charts / fact_check_draft."""
    finding = _seed_to_finding(seed)
    return LessonPick(
        kind="seed",
        finding=finding,
        release_context=release_context,
        reason=f"Seed: {seed.id}",
    )
