# AI Generated Code - Start
"""
Full end-to-end pipeline:
  ProfileStore → 3× FilterAgent (parallel) → GraphBuilder
  → RankingAgent → WritingAgent → ValidationAgent (retry loop)
  → briefing.txt + briefing.json

Usage:
  python -m src.scripts.run_pipeline
"""

import concurrent.futures
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.core.llm_client import get_client, get_model
from src.core.models import (
    BriefingOutput,
    FilterAgentOutput,
    Priority,
    Section,
    SourceType,
)
from src.core.profile_store import ProfileStore
from src.core.graph_builder import GraphBuilder
from src.agents.calendar_filter_agent import CalendarFilterAgent
from src.agents.email_filter_agent import EmailFilterAgent
from src.agents.news_filter_agent import NewsFilterAgent
from src.agents.ranking_agent import RankingAgent
from src.agents.writing_agent import WritingAgent
from src.agents.validation_agent import ValidationAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s  %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("pipeline")

INPUTS      = Path("inputs")
OUTPUT_TXT  = Path("briefing.txt")
OUTPUT_JSON = Path("briefing.json")

MAX_WRITE_RETRIES = 2   # max WritingAgent retries after validation fails


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _banner(msg: str):
    line = "=" * 65
    print(f"\n{line}\n  {msg}\n{line}", flush=True)


def _collect_covered_ids(sections, source: SourceType) -> list[str]:
    ids = []
    for sec in sections:
        for eid in sec.covered_event_ids:
            if source.value[:3] in eid or (
                source == SourceType.CALENDAR and eid.startswith("cal_")
            ) or (
                source == SourceType.EMAIL and eid.startswith("em_")
            ) or (
                source == SourceType.NEWS and eid.startswith("news_")
            ):
                ids.append(eid)
    return list(dict.fromkeys(ids))   # deduplicate, preserve order


def _build_excluded_list(
    filter_outputs: list[FilterAgentOutput],
) -> list[dict]:
    excluded = []
    for output in filter_outputs:
        for ev in output.excluded:
            excluded.append({
                "id": ev.id,
                "source": ev.source.value,
                "title": ev.title,
                "reason": ev.filter_reason or "",
            })
    return excluded


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    client = get_client()

    # ── Stage 0: ProfileStore ─────────────────────────────────────────────
    _banner("Stage 0 — ProfileStore")
    store = ProfileStore(INPUTS / "profile.json", client, get_model("profile"))
    store.load()
    ctx = store.context
    target_words = int(ctx.audio_target_seconds * 2.5)
    print(
        f"  User: {ctx.user_name}  |  Target: {ctx.audio_target_seconds}s "
        f"(~{target_words} words, range {int(ctx.audio_min_seconds*2.5)}–{int(ctx.audio_max_seconds*2.5)})",
        flush=True,
    )

    # ── Stage 1: Filter Agents ────────────────────────────────────────────
    _banner("Stage 1 — Filter Agents (parallel)")
    cal   = CalendarFilterAgent(INPUTS / "calendar.json", store, client, get_model("filter"))
    email = EmailFilterAgent(   INPUTS / "emails.json",   store, client, get_model("filter"))
    news  = NewsFilterAgent(    INPUTS / "news.json",     store, client, get_model("filter"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        fc, fe, fn = ex.submit(cal.run), ex.submit(email.run), ex.submit(news.run)
        cal_out, email_out, news_out = fc.result(), fe.result(), fn.result()

    print(
        f"  Calendar : {len(cal_out.candidates)} candidates, {len(cal_out.excluded)} excluded\n"
        f"  Email    : {len(email_out.candidates)} candidates, {len(email_out.excluded)} excluded\n"
        f"  News     : {len(news_out.candidates)} candidates, {len(news_out.excluded)} excluded",
        flush=True,
    )

    # ── Stage 2: GraphBuilder ─────────────────────────────────────────────
    _banner("Stage 2 — GraphBuilder")
    builder = GraphBuilder(store)
    graph   = builder.build([cal_out, email_out, news_out])
    print(
        f"  briefing_date = {graph.briefing_date}\n"
        f"  events = {len(graph.events)}, "
        f"topic_tags = {len(graph.topic_tags)}, entity_tags = {len(graph.entity_tags)}",
        flush=True,
    )

    # ── Stage 3: RankingAgent ─────────────────────────────────────────────
    _banner("Stage 3 — RankingAgent")
    ranking_agent  = RankingAgent(store, client, get_model("ranking"))
    ranking_output = ranking_agent.run(graph)
    topic_summary  = "  ".join(
        f"{t.topic_priority.value}:{t.topic}" for t in ranking_output.ranked_topics
    )
    print(f"  Topics: {topic_summary}", flush=True)
    if ranking_output.agent_notes:
        print(f"  Notes : {ranking_output.agent_notes[:120]}…", flush=True)

    # ── Stage 4: WritingAgent + Validation retry loop ─────────────────────
    _banner("Stage 4 — WritingAgent + ValidationAgent")
    writer    = WritingAgent(store, client, get_model("writing"), INPUTS / "session_history.json")
    validator = ValidationAgent(store, client, get_model("validation"), INPUTS / "session_history.json")

    writing_output = writer.run(ranking_output, briefing_date=graph.briefing_date)
    print(
        f"  Draft #1: {writing_output.word_count} words / {writing_output.estimated_seconds}s",
        flush=True,
    )
    print(f"\n--- Draft #1 text ---\n{writing_output.text}\n--- end ---\n", flush=True)

    for attempt in range(MAX_WRITE_RETRIES + 1):
        val_result = validator.run(writing_output, ranking_output, graph.briefing_date)

        errors   = [i for i in val_result.issues if i.severity == "error"]
        warnings = [i for i in val_result.issues if i.severity == "warning"]
        print(
            f"\n  Validation attempt {attempt + 1}: "
            f"passed={val_result.passed}  errors={len(errors)}  warnings={len(warnings)}",
            flush=True,
        )
        for issue in val_result.issues:
            sym = "❌" if issue.severity == "error" else "⚠️ "
            print(f"    {sym} [{issue.check}] {issue.message}", flush=True)

        if val_result.passed:
            print("  ✅ Validation passed.", flush=True)
            break

        if attempt < MAX_WRITE_RETRIES:
            print(f"\n  retry_hint:\n{val_result.retry_hint}\n", flush=True)
            print(f"  Retrying WritingAgent…", flush=True)
            writing_output = writer.run(
                ranking_output,
                briefing_date=graph.briefing_date,
                retry_hint=val_result.retry_hint,
            )
            print(
                f"  Draft #{attempt + 2}: {writing_output.word_count} words / {writing_output.estimated_seconds}s",
                flush=True,
            )
            print(f"\n--- Draft #{attempt + 2} text ---\n{writing_output.text}\n--- end ---\n", flush=True)
        else:
            print(
                f"  ⚠️  Max retries reached — using last draft (may have remaining issues).",
                flush=True,
            )

    # ── Stage 5: Write output files ───────────────────────────────────────
    _banner("Stage 5 — Write briefing.txt + briefing.json")

    # briefing.txt
    OUTPUT_TXT.write_text(writing_output.text, encoding="utf-8")
    print(f"  ✅ briefing.txt written ({writing_output.word_count} words)", flush=True)

    # ── covered_event_ids: from WritingAgent (what it actually mentioned) ──
    # Primary source: WritingAgent self-reports covered_event_ids
    # These are the events the LLM confirmed it mentioned in the text.
    covered_ids = writing_output.covered_event_ids
    cal_covered   = sorted({e for e in covered_ids if e.startswith("cal_")})
    email_covered = sorted({e for e in covered_ids if e.startswith("em_")})
    news_covered  = sorted({e for e in covered_ids if e.startswith("news_")})

    # ── excluded_items: three-layer merge ─────────────────────────────────
    # Layer 1: FilterAgent hard-excluded (semantic / automated / not-interested)
    filter_excluded = _build_excluded_list([cal_out, email_out, news_out])

    # Layer 2: RankingAgent skipped (below topic budget / no topic match)
    # Build a lookup of all events in graph for source/title
    ranking_excluded = []
    for eid in ranking_output.skipped_event_ids:
        ev = graph.events.get(eid)
        ranking_excluded.append({
            "id": eid,
            "source": ev.source.value if ev else "unknown",
            "title": ev.title if ev else "",
            "reason": "below ranking budget or topic capacity",
            "excluded_by": "ranking_agent",
        })

    # Layer 3: WritingAgent unused (in ranking candidates but not mentioned in text)
    ranking_candidate_ids = {
        ev.id
        for topic in ranking_output.ranked_topics
        for ev in topic.events
    }
    writing_excluded = []
    for eid in ranking_candidate_ids - set(covered_ids):
        ev = graph.events.get(eid)
        writing_excluded.append({
            "id": eid,
            "source": ev.source.value if ev else "unknown",
            "title": ev.title if ev else "",
            "reason": "in ranking candidates but omitted by writing agent (word budget)",
            "excluded_by": "writing_agent",
        })

    all_excluded = (
        [dict(e, excluded_by="filter_agent") for e in filter_excluded]
        + ranking_excluded
        + writing_excluded
    )

    # Sections with real char positions (computed by WritingAgent's anchor search)
    sections_json = [
        {
            "name": sec.name,
            "char_start": sec.char_start,
            "char_end": sec.char_end,
            "covered_event_ids": sec.covered_event_ids,
        }
        for sec in writing_output.sections
    ]

    briefing_json = {
        "briefing_date": graph.briefing_date,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "word_count": writing_output.word_count,
        "estimated_seconds": writing_output.estimated_seconds,
        "estimation_method": "word_count / 2.5 words_per_second (TTS at 150 wpm)",
        "target_seconds": ctx.audio_target_seconds,
        "audio_range_seconds": [ctx.audio_min_seconds, ctx.audio_max_seconds],
        "sections": sections_json,
        "covered_calendar_ids": cal_covered,
        "covered_email_ids":    email_covered,
        "covered_news_ids":     news_covered,
        "excluded_items": all_excluded,
        "validation": {
            "passed": val_result.passed,
            "issues": [
                {"severity": i.severity, "check": i.check, "message": i.message}
                for i in val_result.issues
            ],
        },
        "pipeline_stats": {
            "filter_candidates": {
                "calendar": len(cal_out.candidates),
                "email":    len(email_out.candidates),
                "news":     len(news_out.candidates),
            },
            "filter_excluded": {
                "calendar": len(cal_out.excluded),
                "email":    len(email_out.excluded),
                "news":     len(news_out.excluded),
            },
            "graph_events": len(graph.events),
            "ranked_topics": len(ranking_output.ranked_topics),
            "ranking_skipped": ranking_output.skipped_event_ids,
            "writing_covered": len(covered_ids),
            "writing_omitted": len(writing_excluded),
        },
    }

    OUTPUT_JSON.write_text(
        json.dumps(briefing_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"  ✅ briefing.json written  "
        f"(covered: {len(covered_ids)}, excluded total: {len(all_excluded)})",
        flush=True,
    )

    # ── Final print ───────────────────────────────────────────────────────
    _banner("FINAL BRIEFING TEXT")
    print(writing_output.text, flush=True)
    print(
        f"\n  Words: {writing_output.word_count}  |  Est: {writing_output.estimated_seconds}s  "
        f"|  Range: {ctx.audio_min_seconds}–{ctx.audio_max_seconds}s",
        flush=True,
    )
    print(f"\n  Covered  : cal={cal_covered}  email={email_covered}  news={news_covered}", flush=True)
    print(f"  Excluded : filter={len(filter_excluded)}  ranking={len(ranking_excluded)}  writing={len(writing_excluded)}", flush=True)


if __name__ == "__main__":
    main()

# AI Generated Code - End
