# AI Generated Code - Start
"""
Pipeline validation script: Filter Agents → GraphBuilder → TopicSelector → RankingAgent

Shows all intermediate outputs clearly.
Usage: python -m src.scripts.validate_pipeline
"""

import json
import logging
import concurrent.futures
from pathlib import Path

from src.core.llm_client import get_client, get_model
from src.core.profile_store import ProfileStore
from src.core.graph_builder import GraphBuilder
from src.core.topic_selector import TopicSelector
from src.agents.calendar_filter_agent import CalendarFilterAgent
from src.agents.email_filter_agent import EmailFilterAgent
from src.agents.news_filter_agent import NewsFilterAgent
from src.agents.ranking_agent import RankingAgent
from src.core.models import FilterAgentOutput, Priority, SourceType, SpecialFlag

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(name)s  %(message)s")

INPUTS = Path("inputs")

SEP  = "=" * 70
SEP2 = "─" * 70
P_EMOJI = {Priority.P0: "🔴", Priority.P1: "🟡", Priority.P2: "🟢", Priority.P3: "⚪"}
S_EMOJI = {SourceType.CALENDAR: "📅", SourceType.EMAIL: "📧", SourceType.NEWS: "📰"}


def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def subsection(title: str):
    print(f"\n  ── {title} ──")


# ─────────────────────────────────────────────
# STAGE 1  Filter Agents
# ─────────────────────────────────────────────

def print_filter_output(output: FilterAgentOutput):
    src_emoji = S_EMOJI.get(output.source, "?")
    src_name  = output.source.value.upper()
    cands     = output.candidates
    excl      = output.excluded
    print(f"\n  {src_emoji} {src_name}  ✅ {len(cands)} candidates  ❌ {len(excl)} excluded")
    print(f"  {SEP2}")

    for e in sorted(cands, key=lambda x: x.id):
        flags = [f.value for f in e.special_flags]
        dec   = {"include": "✅", "deprioritize": "⬇ "}.get(e.filter_decision.value, "?")
        print(f"  {dec} {e.id:<12}  {e.title[:52]}")
        if flags:
            print(f"              flags:   {flags}")
        print(f"              entity:  {e.entity_tags}")
        print(f"              topic:   {e.topic_tags}")
        if e.filter_reason and e.filter_decision.value == "deprioritize":
            print(f"              reason:  {e.filter_reason}")

    if excl:
        print(f"\n  ── Excluded ({len(excl)}) ──")
        for e in excl:
            print(f"  ❌ {e.id:<12}  {e.title[:52]}")
            print(f"              reason: {e.filter_reason}")

    if output.agent_notes:
        print(f"\n  📝 {output.agent_notes}")


# ─────────────────────────────────────────────
# STAGE 2  GraphBuilder
# ─────────────────────────────────────────────

def print_graph_summary(graph):
    print(f"\n  Briefing date : {graph.briefing_date}")
    print(f"  Events in graph : {len(graph.events)}")
    print(f"  Topic tags : {len(graph.topic_tags)}  |  Entity tags : {len(graph.entity_tags)}")

    print(f"\n  Top 10 topic tags by effective_weight:")
    top_topics = sorted(graph.topic_tags.values(), key=lambda t: t.effective_weight, reverse=True)[:10]
    for t in top_topics:
        bar = "█" * min(int(t.effective_weight / 2), 15)
        print(f"    {t.name:<34}  in_degree={t.in_degree}  eff={t.effective_weight:>5.1f}  {bar}")

    print(f"\n  Topic clusters (in_degree > 1) — cross-source signal:")
    clusters = [(n, node) for n, node in graph.topic_tags.items() if node.in_degree > 1]
    clusters.sort(key=lambda x: x[1].effective_weight, reverse=True)
    for name, node in clusters[:8]:
        ids = [f"{eid}({graph.events[eid].source.value[:3]})" for eid in node.event_ids if eid in graph.events]
        print(f"    [{name}]  eff={node.effective_weight:.1f}  →  {', '.join(ids)}")


# ─────────────────────────────────────────────
# STAGE 3  TopicSelector
# ─────────────────────────────────────────────

def print_topic_selector(topic_slots, overflow_events):
    total_events = sum(s.event_count for s in topic_slots)
    print(f"\n  {len(topic_slots)} topics selected  |  {total_events} events  |  {len(overflow_events)} overflow")
    print()
    for slot in topic_slots:
        print(f"  [{slot.topic:<34}]  eff={slot.effective_weight:>5.1f}  ({slot.event_count} events)")
        for e in slot.events:
            flags = [f.value for f in e.special_flags]
            forced = "⚠️ " if any(f in {"action-required","from-ceo","deadline-today"} for f in [f.value for f in e.special_flags]) else "   "
            print(f"    {forced}{e.id:<12}  {S_EMOJI.get(e.source,'?')}  {e.title[:48]}")
            if flags:
                print(f"                  flags: {flags}")

    if overflow_events:
        print(f"\n  ⚠️  Special-flag overflow (not in top topics):")
        for e in overflow_events:
            print(f"    {e.id}  {[f.value for f in e.special_flags]}  {e.title[:50]}")


# ─────────────────────────────────────────────
# STAGE 4  RankingAgent final output
# ─────────────────────────────────────────────

def print_ranking_output(ranking_output):
    print(f"\n  {len(ranking_output.ranked_topics)} topics  |  skipped: {ranking_output.skipped_event_ids}")
    if ranking_output.agent_notes:
        print(f"\n  📝 WritingAgent notes:\n     {ranking_output.agent_notes}\n")

    total_p = {Priority.P0: 0, Priority.P1: 0, Priority.P2: 0}
    for topic in ranking_output.ranked_topics:
        for ev in topic.events:
            total_p[ev.event_priority] = total_p.get(ev.event_priority, 0) + 1

    print(f"  Event distribution  🔴 P0={total_p[Priority.P0]}  🟡 P1={total_p[Priority.P1]}  🟢 P2={total_p.get(Priority.P2,0)}")
    print()

    for topic in ranking_output.ranked_topics:
        tp  = P_EMOJI.get(topic.topic_priority, "?")
        print(f"  {tp} [{topic.topic}]  ({topic.topic_priority.value})  eff={topic.effective_weight:.1f}")
        if topic.topic_summary:
            print(f"     💡 {topic.topic_summary}")
        for ev in topic.events:
            ep    = P_EMOJI.get(ev.event_priority, "?")
            flags = [f.value for f in ev.special_flags]
            src   = S_EMOJI.get(ev.source, "?")
            print(f"     {ep} {src} {ev.id:<12}  {ev.title[:48]}")
            if flags:
                print(f"                   flags: {flags}")
            if ev.narration_hint:
                print(f"                   hint : {ev.narration_hint}")
        print()


# ─────────────────────────────────────────────
# FINAL  What WritingAgent receives
# ─────────────────────────────────────────────

def print_writing_agent_input(ranking_output):
    """Print a clean JSON of exactly what WritingAgent will consume."""
    payload = {
        "briefing_date": "",   # filled below
        "agent_notes": ranking_output.agent_notes,
        "ranked_topics": []
    }
    for topic in ranking_output.ranked_topics:
        t_dict = {
            "topic": topic.topic,
            "topic_priority": topic.topic_priority.value,
            "topic_summary": topic.topic_summary,
            "events": []
        }
        for ev in topic.events:
            t_dict["events"].append({
                "id": ev.id,
                "source": ev.source.value,
                "event_priority": ev.event_priority.value,
                "special_flags": [f.value for f in ev.special_flags],
                "title": ev.title,
                "summary": ev.summary,
                "narration_hint": ev.narration_hint,
                "timestamp": ev.timestamp,
            })
        payload["ranked_topics"].append(t_dict)
    payload["skipped_event_ids"] = ranking_output.skipped_event_ids
    print(json.dumps(payload, indent=2, ensure_ascii=False))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    client = get_client()

    # ── ProfileStore ──────────────────────────────────────────────────
    section("STAGE 0 — ProfileStore")
    store = ProfileStore(INPUTS / "profile.json", client, get_model("profile"))
    ctx = store.load()
    print(f"\n  User    : {ctx.user_name}  ({ctx.user_role} @ {ctx.user_company})")
    print(f"  Tone    : {ctx.tone_style}")
    print(f"  Target  : {ctx.audio_target_seconds}s  (~{int(ctx.audio_target_seconds*2.5)} words)")
    print(f"  Interests  : {[t.tag for t in ctx.interest_tags]}")
    print(f"  Blocked    : {[t.tag for t in ctx.blocked_tags]}")
    print(f"  Entities   : {[t.tag for t in ctx.entity_tags]}")

    # ── Filter Agents (concurrent) ────────────────────────────────────
    section("STAGE 1 — Filter Agents (concurrent)")
    print("  Running Calendar / Email / News agents in parallel…")

    cal_agent   = CalendarFilterAgent(INPUTS / "calendar.json", store, client, get_model("filter"))
    email_agent = EmailFilterAgent(INPUTS / "emails.json",      store, client, get_model("filter"))
    news_agent  = NewsFilterAgent(INPUTS / "news.json",         store, client, get_model("filter"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        fc = executor.submit(cal_agent.run)
        fe = executor.submit(email_agent.run)
        fn = executor.submit(news_agent.run)
        cal_output   = fc.result()
        email_output = fe.result()
        news_output  = fn.result()

    print_filter_output(cal_output)
    print_filter_output(email_output)
    print_filter_output(news_output)

    filter_summary = (
        f"\n  FILTER TOTAL  ✅ candidates: "
        f"{len(cal_output.candidates)} cal + {len(email_output.candidates)} email + {len(news_output.candidates)} news"
        f" = {len(cal_output.candidates)+len(email_output.candidates)+len(news_output.candidates)}"
        f"   ❌ excluded: {len(cal_output.excluded)+len(email_output.excluded)+len(news_output.excluded)}"
    )
    print(filter_summary)

    # ── GraphBuilder ─────────────────────────────────────────────────
    section("STAGE 2 — GraphBuilder")
    builder = GraphBuilder(store)
    graph = builder.build([cal_output, email_output, news_output])
    print_graph_summary(graph)

    # ── TopicSelector ────────────────────────────────────────────────
    section("STAGE 3 — TopicSelector  (Python, deterministic)")
    selector = TopicSelector(graph, max_events=22)
    topic_slots, overflow_events = selector.select()
    print_topic_selector(topic_slots, overflow_events)

    # ── RankingAgent ─────────────────────────────────────────────────
    section("STAGE 4 — RankingAgent  (LLM)")
    ranking_agent = RankingAgent(store, client, get_model("ranking"))
    ranking_output = ranking_agent.run(graph)
    print_ranking_output(ranking_output)

    # ── Final payload for WritingAgent ───────────────────────────────
    section("STAGE 4 OUTPUT — WritingAgent input payload (JSON)")
    # inject briefing_date
    import json as _json
    payload_obj = _json.loads(
        _json.dumps({
            "briefing_date": graph.briefing_date,
            "agent_notes": ranking_output.agent_notes,
            "ranked_topics": [
                {
                    "topic": t.topic,
                    "topic_priority": t.topic_priority.value,
                    "topic_summary": t.topic_summary,
                    "events": [
                        {
                            "id": e.id,
                            "source": e.source.value,
                            "event_priority": e.event_priority.value,
                            "special_flags": [f.value for f in e.special_flags],
                            "title": e.title,
                            "summary": e.summary,
                            "narration_hint": e.narration_hint,
                            "timestamp": e.timestamp,
                        }
                        for e in t.events
                    ],
                }
                for t in ranking_output.ranked_topics
            ],
            "skipped_event_ids": ranking_output.skipped_event_ids,
        }, ensure_ascii=False)
    )
    print(json.dumps(payload_obj, indent=2, ensure_ascii=False))

    # ── Sanity checks ────────────────────────────────────────────────
    section("SANITY CHECKS")
    errors = []
    warnings = []

    # 1. Date consistency
    if graph.briefing_date:
        print(f"  ✅ briefing_date = {graph.briefing_date}")
    else:
        errors.append("briefing_date is empty")

    # 2. All action-required events appear somewhere in ranked output
    action_ids = {
        e.id for e in graph.events.values()
        if SpecialFlag.ACTION_REQUIRED in e.special_flags
    }
    ranked_ids = {
        ev.id
        for topic in ranking_output.ranked_topics
        for ev in topic.events
    }
    skipped_ids = set(ranking_output.skipped_event_ids)
    missing_action = action_ids - ranked_ids - skipped_ids
    if missing_action:
        errors.append(f"action-required events not in output: {missing_action}")
    else:
        print(f"  ✅ All action-required events accounted for ({len(action_ids)} total)")

    # 3. No P0 topic has 0 P0 events
    for topic in ranking_output.ranked_topics:
        if topic.topic_priority == Priority.P0:
            p0_events = [e for e in topic.events if e.event_priority == Priority.P0]
            if not p0_events:
                warnings.append(f"P0 topic '{topic.topic}' has no P0 events — unusual")

    # 4. Ranked topics are ordered P0 → P1 → P2
    prev_p = Priority.P0
    priority_order = {Priority.P0: 0, Priority.P1: 1, Priority.P2: 2, Priority.P3: 3}
    ordered_ok = True
    for t in ranking_output.ranked_topics:
        if priority_order[t.topic_priority] < priority_order[prev_p]:
            ordered_ok = False
            errors.append(f"Topic ordering violated: {t.topic} ({t.topic_priority}) after {prev_p}")
        prev_p = t.topic_priority
    if ordered_ok:
        print(f"  ✅ Topics ordered P0 → P1 → P2 correctly")

    # 5. narration_hint present for all P0 events
    missing_hints = [
        f"{ev.id}({topic.topic})"
        for topic in ranking_output.ranked_topics
        for ev in topic.events
        if ev.event_priority == Priority.P0 and not ev.narration_hint.strip()
    ]
    if missing_hints:
        warnings.append(f"P0 events missing narration_hint: {missing_hints}")
    else:
        print(f"  ✅ All P0 events have narration_hint")

    if warnings:
        for w in warnings:
            print(f"  ⚠️  {w}")
    if errors:
        for e in errors:
            print(f"  ❌ ERROR: {e}")
        print(f"\n  Pipeline has {len(errors)} error(s) — review before passing to WritingAgent")
    else:
        print(f"\n  ✅ All checks passed — output is ready for WritingAgent")

    print(f"\n{SEP}\n  Done.\n{SEP}\n")


if __name__ == "__main__":
    main()

# AI Generated Code - End
