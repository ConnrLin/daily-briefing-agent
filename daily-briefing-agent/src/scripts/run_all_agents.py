# AI Generated Code - Start
"""
Full runner: executes all three Filter Agents concurrently, then builds the merged KnowledgeGraph,
then runs TopicSelector + RankingAgent.

Usage:
  python -m src.scripts.run_all_agents
"""

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
from src.core.models import FilterAgentOutput, Priority, SourceType

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

INPUTS_DIR = Path("inputs")

# ──────────────────────────────────────────────────────────
# Print helpers
# ──────────────────────────────────────────────────────────

SOURCE_EMOJI = {
    SourceType.CALENDAR: "📅",
    SourceType.EMAIL:    "📧",
    SourceType.NEWS:     "📰",
}

DECISION_LABEL = {
    "include":      "✅",
    "deprioritize": "⬇ ",
    "exclude":      "❌",
}


def print_filter_output(output: FilterAgentOutput):
    emoji = SOURCE_EMOJI.get(output.source, "?")
    src = output.source.value.upper()
    print(f"\n{emoji} {src} FILTER  |  ✅ {len(output.candidates)} candidates  ❌ {len(output.excluded)} excluded")
    print("─" * 72)
    for c in sorted(output.candidates, key=lambda x: x.id):
        label = DECISION_LABEL.get(c.filter_decision.value, "?")
        flags_str = ", ".join(f.value for f in c.special_flags) or "—"
        print(f"  {label} {c.id:<10}  {c.title[:50]}")
        print(f"             flags       : {flags_str}")
        print(f"             entity_tags : {c.entity_tags}")
        print(f"             topic_tags  : {c.topic_tags}")

    if output.excluded:
        print(f"\n  ── Excluded ──")
        for e in output.excluded:
            print(f"  ❌ {e.id:<10}  {e.title[:50]}")
            print(f"             reason: {e.filter_reason}")

    if output.agent_notes:
        print(f"\n  📝 Agent notes: {output.agent_notes}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    client = get_client()

    # ── Step 1: ProfileStore ──
    print("\n" + "=" * 72)
    print("  Step 1: ProfileStore — LLM tag extraction")
    print("=" * 72)

    store = ProfileStore(
        profile_path=INPUTS_DIR / "profile.json",
        client=client,
        model=get_model("profile"),
    )
    ctx = store.load()

    print(f"\n✓ User: {ctx.user_name} ({ctx.user_role} @ {ctx.user_company})")
    print(f"  Interest tags  : {[t.tag for t in ctx.interest_tags]}")
    print(f"  Blocked tags   : {[t.tag for t in ctx.blocked_tags]}")
    print(f"  Entity tags    : {[t.tag for t in ctx.entity_tags]}")

    # ── Step 2: Run three Filter Agents concurrently ──
    print("\n" + "=" * 72)
    print("  Step 2: Running all Filter Agents (concurrent)")
    print("=" * 72)

    cal_agent   = CalendarFilterAgent(INPUTS_DIR / "calendar.json", store, client, get_model("filter"))
    email_agent = EmailFilterAgent(INPUTS_DIR / "emails.json",   store, client, get_model("filter"))
    news_agent  = NewsFilterAgent(INPUTS_DIR / "news.json",      store, client, get_model("filter"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_cal   = executor.submit(cal_agent.run)
        future_email = executor.submit(email_agent.run)
        future_news  = executor.submit(news_agent.run)

        cal_output   = future_cal.result()
        email_output = future_email.result()
        news_output  = future_news.result()

    # Print per-agent filter results
    print_filter_output(cal_output)
    print_filter_output(email_output)
    print_filter_output(news_output)

    # ── Step 3: Build merged Knowledge Graph ──
    print("\n" + "=" * 72)
    print("  Step 3: GraphBuilder — merged knowledge graph (all 3 sources)")
    print("=" * 72)

    builder = GraphBuilder(store)
    graph = builder.build([cal_output, email_output, news_output])

    print("\n" + builder.render_ascii(graph, top_n_tags=20))

    # ── Step 4: TopicSelector preview (Python, no LLM) ──
    print("\n" + "=" * 72)
    print("  Step 4: TopicSelector — topic budget allocation (Python)")
    print("=" * 72)

    selector = TopicSelector(graph, max_events=22)
    topic_slots, overflow_events = selector.select()

    print(f"\n  Selected {len(topic_slots)} topics  |  {sum(s.event_count for s in topic_slots)} events  |  {len(overflow_events)} overflow")
    print()
    for slot in topic_slots:
        print(f"  [{slot.topic:<32}]  eff={slot.effective_weight:>5.1f}  events={slot.event_count}")
        for e in slot.events:
            flags = ",".join(f.value[:3] for f in e.special_flags) or "—"
            print(f"    {e.id:<12}  {e.source.value:<8}  [{flags}]  {e.title[:50]}")
    if overflow_events:
        print(f"\n  ⚠️  Overflow (special-flag, no top topic):")
        for e in overflow_events:
            print(f"    {e.id}  {[f.value for f in e.special_flags]}  {e.title[:50]}")

    # ── Step 5: RankingAgent ──
    print("\n" + "=" * 72)
    print("  Step 5: RankingAgent — LLM topic + event prioritisation")
    print("=" * 72)

    ranking_agent = RankingAgent(
        profile_store=store,
        client=client,
        model=get_model("ranking"),
    )
    ranking_output = ranking_agent.run(graph)

    PRIORITY_EMOJI = {Priority.P0: "🔴", Priority.P1: "🟡", Priority.P2: "🟢", Priority.P3: "⚪"}

    print(f"\n  {len(ranking_output.ranked_topics)} ranked topics  |  {len(ranking_output.skipped_event_ids)} skipped events")
    if ranking_output.agent_notes:
        print(f"  📝 Notes: {ranking_output.agent_notes}")
    print()

    for topic in ranking_output.ranked_topics:
        tp_emoji = PRIORITY_EMOJI.get(topic.topic_priority, "?")
        print(f"  {tp_emoji} [{topic.topic}]  ({topic.topic_priority.value})  eff={topic.effective_weight:.1f}")
        if topic.topic_summary:
            print(f"     → {topic.topic_summary}")
        for ev in topic.events:
            ep_emoji = PRIORITY_EMOJI.get(ev.event_priority, "?")
            flags = ",".join(f.value[:3] for f in ev.special_flags) or "—"
            print(f"     {ep_emoji} {ev.id:<12} [{ev.source.value:<8}] [{flags}]  {ev.title[:45]}")
            if ev.narration_hint:
                print(f"        💬 {ev.narration_hint}")
        print()

    if ranking_output.skipped_event_ids:
        print(f"  ⚪ Skipped: {ranking_output.skipped_event_ids}")

    # ── Summary stats ──
    print("\n" + "=" * 72)
    print("  Summary")
    print("=" * 72)

    p_counts = {p: 0 for p in Priority}
    for event in graph.events.values():
        if event.priority_level:
            p_counts[event.priority_level] += 1

    total_candidates = len(graph.events)
    total_excluded   = sum(len(o.excluded) for o in [cal_output, email_output, news_output])

    print(f"\n  Total candidates in graph : {total_candidates}")
    print(f"  Total excluded            : {total_excluded}")
    print(f"\n  GraphBuilder priority distribution:")
    for p, count in sorted(p_counts.items(), key=lambda x: x[0].value):
        bar = "█" * count
        print(f"    {p.value} : {count:>3}  {bar}")

    print(f"\n  Topic tags (unique)  : {len(graph.topic_tags)}")
    print(f"  Entity tags (unique) : {len(graph.entity_tags)}")

    print(f"\n  RankingAgent output:")
    for p in [Priority.P0, Priority.P1, Priority.P2]:
        topics_at_p = [t for t in ranking_output.ranked_topics if t.topic_priority == p]
        events_at_p = sum(
            sum(1 for e in t.events if e.event_priority == p)
            for t in ranking_output.ranked_topics
        )
        print(f"    {p.value} topics: {len(topics_at_p)}  |  {p.value} events: {events_at_p}")

    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()

# AI Generated Code - End
