# AI Generated Code - Start
"""
Quick runner: executes CalendarFilterAgent → GraphBuilder and prints results.
Usage: python -m src.scripts.run_calendar_agent
"""

import logging
from pathlib import Path

from src.core.llm_client import get_client, get_model
from src.core.profile_store import ProfileStore
from src.core.graph_builder import GraphBuilder
from src.agents.calendar_filter_agent import CalendarFilterAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

INPUTS_DIR = Path("inputs")


def main():
    # ------------------------------------------------------------------
    # Step 1: ProfileStore
    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("  Step 1: ProfileStore — LLM tag extraction")
    print("=" * 68)

    store = ProfileStore(
        profile_path=INPUTS_DIR / "profile.json",
        client=get_client(),
        model=get_model("profile"),
    )
    ctx = store.load()

    print(f"\n✓ User: {ctx.user_name} ({ctx.user_role} @ {ctx.user_company})")
    print(f"\n📌 Interest tags ({len(ctx.interest_tags)}):")
    for t in ctx.interest_tags:
        print(f"   [{t.tag:<30}]  weight={t.base_weight}")

    print(f"\n🚫 Blocked tags ({len(ctx.blocked_tags)}):")
    for t in ctx.blocked_tags:
        print(f"   [{t.tag}]")

    print(f"\n🎯 Entity tags ({len(ctx.entity_tags)}):")
    for t in ctx.entity_tags:
        aliases = getattr(t, "aliases", [])
        alias_str = f"  aliases={aliases}" if aliases else ""
        print(f"   [{t.tag}]{alias_str}")

    # ------------------------------------------------------------------
    # Step 2: CalendarFilterAgent
    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("  Step 2: CalendarFilterAgent — filter & tag calendar events")
    print("=" * 68)

    agent = CalendarFilterAgent(
        calendar_path=INPUTS_DIR / "calendar.json",
        profile_store=store,
        client=get_client(),
        model=get_model("filter"),
    )
    cal_output = agent.run()

    print(f"\n✅ CANDIDATES ({len(cal_output.candidates)})  |  ❌ EXCLUDED ({len(cal_output.excluded)})")
    print("-" * 68)
    for c in sorted(cal_output.candidates, key=lambda x: x.id):
        flags_str = ", ".join(f.value for f in c.special_flags) or "—"
        decision_label = {
            "include":      "✅",
            "deprioritize": "⬇ ",
            "exclude":      "❌",
        }.get(c.filter_decision.value, "?")
        print(f"  {decision_label} {c.id}  {c.title}")
        print(f"       flags        : {flags_str}")
        print(f"       entity_tags  : {c.entity_tags}")
        print(f"       topic_tags   : {c.topic_tags}")
        print()

    for e in cal_output.excluded:
        print(f"  ❌ {e.id}  {e.title}")
        print(f"       reason: {e.filter_reason}")
        print()

    if cal_output.agent_notes:
        print(f"\n📝 Agent notes:\n  {cal_output.agent_notes}")

    # ------------------------------------------------------------------
    # Step 3: GraphBuilder
    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("  Step 3: GraphBuilder — build knowledge graph")
    print("=" * 68)

    builder = GraphBuilder(store)
    graph = builder.build([cal_output])

    print("\n" + builder.render_ascii(graph, top_n_tags=15))

    print("\n" + "=" * 68)
    print("  Done.")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()

# AI Generated Code - End
