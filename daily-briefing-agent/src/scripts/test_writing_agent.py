#!/usr/bin/env python3
# AI Generated Code - Start
"""
Quick WritingAgent test: runs full pipeline and prints the final briefing.
Usage: python src/scripts/test_writing_agent.py
"""
import logging, concurrent.futures, json, sys
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s  %(message)s")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.llm_client import get_client, get_model
from src.core.profile_store import ProfileStore
from src.core.graph_builder import GraphBuilder
from src.agents.calendar_filter_agent import CalendarFilterAgent
from src.agents.email_filter_agent import EmailFilterAgent
from src.agents.news_filter_agent import NewsFilterAgent
from src.agents.ranking_agent import RankingAgent
from src.agents.writing_agent import WritingAgent

INPUTS = Path("inputs")


def main():
    client = get_client()

    print("Step 1/5  ProfileStore …")
    store = ProfileStore(INPUTS / "profile.json", client, get_model("profile"))
    store.load()
    ctx = store.context
    target_words = int(ctx.audio_target_seconds * 2.5)
    print(f"          target {target_words} words  ({ctx.audio_min_seconds}–{ctx.audio_max_seconds}s range)")

    print("Step 2/5  Filter agents (parallel) …")
    cal   = CalendarFilterAgent(INPUTS / "calendar.json", store, client, get_model("filter"))
    email = EmailFilterAgent(INPUTS / "emails.json",      store, client, get_model("filter"))
    news  = NewsFilterAgent(INPUTS / "news.json",         store, client, get_model("filter"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        fc, fe, fn = ex.submit(cal.run), ex.submit(email.run), ex.submit(news.run)
        co, eo, no = fc.result(), fe.result(), fn.result()

    print(f"          cal={len(co.candidates)} email={len(eo.candidates)} news={len(no.candidates)}")

    print("Step 3/5  GraphBuilder …")
    graph = GraphBuilder(store).build([co, eo, no])
    print(f"          briefing_date={graph.briefing_date}  events={len(graph.events)}")

    print("Step 4/5  RankingAgent …")
    ranking_output = RankingAgent(store, client, get_model("ranking")).run(graph)
    topics_summary = "  ".join(
        f"{t.topic_priority.value}:{t.topic}" for t in ranking_output.ranked_topics
    )
    print(f"          {topics_summary}")

    print("Step 5/5  WritingAgent …")
    writer = WritingAgent(store, client, get_model("writing"), INPUTS / "session_history.json")
    result = writer.run(ranking_output, briefing_date=graph.briefing_date)

    SEP = "=" * 65
    print()
    print(SEP)
    print("  FINAL BRIEFING")
    print(SEP)
    print(f"  Words     : {result.word_count}  (target {target_words},  range {int(ctx.audio_min_seconds*2.5)}–{int(ctx.audio_max_seconds*2.5)})")
    print(f"  Duration  : {result.estimated_seconds}s  (target {ctx.audio_target_seconds}s)")
    print(f"  Sections  : {[s.name for s in result.sections]}")
    print()
    print(result.text)
    print()
    print(SEP)

    # Check session_history was updated
    history = json.loads((INPUTS / "session_history.json").read_text())
    today_entry = next((s for s in history["sessions"] if s["date"] == graph.briefing_date), None)
    if today_entry:
        print(f"  Session history updated: opening={today_entry['opening_style']}  closing={today_entry['closing_style']}")
        print(f"  Opening fragment: {today_entry['opening_fragment']}")
        print(f"  Closing fragment: {today_entry['closing_fragment']}")
    print(SEP)


if __name__ == "__main__":
    main()
# AI Generated Code - End
