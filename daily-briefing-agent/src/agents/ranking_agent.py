# AI Generated Code - Start
"""
RankingAgent: Topic-by-topic analysis and event prioritisation.

Responsibilities:
  - Receive pre-selected topics (from TopicSelector) and their events
  - For each topic: decide topic_priority (P0/P1/P2), summarise why it matters today
  - For each event within a topic: assign event_priority (P0/P1/P2) and write a narration_hint
  - Produce RankingAgentOutput consumed directly by WritingAgent

Prompt design (per user spec):
  Role → Background (user profile + preferences) → Knowledge (topic slots) →
  Task → Example output
"""

import json
import logging
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.core.llm_client import extract_json, get_model
from src.core.models import (
    EventNode,
    KnowledgeGraph,
    Priority,
    RankedEvent,
    RankedTopic,
    RankingAgentOutput,
    SourceType,
    SpecialFlag,
)
from src.core.profile_store import ProfileStore
from src.core.topic_selector import TopicSelector, TopicSlot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

RANKING_SYSTEM_PROMPT = """## Role
You are the Briefing Strategist for a personal morning audio briefing system.
Your job is to take pre-filtered information grouped by topic and decide:
  1. Which topics deserve the most airtime (P0 / P1 / P2)
  2. Within each topic, which specific events to highlight and how

## Background
{user_background}

## User Preferences
- Interested in: {interest_tags}
- Does NOT want: {blocked_tags}
- Tracked entities (always relevant): {entity_tags}
- Tone: {tone_style}
- Target audio length: {target_seconds} seconds ({target_words} words approx)

## Knowledge: Today's Pre-Selected Topics and Events
{topic_slots_json}

{special_flag_section}

## Task

### Step 1 — Rank topics
For each topic, assign a topic_priority:
- **P0**: Critical today — directly affects user's work, has action items, or involves a tracked entity with major news
- **P1**: Important background — relevant to user's interests, worth a brief mention
- **P2**: Nice-to-know — loosely relevant, include only if time permits

Write a topic_summary (1-2 sentences) explaining WHY this topic matters today (not what it is — the user knows; WHY it's relevant NOW).

### Step 2 — Rank events within each topic
For each event under a topic, assign an event_priority:
- **P0**: Must mention — action required, key insight, or the definitive source on this topic
- **P1**: Should mention — adds context or confirms a pattern
- **P2**: Can skip — redundant, low new information

Aim for roughly balanced distribution across P0/P1/P2. Avoid marking everything P0.
Maximum 2-3 P0 events per topic.

Write a narration_hint (one sentence) guiding the WritingAgent:
  - For calendar events: what to say about the meeting (prep needed? key attendee? conflict?)
  - For emails: the core ask or insight, not just the subject line
  - For news: the "so what" for this user specifically (not a generic headline summary)

### Step 3 — Identify events to skip
List any event IDs that are redundant, too low-value, or would just eat up airtime with no benefit.

## Output Format

Return a JSON object with this EXACT structure:
{{
  "ranked_topics": [
    {{
      "topic": "payments-product-launch",
      "topic_priority": "P0",
      "topic_summary": "Today is a convergence point: the GA announcement just hit TechCrunch, the CEO wants the board deck reviewed before 2pm, and Priya is pushing to pull the API launch forward. This needs the most airtime.",
      "events": [
        {{
          "id": "em_001",
          "source": "email",
          "event_priority": "P0",
          "narration_hint": "CEO asks you to review slides 8-14 before 2pm — the ARR projection chart is flagged as the key one. Hard deadline."
        }},
        {{
          "id": "news_001",
          "source": "news",
          "event_priority": "P1",
          "narration_hint": "Cobalt Labs beta exit in TechCrunch — external validation that aligns with today's board narrative. Brief mention."
        }},
        {{
          "id": "cal_002",
          "source": "calendar",
          "event_priority": "P0",
          "narration_hint": "Q2 roadmap review at 9am — first meeting of the day, feeds directly into board prep. Confirm Priya's pull-forward ask."
        }}
      ]
    }},
    {{
      "topic": "fintech-regulation",
      "topic_priority": "P0",
      "topic_summary": "PSD3 final text dropped overnight and you have a 3pm compliance sync with Rahul. You need to read the email summary before that meeting.",
      "events": [
        {{
          "id": "em_011",
          "source": "email",
          "event_priority": "P0",
          "narration_hint": "Rahul's email has the key SCA changes — read before 3pm sync. Deadline-today flag."
        }},
        {{
          "id": "news_002",
          "source": "news",
          "event_priority": "P1",
          "narration_hint": "Reuters published the full PSD3 text — corroborates Rahul's summary. Note the 2027 implementation deadline."
        }},
        {{
          "id": "cal_009",
          "source": "calendar",
          "event_priority": "P1",
          "narration_hint": "3pm compliance sync at 3pm — Rahul will walk through the SCA flow impact. Prep by reading em_011."
        }}
      ]
    }}
  ],
  "skipped_event_ids": ["em_017", "cal_003"],
  "agent_notes": "The board deck review (em_001) and PSD3 sync (em_011) are the two most time-sensitive action items. The WritingAgent should open with these."
}}

## Important Rules
- ranked_topics MUST be ordered P0 first, then P1, then P2
- Within topics, events MUST be ordered P0 first, then P1, then P2
- Do NOT put all events as P0 — prioritise ruthlessly
- narration_hint must be specific to THIS user's context, not a generic summary
- topic_summary should explain relevance TODAY, not just describe the topic
- If a topic only has P2 events, mark the topic itself as P2
"""


# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

RANKING_USER_TEMPLATE = """Today is {date} ({day_of_week}).

Please analyse the topics and events above, then output your ranked_topics JSON.

Key constraints:
- Total audio target: {target_seconds} seconds ≈ {target_words} words
- Maximum P0 topics: 3 (focus forces better briefing)
- Ensure all events with special flags (action-required, from-ceo, deadline-today) get event_priority P0
- The special_flag_events section lists any high-priority events not tied to a top topic — decide whether to add them to an existing topic or create a catch-all topic for them
"""


# ---------------------------------------------------------------------------
# RankingAgent
# ---------------------------------------------------------------------------

class RankingAgent:
    """
    Ranking Agent.

    Pipeline:
      1. TopicSelector (Python) → topic_slots, special_flag_events
      2. Build prompt with user background + topic knowledge
      3. LLM call → ranked_topics JSON
      4. Parse + return RankingAgentOutput
    """

    def __init__(
        self,
        profile_store: ProfileStore,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        max_events: int = 22,
    ):
        self.profile_store = profile_store
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.max_events = max_events

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, graph: KnowledgeGraph) -> RankingAgentOutput:
        """Execute topic selection + LLM ranking."""
        # Step 1: Python pre-processing
        selector = TopicSelector(graph, max_events=self.max_events)
        topic_slots, special_flag_events = selector.select()

        logger.info(
            "RankingAgent: %d topic slots, %d special-flag overflow events",
            len(topic_slots), len(special_flag_events),
        )

        # Step 2: LLM call
        llm_result = self._call_llm(topic_slots, special_flag_events, graph.briefing_date)

        # Step 3: Parse output
        output = self._parse_output(llm_result, graph)
        logger.info(
            "RankingAgent: %d ranked topics, %d skipped events",
            len(output.ranked_topics), len(output.skipped_event_ids),
        )
        return output

    # ------------------------------------------------------------------
    # Internal: build prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        ctx = self.profile_store.context
        target_words = int(ctx.audio_target_seconds * 2.5)  # ~2.5 words/second for clear speech

        user_background = (
            f"{ctx.user_name} is a {ctx.user_role} at {ctx.user_company}. "
            f"{ctx.user_background if hasattr(ctx, 'user_background') else ''}"
        )

        return RANKING_SYSTEM_PROMPT.format(
            user_background=user_background,
            interest_tags=", ".join(t.tag for t in ctx.interest_tags),
            blocked_tags=", ".join(t.tag for t in ctx.blocked_tags),
            entity_tags=", ".join(t.tag for t in ctx.entity_tags),
            tone_style=ctx.tone_style,
            target_seconds=ctx.audio_target_seconds,
            target_words=target_words,
            topic_slots_json="{topic_slots_json}",   # filled in user prompt
            special_flag_section="{special_flag_section}",
        )

    def _call_llm(
        self,
        topic_slots: list[TopicSlot],
        special_flag_events: list[EventNode],
        briefing_date: str = "",
    ) -> dict:
        from datetime import datetime, date
        ctx = self.profile_store.context
        target_words = int(ctx.audio_target_seconds * 2.5)

        # Use briefing_date from data; parse day-of-week from it
        if briefing_date:
            try:
                dt = datetime.strptime(briefing_date, "%Y-%m-%d")
                date_str = briefing_date
                day_of_week = dt.strftime("%A")
            except ValueError:
                date_str = briefing_date
                day_of_week = ""
        else:
            # Fallback: infer from first event with a timestamp in any slot
            date_str = ""
            for slot in topic_slots:
                for e in slot.events:
                    if e.timestamp:
                        date_str = e.timestamp[:10]
                        try:
                            dt = datetime.strptime(date_str, "%Y-%m-%d")
                            day_of_week = dt.strftime("%A")
                        except ValueError:
                            day_of_week = ""
                        break
                if date_str:
                    break
            if not date_str:
                date_str = datetime.now().strftime("%Y-%m-%d")
                day_of_week = datetime.now().strftime("%A")
                logger.warning("RankingAgent: no briefing_date found in graph — using system clock")

        # Serialize topic slots
        topic_slots_json = json.dumps(
            [slot.to_dict() for slot in topic_slots],
            indent=2, ensure_ascii=False
        )

        # Special flag overflow section
        if special_flag_events:
            overflow_items = [
                {
                    "id": e.id,
                    "source": e.source.value,
                    "title": e.title,
                    "summary": e.summary,
                    "special_flags": [f.value for f in e.special_flags],
                }
                for e in special_flag_events
            ]
            special_flag_section = (
                "## ⚠️ Special-Flag Events (not in top topics — must not be dropped)\n"
                + json.dumps(overflow_items, indent=2, ensure_ascii=False)
            )
        else:
            special_flag_section = ""

        # Build system prompt (fill in topic slots + special flags)
        user_background = (
            f"{ctx.user_name} is a {ctx.user_role} at {ctx.user_company}."
        )
        target_words_calc = int(ctx.audio_target_seconds * 2.5)

        system_prompt = RANKING_SYSTEM_PROMPT.format(
            user_background=user_background,
            interest_tags=", ".join(t.tag for t in ctx.interest_tags),
            blocked_tags=", ".join(t.tag for t in ctx.blocked_tags),
            entity_tags=", ".join(t.tag for t in ctx.entity_tags),
            tone_style=ctx.tone_style,
            target_seconds=ctx.audio_target_seconds,
            target_words=target_words_calc,
            topic_slots_json=topic_slots_json,
            special_flag_section=special_flag_section,
        )

        user_prompt = RANKING_USER_TEMPLATE.format(
            date=date_str,
            day_of_week=day_of_week,
            target_seconds=ctx.audio_target_seconds,
            target_words=target_words,
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                _rf = {"response_format": {"type": "json_object"}} if supports_json_mode() else {}
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    **_rf,
                )
                raw_text = response.choices[0].message.content
                cleaned = extract_json(raw_text)
                return json.loads(cleaned)
            except Exception as e:
                logger.warning("RankingAgent LLM attempt %d failed: %s", attempt, e)
                if attempt == self.max_retries:
                    raise

    # ------------------------------------------------------------------
    # Internal: parse LLM output → RankingAgentOutput
    # ------------------------------------------------------------------

    def _parse_output(self, llm_result: dict, graph: KnowledgeGraph) -> RankingAgentOutput:
        ranked_topics: list[RankedTopic] = []

        for topic_dict in llm_result.get("ranked_topics", []):
            topic_name = topic_dict.get("topic", "unknown")

            # effective_weight from graph
            eff_weight = 0.0
            if topic_name in graph.topic_tags:
                eff_weight = graph.topic_tags[topic_name].effective_weight

            # Parse topic priority
            tp_str = topic_dict.get("topic_priority", "P1")
            try:
                topic_priority = Priority(tp_str)
            except ValueError:
                topic_priority = Priority.P1

            ranked_events: list[RankedEvent] = []
            for ev_dict in topic_dict.get("events", []):
                event_id = ev_dict.get("id", "")
                original = graph.events.get(event_id)

                ep_str = ev_dict.get("event_priority", "P1")
                try:
                    event_priority = Priority(ep_str)
                except ValueError:
                    event_priority = Priority.P1

                source_str = ev_dict.get("source", original.source.value if original else "calendar")
                try:
                    source = SourceType(source_str)
                except ValueError:
                    source = SourceType.CALENDAR

                ranked_events.append(RankedEvent(
                    id=event_id,
                    source=source,
                    event_priority=event_priority,
                    special_flags=original.special_flags if original else [],
                    title=original.title if original else ev_dict.get("title", ""),
                    summary=original.summary if original else "",
                    narration_hint=ev_dict.get("narration_hint", ""),
                    timestamp=original.timestamp if original else None,
                ))

            # Sort events: P0 first, then P1, P2
            priority_order = {Priority.P0: 0, Priority.P1: 1, Priority.P2: 2, Priority.P3: 3}
            ranked_events.sort(key=lambda e: priority_order.get(e.event_priority, 3))

            ranked_topics.append(RankedTopic(
                topic=topic_name,
                topic_priority=topic_priority,
                topic_summary=topic_dict.get("topic_summary", ""),
                effective_weight=eff_weight,
                events=ranked_events,
            ))

        # Sort topics: P0 first, then P1, P2, then by effective_weight desc
        priority_order = {Priority.P0: 0, Priority.P1: 1, Priority.P2: 2, Priority.P3: 3}
        ranked_topics.sort(key=lambda t: (priority_order.get(t.topic_priority, 3), -t.effective_weight))

        return RankingAgentOutput(
            ranked_topics=ranked_topics,
            skipped_event_ids=llm_result.get("skipped_event_ids", []),
            agent_notes=llm_result.get("agent_notes", ""),
        )

# AI Generated Code - End
