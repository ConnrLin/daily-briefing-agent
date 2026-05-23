# AI Generated Code - Start
"""
CalendarFilterAgent: Filters and annotates today's calendar events.

Responsibilities:
  - Apply user preference filtering (what to include/exclude)
  - Detect time conflicts via interval overlap algorithm (no LLM needed)
  - Mark private events (include event exists, but not details)
  - Produce structured EventNode list with tags and special_flags

Design note: Conflict detection is done deterministically in Python,
NOT by the LLM — this avoids hallucination and is O(n log n).
The LLM's job is semantic tag assignment and relevance scoring only.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.core.llm_client import extract_json
from src.core.models import (
    EventNode,
    FilterAgentOutput,
    FilterDecision,
    ProfileContext,
    SourceType,
    SpecialFlag,
)
from src.core.profile_store import ProfileStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

CALENDAR_FILTER_SYSTEM_PROMPT = """You are the Calendar Analyst for a personal daily briefing system.

Your role is to review today's calendar events for a busy professional and determine:
1. Which events are worth mentioning in a 60-90 second audio briefing
2. What semantic tags each event carries (for knowledge graph building)
3. Any special flags that need immediate attention

{profile_context}

## Your Task

You will receive a list of today's calendar events (some may already be flagged 
with conflict or private markers from pre-processing).

For each event, decide:
- **INCLUDE**: This event should be a candidate for the briefing
- **EXCLUDE**: This event is not worth mentioning
- **DEPRIORITIZE**: Include but mark as low priority

## Tag Classification — CRITICAL

You must assign two SEPARATE lists of tags for each event:

### entity_tags — Named specific things
Things that can be uniquely identified in the real world:
- Company names: "cobalt-labs", "stripe", "plaid", "lyra-finance"
- Person names: "maya-chen", "alex-wong"
- Specific law / regulation names: "psd3", "eu-ai-act"
- Product names: "stripe-issuing", "fednow"
Use the EXACT canonical slug from the "Tracked Entity Tags" section in the profile above.
Non-tracked entities may also be added if clearly named (e.g. "acme-bank", "sequoia").

### topic_tags — Semantic content domains
What the event is fundamentally ABOUT — the underlying topic area:
- "fintech-regulation" (not "psd3" — that's an entity)
- "payments-product-launch", "api-design", "developer-tools"
- "board-prep", "competitive-intel", "customer-discovery"
- "team-meeting", "1-on-1", "external-meeting"
Assign 1-4 topic tags that best describe the content domain.

### Disambiguation examples
| Content | entity_tag | topic_tag |
|---------|-----------|-----------|
| Meeting about EU PSD3 regulation | "psd3" | "fintech-regulation" |
| Lunch with Stripe | "stripe" | "external-meeting", "partner-relations" |
| Lyra Finance SDK review | "lyra-finance" | "competitive-intel", "developer-tools" |
| Board deck prep with CEO | "cobalt-labs" | "board-prep", "high-stakes" |

### Special Flags
- `action-required`: User needs to DO something (prepare, review, respond)
- `prep-required`: Prep needed but less urgent than action-required
- `tracked-entity`: A tracked entity is directly involved
- `conflict-detected`: (pre-set by system — keep if present)
- `personal`: Personal / family event
- `private`: Calendar marked private — no details
- `high-stakes`: High-stakes meeting or decision

### Filtering Rules
1. **Always include** any event involving a tracked entity
2. **Always include** any event where action-required or prep-required is evident
3. **Deprioritize** purely routine recurring events (daily standup) unless flagged
4. **Private events**: include as "Personal appointment" with no details, set is_private=true

## Output Format

Return a JSON object with this exact structure:
{{
  "candidates": [
    {{
      "id": "cal_001",
      "title": "Team standup",
      "summary": "Daily team standup — routine, no special prep needed",
      "entity_tags": ["cobalt-labs"],
      "topic_tags": ["team-meeting", "recurring"],
      "special_flags": [],
      "filter_decision": "deprioritize",
      "filter_reason": "Routine recurring standup, no action items flagged"
    }},
    {{
      "id": "cal_009",
      "title": "Compliance sync — EU PSD3 implications",
      "summary": "Rahul walks through PSD3 final text and SCA flow impact.",
      "entity_tags": ["cobalt-labs", "psd3"],
      "topic_tags": ["fintech-regulation", "compliance"],
      "special_flags": ["action-required"],
      "filter_decision": "include",
      "filter_reason": "PSD3 is a key fintech regulation topic — directly affects checkout flow"
    }}
  ],
  "excluded": [],
  "agent_notes": "..."
}}

## Important Notes
- entity_tags and topic_tags are SEPARATE arrays — never mix them
- Do NOT put "psd3" in topic_tags — it's a named entity, put it in entity_tags
- Do NOT put "fintech-regulation" in entity_tags — it's a domain, put it in topic_tags
- Focus on what the user needs to KNOW and DO, not just what's on the calendar
"""


# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

CALENDAR_FILTER_USER_TEMPLATE = """Today is {date} ({day_of_week}).

Here are today's calendar events (pre-processed with conflict flags):

{events_json}

Pre-detected conflicts (from interval overlap analysis):
{conflicts_text}

Please analyze each event and return the structured JSON output."""


# ---------------------------------------------------------------------------
# CalendarFilterAgent
# ---------------------------------------------------------------------------

class CalendarFilterAgent:
    """
    Filter Agent for calendar events.
    
    Processing pipeline:
      1. Load calendar.json
      2. Deterministic conflict detection (Python, no LLM)
      3. Mark private events
      4. Call LLM for semantic tagging and inclusion decision
      5. Return FilterAgentOutput
    """

    # Scoring bonuses applied on top of tag-based scoring (for graph builder)
    FLAG_WEIGHTS = {
        SpecialFlag.ACTION_REQUIRED: 5.0,
        SpecialFlag.CONFLICT_DETECTED: 3.0,
        SpecialFlag.TRACKED_ENTITY: 2.0,
        SpecialFlag.FROM_CEO: 2.0,
        SpecialFlag.PERSONAL: 1.0,
    }

    def __init__(
        self,
        calendar_path: str | Path,
        profile_store: ProfileStore,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
    ):
        self.calendar_path = Path(calendar_path)
        self.profile_store = profile_store
        self.client = client
        self.model = model
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> FilterAgentOutput:
        """Execute the full filtering pipeline and return structured output."""
        ctx = self.profile_store.context
        raw = self._load_calendar()
        events = raw["events"]
        date_str = raw["date"]

        # Step 1: Deterministic pre-processing (no LLM)
        conflict_pairs = self._detect_conflicts(events)
        events_with_flags = self._apply_preprocess_flags(events, conflict_pairs)

        # Step 2: LLM semantic analysis
        llm_result = self._call_llm(ctx, events_with_flags, date_str, conflict_pairs)

        # Step 3: Build EventNode objects
        output = self._build_output(llm_result, events_with_flags)

        logger.info(
            "CalendarFilterAgent: %d candidates, %d excluded, %d conflicts detected",
            len(output.candidates),
            len(output.excluded),
            len(conflict_pairs),
        )
        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_calendar(self) -> dict[str, Any]:
        with open(self.calendar_path, encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Step 1: Deterministic conflict detection
    # ------------------------------------------------------------------

    def _detect_conflicts(self, events: list[dict]) -> list[tuple[str, str]]:
        """
        O(n log n) interval overlap detection.
        Returns list of (event_id_a, event_id_b) pairs that overlap.
        """
        # Filter to events with both start and end times
        timed = [
            e for e in events
            if e.get("start") and e.get("end")
        ]
        timed.sort(key=lambda e: e["start"])

        conflicts = []
        for i in range(len(timed) - 1):
            a = timed[i]
            b = timed[i + 1]
            # Overlap: a.start < b.end AND b.start < a.end
            if a["start"] < b["end"] and b["start"] < a["end"]:
                conflicts.append((a["id"], b["id"]))
                logger.info(
                    "Conflict detected: %s (%s-%s) overlaps %s (%s-%s)",
                    a["id"], a["start"], a["end"],
                    b["id"], b["start"], b["end"],
                )
        return conflicts

    def _apply_preprocess_flags(
        self, events: list[dict], conflict_pairs: list[tuple[str, str]]
    ) -> list[dict]:
        """
        Add pre-computed flags to each event dict before sending to LLM.
        This keeps conflict detection deterministic.
        """
        conflict_ids = {id_ for pair in conflict_pairs for id_ in pair}
        result = []
        for event in events:
            e = dict(event)  # copy
            preflags = []

            # Conflict flag
            if e["id"] in conflict_ids:
                preflags.append("conflict-detected")

            # Private flag
            if e.get("visibility") == "private":
                preflags.append("private")

            e["_pre_flags"] = preflags
            result.append(e)
        return result

    # ------------------------------------------------------------------
    # Step 2: LLM call
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        ctx: ProfileContext,
        events: list[dict],
        date_str: str,
        conflict_pairs: list[tuple[str, str]],
    ) -> dict[str, Any]:
        """Call LLM with retry. Returns parsed JSON dict."""
        profile_context_block = self.profile_store.build_filter_context_block("Calendar Events")

        # Format conflicts for prompt
        if conflict_pairs:
            conflicts_text = "\n".join(
                f"  - {a} and {b} have overlapping times" for a, b in conflict_pairs
            )
        else:
            conflicts_text = "  No conflicts detected."

        # Build events JSON (clean — remove internal _pre_flags for readability, keep as annotation)
        events_for_prompt = []
        for e in events:
            entry = {
                "id": e["id"],
                "title": e["title"],
                "start": e["start"],
                "end": e["end"],
                "location": e.get("location", ""),
                "attendees": e.get("attendees", []),
                "is_recurring": e.get("is_recurring", False),
                "description": e.get("description", ""),
                "pre_flags": e.get("_pre_flags", []),
            }
            # Mask private events before sending to LLM
            if "private" in e.get("_pre_flags", []):
                entry["title"] = "[PRIVATE EVENT]"
                entry["description"] = "Marked private — do not surface details."
                entry["attendees"] = []
                entry["location"] = "Private"
            events_for_prompt.append(entry)

        # Parse date for day of week
        try:
            dt = datetime.fromisoformat(date_str)
            day_of_week = dt.strftime("%A")
        except Exception:
            day_of_week = "Friday"

        system_prompt = CALENDAR_FILTER_SYSTEM_PROMPT.format(
            profile_context=profile_context_block
        )
        user_prompt = CALENDAR_FILTER_USER_TEMPLATE.format(
            date=date_str,
            day_of_week=day_of_week,
            events_json=json.dumps(events_for_prompt, indent=2),
            conflicts_text=conflicts_text,
        )

        for attempt in range(self.max_retries):
            try:
                _rf = {"response_format": {"type": "json_object"}} if supports_json_mode() else {}
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    **_rf,
                )
                raw = response.choices[0].message.content
                return json.loads(extract_json(raw))
            except Exception as e:
                logger.warning("CalendarFilterAgent LLM attempt %d failed: %s", attempt + 1, e)
                if attempt == self.max_retries - 1:
                    logger.error("All LLM attempts failed, returning empty result")
                    return {"candidates": [], "excluded": [], "agent_notes": f"LLM failed: {e}"}

        return {}

    # ------------------------------------------------------------------
    # Step 3: Build EventNode output
    # ------------------------------------------------------------------

    def _build_output(
        self,
        llm_result: dict[str, Any],
        events_with_flags: list[dict],
    ) -> FilterAgentOutput:
        """Convert LLM JSON result into typed EventNode objects."""
        # Build a lookup for original event data (start/end times, etc.)
        event_lookup = {e["id"]: e for e in events_with_flags}

        candidates = []
        excluded = []

        for item in llm_result.get("candidates", []):
            node = self._build_event_node(item, event_lookup)
            candidates.append(node)

        for item in llm_result.get("excluded", []):
            node = self._build_event_node(item, event_lookup)
            node.filter_decision = FilterDecision.EXCLUDE
            excluded.append(node)

        return FilterAgentOutput(
            source=SourceType.CALENDAR,
            candidates=candidates,
            excluded=excluded,
            agent_notes=llm_result.get("agent_notes"),
        )

    def _build_event_node(
        self, item: dict[str, Any], event_lookup: dict[str, dict]
    ) -> EventNode:
        """Build a single EventNode from LLM output + original event data."""
        event_id = item.get("id", "unknown")
        original = event_lookup.get(event_id, {})

        # Parse special flags — deduplicate while preserving order
        raw_flags = item.get("special_flags", []) + original.get("_pre_flags", [])
        seen_flags: set[str] = set()
        special_flags = []
        for f in raw_flags:
            try:
                sf = SpecialFlag(f)
                if sf not in seen_flags:
                    special_flags.append(sf)
                    seen_flags.add(sf)
            except ValueError:
                logger.warning("Unknown special flag '%s' for event %s — skipping", f, event_id)

        # Parse filter decision
        decision_str = item.get("filter_decision", "include")
        try:
            decision = FilterDecision(decision_str)
        except ValueError:
            decision = FilterDecision.INCLUDE

        # Accept both new schema (entity_tags/topic_tags) and old schema (tags) for compatibility
        entity_tags: list[str] = item.get("entity_tags", [])
        topic_tags: list[str] = item.get("topic_tags", [])
        if not entity_tags and not topic_tags and item.get("tags"):
            # Fallback: old flat tags field — treat all as topic_tags
            topic_tags = item.get("tags", [])
            logger.warning("Event %s used legacy flat 'tags' field — treating all as topic_tags", event_id)

        return EventNode(
            id=event_id,
            source=SourceType.CALENDAR,
            title=item.get("title", original.get("title", "")),
            summary=item.get("summary", ""),
            entity_tags=entity_tags,
            topic_tags=topic_tags,
            special_flags=special_flags,
            timestamp=original.get("start"),
            start_time=original.get("start"),
            end_time=original.get("end"),
            is_private="private" in [f.value for f in special_flags],
            filter_decision=decision,
            filter_reason=item.get("filter_reason"),
        )

# AI Generated Code - End
