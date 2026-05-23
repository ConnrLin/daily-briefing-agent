# AI Generated Code - Start
"""
TopicSelector: Deterministic Python pre-processing before RankingAgent.

Responsibilities:
  1. Sort topics by effective_weight (in_degree × base_weight) descending
  2. Greedily pick top topics until cumulative event count reaches MAX_EVENTS cap
  3. Force-include topics that contain special-flag events (action-required, from-ceo, deadline-today)
  4. Assign each event to its SINGLE best topic (highest effective_weight topic it belongs to)
  5. Return a list of (topic, [events]) pairs ready to be serialized into the RankingAgent prompt

Design note:
  This step is deliberately LLM-free. It solves the "budget allocation" problem
  deterministically so the RankingAgent can focus purely on narrative quality.
"""

import logging
from dataclasses import dataclass, field

from src.core.models import (
    EventNode,
    KnowledgeGraph,
    Priority,
    SourceType,
    SpecialFlag,
    TagNode,
)

logger = logging.getLogger(__name__)

# Maximum total events sent to RankingAgent
MAX_EVENTS = 22

# Flags that force a topic to be included regardless of weight
FORCE_INCLUDE_FLAGS = {
    SpecialFlag.ACTION_REQUIRED,
    SpecialFlag.FROM_CEO,
    SpecialFlag.DEADLINE_TODAY,
    SpecialFlag.CONFLICT_DETECTED,
}


@dataclass
class TopicSlot:
    """One topic and the events assigned to it for RankingAgent."""
    topic: str
    effective_weight: float
    events: list[EventNode] = field(default_factory=list)

    @property
    def event_count(self) -> int:
        return len(self.events)

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "effective_weight": round(self.effective_weight, 2),
            "events": [
                {
                    "id": e.id,
                    "source": e.source.value,
                    "title": e.title,
                    "summary": e.summary,
                    "special_flags": [f.value for f in e.special_flags],
                    "entity_tags": e.entity_tags,
                    "topic_tags": e.topic_tags,
                    "timestamp": e.timestamp,
                }
                for e in sorted(
                    self.events,
                    key=lambda x: (
                        # action-required / from-ceo first
                        0 if any(f in FORCE_INCLUDE_FLAGS for f in x.special_flags) else 1,
                        # then by source: calendar > email > news
                        {"calendar": 0, "email": 1, "news": 2}.get(x.source.value, 3),
                    ),
                )
            ],
        }


class TopicSelector:
    """
    Selects the top topics and assigns events to them before calling RankingAgent.

    Usage:
        selector = TopicSelector(graph, max_events=22)
        topic_slots, special_flag_events = selector.select()
    """

    def __init__(self, graph: KnowledgeGraph, max_events: int = MAX_EVENTS):
        self.graph = graph
        self.max_events = max_events

    def select(self) -> tuple[list[TopicSlot], list[EventNode]]:
        """
        Returns:
          topic_slots          — ordered list of TopicSlot (topic → assigned events)
          special_flag_events  — events with FORCE_INCLUDE_FLAGS not already in any slot
                                 (ensures none are silently dropped)
        """
        graph = self.graph

        # ── Step 1: Build event → best_topic mapping ──────────────────────────
        # Each event is assigned to the ONE topic with highest effective_weight.
        event_best_topic: dict[str, str] = {}   # event_id → topic_name

        for event_id, event in graph.events.items():
            best_topic = None
            best_ew = -1.0
            for tag in event.topic_tags:
                if tag in graph.topic_tags:
                    ew = graph.topic_tags[tag].effective_weight
                    if ew > best_ew:
                        best_ew = ew
                        best_topic = tag
            if best_topic:
                event_best_topic[event_id] = best_topic

        # ── Step 2: Sort topics by effective_weight ───────────────────────────
        sorted_topics: list[TagNode] = sorted(
            graph.topic_tags.values(),
            key=lambda t: t.effective_weight,
            reverse=True,
        )

        # ── Step 3: Identify topics that MUST be included (force-include flags) ──
        force_include_topics: set[str] = set()
        for event in graph.events.values():
            if any(f in FORCE_INCLUDE_FLAGS for f in event.special_flags):
                best = event_best_topic.get(event.id)
                if best:
                    force_include_topics.add(best)

        # ── Step 4: Greedy topic selection up to MAX_EVENTS ───────────────────
        selected_topic_names: list[str] = []
        cumulative_events: int = 0

        # First pass: force-include topics
        for topic_node in sorted_topics:
            if topic_node.name in force_include_topics:
                count = sum(
                    1 for eid, t in event_best_topic.items() if t == topic_node.name
                )
                selected_topic_names.append(topic_node.name)
                cumulative_events += count
                logger.debug(
                    "TopicSelector force-include '%s' (%d events)", topic_node.name, count
                )

        # Second pass: add remaining topics by weight until budget fills up
        for topic_node in sorted_topics:
            if topic_node.name in selected_topic_names:
                continue
            count = sum(
                1 for eid, t in event_best_topic.items() if t == topic_node.name
            )
            if count == 0:
                continue
            if cumulative_events + count > self.max_events:
                logger.debug(
                    "TopicSelector budget cap: skipping '%s' (would reach %d > %d)",
                    topic_node.name, cumulative_events + count, self.max_events,
                )
                break
            selected_topic_names.append(topic_node.name)
            cumulative_events += count

        logger.info(
            "TopicSelector: %d topics selected, %d events total (cap=%d)",
            len(selected_topic_names), cumulative_events, self.max_events,
        )

        # ── Step 5: Build TopicSlot objects ───────────────────────────────────
        topic_slots: list[TopicSlot] = []
        assigned_event_ids: set[str] = set()

        for topic_name in selected_topic_names:
            node = graph.topic_tags[topic_name]
            slot = TopicSlot(topic=topic_name, effective_weight=node.effective_weight)
            for event_id, best_topic in event_best_topic.items():
                if best_topic == topic_name:
                    slot.events.append(graph.events[event_id])
                    assigned_event_ids.add(event_id)
            topic_slots.append(slot)

        # ── Step 6: Collect special-flag events not yet assigned ───────────────
        special_flag_events: list[EventNode] = []
        for event in graph.events.values():
            if any(f in FORCE_INCLUDE_FLAGS for f in event.special_flags):
                if event.id not in assigned_event_ids:
                    special_flag_events.append(event)
                    logger.warning(
                        "TopicSelector: special-flag event '%s' has no topic — added to overflow",
                        event.id,
                    )

        return topic_slots, special_flag_events

# AI Generated Code - End
