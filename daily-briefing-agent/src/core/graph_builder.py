# AI Generated Code - Start
"""
GraphBuilder: Builds an in-memory knowledge graph from Filter Agent outputs.

Graph structure:
  - Event nodes: individual items (calendar / email / news)
  - Tag nodes: semantic labels connecting events
  - Edges: Event --[tagged_by]--> Tag

Scoring formula:
  priority(event) = Σ tag.effective_weight  +  Σ special_flag_bonus
  tag.effective_weight = tag.in_degree × tag.base_weight

Special flag bonuses (additive):
  action-required    +5.0
  conflict-detected  +3.0
  tracked-entity     +2.0
  from-ceo           +2.0
  deadline-today     +4.0
  prep-required      +1.5
  high-stakes        +1.5
  personal           +1.0
"""

import logging
from collections import defaultdict

from src.core.models import (
    EventNode,
    FilterAgentOutput,
    KnowledgeGraph,
    Priority,
    SpecialFlag,
    TagNode,
)
from src.core.profile_store import ProfileStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Special flag bonus scores
# ---------------------------------------------------------------------------

FLAG_BONUS: dict[SpecialFlag, float] = {
    SpecialFlag.ACTION_REQUIRED:   5.0,
    SpecialFlag.DEADLINE_TODAY:    4.0,
    SpecialFlag.CONFLICT_DETECTED: 3.0,
    SpecialFlag.TRACKED_ENTITY:    2.0,
    SpecialFlag.FROM_CEO:          2.0,
    SpecialFlag.HIGH_STAKES:       1.5,
    SpecialFlag.PREP_REQUIRED:     1.5,
    SpecialFlag.PERSONAL:          1.0,
    SpecialFlag.EXTERNAL:          0.5,
    SpecialFlag.PRIVATE:           0.0,
}

# Priority thresholds (based on final priority_score)
P0_THRESHOLD = 8.0
P1_THRESHOLD = 5.0
P2_THRESHOLD = 2.0


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------

class GraphBuilder:
    """
    Builds the KnowledgeGraph from one or more FilterAgentOutput objects.
    
    Usage:
        builder = GraphBuilder(profile_store)
        graph = builder.build([calendar_output, email_output, news_output])
    """

    def __init__(self, profile_store: ProfileStore):
        self.profile_store = profile_store

    def build(self, filter_outputs: list[FilterAgentOutput]) -> KnowledgeGraph:
        """
        Consume all filter outputs and produce a scored KnowledgeGraph.
        Only INCLUDE and DEPRIORITIZE events enter the graph (EXCLUDE skipped).

        Scoring design:
          priority_score = topic_score + entity_bonus + flag_bonus

          topic_score  = Σ topic_tag.effective_weight
                       = Σ (in_degree × base_weight)
                       Primary driver — reflects true topic density across all sources

          entity_bonus = Σ entity_tag.base_weight  (flat, no in_degree amplification)
                       Entity tags like "cobalt-labs" appear everywhere so we don't
                       amplify by in_degree; instead each entity contributes a fixed
                       bonus only when it's a tracked entity with strategic significance.
                       Non-tracked entities (e.g. "acme-bank") contribute 0.

          flag_bonus   = Σ FLAG_BONUS[flag]
        """
        graph = KnowledgeGraph()
        ctx = self.profile_store.context

        # Collect all candidate events
        all_candidates: list[EventNode] = []
        for output in filter_outputs:
            for event in output.candidates:
                all_candidates.append(event)
                graph.events[event.id] = event

        # Infer briefing_date from calendar timestamps (most authoritative),
        # then fall back to the earliest timestamp found across all events.
        graph.briefing_date = self._infer_briefing_date(filter_outputs)
        logger.info("GraphBuilder: briefing_date = %s", graph.briefing_date)

        logger.info("GraphBuilder: %d candidate events collected", len(all_candidates))

        # Build Topic Tag nodes (in_degree = how many events share this topic)
        topic_event_map: dict[str, list[str]] = defaultdict(list)
        for event in all_candidates:
            for tag in event.topic_tags:
                topic_event_map[tag].append(event.id)

        for tag_name, event_ids in topic_event_map.items():
            base_weight = ctx.get_tag_weight(tag_name)
            graph.topic_tags[tag_name] = TagNode(
                name=tag_name,
                tag_type="topic",
                base_weight=base_weight,
                in_degree=len(event_ids),
                event_ids=event_ids,
            )

        # Build Entity Tag nodes (in_degree tracked for visibility, not for scoring)
        entity_event_map: dict[str, list[str]] = defaultdict(list)
        for event in all_candidates:
            for tag in event.entity_tags:
                entity_event_map[tag].append(event.id)

        for tag_name, event_ids in entity_event_map.items():
            base_weight = ctx.get_tag_weight(tag_name)
            graph.entity_tags[tag_name] = TagNode(
                name=tag_name,
                tag_type="entity",
                base_weight=base_weight,
                in_degree=len(event_ids),
                event_ids=event_ids,
            )

        logger.info(
            "GraphBuilder: %d topic tags, %d entity tags",
            len(graph.topic_tags), len(graph.entity_tags),
        )

        # Score each event
        for event in all_candidates:
            # 1. Topic score: in_degree × base_weight (amplified by cross-source clustering)
            topic_score = sum(
                graph.topic_tags[tag].effective_weight
                for tag in event.topic_tags
                if tag in graph.topic_tags
            )

            # 2. Entity bonus: flat base_weight for each tracked entity only (no in_degree)
            entity_bonus = sum(
                ctx.get_tag_weight(tag)
                for tag in event.entity_tags
                if ctx.get_tag_weight(tag) > 1.0  # only tracked entities have weight > 1.0
            )

            # 3. Flag bonus
            flag_bonus = sum(
                FLAG_BONUS.get(flag, 0.0) for flag in event.special_flags
            )

            event.priority_score = round(topic_score + entity_bonus + flag_bonus, 2)

            if event.priority_score >= P0_THRESHOLD:
                event.priority_level = Priority.P0
            elif event.priority_score >= P1_THRESHOLD:
                event.priority_level = Priority.P1
            elif event.priority_score >= P2_THRESHOLD:
                event.priority_level = Priority.P2
            else:
                event.priority_level = Priority.P3

        logger.info(
            "GraphBuilder scoring done. P0=%d P1=%d P2=%d P3=%d",
            sum(1 for e in all_candidates if e.priority_level == Priority.P0),
            sum(1 for e in all_candidates if e.priority_level == Priority.P1),
            sum(1 for e in all_candidates if e.priority_level == Priority.P2),
            sum(1 for e in all_candidates if e.priority_level == Priority.P3),
        )
        return graph

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def render_ascii(self, graph: KnowledgeGraph, top_n_tags: int = 15) -> str:
        """
        Render an ASCII summary of the knowledge graph.
        Shows topic tags (primary scoring) and entity tags (structural) separately.
        """
        lines: list[str] = []

        # ---- Topic Tag leaderboard ----
        sorted_topic_tags = sorted(
            graph.topic_tags.values(),
            key=lambda t: t.effective_weight,
            reverse=True,
        )[:top_n_tags]

        lines.append("┌──────────────────────────────────────────────────────────────────┐")
        lines.append("│  TOPIC TAGS — primary scoring drivers (in_degree × base_weight)  │")
        lines.append("├──────────────────────────┬───────────┬─────────────┬─────────────┤")
        lines.append("│  Tag                     │ in_degree │ base_weight │ eff_weight  │")
        lines.append("├──────────────────────────┼───────────┼─────────────┼─────────────┤")
        for t in sorted_topic_tags:
            bar = "█" * min(int(t.effective_weight), 16)
            lines.append(
                f"│  {t.name:<24}│ {t.in_degree:^9} │ {t.base_weight:^11.1f} │ {t.effective_weight:^11.1f}│  {bar}"
            )
        lines.append("└──────────────────────────┴───────────┴─────────────┴─────────────┘")

        # ---- Entity Tag overview ----
        sorted_entity_tags = sorted(
            graph.entity_tags.values(),
            key=lambda t: (t.in_degree, t.base_weight),
            reverse=True,
        )
        lines.append("")
        lines.append("┌──────────────────────────────────────────────────────────────────┐")
        lines.append("│  ENTITY TAGS — flat bonus (base_weight, not amplified)           │")
        lines.append("├──────────────────────────┬───────────┬─────────────┬─────────────┤")
        lines.append("│  Tag                     │ in_degree │ base_weight │ bonus/event │")
        lines.append("├──────────────────────────┼───────────┼─────────────┼─────────────┤")
        for t in sorted_entity_tags:
            bonus = t.base_weight if t.base_weight > 1.0 else 0.0
            lines.append(
                f"│  {t.name:<24}│ {t.in_degree:^9} │ {t.base_weight:^11.1f} │ {bonus:^11.1f}│"
            )
        lines.append("└──────────────────────────┴───────────┴─────────────┴─────────────┘")

        # ---- Event priority ranking ----
        sorted_events = sorted(
            graph.events.values(),
            key=lambda e: e.priority_score,
            reverse=True,
        )
        lines.append("")
        lines.append("┌──────────────────────────────────────────────────────────────────┐")
        lines.append("│  EVENT RANKING (topic_score + entity_bonus + flag_bonus)         │")
        lines.append("├──────────┬────┬────────┬───────────────────────────────────────┤")
        lines.append("│  id      │ P# │ score  │ title                                 │")
        lines.append("├──────────┼────┼────────┼───────────────────────────────────────┤")
        for e in sorted_events:
            lvl = e.priority_level.value if e.priority_level else "??"
            flags_abbr = ",".join(f.value[:3] for f in e.special_flags) or "—"
            title_trunc = e.title[:39] if len(e.title) > 39 else e.title
            lines.append(
                f"│  {e.id:<8} │ {lvl:<2} │ {e.priority_score:>6.1f} │ {title_trunc:<39}│  [{flags_abbr}]"
            )
        lines.append("└──────────┴────┴────────┴───────────────────────────────────────┘")

        # ---- Cross-source topic clusters ----
        clusters = {
            name: node for name, node in graph.topic_tags.items()
            if node.in_degree > 1
        }
        if clusters:
            lines.append("")
            lines.append("┌──────────────────────────────────────────────────────────────────┐")
            lines.append("│  TOPIC CLUSTERS (shared by multiple events — dedup candidates)   │")
            lines.append("└──────────────────────────────────────────────────────────────────┘")
            for tag_name, node in sorted(clusters.items(), key=lambda x: x[1].in_degree, reverse=True):
                event_titles = [
                    f"{eid}({graph.events[eid].source.value[:3]})"
                    for eid in node.event_ids
                    if eid in graph.events
                ]
                lines.append(f"  [{tag_name}]  in_degree={node.in_degree}  eff_weight={node.effective_weight:.1f}")
                lines.append(f"    └─ {', '.join(event_titles)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal: infer briefing date from data sources
    # ------------------------------------------------------------------

    def _infer_briefing_date(self, filter_outputs: list) -> str:
        """
        Infer the canonical briefing date from Filter Agent outputs.

        Priority:
          1. Calendar events have the most explicit date — use their date portion.
          2. If no calendar events, use the earliest timestamp from any event.
          3. Fallback: empty string (callers should handle this gracefully).
        """
        from src.core.models import SourceType

        # 1. Look for calendar events — their start timestamps are the most reliable
        for output in filter_outputs:
            if output.source == SourceType.CALENDAR:
                for event in output.candidates:
                    if event.timestamp:
                        return event.timestamp[:10]   # "YYYY-MM-DD"
                # Also check excluded calendar events
                for event in output.excluded:
                    if event.timestamp:
                        return event.timestamp[:10]

        # 2. Fallback: earliest timestamp across all events
        timestamps = []
        for output in filter_outputs:
            for event in list(output.candidates) + list(output.excluded):
                if event.timestamp:
                    timestamps.append(event.timestamp[:10])
        if timestamps:
            return min(timestamps)

        return ""

# AI Generated Code - End
