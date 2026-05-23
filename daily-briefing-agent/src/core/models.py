# AI Generated Code - Start
"""
Core data models for the Daily Briefing Agent system.
These models define the contract between all agents.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    CALENDAR = "calendar"
    EMAIL = "email"
    NEWS = "news"


class SpecialFlag(str, Enum):
    """Special attention flags attached to Event nodes."""
    ACTION_REQUIRED   = "action-required"    # Requires user action today
    CONFLICT_DETECTED = "conflict-detected"  # Calendar time overlap
    TRACKED_ENTITY    = "tracked-entity"     # Mentions a tracked entity
    FROM_CEO          = "from-ceo"           # Email from CEO
    PERSONAL          = "personal"           # Personal / family item
    PRIVATE           = "private"            # Calendar event marked private
    DEADLINE_TODAY    = "deadline-today"     # Hard deadline is today
    PREP_REQUIRED     = "prep-required"      # Pre-meeting preparation needed
    HIGH_STAKES       = "high-stakes"        # High-stakes meeting or decision
    EXTERNAL          = "external"           # Involves external party


class FilterDecision(str, Enum):
    INCLUDE = "include"       # Include as candidate
    EXCLUDE = "exclude"       # Hard exclude (not_interested / automated)
    DEPRIORITIZE = "deprioritize"  # Keep but lower priority


class Priority(str, Enum):
    P0 = "P0"  # Action-required + today deadline
    P1 = "P1"  # Tracked entity + directly linked to today's schedule
    P2 = "P2"  # High-weight tag, no immediate deadline
    P3 = "P3"  # Background / low relevance


# ---------------------------------------------------------------------------
# ProfileStore models
# ---------------------------------------------------------------------------

class TagWeight(BaseModel):
    """A semantic tag extracted from user profile, with base weight."""
    tag: str = Field(..., description="Normalized tag label, snake_case")
    display: str = Field(..., description="Human-readable display name")
    base_weight: float = Field(..., ge=0.0, description="Base importance weight from profile")
    source: str = Field(..., description="Which profile field this came from: interest/blocked/entity")


class ProfileContext(BaseModel):
    """
    Preprocessed user profile ready for injection into agent prompts.
    Generated once per run by ProfileStore.
    """
    # User identity
    user_name: str
    user_role: str
    user_company: str
    user_team: str
    user_timezone: str

    # Raw profile text (for prompt injection)
    interests_text: list[str]
    not_interested_text: list[str]

    # Structured tags (for graph building)
    interest_tags: list[TagWeight]   # tags derived from interests
    blocked_tags: list[TagWeight]    # tags derived from not_interested
    entity_tags: list[TagWeight]     # tags from tracked_entities (LLM-generated canonical slugs)

    # Canonical entity tag map: any name/alias → canonical tag slug
    # e.g. {"cobalt labs": "cobalt-labs", "cobalt": "cobalt-labs"}
    # Used by GraphBuilder and Filter Agents to ensure cross-agent tag consistency.
    entity_tag_map: dict[str, str] = Field(
        default_factory=dict,
        description="entity display name / alias (lowercase) → canonical tag slug"
    )

    # Tracked entities for quick lookup
    tracked_entities: dict[str, str]  # name -> reason

    # Tone and delivery
    tone_style: str
    tone_rules: list[str]
    audio_min_seconds: int
    audio_max_seconds: int
    audio_target_seconds: int
    delivery_notes: str

    # Convenience: all tags in one flat list
    @property
    def all_tags(self) -> list[TagWeight]:
        return self.interest_tags + self.blocked_tags + self.entity_tags

    def get_tag_weight(self, tag: str) -> float:
        """Look up base weight for a tag, default 1.0 if not found."""
        for t in self.all_tags:
            if t.tag == tag:
                return t.base_weight
        return 1.0

    def resolve_entity_tag(self, name: str) -> Optional[str]:
        """
        Resolve an entity name or alias to its canonical tag slug.
        Returns None if the name is not a tracked entity.
        e.g. resolve_entity_tag("Cobalt") → "cobalt-labs"
        """
        return self.entity_tag_map.get(name.lower())

    def is_entity_tracked(self, name: str) -> bool:
        """Check if a named entity is in the tracked list (case-insensitive)."""
        return self.resolve_entity_tag(name) is not None


# ---------------------------------------------------------------------------
# Event node (output of Filter Agents, input to GraphBuilder)
# ---------------------------------------------------------------------------

class EventNode(BaseModel):
    """
    A candidate information item produced by a Filter Agent.
    Becomes a node in the knowledge graph.

    Tags are split into two distinct lists:
      entity_tags  — named specific things (company, person, product, regulation name)
                     e.g. "cobalt-labs", "stripe", "psd3", "maya-chen"
                     Role: trigger always-include; contribute entity-based scoring

      topic_tags   — semantic domains / content categories
                     e.g. "fintech-regulation", "api-design", "payments-product-launch"
                     Role: primary driver of priority_score via effective_weight
    """
    id: str = Field(..., description="Original source id, e.g. cal_001, em_011, news_002")
    source: SourceType
    title: str
    summary: str
    entity_tags: list[str] = Field(
        default_factory=list,
        description="Named entities: company, person, product, specific law/regulation name"
    )
    topic_tags: list[str] = Field(
        default_factory=list,
        description="Semantic topic domains: what this event is about"
    )
    special_flags: list[SpecialFlag] = Field(default_factory=list)
    timestamp: Optional[str] = Field(None, description="ISO8601 timestamp of event/email/news")

    # Set by GraphBuilder after tag scoring
    priority_score: float = Field(default=0.0)
    priority_level: Optional[Priority] = None

    # Filter decision (for traceability)
    filter_decision: FilterDecision = FilterDecision.INCLUDE
    filter_reason: Optional[str] = None

    # For calendar: start/end for conflict detection
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    is_private: bool = False


# ---------------------------------------------------------------------------
# Filter Agent output
# ---------------------------------------------------------------------------

class FilterAgentOutput(BaseModel):
    """
    Complete output from one Filter Agent (Calendar / Email / News).
    """
    source: SourceType
    candidates: list[EventNode] = Field(
        default_factory=list,
        description="Items that passed filtering and are candidates for the briefing"
    )
    excluded: list[EventNode] = Field(
        default_factory=list,
        description="Items that were hard-excluded, with filter_reason explaining why"
    )
    agent_notes: Optional[str] = Field(
        None,
        description="Any special observations the agent wants to pass to the next stage"
    )


# ---------------------------------------------------------------------------
# Knowledge Graph models
# ---------------------------------------------------------------------------

class TagNode(BaseModel):
    """
    A Tag node in the knowledge graph.
    tag_type distinguishes entity tags from topic tags for scoring purposes.
    """
    name: str
    tag_type: str = "topic"            # "entity" | "topic"
    base_weight: float = 1.0           # From ProfileStore
    in_degree: int = 0                 # How many Events reference this tag
    event_ids: list[str] = Field(default_factory=list)

    @property
    def effective_weight(self) -> float:
        return self.in_degree * self.base_weight


class KnowledgeGraph(BaseModel):
    """The in-memory knowledge graph built from Filter Agent outputs."""
    events: dict[str, EventNode] = Field(default_factory=dict)
    entity_tags: dict[str, TagNode] = Field(default_factory=dict)  # entity tag_name -> TagNode
    topic_tags: dict[str, TagNode] = Field(default_factory=dict)   # topic tag_name -> TagNode

    # The canonical "today" date for this briefing, derived from input data (not system clock).
    # Format: "YYYY-MM-DD", e.g. "2026-05-15"
    briefing_date: str = Field(
        default="",
        description="The briefing date inferred from input data timestamps (calendar.date or news.generated_at)"
    )

    @property
    def all_tags(self) -> dict[str, TagNode]:
        return {**self.entity_tags, **self.topic_tags}


# ---------------------------------------------------------------------------
# Ranking Agent output
# ---------------------------------------------------------------------------

class RankedEvent(BaseModel):
    """
    A single event inside a RankedTopic, annotated with event-level priority
    and a narration hint for the Writing Agent.
    """
    id: str
    source: SourceType
    event_priority: Priority          # P0 / P1 / P2 assigned by RankingAgent LLM
    special_flags: list[SpecialFlag] = Field(default_factory=list)
    title: str
    summary: str
    narration_hint: str = Field(
        "",
        description=(
            "One-sentence guidance for WritingAgent: what angle to take, "
            "what to emphasise, whether to merge with another event."
        ),
    )
    timestamp: Optional[str] = None


class RankedTopic(BaseModel):
    """
    A topic cluster produced by RankingAgent, ready for WritingAgent to narrate.

    topic_priority reflects the importance of this theme today (P0/P1/P2).
    events are sorted P0 → P1 → P2 within the topic.
    """
    topic: str                          # e.g. "payments-product-launch"
    topic_priority: Priority            # P0: must cover; P1: should cover; P2: time permitting
    topic_summary: str = Field(
        "",
        description="1-2 sentence overview of why this topic matters today (for WritingAgent context)",
    )
    effective_weight: float = 0.0       # From GraphBuilder (for reference / audit)
    events: list[RankedEvent] = Field(default_factory=list)

    @property
    def must_mention_count(self) -> int:
        return sum(1 for e in self.events if e.event_priority == Priority.P0)


class RankingAgentOutput(BaseModel):
    """
    Complete output from RankingAgent.
    ranked_topics is ordered by topic_priority (P0 first), then effective_weight desc.
    WritingAgent consumes this directly.
    """
    ranked_topics: list[RankedTopic]
    skipped_event_ids: list[str] = Field(
        default_factory=list,
        description="Event IDs that were dropped (too low priority / redundant)",
    )
    agent_notes: str = Field(
        "",
        description="Any observations for the WritingAgent (e.g. conflict warnings, tone suggestions)",
    )


# ---------------------------------------------------------------------------
# Writing Agent output
# ---------------------------------------------------------------------------

class Section(BaseModel):
    """A chapter/section within the briefing."""
    name: str                           # e.g. "opening", "action-items", "news", "schedule", "closing"
    char_start: int
    char_end: int
    covered_event_ids: list[str]


class WritingAgentOutput(BaseModel):
    """Draft output from the Writing Agent."""
    text: str = Field(..., description="Plain text for TTS, no markdown")
    word_count: int
    estimated_seconds: float
    sections: list[Section]
    # Event IDs actually mentioned in the text (self-reported by LLM)
    covered_event_ids: list[str] = Field(
        default_factory=list,
        description="IDs of events the writing agent actually included in the text"
    )


# ---------------------------------------------------------------------------
# Validation Agent output
# ---------------------------------------------------------------------------

class ValidationIssue(BaseModel):
    severity: str  # "error" | "warning"
    check: str     # e.g. "word_count", "not_interested_topic", "markdown_detected"
    message: str


class ValidationAgentOutput(BaseModel):
    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    retry_hint: str = Field("", description="Instruction for the next retry attempt")


# ---------------------------------------------------------------------------
# Final output
# ---------------------------------------------------------------------------

class BriefingOutput(BaseModel):
    """The final deliverable written to briefing.txt and briefing.json."""
    text: str
    word_count: int
    estimated_seconds: float
    sections: list[Section]
    covered_calendar_ids: list[str]
    covered_email_ids: list[str]
    covered_news_ids: list[str]
    excluded_items: list[dict]   # [{id, source, reason}]
    generation_timestamp: str

# AI Generated Code - End
