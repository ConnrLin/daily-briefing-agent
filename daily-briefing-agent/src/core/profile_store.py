# AI Generated Code - Start
"""
ProfileStore: Preprocesses profile.json into a structured ProfileContext.

Design: ONE LLM call extracts normalized tags for all three categories:
  - interests        → interest_tags (base_weight 1.5-3.0)
  - not_interested   → blocked_tags  (base_weight 0.0)
  - tracked_entities → entity_tags   (base_weight 3.0, LLM-generated canonical slug)

Critically, tracked_entities go through the LLM too so that:
  1. The canonical tag slug is consistent ("cobalt-labs" not "cobaltlabs")
  2. Aliases / alternate spellings are captured (e.g. "Cobalt" → "cobalt-labs")
  3. The same slug can be used across all Filter Agents and the GraphBuilder
     with no string-matching mismatch

The resulting ProfileContext is the single source of truth shared by all agents.
"""

import json
import logging
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.core.llm_client import extract_json
from src.core.models import ProfileContext, TagWeight

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt: unified tag extraction (interests + blocked + entities in one call)
# ---------------------------------------------------------------------------

TAG_EXTRACTION_SYSTEM_PROMPT = """You are a tag extraction assistant for a personal AI briefing system.

Your job is to analyze a user's profile and produce normalized semantic tags for three categories.
These tags will be used to filter, score, and match incoming news/email/calendar items.

Output rules:
- Tags must be SHORT, lowercase, hyphenated English strings: "fintech-regulation", "cobalt-labs", "crypto-price-movement"
- Tags must be CONSISTENT — the same concept must always produce the same tag string
- Tags must be SPECIFIC enough to match content, but GENERAL enough to work across domains
- Return ONLY valid JSON, no markdown fences, no prose

Weight guidelines:
- interest tags:  base_weight 1.5 to 3.0  (higher = more important to user)
- blocked tags:   base_weight 0.0          (hard blockers, always excluded)
- entity tags:    base_weight 3.0          (always-include, highest priority)
"""

TAG_EXTRACTION_USER_TEMPLATE = """Extract semantic tags from this user's profile. All three sections below.

=== INTERESTS (assign base_weight 1.5–3.0 each) ===
{interests}

=== NOT INTERESTED (assign base_weight 0.0 — these are hard blockers) ===
{not_interested}

=== TRACKED ENTITIES (assign base_weight 3.0 — always include if mentioned) ===
{entities}

Return a JSON object with EXACTLY this structure:
{{
  "interest_tags": [
    {{"tag": "fintech-regulation", "display": "Fintech Regulation", "base_weight": 2.5, "source": "interest"}},
    {{"tag": "ai-product", "display": "AI Product Applications", "base_weight": 2.0, "source": "interest"}}
  ],
  "blocked_tags": [
    {{"tag": "celebrity-news", "display": "Celebrity News", "base_weight": 0.0, "source": "blocked"}},
    {{"tag": "crypto-price-movement", "display": "Crypto Price Movement (not regulation)", "base_weight": 0.0, "source": "blocked"}}
  ],
  "entity_tags": [
    {{
      "tag": "cobalt-labs",
      "display": "Cobalt Labs",
      "base_weight": 3.0,
      "source": "entity",
      "aliases": ["Cobalt", "cobaltlabs"],
      "reason": "my company — always include"
    }}
  ]
}}

Important nuances:
- "day-to-day cryptocurrency price movements" → blocked tag "crypto-price-movement", but crypto REGULATION is an interest
- "AI product applications (not pure research benchmarks)" → high-weight "ai-product" tag, lower-weight "ai-research" tag
- For entities, include common abbreviations or alternate spellings in "aliases" (e.g. "Cobalt Labs" → aliases: ["Cobalt"])
- Each entity gets BOTH a canonical tag AND an aliases list for fuzzy matching
- Do NOT merge interests and entities — keep them in separate arrays
"""


# ---------------------------------------------------------------------------
# Extended TagWeight for entities (aliases support)
# ---------------------------------------------------------------------------

class EntityTagWeight(TagWeight):
    """TagWeight with aliases for entity matching across different spellings."""
    aliases: list[str] = []
    reason: str = ""


# ---------------------------------------------------------------------------
# ProfileStore
# ---------------------------------------------------------------------------

class ProfileStore:
    """
    Loads profile.json, calls LLM once to extract ALL tags (interests + blocked + entities),
    and exposes a ProfileContext for use by all downstream agents.
    
    Key guarantee: entity canonical tag slugs are LLM-generated and consistent,
    so all Filter Agents and the GraphBuilder can reference the same slug string.
    """

    def __init__(self, profile_path: str | Path, client: OpenAI, model: str = "gpt-4o-mini"):
        self.profile_path = Path(profile_path)
        self.client = client
        self.model = model
        self._context: ProfileContext | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> ProfileContext:
        """
        Load and process the profile. Idempotent — caches after first call.
        """
        if self._context is not None:
            return self._context

        raw = self._read_profile()
        interest_tags, blocked_tags, entity_tags = self._extract_all_tags_via_llm(raw)

        # Build entity_tag_map: entity display name → canonical tag slug
        # (and aliases → same slug). Used for cross-agent entity matching.
        entity_tag_map: dict[str, str] = {}
        for et in entity_tags:
            entity_tag_map[et.display.lower()] = et.tag
            if isinstance(et, EntityTagWeight):
                for alias in et.aliases:
                    entity_tag_map[alias.lower()] = et.tag

        self._context = ProfileContext(
            user_name=raw["user"]["name"],
            user_role=raw["user"]["role"],
            user_company=raw["user"]["company"],
            user_team=raw["user"]["team"],
            user_timezone=raw["user"]["timezone"],

            interests_text=raw["interests"],
            not_interested_text=raw["not_interested"],

            interest_tags=interest_tags,
            blocked_tags=blocked_tags,
            entity_tags=entity_tags,
            entity_tag_map=entity_tag_map,

            tracked_entities={
                e["name"]: e["reason"] for e in raw["tracked_entities"]
            },

            tone_style=raw["tone"]["style"],
            tone_rules=raw["tone"]["rules"],
            audio_min_seconds=raw["audio_length_seconds"]["min"],
            audio_max_seconds=raw["audio_length_seconds"]["max"],
            audio_target_seconds=raw["audio_length_seconds"]["target"],
            delivery_notes=raw["delivery_notes"],
        )

        logger.info(
            "ProfileStore loaded: %d interest tags, %d blocked tags, %d entity tags | "
            "entity_tag_map: %s",
            len(interest_tags), len(blocked_tags), len(entity_tags),
            entity_tag_map,
        )
        return self._context

    @property
    def context(self) -> ProfileContext:
        if self._context is None:
            raise RuntimeError("ProfileStore.load() must be called before accessing .context")
        return self._context

    # ------------------------------------------------------------------
    # Internal: LLM call
    # ------------------------------------------------------------------

    def _read_profile(self) -> dict[str, Any]:
        with open(self.profile_path, encoding="utf-8") as f:
            return json.load(f)

    def _extract_all_tags_via_llm(
        self, raw: dict[str, Any]
    ) -> tuple[list[TagWeight], list[TagWeight], list[TagWeight]]:
        """
        Single LLM call extracts tags for all three categories.
        Falls back to hardcoded minimal set on failure.
        """
        interests_text = "\n".join(f"- {i}" for i in raw["interests"])
        not_interested_text = "\n".join(f"- {n}" for n in raw["not_interested"])
        entities_text = "\n".join(
            f'- {e["name"]}: {e["reason"]}' for e in raw["tracked_entities"]
        )

        user_prompt = TAG_EXTRACTION_USER_TEMPLATE.format(
            interests=interests_text,
            not_interested=not_interested_text,
            entities=entities_text,
        )

        for attempt in range(3):
            try:
                _rf = {"response_format": {"type": "json_object"}} if supports_json_mode() else {}
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": TAG_EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    **_rf,
                )
                raw = response.choices[0].message.content
                data = json.loads(extract_json(raw))
                return self._parse_llm_response(data)

            except Exception as e:
                logger.warning("Tag extraction attempt %d failed: %s", attempt + 1, e)
                if attempt == 2:
                    logger.error("All tag extraction attempts failed, using fallback tags")
                    return self._fallback_tags(raw)

        return [], [], []  # unreachable

    def _parse_llm_response(
        self, data: dict[str, Any]
    ) -> tuple[list[TagWeight], list[TagWeight], list[TagWeight]]:
        """Parse LLM JSON response into typed TagWeight / EntityTagWeight lists."""
        interest_tags = [
            TagWeight(**t) for t in data.get("interest_tags", [])
        ]
        blocked_tags = [
            TagWeight(**t) for t in data.get("blocked_tags", [])
        ]

        # Entity tags use the extended EntityTagWeight (with aliases)
        entity_tags: list[TagWeight] = []
        for t in data.get("entity_tags", []):
            try:
                entity_tags.append(EntityTagWeight(
                    tag=t["tag"],
                    display=t["display"],
                    base_weight=t.get("base_weight", 3.0),
                    source=t.get("source", "entity"),
                    aliases=t.get("aliases", []),
                    reason=t.get("reason", ""),
                ))
            except Exception as e:
                logger.warning("Failed to parse entity tag %s: %s", t, e)

        logger.info(
            "LLM extracted %d interest tags, %d blocked tags, %d entity tags",
            len(interest_tags), len(blocked_tags), len(entity_tags),
        )
        return interest_tags, blocked_tags, entity_tags

    def _fallback_tags(
        self, raw: dict[str, Any]
    ) -> tuple[list[TagWeight], list[TagWeight], list[TagWeight]]:
        """
        Hardcoded fallback when LLM is unavailable.
        Entity slugs use simple lowercase-hyphen normalization as last resort.
        """
        interest_tags = [
            TagWeight(tag="ai-product", display="AI Product Applications", base_weight=2.0, source="interest"),
            TagWeight(tag="fintech-regulation", display="Fintech Regulation", base_weight=2.5, source="interest"),
            TagWeight(tag="developer-tools", display="Developer Tools", base_weight=2.0, source="interest"),
            TagWeight(tag="payments", display="Payments", base_weight=2.5, source="interest"),
        ]
        blocked_tags = [
            TagWeight(tag="celebrity-news", display="Celebrity News", base_weight=0.0, source="blocked"),
            TagWeight(tag="sports", display="Sports Scores", base_weight=0.0, source="blocked"),
            TagWeight(tag="crypto-price-movement", display="Crypto Price Movement", base_weight=0.0, source="blocked"),
        ]
        # Fallback entity tags: simple slug normalization
        entity_tags: list[TagWeight] = [
            EntityTagWeight(
                tag=e["name"].lower().replace(" ", "-"),
                display=e["name"],
                base_weight=3.0,
                source="entity",
                aliases=[],
                reason=e.get("reason", ""),
            )
            for e in raw.get("tracked_entities", [])
        ]
        logger.warning("Using fallback tags — entity tag slugs may not match LLM-generated ones")
        return interest_tags, blocked_tags, entity_tags

    # ------------------------------------------------------------------
    # Prompt assembly helper (used by Filter Agents)
    # ------------------------------------------------------------------

    def build_filter_context_block(self, source_type: str) -> str:
        """
        Returns a formatted text block ready for injection into a Filter Agent's system prompt.
        
        Critically, entity tags here show both the canonical tag slug AND aliases,
        so the Filter Agent LLM can match content mentions to the correct canonical tag.
        """
        ctx = self.context

        interest_tag_lines = "\n".join(
            f'  - [{t.tag}] {t.display} (weight: {t.base_weight})'
            for t in ctx.interest_tags
        )

        blocked_tag_lines = "\n".join(
            f'  - [{t.tag}] {t.display} — HARD BLOCK'
            for t in ctx.blocked_tags
        )

        entity_tag_lines_parts = []
        for t in ctx.entity_tags:
            aliases_str = ""
            if isinstance(t, EntityTagWeight) and t.aliases:
                aliases_str = f" (also known as: {', '.join(t.aliases)})"
            reason = t.reason if isinstance(t, EntityTagWeight) else ""
            entity_tag_lines_parts.append(
                f'  - [{t.tag}] {t.display}{aliases_str} — ALWAYS INCLUDE | reason: {reason}'
            )
        entity_tag_lines = "\n".join(entity_tag_lines_parts)

        return f"""## User Profile Context

**User**: {ctx.user_name}, {ctx.user_role} at {ctx.user_company} ({ctx.user_team} team)
**Timezone**: {ctx.user_timezone}
**Data source being filtered**: {source_type}

### Tracked Entities — ALWAYS include if mentioned, use the canonical tag slug shown
{entity_tag_lines}

### Interest Tags — include and score if relevant (higher weight = more important)
{interest_tag_lines}

### Blocked Topics — EXCLUDE unless a tracked entity override applies
{blocked_tag_lines}

### CRITICAL: Tag Consistency Rule
When you assign tags in your output, you MUST use the EXACT tag slug shown above
(e.g. use "cobalt-labs" not "cobalt" or "cobaltlabs").
This ensures all agents build a consistent knowledge graph.

### Tone & Delivery
- Style: {ctx.tone_style}
- Delivery: {ctx.delivery_notes}
"""

# AI Generated Code - End
