# AI Generated Code - Start
"""
NewsFilterAgent: Filters and annotates news items.

Responsibilities:
  - Hard-filter blocked topics (sports, entertainment-gossip, celebrity-news)
    deterministically via keyword/source matching — no LLM needed
  - Detect tracked entity mentions in headline and summary
  - Assess relevance to user's interest topics
  - Produce structured EventNode list with entity_tags, topic_tags, special_flags

Pipeline:
  Step 1 (Python): Hard-filter obvious blocked topics, flag tracked entity mentions
  Step 2 (LLM):    Relevance scoring, tag assignment, inclusion decision for remainder
  Step 3 (Python): Build typed EventNode objects
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.core.llm_client import extract_json
from src.core.models import (
    EventNode,
    FilterAgentOutput,
    FilterDecision,
    SourceType,
    SpecialFlag,
)
from src.core.profile_store import ProfileStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hard-exclude news sources (always irrelevant regardless of content)
# Only ban sources that are PURELY entertainment/sports — never fintech/tech
# ---------------------------------------------------------------------------

BLOCKED_SOURCES = {
    "ESPN", "Variety", "TMZ", "People", "Entertainment Weekly",
    "GamesIndustry.biz",  # gaming
    "BBC Sport",           # sports-specific outlet (not general BBC)
}

# Keywords in title that trigger hard-exclude (must be clearly off-topic for a fintech PM)
BLOCKED_TITLE_PATTERNS = [
    # Sports
    re.compile(r"\b(NBA|NFL|MLB|NHL|soccer|football|basketball|tennis|golf|cricket)\b", re.I),
    re.compile(r"\b(eliminated|playoff|championship|super bowl|world cup|game 7|FA Cup|wins the)\b", re.I),
    # Celebrity / entertainment
    re.compile(r"\b(celebrity|gossip|dating|divorced|married|baby|pregnant)\b", re.I),
    re.compile(r"\b(album|tour|box office|oscars|grammys|emmys|cannes film)\b", re.I),
    # Climate / health (not in user interest)
    re.compile(r"\b(hottest year on record|wmo confirms|long Covid symptoms|cancer treatment|clinical trial)\b", re.I),
    # Generic earnings / layoffs from companies user doesn't track
    # Only fires if BOTH a non-tracked company AND a generic corporate event are in the title
    re.compile(r"\b(Snowflake|Salesforce|Walmart|Oracle|SAP|Workday)\b.{0,30}\b(earnings|layoffs?|restructur|acqui)\b", re.I),
    re.compile(r"\b(layoffs?|restructur).{0,30}\b(Snowflake|Salesforce|Walmart|Oracle|SAP|Workday)\b", re.I),
    # Game / entertainment industry deals
    re.compile(r"\b(gaming studio|game publisher|esports|console)\b", re.I),
]

# Keywords that OVERRIDE hard-exclude (entity mentions rescue an item)
RESCUE_PATTERNS = [
    re.compile(r"\b(psd3|plaid|cobalt\s*labs?|lyra\s*finance|stripe|fednow|eu\s*ai\s*act)\b", re.I),
]


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

NEWS_FILTER_SYSTEM_PROMPT = """You are the News Analyst for a personal daily briefing system.

Your role is to review a morning news digest and determine:
1. Which news items are relevant for a busy fintech professional's morning briefing
2. What semantic tags each item carries (entity vs. topic — MUST be separate lists)
3. Any special flags warranting immediate attention

{profile_context}

## Your Task

You will receive a list of news items. Some have been pre-marked as blocked by the system
(clear off-topic content). Focus on deciding which of the remaining items deserve a mention
and how to tag them accurately.

For each item, decide:
- **INCLUDE**: Directly relevant to the user's work or tracked entities — must be in briefing
- **EXCLUDE**: Not relevant enough (tangential tech news, general market news with no direct impact)
- **DEPRIORITIZE**: Loosely interesting, include only if time allows

## Tag Classification — CRITICAL

Two SEPARATE lists for every item:

### entity_tags — Named specific things
- Company names: "cobalt-labs", "stripe", "plaid", "lyra-finance", "anthropic", "openai"
- Specific regulation names: "psd3", "eu-ai-act", "sec"
- Person names only if they are key actors in the story
Use canonical slugs from "Tracked Entity Tags" when applicable.

### topic_tags — Content domains
What the news is fundamentally about:
- "fintech-regulation", "competitive-intel", "crypto-regulation"
- "developer-tools", "ai-product", "payments-infrastructure"
- "funding-round", "lawsuit", "product-launch"

### Examples
| News | entity_tags | topic_tags |
|------|------------|-----------|
| Lyra Finance raises $80M Series C | "lyra-finance", "sequoia" | "competitive-intel", "funding-round" |
| EU publishes final PSD3 text | "psd3" | "fintech-regulation", "compliance" |
| Claude 5.0 with 2M context | "anthropic" | "ai-product", "large-language-models" |
| Stripe Issuing API in 5 new markets | "stripe" | "payments-infrastructure", "product-launch" |

## Special Flags
- `tracked-entity`: A tracked entity (Cobalt Labs, Stripe, Plaid, Lyra Finance, Maya Chen) is
  the direct subject of the story — NOT just mentioned in passing
- `action-required`: This news requires the user to take action today (e.g. a regulation they
  must read before a sync meeting)

## Filtering Rules
1. **Always include** news where a tracked entity is the MAIN subject
2. **Always include** regulation/compliance news in the user's interest domains
3. **Include** competitive intelligence about fintech companies
4. **Deprioritize** general tech/AI news unless it has direct fintech product relevance
5. **Exclude** sports, celebrity, entertainment, crypto price movements
6. **Exclude** news items pre-marked as [HARD_EXCLUDED] by the system

## Output Format

Return a JSON object:
{{
  "candidates": [
    {{
      "id": "news_001",
      "title": "Cobalt Labs payments platform exits beta with $4B transaction volume",
      "summary": "Cobalt Labs announced GA of its payments platform after 6-month beta processing $4B with Stripe.",
      "entity_tags": ["cobalt-labs", "stripe"],
      "topic_tags": ["payments-product-launch", "product-launch"],
      "special_flags": ["tracked-entity"],
      "filter_decision": "include",
      "filter_reason": "Cobalt Labs (user's company) is the direct subject — highly relevant"
    }}
  ],
  "excluded": [
    {{
      "id": "news_006",
      "title": "Lakers eliminated in Game 7",
      "summary": "Sports news — pre-excluded.",
      "entity_tags": [],
      "topic_tags": ["sports"],
      "special_flags": [],
      "filter_decision": "exclude",
      "filter_reason": "Sports news — outside user's interests"
    }}
  ],
  "agent_notes": "news_002 (PSD3 final text) is directly linked to cal_009 compliance sync at 3pm — the user needs to read it before that meeting."
}}
"""


# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

NEWS_FILTER_USER_TEMPLATE = """Today is {date} ({day_of_week}).

Here are today's news items (some pre-marked as [HARD_EXCLUDED]):

{news_json}

Decide include/exclude/deprioritize for each item. Assign entity_tags and topic_tags separately.
"""


# ---------------------------------------------------------------------------
# NewsFilterAgent
# ---------------------------------------------------------------------------

class NewsFilterAgent:
    """
    Filter Agent for news items.

    Processing pipeline:
      1. Load news.json
      2. Deterministic pre-filter: hard-exclude blocked sources/patterns, flag entity mentions
      3. Call LLM for relevance scoring and tag assignment
      4. Return FilterAgentOutput
    """

    def __init__(
        self,
        news_path: str | Path,
        profile_store: ProfileStore,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
    ):
        self.news_path = Path(news_path)
        self.profile_store = profile_store
        self.client = client
        self.model = model
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> FilterAgentOutput:
        """Execute the full news filtering pipeline."""
        ctx = self.profile_store.context
        raw = self._load_news()
        items = raw["items"]
        generated_at = raw.get("generated_at", "")

        # Step 1: Deterministic pre-filtering
        items_with_hints = self._apply_preprocess_flags(items)

        # Step 2: LLM analysis
        llm_result = self._call_llm(ctx, items_with_hints, generated_at)

        # Step 3: Build EventNode objects
        output = self._build_output(llm_result, items_with_hints)

        logger.info(
            "NewsFilterAgent: %d candidates, %d excluded",
            len(output.candidates),
            len(output.excluded),
        )
        return output

    # ------------------------------------------------------------------
    # Step 1: Deterministic pre-filtering
    # ------------------------------------------------------------------

    def _apply_preprocess_flags(self, items: list[dict]) -> list[dict]:
        """
        Tag each news item with:
          _hard_excluded: True  → system has already decided this is off-topic
          _pre_flags: list of pre-detected flags (e.g. tracked-entity)

        Rescue rule: if a blocked item explicitly mentions a tracked entity in the title,
        it is NOT hard-excluded.
        """
        ctx = self.profile_store.context
        result = []
        for item in items:
            n = dict(item)
            n["_hard_excluded"] = False
            n["_pre_flags"] = []
            title = item.get("title", "")
            full_text = f"{title} {item.get('summary', '')}"

            # Check if a tracked entity rescues this item
            is_rescued = any(p.search(full_text) for p in RESCUE_PATTERNS)

            # Hard-exclude by source (unless rescued)
            if not is_rescued and item.get("source") in BLOCKED_SOURCES:
                n["_hard_excluded"] = True
                n["_exclude_reason"] = f"Blocked source: {item['source']}"
                result.append(n)
                continue

            # Hard-exclude by title pattern (unless rescued)
            if not is_rescued:
                for pattern in BLOCKED_TITLE_PATTERNS:
                    if pattern.search(title):
                        n["_hard_excluded"] = True
                        n["_exclude_reason"] = "Blocked topic pattern in title"
                        break

            # Pre-flag tracked entity mentions in title + summary
            for alias, canonical_slug in ctx.entity_tag_map.items():
                if re.search(r'\b' + re.escape(alias) + r'\b', full_text, re.I):
                    if "tracked-entity" not in n["_pre_flags"]:
                        n["_pre_flags"].append("tracked-entity")
                    break

            result.append(n)

        n_excluded = sum(1 for n in result if n["_hard_excluded"])
        logger.info("NewsFilterAgent pre-filter: %d hard-excluded of %d items", n_excluded, len(result))
        return result

    # ------------------------------------------------------------------
    # Step 2: LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, ctx, items: list[dict], generated_at: str) -> dict:
        from datetime import datetime
        date_str = generated_at[:10] if generated_at else "unknown"
        try:
            dt = datetime.fromisoformat(generated_at)
            day_of_week = dt.strftime("%A")
        except Exception:
            day_of_week = ""

        # Split hard-excluded vs items to send to LLM
        to_llm = []
        pre_excluded = []
        for item in items:
            if item.get("_hard_excluded"):
                pre_excluded.append(item)
            else:
                # Only send fields the LLM needs — strip internal hints to save tokens
                to_llm.append({
                    "id": item["id"],
                    "source": item.get("source", ""),
                    "title": item.get("title", ""),
                    "published_at": item.get("published_at", ""),
                    "summary": item.get("summary", ""),
                    "_pre_flags": item.get("_pre_flags", []),
                })

        profile_context = self.profile_store.build_filter_context_block("news")
        system_prompt = NEWS_FILTER_SYSTEM_PROMPT.format(
            profile_context=profile_context
        )
        user_prompt = NEWS_FILTER_USER_TEMPLATE.format(
            date=date_str,
            day_of_week=day_of_week,
            news_json=json.dumps(to_llm, indent=2, ensure_ascii=False),
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
                    temperature=0.2,
                    **_rf,
                )
                raw_text = response.choices[0].message.content
                cleaned = extract_json(raw_text)
                result = json.loads(cleaned)

                # Append hard-excluded items as excluded entries
                for ex in pre_excluded:
                    result.setdefault("excluded", []).append({
                        "id": ex["id"],
                        "title": ex.get("title", ""),
                        "summary": ex.get("summary", ""),
                        "entity_tags": [],
                        "topic_tags": [],
                        "special_flags": [],
                        "filter_decision": "exclude",
                        "filter_reason": ex.get("_exclude_reason", "Hard-excluded by pre-filter"),
                    })
                return result
            except Exception as e:
                logger.warning("NewsFilterAgent LLM attempt %d failed: %s", attempt, e)
                if attempt == self.max_retries:
                    raise

    # ------------------------------------------------------------------
    # Step 3: Build output
    # ------------------------------------------------------------------

    def _build_output(self, llm_result: dict, items: list[dict]) -> FilterAgentOutput:
        item_lookup = {n["id"]: n for n in items}

        candidates = [
            self._build_event_node(item, item_lookup)
            for item in llm_result.get("candidates", [])
        ]
        excluded = [
            self._build_event_node(item, item_lookup)
            for item in llm_result.get("excluded", [])
        ]

        return FilterAgentOutput(
            source=SourceType.NEWS,
            candidates=candidates,
            excluded=excluded,
            agent_notes=llm_result.get("agent_notes"),
        )

    def _build_event_node(
        self, item: dict[str, Any], item_lookup: dict[str, dict]
    ) -> EventNode:
        item_id = item.get("id", "unknown")
        original = item_lookup.get(item_id, {})

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
                logger.warning("Unknown special flag '%s' for news %s — skipping", f, item_id)

        decision_str = item.get("filter_decision", "include")
        try:
            decision = FilterDecision(decision_str)
        except ValueError:
            decision = FilterDecision.INCLUDE

        entity_tags: list[str] = item.get("entity_tags", [])
        topic_tags: list[str] = item.get("topic_tags", [])
        if not entity_tags and not topic_tags and item.get("tags"):
            topic_tags = item.get("tags", [])
            logger.warning("News %s used legacy flat 'tags' — treating as topic_tags", item_id)

        return EventNode(
            id=item_id,
            source=SourceType.NEWS,
            title=item.get("title", original.get("title", "")),
            summary=item.get("summary", ""),
            entity_tags=entity_tags,
            topic_tags=topic_tags,
            special_flags=special_flags,
            timestamp=original.get("published_at"),
            filter_decision=decision,
            filter_reason=item.get("filter_reason"),
        )

    # ------------------------------------------------------------------
    # Internal loader
    # ------------------------------------------------------------------

    def _load_news(self) -> dict[str, Any]:
        with open(self.news_path, encoding="utf-8") as f:
            return json.load(f)

# AI Generated Code - End
