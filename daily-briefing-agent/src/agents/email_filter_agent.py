# AI Generated Code - Start
"""
EmailFilterAgent: Filters and annotates incoming emails.

Responsibilities:
  - Hard-filter automated/promotional/newsletter emails (no LLM needed for clear cases)
  - Always include action-required labeled emails
  - Detect tracked entity mentions (from, cc, or body)
  - Produce structured EventNode list with entity_tags, topic_tags, and special_flags

Pipeline:
  Step 1 (Python): Label-based pre-filtering — mark automated as exclude_candidate,
                   mark action-required as always_include
  Step 2 (LLM):    Semantic analysis for borderline cases, tag assignment, decision
  Step 3 (Python): Build typed EventNode objects
"""

import json
import logging
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
# Label sets for deterministic pre-filtering (no LLM)
# ---------------------------------------------------------------------------

# Emails with ANY of these labels are hard-excluded (never need LLM)
HARD_EXCLUDE_LABELS = {"automated", "promotions", "spam"}

# Emails with ANY of these labels are always candidates (override other checks)
ALWAYS_INCLUDE_LABELS = {"action-required", "from-ceo", "primary"}


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

EMAIL_FILTER_SYSTEM_PROMPT = """You are the Email Analyst for a personal daily briefing system.

Your role is to review a professional's recent emails and determine:
1. Which emails are worth mentioning in a 60-90 second morning audio briefing
2. What semantic tags each email carries (entity vs. topic — they MUST be separate)
3. Any special flags that need immediate attention

{profile_context}

## Your Task

You will receive a list of emails. Some may be pre-marked with hints from the system.
For each email, decide:
- **INCLUDE**: This email is relevant and should be in the briefing
- **EXCLUDE**: Not worth mentioning (routine notification, newsletter with no relevance, etc.)
- **DEPRIORITIZE**: Loosely relevant, include only if time allows

## Tag Classification — CRITICAL

You must assign two SEPARATE lists:

### entity_tags — Named specific things
- Company/product names: "stripe", "plaid", "cobalt-labs", "aws", "linkedin"
- Person names: "maya-chen", "alex-wong", "ben-thompson"
- Specific law names: "psd3"
Use canonical slugs from the "Tracked Entity Tags" section. For non-tracked companies/people
that are clearly named, add them too (e.g. "plaid", "stripe" from email body).

### topic_tags — Content domains
What the email is fundamentally about:
- "action-item", "board-prep", "api-credentials", "fintech-regulation"
- "partner-relations", "newsletter", "billing-alert", "competitive-intel"
- "personal", "1-on-1-prep", "payments-product-launch"

### Disambiguation reminder
| Content | entity_tag | topic_tag |
|---------|-----------|-----------|
| Email about PSD3 | "psd3" | "fintech-regulation" |
| Email from Stripe's Sam Reyes | "stripe", "sam-reyes" | "partner-relations" |
| CEO board deck request | "cobalt-labs" | "board-prep", "action-item" |

## Special Flags
- `action-required`: User must DO something (reply, review, rotate credentials)
- `from-ceo`: Email is from the CEO (highest priority signal)
- `tracked-entity`: A tracked entity is the sender or key subject
- `personal`: Personal/family email
- `deadline-today`: Deadline mentioned is today

## Filtering Rules
1. **Always include** emails from tracked entities (company or person)
2. **Always include** emails with `action-required` label — these are the most important
3. **Always include** emails from CEO
4. **Include** newsletters only if they directly mention a tracked entity or key interest topic
5. **Exclude** routine automated notifications (billing digests, LinkedIn, HN weekly, etc.)
   UNLESS they mention something directly actionable or involve a tracked entity prominently
6. **Exclude** internal automated tool notifications (Notion doc updates, etc.) unless critical

## Output Format

Return a JSON object with this exact structure:
{{
  "candidates": [
    {{
      "id": "em_001",
      "title": "Board deck review request from CEO",
      "summary": "Alex Wong (CEO) asks Jordan to review payments revenue slides 8-14 before 2pm.",
      "entity_tags": ["cobalt-labs", "alex-wong"],
      "topic_tags": ["board-prep", "action-item"],
      "special_flags": ["action-required", "from-ceo"],
      "filter_decision": "include",
      "filter_reason": "CEO action-required — must review slides before 2pm board prep"
    }}
  ],
  "excluded": [
    {{
      "id": "em_005",
      "title": "LinkedIn profile views digest",
      "summary": "Weekly LinkedIn digest — no actionable content.",
      "entity_tags": ["linkedin"],
      "topic_tags": ["newsletter"],
      "special_flags": [],
      "filter_decision": "exclude",
      "filter_reason": "Automated promotional digest — not actionable"
    }}
  ],
  "agent_notes": "em_009 Plaid credential rotation has a hard deadline of May 20 — this is the most time-sensitive item."
}}
"""


# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

EMAIL_FILTER_USER_TEMPLATE = """Today is {date} ({day_of_week}).
Review window: {window_start} → {window_end}

Here are the emails in Jordan's inbox:

{emails_json}

For each email above, output your include/exclude/deprioritize decision with entity_tags and topic_tags.
Remember: entity_tags = specific named things; topic_tags = content domains. Never mix them.
"""


# ---------------------------------------------------------------------------
# EmailFilterAgent
# ---------------------------------------------------------------------------

class EmailFilterAgent:
    """
    Filter Agent for emails.

    Processing pipeline:
      1. Load emails.json
      2. Deterministic pre-filter: label-based hard exclude / always-include marking
      3. Call LLM for semantic analysis of borderline emails + tag assignment
      4. Return FilterAgentOutput
    """

    def __init__(
        self,
        email_path: str | Path,
        profile_store: ProfileStore,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
    ):
        self.email_path = Path(email_path)
        self.profile_store = profile_store
        self.client = client
        self.model = model
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> FilterAgentOutput:
        """Execute the full email filtering pipeline."""
        ctx = self.profile_store.context
        raw = self._load_emails()
        emails = raw["emails"]
        window_start = raw.get("window_start", "")
        window_end = raw.get("window_end", "")

        # Step 1: Deterministic label pre-processing
        emails_with_hints = self._apply_label_hints(emails)

        # Step 2: LLM semantic analysis
        llm_result = self._call_llm(ctx, emails_with_hints, window_start, window_end)

        # Step 3: Build EventNode objects
        output = self._build_output(llm_result, emails_with_hints)

        logger.info(
            "EmailFilterAgent: %d candidates, %d excluded",
            len(output.candidates),
            len(output.excluded),
        )
        return output

    # ------------------------------------------------------------------
    # Step 1: Label-based pre-processing
    # ------------------------------------------------------------------

    def _apply_label_hints(self, emails: list[dict]) -> list[dict]:
        """
        Add _pre_hint to each email:
          "always_include" — labeled action-required, from-ceo, or primary
          "likely_exclude" — labeled automated, promotions, spam
          "neutral"        — everything else
        These hints are visible to the LLM but the final decision is its own.
        """
        result = []
        for email in emails:
            e = dict(email)
            labels = set(e.get("labels", []))

            if labels & HARD_EXCLUDE_LABELS:
                e["_pre_hint"] = "likely_exclude"
            elif labels & ALWAYS_INCLUDE_LABELS:
                e["_pre_hint"] = "always_include"
            else:
                e["_pre_hint"] = "neutral"

            # Also pre-flag if sender is a tracked entity
            sender_email = e.get("from", {}).get("email", "")
            sender_name = e.get("from", {}).get("name", "")
            ctx = self.profile_store.context
            if ctx.resolve_entity_tag(sender_name) or ctx.resolve_entity_tag(sender_email.split("@")[0]):
                if "_pre_flags" not in e:
                    e["_pre_flags"] = []
                e["_pre_flags"].append("tracked-entity")

            result.append(e)
        return result

    # ------------------------------------------------------------------
    # Step 2: LLM call
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        ctx,
        emails: list[dict],
        window_start: str,
        window_end: str,
    ) -> dict:
        from datetime import datetime
        date_str = window_end[:10] if window_end else "unknown"
        try:
            dt = datetime.fromisoformat(window_end)
            day_of_week = dt.strftime("%A")
        except Exception:
            day_of_week = ""

        profile_context = self.profile_store.build_filter_context_block("email")
        system_prompt = EMAIL_FILTER_SYSTEM_PROMPT.format(
            profile_context=profile_context
        )

        # Slim down emails for LLM — keep only fields it needs
        emails_for_llm = [
            {
                "id": e["id"],
                "from": e.get("from", {}),
                "subject": e.get("subject", ""),
                "received_at": e.get("received_at", ""),
                "summary": e.get("summary", ""),
                "labels": e.get("labels", []),
                "_pre_hint": e.get("_pre_hint", "neutral"),
                "_pre_flags": e.get("_pre_flags", []),
            }
            for e in emails
        ]

        user_prompt = EMAIL_FILTER_USER_TEMPLATE.format(
            date=date_str,
            day_of_week=day_of_week,
            window_start=window_start,
            window_end=window_end,
            emails_json=json.dumps(emails_for_llm, indent=2, ensure_ascii=False),
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
                return json.loads(cleaned)
            except Exception as e:
                logger.warning("EmailFilterAgent LLM attempt %d failed: %s", attempt, e)
                if attempt == self.max_retries:
                    raise

    # ------------------------------------------------------------------
    # Step 3: Build output
    # ------------------------------------------------------------------

    def _build_output(self, llm_result: dict, emails: list[dict]) -> FilterAgentOutput:
        email_lookup = {e["id"]: e for e in emails}

        candidates = [
            self._build_event_node(item, email_lookup)
            for item in llm_result.get("candidates", [])
        ]
        excluded = [
            self._build_event_node(item, email_lookup)
            for item in llm_result.get("excluded", [])
        ]

        return FilterAgentOutput(
            source=SourceType.EMAIL,
            candidates=candidates,
            excluded=excluded,
            agent_notes=llm_result.get("agent_notes"),
        )

    def _build_event_node(
        self, item: dict[str, Any], email_lookup: dict[str, dict]
    ) -> EventNode:
        email_id = item.get("id", "unknown")
        original = email_lookup.get(email_id, {})

        # Special flags — merge LLM output with pre_flags, deduplicate
        raw_flags = item.get("special_flags", []) + original.get("_pre_flags", [])

        # Also promote from-ceo label to flag
        labels = set(original.get("labels", []))
        if "from-ceo" in labels:
            raw_flags.append("from-ceo")

        seen_flags: set[str] = set()
        special_flags = []
        for f in raw_flags:
            try:
                sf = SpecialFlag(f)
                if sf not in seen_flags:
                    special_flags.append(sf)
                    seen_flags.add(sf)
            except ValueError:
                logger.warning("Unknown special flag '%s' for email %s — skipping", f, email_id)

        decision_str = item.get("filter_decision", "include")
        try:
            decision = FilterDecision(decision_str)
        except ValueError:
            decision = FilterDecision.INCLUDE

        entity_tags: list[str] = item.get("entity_tags", [])
        topic_tags: list[str] = item.get("topic_tags", [])
        if not entity_tags and not topic_tags and item.get("tags"):
            topic_tags = item.get("tags", [])
            logger.warning("Email %s used legacy flat 'tags' — treating as topic_tags", email_id)

        return EventNode(
            id=email_id,
            source=SourceType.EMAIL,
            title=item.get("title", original.get("subject", "")),
            summary=item.get("summary", ""),
            entity_tags=entity_tags,
            topic_tags=topic_tags,
            special_flags=special_flags,
            timestamp=original.get("received_at"),
            filter_decision=decision,
            filter_reason=item.get("filter_reason"),
        )

    # ------------------------------------------------------------------
    # Internal loader
    # ------------------------------------------------------------------

    def _load_emails(self) -> dict[str, Any]:
        with open(self.email_path, encoding="utf-8") as f:
            return json.load(f)

# AI Generated Code - End
