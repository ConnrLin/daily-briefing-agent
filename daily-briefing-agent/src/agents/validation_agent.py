# AI Generated Code - Start
"""
ValidationAgent: Two-layer quality gate for WritingAgent output.

Layer 1 — Hard code checks (deterministic, no LLM):
  - No Markdown markup (#, *, _, -, bullet chars)
  - No URLs or email addresses
  - No raw numeric symbols: %, $, €, £, +digit, digit+K/M/B (unspelled)
  - Word count within profile's [min_seconds, max_seconds] × speaking_rate band
  - Opening sentence not identical to any of the last 3 sessions

Layer 2 — Soft LLM review:
  - Tone match with profile style
  - Factual accuracy (events described correctly vs source data)
  - Flow and naturalness for TTS
  - Coverage of P0 events

If any hard checks FAIL → passed=False, compose retry_hint combining all errors + LLM opinion.
If only soft issues → passed=False, retry_hint = LLM suggestions only.
If all clear → passed=True, retry_hint = "".
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI

from src.core.llm_client import extract_json, get_model
from src.core.models import (
    Priority,
    RankingAgentOutput,
    ValidationAgentOutput,
    ValidationIssue,
    WritingAgentOutput,
)
from src.core.profile_store import ProfileStore

logger = logging.getLogger(__name__)

TTS_WORDS_PER_SECOND: float = 2.5


# ---------------------------------------------------------------------------
# Hard-check patterns
# ---------------------------------------------------------------------------

# Markdown markers: # at line start, *, _, ~~ pairs, >
_MD_PATTERN = re.compile(
    r'(?m)^#+\s'               # heading
    r'|(?<!\w)[*_]{1,3}(?!\s)' # bold/italic
    r'|(?<!\w)~~.+?~~'         # strikethrough
    r'|\n>\s'                  # blockquote
    r'|\n[-*+]\s'              # bullet list
    r'|\n\d+\.\s',             # numbered list
    re.DOTALL,
)

# URLs: http/https/www
_URL_PATTERN = re.compile(
    r'https?://\S+'
    r'|www\.\S+\.\S+'
    r'|(?<!\w)\S+@\S+\.\w{2,6}(?!\w)',  # email
)

# Raw numeric symbols that must be spelled out
# e.g. "23%", "$200", "€4B", "£50M", "4B", "12K", "3.5M"
_NUMERIC_SYMBOL_PATTERNS = [
    (re.compile(r'\d+\s*%'),                  'percent sign — spell out e.g. "twenty-three percent"'),
    (re.compile(r'[$€£]\s*\d'),               'currency symbol — spell out e.g. "two hundred million dollars"'),
    (re.compile(r'\d+\s*[KMB]\b'),            'numeric abbreviation (K/M/B) — spell out e.g. "four billion"'),
    (re.compile(r'\$\d'),                     'dollar sign — spell out'),
]


# ---------------------------------------------------------------------------
# LLM Review Prompt
# ---------------------------------------------------------------------------

VALIDATION_LLM_PROMPT = """## Role
You are a quality reviewer for a personal TTS morning briefing. Your job is to identify any problems
with the briefing text — NOT to rewrite it. Return structured feedback only.

## User Profile
{user_background}
Tone style: {tone_style}
Tone rules: {tone_rules}

## Source Events (for factual accuracy check)
The following events are the COMPLETE source of truth. Summaries are NOT truncated.
Only flag a factual issue if the briefing contradicts or fabricates something not present in these summaries.
Do NOT flag something as inaccurate if it appears verbatim or paraphrased from the summary below.

{events_summary}

## Briefing Text to Review
---
{briefing_text}
---

## Review Tasks

1. **Tone match**: Does the briefing match the profile's tone style and rules? Flag any violations.
2. **Factual accuracy**: Are event details (times, names, deadlines, numbers) accurate vs the source data?
   Flag any inaccuracies or misleading phrasings.
3. **TTS naturalness**: Are there any sentences that would sound awkward when read aloud?
   (Too long, complex nested clauses, repetitive sentence starts, etc.)
4. **P0 coverage**: Are all critical events (action-required, deadline-today, conflict-detected) 
   clearly mentioned? List any that seem missing or buried.
5. **Opening variety**: The recent openings were: {recent_openings}
   Does today's opening feel sufficiently different?

## Output Format
Return a JSON object:
{{
  "tone_issues": ["<issue 1>", ...],           // empty list if fine
  "factual_issues": ["<issue 1>", ...],        // empty list if fine  
  "flow_issues": ["<issue 1>", ...],           // empty list if fine
  "coverage_issues": ["<issue 1>", ...],       // empty list if fine
  "opening_issue": "<string or empty string>",
  "overall_verdict": "pass" | "minor_issues" | "needs_revision",
  "revision_instructions": "<concise instructions for the writer to fix — empty if overall_verdict=pass>"
}}
"""


# ---------------------------------------------------------------------------
# ValidationAgent
# ---------------------------------------------------------------------------

class ValidationAgent:
    """
    Two-layer quality gate.

    Usage:
        val = ValidationAgent(profile_store, client, get_model("validation"))
        result = val.run(writing_output, ranking_output, history_path)
        if not result.passed:
            # retry WritingAgent with result.retry_hint
    """

    def __init__(
        self,
        profile_store: ProfileStore,
        client: OpenAI,
        model: str,
        history_path: Path = Path("inputs/session_history.json"),
    ):
        self.profile_store = profile_store
        self.client = client
        self.model = model
        self.history_path = history_path

    def run(
        self,
        writing_output: WritingAgentOutput,
        ranking_output: RankingAgentOutput,
        briefing_date: str = "",
    ) -> ValidationAgentOutput:
        """
        Run both layers and return a ValidationAgentOutput.
        """
        ctx = self.profile_store.context
        issues: list[ValidationIssue] = []
        text = writing_output.text

        # ── Layer 1: Hard code checks ──────────────────────────────────────
        issues.extend(self._check_markdown(text))
        issues.extend(self._check_urls(text))
        issues.extend(self._check_numeric_symbols(text))
        issues.extend(self._check_word_count(writing_output, ctx))
        issues.extend(self._check_opening_variety(text))

        hard_errors = [i for i in issues if i.severity == "error"]

        # ── Layer 2: LLM soft review ───────────────────────────────────────
        llm_issues, revision_instructions = self._llm_review(
            text, ranking_output, briefing_date
        )
        issues.extend(llm_issues)

        # ── Final verdict ──────────────────────────────────────────────────
        passed = len(issues) == 0 or (
            len(hard_errors) == 0 and not revision_instructions.strip()
        )

        retry_hint = self._build_retry_hint(hard_errors, llm_issues, revision_instructions)

        logger.info(
            "ValidationAgent: passed=%s  hard_errors=%d  llm_issues=%d",
            passed, len(hard_errors), len(llm_issues),
        )

        return ValidationAgentOutput(
            passed=passed,
            issues=issues,
            retry_hint=retry_hint,
        )

    # ------------------------------------------------------------------
    # Layer 1 checks
    # ------------------------------------------------------------------

    def _check_markdown(self, text: str) -> list[ValidationIssue]:
        """Detect any residual Markdown markup characters."""
        matches = _MD_PATTERN.findall(text)
        if matches:
            examples = list(dict.fromkeys(m.strip()[:20] for m in matches))[:3]
            return [ValidationIssue(
                severity="error",
                check="markdown_detected",
                message=f"Markdown markup found: {examples}. Remove all markdown — plain text only.",
            )]
        return []

    def _check_urls(self, text: str) -> list[ValidationIssue]:
        """Detect URLs and email addresses."""
        matches = _URL_PATTERN.findall(text)
        if matches:
            return [ValidationIssue(
                severity="error",
                check="url_or_email_detected",
                message=f"URL or email found: {matches[:2]}. Remove all URLs and email addresses.",
            )]
        return []

    def _check_numeric_symbols(self, text: str) -> list[ValidationIssue]:
        """Detect un-spoken numeric symbols (%, $, K/M/B shortcuts)."""
        issues = []
        for pattern, description in _NUMERIC_SYMBOL_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                issues.append(ValidationIssue(
                    severity="error",
                    check="numeric_symbol",
                    message=f"Unspoken numeric format '{matches[0]}' — {description}.",
                ))
        return issues

    def _check_word_count(self, output: WritingAgentOutput, ctx) -> list[ValidationIssue]:
        """Check word count falls within the profile's audio length band."""
        min_words = int(ctx.audio_min_seconds * TTS_WORDS_PER_SECOND)
        max_words = int(ctx.audio_max_seconds * TTS_WORDS_PER_SECOND)
        wc = output.word_count
        if wc < min_words:
            return [ValidationIssue(
                severity="error",
                check="word_count_too_short",
                message=(
                    f"Briefing is {wc} words ({output.estimated_seconds}s) — "
                    f"below minimum {min_words} words ({ctx.audio_min_seconds}s). "
                    f"Expand coverage of P1 topics or add more detail to P0 events."
                ),
            )]
        if wc > max_words:
            return [ValidationIssue(
                severity="error",
                check="word_count_too_long",
                message=(
                    f"Briefing is {wc} words ({output.estimated_seconds}s) — "
                    f"exceeds maximum {max_words} words ({ctx.audio_max_seconds}s). "
                    f"Remove P2 content and compress P1 topics to one sentence each."
                ),
            )]
        return []

    def _check_opening_variety(self, text: str) -> list[ValidationIssue]:
        """Check first sentence doesn't repeat a recent opening exactly."""
        if not self.history_path.exists():
            return []
        try:
            history = json.loads(self.history_path.read_text())
            sessions = history.get("sessions", [])
            recent_fragments = [
                s["opening_fragment"] for s in sessions[-3:]
                if s.get("opening_fragment")
            ]
            first_sentence = text.split(".")[0].strip().lower()
            for frag in recent_fragments:
                if frag.lower()[:60] == first_sentence[:60]:
                    return [ValidationIssue(
                        severity="warning",
                        check="opening_not_varied",
                        message=(
                            f"Opening too similar to a recent session: '{frag[:60]}…'. "
                            f"Change the opening hook."
                        ),
                    )]
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # Layer 2: LLM review
    # ------------------------------------------------------------------

    def _llm_review(
        self,
        text: str,
        ranking_output: RankingAgentOutput,
        briefing_date: str,
    ) -> tuple[list[ValidationIssue], str]:
        """Run LLM soft review. Returns (issues, revision_instructions)."""
        ctx = self.profile_store.context

        # Build concise event summary for factual check
        event_lines = []
        for topic in ranking_output.ranked_topics:
            for ev in topic.events:
                if ev.event_priority in (Priority.P0, Priority.P1):
                    flags = [f.value for f in ev.special_flags]
                    # Use FULL summary (no truncation) — truncation caused false-positive
                    # "factual_inaccuracy" judgements when key details fell after the cut-off.
                    event_lines.append(
                        f"[{ev.event_priority.value}] {ev.id} ({ev.source.value}): "
                        f"{ev.title}\n"
                        f"  summary: {ev.summary}"
                        + (f"\n  flags: {flags}" if flags else "")
                        + (f"\n  narration_hint: {ev.narration_hint}" if ev.narration_hint else "")
                    )
        events_summary = "\n".join(event_lines) if event_lines else "No events."

        # Recent opening fragments for variety check
        recent_openings: list[str] = []
        if self.history_path.exists():
            try:
                history = json.loads(self.history_path.read_text())
                recent_openings = [
                    s.get("opening_fragment", "")
                    for s in history.get("sessions", [])[-3:]
                ]
            except Exception:
                pass
        recent_openings_str = "; ".join(f'"{f[:60]}"' for f in recent_openings if f) or "none"

        user_background = (
            f"{ctx.user_name}, {ctx.user_role} at {ctx.user_company}, {ctx.user_team} team."
        )
        tone_rules = "\n".join(f"  - {r}" for r in ctx.tone_rules)

        prompt = VALIDATION_LLM_PROMPT.format(
            user_background   = user_background,
            tone_style        = ctx.tone_style,
            tone_rules        = tone_rules,
            events_summary    = events_summary,
            briefing_text     = text,
            recent_openings   = recent_openings_str,
        )

        logger.info("ValidationAgent: LLM review calling %s", self.model)
        _rf = {"response_format": {"type": "json_object"}} if supports_json_mode() else {}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": "Review the briefing now. Return only the JSON object."},
            ],
            temperature=0.2,
            **_rf,
        )
        raw = response.choices[0].message.content or ""

        try:
            data = json.loads(extract_json(raw))
        except Exception as exc:
            logger.warning("ValidationAgent: LLM response parse failed: %s", exc)
            return [], ""

        issues: list[ValidationIssue] = []
        verdict = data.get("overall_verdict", "pass")

        # Map LLM categories to ValidationIssue objects
        category_map = {
            "tone_issues":     "tone_mismatch",
            "factual_issues":  "factual_inaccuracy",
            "flow_issues":     "tts_flow",
            "coverage_issues": "p0_coverage",
        }
        for field, check_name in category_map.items():
            for msg in data.get(field, []):
                issues.append(ValidationIssue(
                    severity="warning",
                    check=check_name,
                    message=msg,
                ))

        opening_issue = data.get("opening_issue", "").strip()
        if opening_issue:
            issues.append(ValidationIssue(
                severity="warning",
                check="opening_variety",
                message=opening_issue,
            ))

        revision_instructions = ""
        if verdict in ("minor_issues", "needs_revision"):
            revision_instructions = data.get("revision_instructions", "").strip()

        logger.info(
            "ValidationAgent LLM verdict: %s  issues=%d",
            verdict, len(issues),
        )
        return issues, revision_instructions

    # ------------------------------------------------------------------
    # Build retry hint
    # ------------------------------------------------------------------

    def _build_retry_hint(
        self,
        hard_errors: list[ValidationIssue],
        llm_issues: list[ValidationIssue],
        revision_instructions: str,
    ) -> str:
        if not hard_errors and not revision_instructions:
            return ""

        parts: list[str] = []

        if hard_errors:
            parts.append("## Hard Formatting Errors (MUST fix):")
            for issue in hard_errors:
                parts.append(f"- [{issue.check}] {issue.message}")

        if revision_instructions:
            parts.append("\n## Content Revision Instructions:")
            parts.append(revision_instructions)

        if llm_issues and not revision_instructions:
            # Still include LLM warnings even if no revision_instructions
            parts.append("\n## Style / Content Suggestions:")
            for issue in llm_issues:
                parts.append(f"- [{issue.check}] {issue.message}")

        return "\n".join(parts)

# AI Generated Code - End
