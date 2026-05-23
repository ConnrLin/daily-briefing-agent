# AI Generated Code - Start
"""
WritingAgent: Converts RankingAgentOutput into a TTS-ready plain-text briefing.

Responsibilities:
  - Read session_history to select a fresh opening/closing style (not used in last 3 days)
  - Build a compressed, conversational narrative respecting topic P0 → P1 → P2 order
  - Enforce word-count budget derived from audio_target_seconds × speaking_rate
  - Output plain text only — no markdown, no bullets, no visual formatting
  - Update session_history.json with today's used style

Prompt design:
  Role → User background → Tone rules → Today's ranked content →
  Word budget → Style instructions (opening/closing) → Output format
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

from src.core.llm_client import extract_json, get_model
from src.core.models import (
    Priority,
    RankingAgentOutput,
    RankedTopic,
    Section,
    SourceType,
    WritingAgentOutput,
)
from src.core.profile_store import ProfileStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TTS speaking rate calibration
# ---------------------------------------------------------------------------

# Normal clear TTS narration rate for English (words per second)
# 150 wpm / 60 = 2.5 words per second
TTS_WORDS_PER_SECOND: float = 2.5

# Tolerance band: allow ±12% on either side of target word count
WORD_COUNT_TOLERANCE: float = 0.12


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

WRITING_SYSTEM_PROMPT = """## Role
You are a personal briefing writer. You produce a single block of plain spoken English
that will be read aloud by a TTS voice at the start of the user's day.

## User
{user_background}

## Tone Rules
{tone_rules_text}

## Mandatory Format Rules
- Plain text only. Zero markdown — no asterisks, dashes, hyphens as bullets, pound signs.
- No visual lists. Every sentence flows naturally when spoken aloud.
- Read numbers as words: "four billion", "twenty-three percent", "two PM", "May fifteenth".
- Spell out abbreviations the user would say aloud: "PSD3" → say "PSD Three", "GA" → "general availability", "CEO" → "CEO" (fine to keep), "API" → "A P I" or "the API".
- Do not start sentences with "So", "Well", "Actually", "Basically".
- One paragraph per topic is the default; merge only if it flows naturally.
- Do NOT use "Good morning" as the opening — vary it every day.

## Word Budget
Target: **{target_words} words** (≈ {target_seconds} seconds at {wps} words per second).
Acceptable range: {min_words}–{max_words} words.
This is the most important constraint. Count carefully before finalising.

## Opening Style for Today
Use style: **{opening_style}**
Style descriptions:
- calendar-anchor: Lead with the most time-critical calendar event today.
- news-hook: Open with the most surprising or impactful piece of external news.
- time-urgency: Open by naming the number of hard deadlines or conflicts that need resolution.
- entity-spotlight: Open by spotlighting the biggest tracked-entity development.
- question-hook: Open with a rhetorical question that frames the day's central tension.
- conflict-alert: Open by naming a specific scheduling or decision conflict that needs immediate action.

## Closing Style for Today
Use style: **{closing_style}**
Style descriptions:
- action-list: End by listing 2-3 concrete next actions the user should take first.
- encouragement: End with a brief, warm but direct closing sentence (1 sentence only).
- tomorrow-preview: End by flagging one thing worth preparing for tomorrow.
- single-focus: End by naming the single most important thing to focus on right now.
- time-check: End with a time-based reminder ("You have until two PM to…").

## Today's Content (ranked by priority)

Briefing date: {briefing_date}
RankingAgent notes: {agent_notes}

{ranked_content}

## P0 / P1 / P2 Coverage Rules
- **P0 topics**: Must be covered. Every P0 event within should be mentioned (even briefly).
- **P1 topics**: Should be mentioned. Compress to 1-2 sentences per topic.
- **P2 topics**: Include only if word budget allows. One sentence maximum.
- Merge closely related P0 events into a single sentence if it saves words without losing key facts.
- Conflicts (special_flag: conflict-detected) must be stated explicitly — name both parties and the time.
- Action items (special_flag: action-required, deadline-today) must state the deadline or required action.
- narration_hint fields give you the angle to take — follow them when they add clarity.

## Output Format
Return a JSON object with exactly these fields:
{{
  "text": "<the complete briefing, plain text, no markdown>",
  "covered_event_ids": ["<id of every event you actually mention in the text>"],
  "sections": [
    {{"name": "opening",   "anchor": "<exact first 6+ words of the opening sentence>",  "covered_event_ids": []}},
    {{"name": "<topic_slug>", "anchor": "<exact first 6+ words of the first sentence of this topic segment>", "covered_event_ids": ["ev_id1", "ev_id2"]}},
    ...
    {{"name": "closing",   "anchor": "<exact first 6+ words of the closing sentence>",  "covered_event_ids": []}}
  ],
  "opening_style_used": "{opening_style}",
  "closing_style_used": "{closing_style}",
  "opening_fragment": "<first sentence of the briefing>",
  "closing_fragment": "<last sentence of the briefing>"
}}

IMPORTANT for covered_event_ids:
- List ONLY the event ids whose content you actually included in the text.
- If you skipped an event due to word budget, do NOT include its id.
- The top-level covered_event_ids is the union of all section covered_event_ids.
- The anchor string is used by code to find the char position — it must match the text exactly.
"""


# ---------------------------------------------------------------------------
# Helper: build ranked content block for prompt
# ---------------------------------------------------------------------------

def _format_ranked_content(ranking_output: RankingAgentOutput) -> str:
    """Serialise ranked topics into a clean prompt block for the LLM."""
    PRIORITY_LABEL = {Priority.P0: "[MUST COVER]", Priority.P1: "[SHOULD COVER]", Priority.P2: "[TIME PERMITTING]"}
    lines = []

    for topic in ranking_output.ranked_topics:
        p_label = PRIORITY_LABEL.get(topic.topic_priority, "")
        lines.append(f"### Topic: {topic.topic}  {p_label}")
        if topic.topic_summary:
            lines.append(f"Why it matters today: {topic.topic_summary}")
        lines.append("")

        for ev in topic.events:
            ep = ev.event_priority.value
            flags = [f.value for f in ev.special_flags]
            flag_str = "  flags: " + ", ".join(flags) if flags else ""
            lines.append(f"  [{ep}] id={ev.id}  source={ev.source.value}{flag_str}")
            lines.append(f"       title: {ev.title}")
            lines.append(f"       summary: {ev.summary}")
            if ev.narration_hint:
                lines.append(f"       narration_hint: {ev.narration_hint}")
            if ev.timestamp:
                lines.append(f"       timestamp: {ev.timestamp}")
            lines.append("")

    if ranking_output.skipped_event_ids:
        lines.append(f"(Skipped / omit entirely: {ranking_output.skipped_event_ids})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session History helpers
# ---------------------------------------------------------------------------

def _load_session_history(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"sessions": [], "available_opening_styles": [], "available_closing_styles": []}


def _pick_style(available: list[str], recent_used: list[str]) -> str:
    """Pick a style not used in the last 3 sessions. Falls back to least-recently-used."""
    recent_set = set(recent_used[-3:])
    fresh = [s for s in available if s not in recent_set]
    if fresh:
        return fresh[0]
    # All recently used — pick the one used longest ago
    for s in available:
        if s not in recent_set:
            return s
    return available[0] if available else "entity-spotlight"


def _choose_styles(history: dict, briefing_date: str) -> tuple[str, str]:
    """Return (opening_style, closing_style) for today, avoiding recent repeats."""
    sessions = history.get("sessions", [])
    # Exclude today if already recorded (idempotent re-run)
    sessions = [s for s in sessions if s.get("date") != briefing_date]

    recent_openings = [s["opening_style"] for s in sessions[-3:] if "opening_style" in s]
    recent_closings = [s["closing_style"] for s in sessions[-3:] if "closing_style" in s]

    opening_styles = history.get("available_opening_styles", [
        "calendar-anchor", "news-hook", "time-urgency",
        "entity-spotlight", "question-hook", "conflict-alert",
    ])
    closing_styles = history.get("available_closing_styles", [
        "action-list", "encouragement", "tomorrow-preview",
        "single-focus", "time-check",
    ])

    opening = _pick_style(opening_styles, recent_openings)
    closing  = _pick_style(closing_styles, recent_closings)
    return opening, closing


def _update_session_history(
    history_path: Path,
    history: dict,
    briefing_date: str,
    opening_style: str,
    closing_style: str,
    opening_fragment: str,
    closing_fragment: str,
) -> None:
    """Append today's session entry and persist. Keep last 30 days."""
    sessions = [s for s in history.get("sessions", []) if s.get("date") != briefing_date]
    sessions.append({
        "date": briefing_date,
        "opening_style": opening_style,
        "closing_style": closing_style,
        "opening_fragment": opening_fragment,
        "closing_fragment": closing_fragment,
    })
    history["sessions"] = sessions[-30:]
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False))
    logger.info("WritingAgent: session_history updated for %s", briefing_date)


# ---------------------------------------------------------------------------
# WritingAgent
# ---------------------------------------------------------------------------

class WritingAgent:
    """
    Converts RankingAgentOutput → plain TTS text via one LLM call.

    Args:
        profile_store:    Loaded ProfileStore (provides user context + tone rules)
        client:           OpenAI client
        model:            Model identifier
        history_path:     Path to session_history.json (for style rotation)
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
        ranking_output: RankingAgentOutput,
        briefing_date: str = "",
        retry_hint: str = "",
    ) -> WritingAgentOutput:
        """
        Main entry point.
        If retry_hint is non-empty it is appended to the user turn so the
        LLM knows what to fix from the previous attempt.

        1. Load session history → choose opening/closing style
        2. Build system prompt with ranked content
        3. Call LLM
        4. Parse output → WritingAgentOutput
        5. Update session history
        """
        ctx = self.profile_store.context

        # ── Word budget ────────────────────────────────────────────────────
        target_words = int(ctx.audio_target_seconds * TTS_WORDS_PER_SECOND)
        min_words    = int(ctx.audio_min_seconds    * TTS_WORDS_PER_SECOND)
        max_words    = int(ctx.audio_max_seconds    * TTS_WORDS_PER_SECOND)

        # ── Style selection ────────────────────────────────────────────────
        history = _load_session_history(self.history_path)
        opening_style, closing_style = _choose_styles(history, briefing_date)
        logger.info(
            "WritingAgent: date=%s  opening=%s  closing=%s  budget=%d words (%d–%d)",
            briefing_date, opening_style, closing_style, target_words, min_words, max_words,
        )

        # ── Prompt construction ────────────────────────────────────────────
        user_background = (
            f"{ctx.user_name} is a {ctx.user_role} at {ctx.user_company}, "
            f"working on the {ctx.user_team} team."
        )
        tone_rules_text = "\n".join(f"- {r}" for r in ctx.tone_rules)
        ranked_content  = _format_ranked_content(ranking_output)
        agent_notes     = ranking_output.agent_notes or "None."

        system_prompt = WRITING_SYSTEM_PROMPT.format(
            user_background  = user_background,
            tone_rules_text  = tone_rules_text,
            target_words     = target_words,
            target_seconds   = ctx.audio_target_seconds,
            wps              = TTS_WORDS_PER_SECOND,
            min_words        = min_words,
            max_words        = max_words,
            opening_style    = opening_style,
            closing_style    = closing_style,
            briefing_date    = briefing_date or "unknown",
            agent_notes      = agent_notes,
            ranked_content   = ranked_content,
        )

        # ── LLM call ───────────────────────────────────────────────────────
        logger.info("WritingAgent: calling LLM (%s)%s", self.model, " [retry]" if retry_hint else "")
        user_msg = "Write the briefing now. Return only the JSON object."
        if retry_hint:
            user_msg = (
                "The previous attempt had issues. Fix them and rewrite the briefing.\n\n"
                f"## Issues to fix:\n{retry_hint}\n\n"
                "Return only the corrected JSON object."
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ]
        _rf = {"response_format": {"type": "json_object"}} if supports_json_mode() else {}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,   # slightly creative for natural prose
            **_rf,
        )
        raw = response.choices[0].message.content or ""
        logger.debug("WritingAgent raw response length: %d chars", len(raw))

        # ── Trim retry: if LLM exceeded word budget, ask it to cut down ───
        result = self._parse_output(
            raw, target_words, min_words, max_words, ctx.audio_target_seconds,
            history, briefing_date, opening_style, closing_style,
        )
        if result.word_count > max_words:
            overshoot = result.word_count - target_words
            logger.warning(
                "WritingAgent: %d words exceeds max %d — retrying with trim prompt (overshoot=%d)",
                result.word_count, max_words, overshoot,
            )
            trim_user_msg = (
                f"The briefing you wrote is {result.word_count} words, but the hard maximum is "
                f"{max_words} words. Cut it down to exactly {target_words} words by:\n"
                f"  1. Removing or merging P2 events entirely.\n"
                f"  2. Compressing P1 events to one sentence each.\n"
                f"  3. Keeping all P0 events and the conflict warning.\n"
                f"  4. Keep the same opening and closing style.\n"
                f"Return the same JSON format with the trimmed 'text' field."
            )
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": trim_user_msg})
            retry_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                **_rf,
            )
            raw = retry_response.choices[0].message.content or raw
            result = self._parse_output(
                raw, target_words, min_words, max_words, ctx.audio_target_seconds,
                history, briefing_date, opening_style, closing_style,
            )
            logger.info("WritingAgent trim retry: %d words", result.word_count)

        return result

    # ------------------------------------------------------------------

    def _parse_output(
        self,
        raw: str,
        target_words: int,
        min_words: int,
        max_words: int,
        target_seconds: float,
        history: dict,
        briefing_date: str,
        opening_style: str,
        closing_style: str,
    ) -> WritingAgentOutput:
        try:
            data = json.loads(extract_json(raw))
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error("WritingAgent: JSON parse failed: %s", exc)
            # Treat the raw text as the briefing text (graceful fallback)
            text = raw.strip()
            data = {"text": text, "sections": [], "opening_fragment": "", "closing_fragment": ""}

        text: str = data.get("text", "").strip()

        # Actual word count (split on whitespace)
        word_count = len(text.split())
        estimated_seconds = round(word_count / TTS_WORDS_PER_SECOND, 1)

        # Top-level covered_event_ids — what the LLM says it actually used
        covered_event_ids: list[str] = data.get("covered_event_ids", [])

        # Build Section objects with real char positions via anchor search
        sections: list[Section] = []
        raw_sections = data.get("sections", [])
        text_lower = text.lower()
        prev_end = 0
        for i, sec in enumerate(raw_sections):
            name    = sec.get("name", "unknown")
            anchor  = sec.get("anchor", "").strip()
            covered = sec.get("covered_event_ids", [])

            # Find char_start by searching anchor string in text
            char_start = 0
            if anchor:
                # Try exact match first, then first 30 chars
                pos = text_lower.find(anchor.lower())
                if pos == -1:
                    pos = text_lower.find(anchor[:30].lower())
                char_start = pos if pos != -1 else prev_end
            else:
                char_start = prev_end

            # char_end = start of NEXT section (or end of text for last section)
            char_end = len(text)  # default: to end
            if i + 1 < len(raw_sections):
                next_anchor = raw_sections[i + 1].get("anchor", "").strip()
                if next_anchor:
                    npos = text_lower.find(next_anchor.lower())
                    if npos == -1:
                        npos = text_lower.find(next_anchor[:30].lower())
                    if npos != -1:
                        char_end = npos

            prev_end = char_start
            sections.append(Section(
                name=name,
                char_start=char_start,
                char_end=char_end,
                covered_event_ids=covered,
            ))

        logger.info(
            "WritingAgent: %d words / %.1fs  (target %d words / %ds)",
            word_count, estimated_seconds, target_words, target_seconds,
        )

        # ── Persist session history ────────────────────────────────────────
        _update_session_history(
            self.history_path,
            history,
            briefing_date,
            opening_style,
            closing_style,
            data.get("opening_fragment", ""),
            data.get("closing_fragment", ""),
        )

        return WritingAgentOutput(
            text=text,
            word_count=word_count,
            estimated_seconds=estimated_seconds,
            sections=sections,
            covered_event_ids=covered_event_ids,
        )

# AI Generated Code - End
