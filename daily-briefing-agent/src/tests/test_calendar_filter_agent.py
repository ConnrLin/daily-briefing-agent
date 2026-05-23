# AI Generated Code - Start
"""
Tests for ProfileStore and CalendarFilterAgent.

Test strategy:
- ProfileStore: entity_tag_map consistency, alias resolution, LLM unified call
- CalendarFilterAgent deterministic logic: conflict detection, private masking (no LLM)
- CalendarFilterAgent LLM-dependent: EventNode building, flag parsing
- Integration: full pipeline with real input files + mocked LLM
"""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.agents.calendar_filter_agent import CalendarFilterAgent
from src.core.models import FilterDecision, SourceType, SpecialFlag
from src.core.profile_store import ProfileStore, EntityTagWeight

INPUTS_DIR = Path(__file__).parent.parent.parent / "inputs"

# ---------------------------------------------------------------------------
# Shared mock LLM response (entity_tags now included — LLM-generated slugs)
# ---------------------------------------------------------------------------

STANDARD_TAG_RESPONSE = {
    "interest_tags": [
        {"tag": "ai-product", "display": "AI Product", "base_weight": 2.0, "source": "interest"},
        {"tag": "fintech-regulation", "display": "Fintech Regulation", "base_weight": 2.5, "source": "interest"},
        {"tag": "payments", "display": "Payments", "base_weight": 2.5, "source": "interest"},
        {"tag": "developer-tools", "display": "Developer Tools", "base_weight": 2.0, "source": "interest"},
    ],
    "blocked_tags": [
        {"tag": "celebrity-news", "display": "Celebrity News", "base_weight": 0.0, "source": "blocked"},
        {"tag": "sports", "display": "Sports", "base_weight": 0.0, "source": "blocked"},
        {"tag": "crypto-price-movement", "display": "Crypto Price Movement", "base_weight": 0.0, "source": "blocked"},
    ],
    # Entity tags now go through LLM for canonical slug + alias generation
    "entity_tags": [
        {"tag": "cobalt-labs", "display": "Cobalt Labs", "base_weight": 3.0, "source": "entity",
         "aliases": ["Cobalt"], "reason": "my company — always include"},
        {"tag": "stripe", "display": "Stripe", "base_weight": 3.0, "source": "entity",
         "aliases": [], "reason": "key partner, we co-build infra"},
        {"tag": "plaid", "display": "Plaid", "base_weight": 3.0, "source": "entity",
         "aliases": [], "reason": "core integration vendor — operational issues affect us"},
        {"tag": "lyra-finance", "display": "Lyra Finance", "base_weight": 3.0, "source": "entity",
         "aliases": ["Lyra"], "reason": "main competitor, watch closely"},
        {"tag": "maya-chen", "display": "Maya Chen", "base_weight": 3.0, "source": "entity",
         "aliases": ["Maya"], "reason": "my sister — anything mentioning her goes in"},
    ],
}


def make_profile_store(mock_client: MagicMock) -> ProfileStore:
    """Create a ProfileStore with mocked LLM (uses real profile.json)."""
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(STANDARD_TAG_RESPONSE)))]
    )
    store = ProfileStore(INPUTS_DIR / "profile.json", mock_client)
    store.load()
    return store


# ---------------------------------------------------------------------------
# Test: ProfileStore — entity tag map and alias resolution
# ---------------------------------------------------------------------------

class TestProfileStore(unittest.TestCase):
    """Tests for ProfileStore, focusing on entity_tag_map and cross-agent consistency."""

    def setUp(self):
        self.client = MagicMock()
        self.store = make_profile_store(self.client)
        self.ctx = self.store.context

    def test_single_llm_call_for_all_tags(self):
        """ProfileStore should call LLM exactly once, covering all three tag categories."""
        self.assertEqual(self.client.chat.completions.create.call_count, 1)

    def test_entity_tags_are_llm_generated(self):
        """Entity tags should come from LLM output, not simple slug normalization."""
        entity_tag_names = {t.tag for t in self.ctx.entity_tags}
        self.assertIn("cobalt-labs", entity_tag_names)
        self.assertIn("lyra-finance", entity_tag_names)
        self.assertIn("maya-chen", entity_tag_names)
        # Should NOT contain naive slug like "cobalt_labs"
        self.assertNotIn("cobalt_labs", entity_tag_names)

    def test_entity_tag_map_built_from_display_names(self):
        """entity_tag_map should map display names (lowercase) to canonical slugs."""
        self.assertEqual(self.ctx.entity_tag_map.get("cobalt labs"), "cobalt-labs")
        self.assertEqual(self.ctx.entity_tag_map.get("stripe"), "stripe")
        self.assertEqual(self.ctx.entity_tag_map.get("lyra finance"), "lyra-finance")

    def test_entity_tag_map_includes_aliases(self):
        """entity_tag_map should also map aliases to the same canonical slug."""
        # "Cobalt" is an alias for "Cobalt Labs"
        self.assertEqual(self.ctx.entity_tag_map.get("cobalt"), "cobalt-labs")
        # "Lyra" is an alias for "Lyra Finance"
        self.assertEqual(self.ctx.entity_tag_map.get("lyra"), "lyra-finance")
        # "Maya" is an alias for "Maya Chen"
        self.assertEqual(self.ctx.entity_tag_map.get("maya"), "maya-chen")

    def test_resolve_entity_tag_case_insensitive(self):
        """resolve_entity_tag should work regardless of input case."""
        self.assertEqual(self.ctx.resolve_entity_tag("STRIPE"), "stripe")
        self.assertEqual(self.ctx.resolve_entity_tag("Cobalt Labs"), "cobalt-labs")
        self.assertEqual(self.ctx.resolve_entity_tag("cobalt"), "cobalt-labs")

    def test_is_entity_tracked(self):
        """is_entity_tracked should return True for known entities and aliases."""
        self.assertTrue(self.ctx.is_entity_tracked("Stripe"))
        self.assertTrue(self.ctx.is_entity_tracked("Lyra"))       # alias
        self.assertTrue(self.ctx.is_entity_tracked("Maya Chen"))
        self.assertFalse(self.ctx.is_entity_tracked("Google"))
        self.assertFalse(self.ctx.is_entity_tracked("RandomCorp"))

    def test_entity_tags_have_high_base_weight(self):
        """All entity tags must have base_weight 3.0."""
        for t in self.ctx.entity_tags:
            self.assertEqual(t.base_weight, 3.0, f"{t.tag} should have weight 3.0")

    def test_blocked_tags_have_zero_weight(self):
        """All blocked tags must have base_weight 0.0."""
        for t in self.ctx.blocked_tags:
            self.assertEqual(t.base_weight, 0.0, f"{t.tag} should have weight 0.0")

    def test_entity_tags_are_entity_tag_weight_instances(self):
        """Entity tags should be parsed as EntityTagWeight (with aliases field)."""
        for t in self.ctx.entity_tags:
            self.assertIsInstance(t, EntityTagWeight, f"{t.tag} should be EntityTagWeight")

    def test_filter_context_block_includes_canonical_tags(self):
        """build_filter_context_block should show canonical slugs in brackets for LLM injection."""
        block = self.store.build_filter_context_block("Calendar Events")
        self.assertIn("[cobalt-labs]", block)
        self.assertIn("[lyra-finance]", block)
        self.assertIn("[maya-chen]", block)
        # Aliases should also appear
        self.assertIn("also known as", block)

    def test_profile_store_idempotent(self):
        """Calling load() multiple times should not trigger additional LLM calls."""
        call_count_before = self.client.chat.completions.create.call_count
        self.store.load()
        self.store.load()
        self.assertEqual(self.client.chat.completions.create.call_count, call_count_before)


# ---------------------------------------------------------------------------
# Test: Conflict Detection (pure Python, no LLM)
# ---------------------------------------------------------------------------

class TestConflictDetection(unittest.TestCase):
    """Tests for the deterministic interval overlap algorithm."""

    def setUp(self):
        client = MagicMock()
        profile_store = make_profile_store(client)
        self.agent = CalendarFilterAgent(
            calendar_path=INPUTS_DIR / "calendar.json",
            profile_store=profile_store,
            client=client,
        )

    def test_detects_real_conflict(self):
        """cal_006 (13:00-13:30) and cal_007 (13:15-14:00) should conflict."""
        events = [
            {"id": "cal_006", "start": "2026-05-15T13:00:00-07:00", "end": "2026-05-15T13:30:00-07:00"},
            {"id": "cal_007", "start": "2026-05-15T13:15:00-07:00", "end": "2026-05-15T14:00:00-07:00"},
        ]
        conflicts = self.agent._detect_conflicts(events)
        self.assertEqual(len(conflicts), 1)
        self.assertIn(("cal_006", "cal_007"), conflicts)

    def test_no_conflict_adjacent(self):
        """Events that touch at boundaries (end == start) should NOT conflict."""
        events = [
            {"id": "cal_001", "start": "2026-05-15T08:30:00-07:00", "end": "2026-05-15T09:00:00-07:00"},
            {"id": "cal_002", "start": "2026-05-15T09:00:00-07:00", "end": "2026-05-15T10:30:00-07:00"},
        ]
        conflicts = self.agent._detect_conflicts(events)
        self.assertEqual(len(conflicts), 0)

    def test_no_conflict_sequential(self):
        """Non-overlapping events should have no conflicts."""
        events = [
            {"id": "cal_002", "start": "2026-05-15T09:00:00-07:00", "end": "2026-05-15T10:30:00-07:00"},
            {"id": "cal_008", "start": "2026-05-15T14:00:00-07:00", "end": "2026-05-15T15:00:00-07:00"},
        ]
        conflicts = self.agent._detect_conflicts(events)
        self.assertEqual(len(conflicts), 0)

    def test_full_calendar_detects_only_known_conflict(self):
        """Only cal_006/cal_007 should conflict in the real calendar data."""
        with open(INPUTS_DIR / "calendar.json") as f:
            raw = json.load(f)
        conflicts = self.agent._detect_conflicts(raw["events"])
        conflict_ids = {frozenset(pair) for pair in conflicts}
        self.assertIn(frozenset({"cal_006", "cal_007"}), conflict_ids)
        self.assertEqual(len(conflicts), 1)


# ---------------------------------------------------------------------------
# Test: Private Event Masking (pure Python)
# ---------------------------------------------------------------------------

class TestPrivateMasking(unittest.TestCase):

    def setUp(self):
        client = MagicMock()
        profile_store = make_profile_store(client)
        self.agent = CalendarFilterAgent(
            calendar_path=INPUTS_DIR / "calendar.json",
            profile_store=profile_store,
            client=client,
        )

    def test_private_event_flagged(self):
        events = [
            {
                "id": "cal_011",
                "title": "Personal appointment",
                "start": "2026-05-15T17:00:00-07:00",
                "end": "2026-05-15T18:00:00-07:00",
                "visibility": "private",
                "description": "Marked private — do not surface details.",
                "attendees": ["jordan@cobaltlabs.com"],
            }
        ]
        flagged = self.agent._apply_preprocess_flags(events, [])
        self.assertIn("private", flagged[0]["_pre_flags"])

    def test_non_private_event_not_flagged(self):
        events = [
            {
                "id": "cal_001",
                "title": "Team standup",
                "start": "2026-05-15T08:30:00-07:00",
                "end": "2026-05-15T09:00:00-07:00",
            }
        ]
        flagged = self.agent._apply_preprocess_flags(events, [])
        self.assertNotIn("private", flagged[0]["_pre_flags"])


# ---------------------------------------------------------------------------
# Test: LLM Output Parsing
# ---------------------------------------------------------------------------

class TestEventNodeBuilding(unittest.TestCase):

    def setUp(self):
        client = MagicMock()
        profile_store = make_profile_store(client)
        self.agent = CalendarFilterAgent(
            calendar_path=INPUTS_DIR / "calendar.json",
            profile_store=profile_store,
            client=client,
        )
        self.event_lookup = {
            "cal_009": {
                "id": "cal_009",
                "start": "2026-05-15T15:00:00-07:00",
                "end": "2026-05-15T16:00:00-07:00",
                "_pre_flags": [],
            }
        }

    def test_builds_event_node_with_tracked_entity_flag(self):
        item = {
            "id": "cal_009",
            "title": "Compliance sync — EU PSD3 implications",
            "summary": "Rahul walks through PSD3 impact on SCA flow.",
            "entity_tags": ["cobalt-labs", "psd3"],
            "topic_tags": ["fintech-regulation", "compliance"],
            "special_flags": ["action-required", "tracked-entity"],
            "filter_decision": "include",
            "filter_reason": "PSD3 is directly relevant to fintech regulation interest",
        }
        node = self.agent._build_event_node(item, self.event_lookup)
        self.assertEqual(node.id, "cal_009")
        self.assertEqual(node.source, SourceType.CALENDAR)
        self.assertIn(SpecialFlag.ACTION_REQUIRED, node.special_flags)
        self.assertIn(SpecialFlag.TRACKED_ENTITY, node.special_flags)
        self.assertEqual(node.filter_decision, FilterDecision.INCLUDE)
        # entity and topic tags should be split correctly
        self.assertIn("psd3", node.entity_tags)
        self.assertIn("fintech-regulation", node.topic_tags)
        self.assertNotIn("psd3", node.topic_tags)      # psd3 is entity, not topic
        self.assertNotIn("fintech-regulation", node.entity_tags)  # fintech-reg is topic, not entity

    def test_unknown_flag_is_skipped(self):
        item = {
            "id": "cal_009",
            "title": "Test",
            "summary": "Test",
            "entity_tags": [],
            "topic_tags": [],
            "special_flags": ["unknown-flag-xyz"],
            "filter_decision": "include",
        }
        node = self.agent._build_event_node(item, self.event_lookup)
        self.assertEqual(node.special_flags, [])

    def test_deprioritize_decision_parsed(self):
        item = {
            "id": "cal_001",
            "title": "Team standup",
            "summary": "Routine daily standup.",
            "entity_tags": ["cobalt-labs"],
            "topic_tags": ["team-meeting", "recurring"],
            "special_flags": [],
            "filter_decision": "deprioritize",
            "filter_reason": "Routine recurring meeting",
        }
        event_lookup = {
            "cal_001": {
                "id": "cal_001",
                "start": "2026-05-15T08:30:00-07:00",
                "end": "2026-05-15T09:00:00-07:00",
                "_pre_flags": [],
            }
        }
        node = self.agent._build_event_node(item, event_lookup)
        self.assertEqual(node.filter_decision, FilterDecision.DEPRIORITIZE)


# ---------------------------------------------------------------------------
# Test: Integration (mock LLM, real input files)
# ---------------------------------------------------------------------------

class TestCalendarFilterAgentIntegration(unittest.TestCase):

    def _make_mock_llm_response(self) -> dict:
        return {
            "candidates": [
                {
                    "id": "cal_001",
                    "title": "Team standup",
                    "summary": "Daily payments team standup.",
                    "entity_tags": ["cobalt-labs"],
                    "topic_tags": ["team-meeting", "recurring"],
                    "special_flags": [],
                    "filter_decision": "deprioritize",
                    "filter_reason": "Routine recurring standup",
                },
                {
                    "id": "cal_005",
                    "title": "Lunch with Sam Reyes (Stripe Partnerships)",
                    "summary": "Stripe partner lunch. Sam bringing Jess Park. Likely Stripe Issuing discussion.",
                    "entity_tags": ["stripe", "cobalt-labs"],
                    "topic_tags": ["external-meeting", "partner-relations"],
                    "special_flags": ["tracked-entity"],
                    "filter_decision": "include",
                    "filter_reason": "Stripe is a tracked entity",
                },
                {
                    "id": "cal_006",
                    "title": "Quick call with Maya",
                    "summary": "Family call — Mom's birthday planning.",
                    "entity_tags": ["maya-chen"],
                    "topic_tags": ["personal"],
                    "special_flags": ["personal", "tracked-entity", "conflict-detected"],
                    "filter_decision": "include",
                    "filter_reason": "Maya Chen is a tracked entity",
                },
                {
                    "id": "cal_007",
                    "title": "Customer interview — ACME Bank product team",
                    "summary": "Discovery interview on multi-rail payments.",
                    "entity_tags": ["cobalt-labs", "acme-bank"],
                    "topic_tags": ["customer-discovery", "payments-product-launch"],
                    "special_flags": ["conflict-detected"],
                    "filter_decision": "include",
                    "filter_reason": "Conflicts with cal_006 — user needs to be aware",
                },
                {
                    "id": "cal_008",
                    "title": "Board prep with Alex Wong (CEO)",
                    "summary": "Review payments revenue slides for Monday board meeting.",
                    "entity_tags": ["cobalt-labs"],
                    "topic_tags": ["board-prep", "high-stakes"],
                    "special_flags": ["action-required", "from-ceo"],
                    "filter_decision": "include",
                    "filter_reason": "CEO meeting, board prep — high priority",
                },
                {
                    "id": "cal_009",
                    "title": "Compliance sync — EU PSD3 implications",
                    "summary": "Rahul walks through PSD3 final text impact on SCA flow.",
                    "entity_tags": ["cobalt-labs", "psd3"],
                    "topic_tags": ["fintech-regulation", "compliance"],
                    "special_flags": ["action-required"],
                    "filter_decision": "include",
                    "filter_reason": "PSD3 is a key fintech regulation topic",
                },
                {
                    "id": "cal_010",
                    "title": "Competitive teardown — Lyra Finance v3 SDK",
                    "summary": "Internal review of Lyra Finance's latest SDK release.",
                    "entity_tags": ["lyra-finance", "cobalt-labs"],
                    "topic_tags": ["competitive-intel", "developer-tools"],
                    "special_flags": ["tracked-entity"],
                    "filter_decision": "include",
                    "filter_reason": "Lyra Finance is a tracked competitor",
                },
                {
                    "id": "cal_011",
                    "title": "Personal appointment",
                    "summary": "Personal commitment at 5pm. No details available.",
                    "entity_tags": [],
                    "topic_tags": ["personal"],
                    "special_flags": ["private"],
                    "filter_decision": "include",
                    "filter_reason": "Private schedule block",
                },
            ],
            "excluded": [
                {
                    "id": "cal_003",
                    "title": "1:1 with Priya Sharma (VP Eng)",
                    "summary": "Weekly 1:1.",
                    "entity_tags": ["cobalt-labs"],
                    "topic_tags": ["1-on-1", "recurring"],
                    "special_flags": [],
                    "filter_decision": "exclude",
                    "filter_reason": "Routine weekly 1:1",
                },
                {
                    "id": "cal_004",
                    "title": "Focus block — draft payments launch comms",
                    "summary": "Self-scheduled focus block.",
                    "entity_tags": ["cobalt-labs"],
                    "topic_tags": ["focus-block", "payments-product-launch"],
                    "special_flags": [],
                    "filter_decision": "exclude",
                    "filter_reason": "Focus block — no action needed from briefing",
                },
            ],
            "agent_notes": "Conflict between cal_006 and cal_007 flagged."
        }

    def test_full_pipeline(self):
        mock_llm_response = self._make_mock_llm_response()
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(STANDARD_TAG_RESPONSE)))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(mock_llm_response)))]),
        ]

        profile_store = ProfileStore(INPUTS_DIR / "profile.json", client)
        profile_store.load()

        agent = CalendarFilterAgent(
            calendar_path=INPUTS_DIR / "calendar.json",
            profile_store=profile_store,
            client=client,
        )
        output = agent.run()

        self.assertEqual(output.source, SourceType.CALENDAR)
        self.assertGreater(len(output.candidates), 0)

        candidates_by_id = {c.id: c for c in output.candidates}

        # Conflict flags
        for cid in ("cal_006", "cal_007"):
            if cid in candidates_by_id:
                self.assertIn(SpecialFlag.CONFLICT_DETECTED, candidates_by_id[cid].special_flags,
                              f"{cid} should have conflict-detected flag")

        # Board prep should be action-required
        if "cal_008" in candidates_by_id:
            self.assertIn(SpecialFlag.ACTION_REQUIRED, candidates_by_id["cal_008"].special_flags)

        # Private event
        if "cal_011" in candidates_by_id:
            self.assertTrue(candidates_by_id["cal_011"].is_private)

        # Stripe event uses canonical entity tag slug (in entity_tags)
        if "cal_005" in candidates_by_id:
            self.assertIn("stripe", candidates_by_id["cal_005"].entity_tags)
            self.assertNotIn("stripe", candidates_by_id["cal_005"].topic_tags)

        # Lyra event uses canonical entity tag slug (in entity_tags)
        if "cal_010" in candidates_by_id:
            self.assertIn("lyra-finance", candidates_by_id["cal_010"].entity_tags)
            self.assertNotIn("lyra-finance", candidates_by_id["cal_010"].topic_tags)

        # PSD3 should be entity_tag, fintech-regulation should be topic_tag
        if "cal_009" in candidates_by_id:
            self.assertIn("psd3", candidates_by_id["cal_009"].entity_tags)
            self.assertIn("fintech-regulation", candidates_by_id["cal_009"].topic_tags)

        # Excluded items
        excluded_by_id = {e.id: e for e in output.excluded}
        self.assertIn("cal_003", excluded_by_id)

        print(f"\n✓ Integration test: {len(output.candidates)} candidates, {len(output.excluded)} excluded")
        for c in sorted(output.candidates, key=lambda x: x.id):
            print(f"  [{c.filter_decision.value.upper():14}] {c.id}: {c.title}")
            print(f"    flags:       {[f.value for f in c.special_flags]}")
            print(f"    entity_tags: {c.entity_tags}")
            print(f"    topic_tags:  {c.topic_tags}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

# AI Generated Code - End
