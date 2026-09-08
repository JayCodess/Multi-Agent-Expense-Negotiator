"""
Tests for hard constraint validation logic.

Verifies that:
- A proposal violating a hard max budget is auto-objected without an LLM call.
- A proposal within budget limits produces no auto-objections.
- Proposal sum re-normalization works correctly.
"""

import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from validators.hard_constraints import check_hard_budget_limits, validate_proposal_sum
from graph.state import PersonConstraint


class TestCheckHardBudgetLimits:
    """Tests for the hard budget limit checker."""

    def test_exceeds_hard_max_generates_objection(self):
        """A proposal exceeding hard_max_budget should auto-generate an objection."""
        people = [
            PersonConstraint(
                name="Alice",
                preferences="Budget-constrained",
                hard_max_budget=5000.0,
                hard_min_budget=None,
            ),
            PersonConstraint(
                name="Bob",
                preferences="No constraints",
                hard_max_budget=None,
                hard_min_budget=None,
            ),
        ]
        proposal = {"Alice": 6000.0, "Bob": 4000.0}

        objections, auto_names = check_hard_budget_limits(proposal, people, round_number=1)

        assert len(objections) == 1
        assert objections[0]["person"] == "Alice"
        assert "Exceeds hard budget limit" in objections[0]["reason"]
        assert objections[0]["round"] == 1
        assert "Alice" in auto_names

    def test_within_budget_no_objection(self):
        """A proposal within budget limits should produce no auto-objections."""
        people = [
            PersonConstraint(
                name="Alice",
                preferences="Budget-constrained",
                hard_max_budget=5000.0,
                hard_min_budget=None,
            ),
        ]
        proposal = {"Alice": 4000.0}

        objections, auto_names = check_hard_budget_limits(proposal, people, round_number=1)

        assert len(objections) == 0
        assert len(auto_names) == 0

    def test_below_hard_min_generates_objection(self):
        """A proposal below hard_min_budget should auto-generate an objection."""
        people = [
            PersonConstraint(
                name="Charlie",
                preferences="Wants to pay fair share",
                hard_max_budget=None,
                hard_min_budget=3000.0,
            ),
        ]
        proposal = {"Charlie": 2000.0}

        objections, auto_names = check_hard_budget_limits(proposal, people, round_number=2)

        assert len(objections) == 1
        assert objections[0]["person"] == "Charlie"
        assert "Below hard minimum budget" in objections[0]["reason"]
        assert "Charlie" in auto_names

    def test_no_hard_limits_no_objection(self):
        """A person with no hard limits should never be auto-objected."""
        people = [
            PersonConstraint(
                name="Dana",
                preferences="Flexible",
                hard_max_budget=None,
                hard_min_budget=None,
            ),
        ]
        proposal = {"Dana": 99999.0}

        objections, auto_names = check_hard_budget_limits(proposal, people, round_number=1)

        assert len(objections) == 0

    def test_auto_objected_person_skips_llm_call(self):
        """
        When a person is auto-objected by hard constraints, the person agent's
        LLM should NOT be called for that person.
        """
        people = [
            PersonConstraint(
                name="Alice",
                preferences="Budget-constrained",
                hard_max_budget=5000.0,
                hard_min_budget=None,
            ),
            PersonConstraint(
                name="Bob",
                preferences="No constraints",
                hard_max_budget=None,
                hard_min_budget=None,
            ),
        ]
        proposal = {"Alice": 7000.0, "Bob": 3000.0}

        # Get auto-objected names
        _, auto_objected_names = check_hard_budget_limits(proposal, people, round_number=1)

        # Now simulate collect_objections behavior: only call LLM for non-auto-objected
        with patch("agents.person.call_llm") as mock_llm:
            mock_llm.return_value = '{"decision": "ACCEPT", "reason": "Looks good"}'
            from agents.person import evaluate_proposal

            for person in people:
                if person["name"] not in auto_objected_names:
                    evaluate_proposal(
                        name=person["name"],
                        preferences=person["preferences"],
                        hard_max_budget=person.get("hard_max_budget"),
                        current_proposal=proposal,
                    )

            # LLM should have been called for Bob but NOT for Alice
            assert mock_llm.call_count == 1
            assert "Alice" in auto_objected_names
            assert "Bob" not in auto_objected_names


class TestValidateProposalSum:
    """Tests for proposal sum validation and re-normalization."""

    def test_correct_sum_unchanged(self):
        """A proposal that sums correctly should be returned unchanged."""
        proposal = {"A": 5000.0, "B": 5000.0}
        result = validate_proposal_sum(proposal, 10000.0)
        assert result == proposal

    def test_within_tolerance_unchanged(self):
        """A proposal within ±1 tolerance should be returned unchanged."""
        proposal = {"A": 5000.5, "B": 5000.0}
        result = validate_proposal_sum(proposal, 10000.0)
        assert result == proposal

    def test_renormalization(self):
        """A proposal with wrong sum should be re-normalized to the correct total."""
        proposal = {"A": 6000.0, "B": 6000.0}  # Sum = 12000
        result = validate_proposal_sum(proposal, 10000.0)
        assert abs(sum(result.values()) - 10000.0) <= 1.0
