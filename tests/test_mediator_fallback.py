"""
Tests for mediator agent fallback behavior.

Verifies that mediator functions never raise, even when the LLM returns
invalid JSON or raises network/timeout errors — they fall back gracefully.
"""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.mediator import mediator_propose_initial, mediator_propose_revision


SAMPLE_PEOPLE = [
    {"name": "Alice", "preferences": "No special preferences", "hard_max_budget": None},
    {"name": "Bob", "preferences": "Budget-conscious", "hard_max_budget": 5000.0},
]


class TestMediatorFallbackOnInvalidJSON:
    """When call_llm returns unparseable text, the mediator should return
    a valid fallback dict instead of raising."""

    def test_initial_proposal_falls_back_to_equal_split(self):
        """mediator_propose_initial returns equal split on persistent invalid JSON."""
        with patch("agents.mediator.call_llm", return_value="not json at all"):
            result = mediator_propose_initial(
                total_amount=10000.0,
                itemized_costs=None,
                people=SAMPLE_PEOPLE,
                max_retries=1,
            )

        # Must return a valid dict, not raise
        assert isinstance(result, dict)
        assert "proposal" in result
        assert "reasoning" in result

        # Equal split: 10000 / 2 = 5000 each
        proposal = result["proposal"]
        assert proposal["Alice"] == 5000.0
        assert proposal["Bob"] == 5000.0
        assert "Fallback" in result["reasoning"]

    def test_revision_falls_back_to_current_proposal(self):
        """mediator_propose_revision keeps current_proposal on persistent invalid JSON."""
        current = {"Alice": 6000.0, "Bob": 4000.0}

        with patch("agents.mediator.call_llm", return_value="not json at all"):
            result = mediator_propose_revision(
                current_proposal=current,
                previous_reasoning="Prior reasoning.",
                objections=[{"person": "Bob", "reason": "Too much", "round": 1}],
                total_amount=10000.0,
                people=SAMPLE_PEOPLE,
                max_retries=1,
            )

        assert isinstance(result, dict)
        assert "proposal" in result
        assert "reasoning" in result
        assert result["proposal"] == current
        assert "Fallback" in result["reasoning"]


class TestMediatorFallbackOnNetworkError:
    """When call_llm raises a network/timeout error, the mediator should
    catch it and return the fallback instead of propagating the exception."""

    def test_initial_proposal_handles_timeout(self):
        """mediator_propose_initial returns equal split on TimeoutError."""
        with patch("agents.mediator.call_llm", side_effect=TimeoutError("LLM timed out")):
            result = mediator_propose_initial(
                total_amount=9000.0,
                itemized_costs=None,
                people=SAMPLE_PEOPLE,
                max_retries=1,
            )

        assert isinstance(result, dict)
        assert "proposal" in result
        assert result["proposal"]["Alice"] == 4500.0
        assert result["proposal"]["Bob"] == 4500.0
        assert "Fallback" in result["reasoning"]

    def test_revision_handles_timeout(self):
        """mediator_propose_revision keeps current_proposal on TimeoutError."""
        current = {"Alice": 3000.0, "Bob": 7000.0}

        with patch("agents.mediator.call_llm", side_effect=TimeoutError("LLM timed out")):
            result = mediator_propose_revision(
                current_proposal=current,
                previous_reasoning="Earlier reasoning.",
                objections=[],
                total_amount=10000.0,
                people=SAMPLE_PEOPLE,
                max_retries=1,
            )

        assert isinstance(result, dict)
        assert result["proposal"] == current
        assert "Fallback" in result["reasoning"]

    def test_revision_handles_connection_error(self):
        """mediator_propose_revision handles ConnectionError gracefully."""
        current = {"Alice": 5000.0, "Bob": 5000.0}

        with patch("agents.mediator.call_llm", side_effect=ConnectionError("No network")):
            result = mediator_propose_revision(
                current_proposal=current,
                previous_reasoning="",
                objections=[],
                total_amount=10000.0,
                people=SAMPLE_PEOPLE,
                max_retries=0,
            )

        assert isinstance(result, dict)
        assert result["proposal"] == current
        assert "Fallback" in result["reasoning"]
