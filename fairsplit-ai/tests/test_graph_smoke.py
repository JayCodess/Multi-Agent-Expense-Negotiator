"""
Smoke test for the full LangGraph negotiation graph.

Runs the graph end-to-end with the example JSON and mocked LLM responses.
Asserts the graph terminates with a valid final state within max_rounds.
"""

import sys
import os
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.build_graph import build_negotiation_graph
from graph.state import NegotiationState


def _make_prompt_based_mock():
    """
    Return a mock function that inspects the prompt content to determine the
    correct canned response. This avoids ordering issues with a sequential iterator.

    Round 1: Mediator proposes equal split (10000 each) → Aisha and Meera auto-objected.
             Raj's person agent evaluates → ACCEPT.
    Round 2: Mediator revises to respect budgets → all 3 accept.
    Explainer: final summary.
    """
    call_count = {"mediator": 0}

    def mock_call_llm(prompt: str) -> str:
        prompt_lower = prompt.lower()

        # Mediator prompts
        if "neutral mediator splitting" in prompt_lower:
            call_count["mediator"] += 1
            return json.dumps({
                "proposal": {"Aisha": 10000, "Raj": 10000, "Meera": 10000},
                "reasoning": "Starting with an equal split of 30000 among 3 people."
            })
        elif "you previously proposed" in prompt_lower or "revise the split" in prompt_lower:
            call_count["mediator"] += 1
            return json.dumps({
                "proposal": {"Aisha": 8000, "Raj": 16000, "Meera": 6000},
                "reasoning": "Adjusted to respect Aisha's 8000 max and Meera's 6000 max."
            })

        # Person agent prompts
        elif "you represent" in prompt_lower:
            return json.dumps({
                "decision": "ACCEPT",
                "reason": "This amount seems fair given my situation."
            })

        # Explainer prompt
        elif "final split" in prompt_lower:
            return (
                "The group reached a fair agreement! Aisha pays 8,000, "
                "Raj pays 16,000, and Meera pays 6,000. The split accounts "
                "for each person's usage and budget constraints."
            )

        # Fallback
        return json.dumps({"decision": "ACCEPT", "reason": "OK"})

    return mock_call_llm


class TestGraphSmoke:
    """End-to-end smoke test with mocked LLM."""

    def test_full_negotiation_terminates(self):
        """
        Run the full graph with canned LLM responses.
        Assert it terminates within max_rounds with a valid final state.
        """
        # Load example data
        example_path = os.path.join(
            os.path.dirname(__file__), "..", "examples", "goa_trip_example.json"
        )
        with open(example_path) as f:
            example = json.load(f)

        mock_fn = _make_prompt_based_mock()

        with patch("agents.mediator.call_llm", side_effect=mock_fn), \
             patch("agents.person.call_llm", side_effect=mock_fn), \
             patch("agents.explainer.call_llm", side_effect=mock_fn):

            graph = build_negotiation_graph()

            initial_state = NegotiationState(
                total_amount=example["total_amount"],
                itemized_costs=example.get("itemized_costs"),
                people=example["people"],
                current_proposal={},
                proposal_history=[],
                objections=[],
                round_number=1,
                max_rounds=example.get("max_rounds", 5),
                status="negotiating",
                final_explanation=None,
                mediator_reasoning=None,
                messages=[],
            )

            final_state = graph.invoke(initial_state)

        # Verify termination
        assert final_state["status"] in ("converged", "unresolved")
        assert final_state["round_number"] <= final_state["max_rounds"]

        # Verify proposal structure
        proposal = final_state["current_proposal"]
        assert len(proposal) == len(example["people"])
        for person in example["people"]:
            assert person["name"] in proposal

        # Verify sum is approximately correct
        assert abs(sum(proposal.values()) - example["total_amount"]) <= 1.0

        # Verify final explanation exists
        assert final_state["final_explanation"] is not None
        assert len(final_state["final_explanation"]) > 0

        # Verify messages were generated
        assert len(final_state["messages"]) > 0

    def test_negotiation_respects_max_rounds(self):
        """
        With a mediator that always produces budget-violating proposals,
        the graph should terminate at max_rounds as 'unresolved'.
        """
        example_path = os.path.join(
            os.path.dirname(__file__), "..", "examples", "goa_trip_example.json"
        )
        with open(example_path) as f:
            example = json.load(f)

        # Always propose equal split (violates budgets), so it never converges
        def always_equal_proposal(prompt: str) -> str:
            if "neutral mediator splitting" in prompt.lower() or "you previously proposed" in prompt.lower():
                return json.dumps({
                    "proposal": {"Aisha": 10000, "Raj": 10000, "Meera": 10000},
                    "reasoning": "Equal split."
                })
            elif "you represent" in prompt.lower():
                return json.dumps({
                    "decision": "OBJECT",
                    "reason": "This is too much for me."
                })
            else:
                return "The negotiation ended without agreement after maximum rounds."

        max_rounds = 3  # Use small number for speed

        with patch("agents.mediator.call_llm", side_effect=always_equal_proposal), \
             patch("agents.person.call_llm", side_effect=always_equal_proposal), \
             patch("agents.explainer.call_llm", side_effect=always_equal_proposal):

            graph = build_negotiation_graph()

            initial_state = NegotiationState(
                total_amount=example["total_amount"],
                itemized_costs=example.get("itemized_costs"),
                people=example["people"],
                current_proposal={},
                proposal_history=[],
                objections=[],
                round_number=1,
                max_rounds=max_rounds,
                status="negotiating",
                final_explanation=None,
                mediator_reasoning=None,
                messages=[],
            )

            final_state = graph.invoke(initial_state)

        assert final_state["status"] == "unresolved"
        assert final_state["round_number"] <= max_rounds
