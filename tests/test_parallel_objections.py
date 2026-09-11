"""
Tests for parallel person-agent evaluation in collect_objections.

Verifies:
- Person-agent LLM calls run concurrently (wall time < sequential time).
- Resulting messages preserve the original state["people"] order.
"""

import sys
import os
import json
import time
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.nodes import collect_objections
from graph.state import NegotiationState, PersonConstraint


def _make_state(people, proposal, auto_objected=None):
    """Build a minimal NegotiationState for collect_objections."""
    return NegotiationState(
        total_amount=sum(proposal.values()),
        itemized_costs=None,
        people=people,
        current_proposal=proposal,
        proposal_history=[proposal],
        objections=[],
        round_number=1,
        max_rounds=5,
        status="negotiating",
        final_explanation=None,
        mediator_reasoning=None,
        messages=[],
        _auto_objected_names=auto_objected or [],
    )


class TestParallelObjections:
    """Tests for parallelized person-agent evaluation."""

    def test_parallel_speedup(self):
        """
        With 3 people and a 0.2s sleep per LLM call, wall time should be
        < 0.5s (parallel) rather than >= 0.6s (sequential).
        """
        people = [
            PersonConstraint(name="Alice", preferences="A", hard_max_budget=None, hard_min_budget=None),
            PersonConstraint(name="Bob", preferences="B", hard_max_budget=None, hard_min_budget=None),
            PersonConstraint(name="Charlie", preferences="C", hard_max_budget=None, hard_min_budget=None),
        ]
        proposal = {"Alice": 3333.0, "Bob": 3333.0, "Charlie": 3334.0}
        state = _make_state(people, proposal)

        def slow_llm(prompt: str) -> str:
            time.sleep(0.2)
            return json.dumps({"decision": "ACCEPT", "reason": "Looks fair."})

        with patch("agents.person.call_llm", side_effect=slow_llm):
            start = time.monotonic()
            collect_objections(state)
            elapsed = time.monotonic() - start

        # Parallel: ~0.2s.  Sequential would be ~0.6s.
        assert elapsed < 0.5, f"Expected < 0.5s (parallel), got {elapsed:.2f}s"

    def test_message_ordering_preserved(self):
        """
        Messages should appear in the original state['people'] order,
        not in whichever thread finishes first.
        """
        people = [
            PersonConstraint(name="Alice", preferences="A", hard_max_budget=None, hard_min_budget=None),
            PersonConstraint(name="Bob", preferences="B", hard_max_budget=None, hard_min_budget=None),
            PersonConstraint(name="Charlie", preferences="C", hard_max_budget=None, hard_min_budget=None),
        ]
        proposal = {"Alice": 3333.0, "Bob": 3333.0, "Charlie": 3334.0}
        state = _make_state(people, proposal)

        # Give each person a different sleep to scramble thread finishing order
        def variable_delay_llm(prompt: str) -> str:
            if "Alice" in prompt:
                time.sleep(0.15)
            elif "Bob" in prompt:
                time.sleep(0.05)
            elif "Charlie" in prompt:
                time.sleep(0.10)
            return json.dumps({"decision": "ACCEPT", "reason": "OK"})

        with patch("agents.person.call_llm", side_effect=variable_delay_llm):
            result = collect_objections(state)

        message_names = [m["role"] for m in result["messages"] if m["type"] == "decision"]
        assert message_names == ["Alice", "Bob", "Charlie"], (
            f"Expected ['Alice', 'Bob', 'Charlie'], got {message_names}"
        )

    def test_auto_objected_skipped(self):
        """
        People in _auto_objected_names should not trigger an LLM call,
        even with parallel execution.
        """
        people = [
            PersonConstraint(name="Alice", preferences="A", hard_max_budget=5000.0, hard_min_budget=None),
            PersonConstraint(name="Bob", preferences="B", hard_max_budget=None, hard_min_budget=None),
        ]
        proposal = {"Alice": 7000.0, "Bob": 3000.0}
        state = _make_state(people, proposal, auto_objected=["Alice"])

        call_count = {"n": 0}

        def counting_llm(prompt: str) -> str:
            call_count["n"] += 1
            return json.dumps({"decision": "ACCEPT", "reason": "Fine."})

        with patch("agents.person.call_llm", side_effect=counting_llm):
            result = collect_objections(state)

        # Only Bob should have been evaluated
        assert call_count["n"] == 1
        message_names = [m["role"] for m in result["messages"]]
        assert "Alice" not in message_names
        assert "Bob" in message_names
