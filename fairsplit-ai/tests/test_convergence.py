"""
Tests for the check_convergence node logic.

Verifies:
- Zero objections this round → status = "converged"
- Objections present and round_number == max_rounds → status = "unresolved"
- Objections present and round_number < max_rounds → status = "negotiating", round incremented
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.nodes import check_convergence
from graph.state import NegotiationState, Objection


class TestCheckConvergence:
    """Tests for convergence routing logic."""

    def _make_state(self, objections, round_number, max_rounds) -> NegotiationState:
        return NegotiationState(
            total_amount=10000.0,
            itemized_costs=None,
            people=[],
            current_proposal={"A": 5000.0, "B": 5000.0},
            proposal_history=[{"A": 5000.0, "B": 5000.0}],
            objections=objections,
            round_number=round_number,
            max_rounds=max_rounds,
            status="negotiating",
            final_explanation=None,
            mediator_reasoning=None,
            messages=[],
        )

    def test_no_objections_converges(self):
        """With zero objections in the current round, status should be 'converged'."""
        state = self._make_state(objections=[], round_number=1, max_rounds=5)
        result = check_convergence(state)
        assert result["status"] == "converged"

    def test_objections_at_max_rounds_unresolved(self):
        """With objections and round == max_rounds, status should be 'unresolved'."""
        objections = [
            Objection(person="A", reason="Too much", round=5),
        ]
        state = self._make_state(objections=objections, round_number=5, max_rounds=5)
        result = check_convergence(state)
        assert result["status"] == "unresolved"

    def test_objections_before_max_continues(self):
        """With objections and round < max_rounds, should continue negotiating."""
        objections = [
            Objection(person="A", reason="Too much", round=2),
        ]
        state = self._make_state(objections=objections, round_number=2, max_rounds=5)
        result = check_convergence(state)
        assert result["status"] == "negotiating"
        assert result["round_number"] == 3

    def test_old_objections_ignored(self):
        """Objections from previous rounds should not prevent convergence."""
        old_objections = [
            Objection(person="A", reason="Was too much", round=1),
        ]
        # Current round is 2, and no round-2 objections exist
        state = self._make_state(
            objections=old_objections, round_number=2, max_rounds=5
        )
        result = check_convergence(state)
        assert result["status"] == "converged"
