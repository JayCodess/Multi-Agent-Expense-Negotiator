"""
FairSplit AI — Hard Constraint Validators

Pure code-level validation: no LLM calls. Checks budget limits and proposal sums.
"""

import logging
from typing import Dict, List, Tuple

from graph.state import PersonConstraint, Objection

logger = logging.getLogger(__name__)


def validate_proposal_sum(
    proposal: Dict[str, float], total_amount: float, tolerance: float = 1.0
) -> Dict[str, float]:
    """
    Ensure proposal values sum to total_amount within ±tolerance.
    If they don't, re-normalize proportionally and log a warning.
    Returns the (possibly corrected) proposal.
    """
    current_sum = sum(proposal.values())
    if abs(current_sum - total_amount) <= tolerance:
        return proposal

    logger.warning(
        "Proposal sum %.2f != total %.2f — re-normalizing proportionally.",
        current_sum,
        total_amount,
    )
    if current_sum == 0:
        # Edge case: everything is zero — split equally
        equal_share = total_amount / len(proposal)
        return {name: round(equal_share, 2) for name in proposal}

    factor = total_amount / current_sum
    normalized = {name: round(amt * factor, 2) for name, amt in proposal.items()}

    # Fix any rounding drift on the last person
    drift = total_amount - sum(normalized.values())
    if drift != 0:
        last_key = list(normalized.keys())[-1]
        normalized[last_key] = round(normalized[last_key] + drift, 2)

    return normalized


def check_hard_budget_limits(
    proposal: Dict[str, float],
    people: List[PersonConstraint],
    round_number: int,
) -> Tuple[List[Objection], List[str]]:
    """
    For each person, check if the proposed amount exceeds hard_max_budget or is
    below hard_min_budget. Returns a list of auto-generated Objections and a list
    of person names that were auto-objected (so they can be skipped in soft eval).
    """
    auto_objections: List[Objection] = []
    auto_objected_names: List[str] = []

    for person in people:
        name = person["name"]
        amount = proposal.get(name, 0.0)
        hard_max = person.get("hard_max_budget")
        hard_min = person.get("hard_min_budget")

        if hard_max is not None and amount > hard_max:
            auto_objections.append(
                Objection(
                    person=name,
                    reason=f"Exceeds hard budget limit (proposed {amount:.2f}, max {hard_max:.2f})",
                    round=round_number,
                )
            )
            auto_objected_names.append(name)
        elif hard_min is not None and amount < hard_min:
            auto_objections.append(
                Objection(
                    person=name,
                    reason=f"Below hard minimum budget (proposed {amount:.2f}, min {hard_min:.2f})",
                    round=round_number,
                )
            )
            auto_objected_names.append(name)

    return auto_objections, auto_objected_names
