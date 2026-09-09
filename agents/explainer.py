"""
FairSplit AI — Explainer Agent

Produces a plain-English summary of the negotiation outcome.
"""

import json
import logging

from agents.llm_client import call_llm
from graph.prompts import EXPLAINER_PROMPT

logger = logging.getLogger(__name__)


def explain_outcome(
    final_proposal: dict,
    status: str,
    proposal_history: list,
    objections: list,
) -> str:
    """
    Call the explainer LLM to produce a friendly summary of the negotiation result.
    Returns the plain-text explanation string.
    """
    prompt = EXPLAINER_PROMPT.format(
        final_proposal=json.dumps(final_proposal),
        status=status,
        proposal_history=json.dumps(proposal_history),
        objections=json.dumps(objections),
    )

    try:
        raw = call_llm(prompt)
        # The explainer output is plain text, not JSON
        return raw.strip()
    except Exception as e:
        logger.error("Explainer agent failed: %s", e)
        if status == "converged":
            return (
                "The group reached an agreement on the expense split. "
                "See the final amounts in the table above."
            )
        else:
            return (
                "The negotiation ended without full agreement after the maximum "
                "number of rounds. See the latest proposal and objections above."
            )
