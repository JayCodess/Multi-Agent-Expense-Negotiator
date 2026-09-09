"""
FairSplit AI — Mediator Agent

Produces initial and revised expense-split proposals via LLM.
"""

import json
import logging
import re
from typing import Dict, Any

from agents.llm_client import call_llm
from graph.prompts import MEDIATOR_INITIAL_PROMPT, MEDIATOR_REVISION_PROMPT

logger = logging.getLogger(__name__)


def _parse_json_response(raw: str) -> Dict[str, Any]:
    """
    Parse an LLM response that should be JSON. Highly defensive.
    """
    cleaned = raw.strip()
    
    # Attempt 1: Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
        
    # Attempt 2: Extract from markdown fences
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
            
    # Attempt 3: Find outermost curly braces
    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
            
    # Final attempt: let it crash with the original error
    return json.loads(cleaned)


def _format_people_block(people: list) -> str:
    lines = []
    for p in people:
        line = f"- {p['name']}: {p['preferences']}"
        if p.get("hard_max_budget") is not None:
            line += f" (hard max: {p['hard_max_budget']})"
        if p.get("hard_min_budget") is not None:
            line += f" (hard min: {p['hard_min_budget']})"
        lines.append(line)
    return "\n".join(lines)


def _format_objections_block(objections: list) -> str:
    lines = []
    for obj in objections:
        lines.append(f"- {obj['person']}: {obj['reason']} (round {obj['round']})")
    return "\n".join(lines) if lines else "(none)"


def mediator_propose_initial(
    total_amount: float,
    itemized_costs: dict | None,
    people: list,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Generate the initial proposal. Returns {"proposal": {...}, "reasoning": "..."}.
    Retries up to max_retries on malformed JSON.

    Never raises — falls back to an equal split on persistent failure.
    """
    raw = None  # Ensure raw is always defined for safe logging

    try:
        prompt = MEDIATOR_INITIAL_PROMPT.format(
            total_amount=total_amount,
            itemized_costs=json.dumps(itemized_costs) if itemized_costs else "N/A",
            people_block=_format_people_block(people),
        )

        for attempt in range(max_retries + 1):
            try:
                raw = call_llm(prompt)
                result = _parse_json_response(raw)
                if "proposal" not in result:
                    raise KeyError("Missing 'proposal' key in mediator response")
                # Ensure all people are in the proposal
                for p in people:
                    if p["name"] not in result["proposal"]:
                        raise KeyError(f"Missing person '{p['name']}' in proposal")
                return result
            except Exception as e:
                logger.warning(
                    "Mediator initial proposal attempt %d failed: %s\nRaw output was: %r",
                    attempt + 1, e, raw
                )
                if attempt == max_retries:
                    break
    except Exception as e:
        logger.warning(
            "Mediator initial proposal outer error: %s\nRaw output was: %r", e, raw
        )

    # Fallback: equal split across all people
    equal_amount = total_amount / len(people) if people else 0
    fallback_proposal = {p["name"]: equal_amount for p in people}
    logger.error(
        "Mediator failed to produce a valid initial proposal after %d retries — "
        "falling back to equal split: %s",
        max_retries + 1, fallback_proposal,
    )
    return {
        "proposal": fallback_proposal,
        "reasoning": "Fallback: could not get a valid response from the mediator agent — defaulting to an equal split.",
    }


def mediator_propose_revision(
    current_proposal: dict,
    previous_reasoning: str,
    objections: list,
    total_amount: float,
    people: list,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Revise the proposal based on objections. Returns {"proposal": {...}, "reasoning": "..."}.

    Never raises — falls back to keeping the current proposal on persistent failure.
    """
    raw = None  # Ensure raw is always defined for safe logging

    try:
        prompt = MEDIATOR_REVISION_PROMPT.format(
            current_proposal=json.dumps(current_proposal),
            previous_reasoning=previous_reasoning or "No prior reasoning.",
            objections_block=_format_objections_block(objections),
            total_amount=total_amount,
        )

        for attempt in range(max_retries + 1):
            try:
                raw = call_llm(prompt)
                result = _parse_json_response(raw)
                if "proposal" not in result:
                    raise KeyError("Missing 'proposal' key in mediator revision")
                for p in people:
                    if p["name"] not in result["proposal"]:
                        raise KeyError(f"Missing person '{p['name']}' in revision proposal")
                return result
            except Exception as e:
                logger.warning(
                    "Mediator revision attempt %d failed: %s\nRaw output was: %r",
                    attempt + 1, e, raw
                )
                if attempt == max_retries:
                    break
    except Exception as e:
        logger.warning(
            "Mediator revision outer error: %s\nRaw output was: %r", e, raw
        )

    # Fallback: keep the current proposal unchanged
    logger.error(
        "Mediator failed to produce a valid revised proposal after %d retries — "
        "keeping the previous proposal: %s",
        max_retries + 1, current_proposal,
    )
    return {
        "proposal": dict(current_proposal),
        "reasoning": "Fallback: could not get a valid revision from the mediator agent — keeping the previous proposal.",
    }
