"""
FairSplit AI — Person Agent

Represents one person in the negotiation. Evaluates proposals and decides
ACCEPT or OBJECT via LLM.
"""

import json
import logging
import re
from typing import Dict, Any

from agents.llm_client import call_llm
from graph.prompts import PERSON_EVALUATE_PROMPT

logger = logging.getLogger(__name__)


def _parse_json_response(raw: str) -> Dict[str, Any]:
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
            
    return json.loads(cleaned)


def evaluate_proposal(
    name: str,
    preferences: str,
    hard_max_budget: float | None,
    current_proposal: dict,
    max_retries: int = 2,
) -> Dict[str, str]:
    """
    Call the person agent LLM to evaluate the current proposal.
    Returns {"decision": "ACCEPT"|"OBJECT", "reason": "..."}.

    On persistent failure, falls back to OBJECT with a system error reason.
    """
    your_amount = current_proposal.get(name, 0)
    prompt = PERSON_EVALUATE_PROMPT.format(
        name=name,
        preferences=preferences,
        hard_max_budget=hard_max_budget if hard_max_budget is not None else "None",
        your_amount=your_amount,
        current_proposal=json.dumps(current_proposal),
    )

    for attempt in range(max_retries + 1):
        try:
            raw = call_llm(prompt)
            result = _parse_json_response(raw)
            decision = result.get("decision", "").upper()
            if decision not in ("ACCEPT", "OBJECT"):
                raise ValueError(f"Invalid decision: {result.get('decision')}")
            return {
                "decision": decision,
                "reason": result.get("reason", "No reason given."),
            }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(
                "Person agent '%s' attempt %d failed: %s", name, attempt + 1, e
            )
            if attempt == max_retries:
                logger.error(
                    "Person agent '%s' failed after %d retries — defaulting to OBJECT.",
                    name,
                    max_retries + 1,
                )
                return {
                    "decision": "OBJECT",
                    "reason": "Agent response error — could not parse LLM output.",
                }

    # Should not reach here
    return {"decision": "OBJECT", "reason": "Agent response error."}
