"""
FairSplit AI — Graph Nodes

Each function is a LangGraph node that receives and returns NegotiationState.
"""

import json
import logging
from typing import Dict, Any

from graph.state import NegotiationState, Objection
from agents.mediator import mediator_propose_initial, mediator_propose_revision
from agents.person import evaluate_proposal
from agents.explainer import explain_outcome
from validators.hard_constraints import validate_proposal_sum, check_hard_budget_limits

logger = logging.getLogger(__name__)


def _log_state(state: dict, label: str):
    """Write state snapshot to negotiation_log.json (append-friendly)."""
    try:
        try:
            with open("negotiation_log.json", "r") as f:
                log_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            log_data = []

        log_data.append({"label": label, "round": state.get("round_number"), "state_snapshot": {
            "current_proposal": state.get("current_proposal"),
            "objections": state.get("objections"),
            "status": state.get("status"),
            "round_number": state.get("round_number"),
        }})

        with open("negotiation_log.json", "w") as f:
            json.dump(log_data, f, indent=2)
    except Exception as e:
        logger.warning("Failed to write negotiation log: %s", e)


def mediator_propose(state: NegotiationState) -> Dict[str, Any]:
    """
    Node 1: Mediator proposes or revises a split.
    On round 1, uses initial prompt. On later rounds, uses revision prompt
    incorporating accumulated objections.
    """
    round_num = state["round_number"]
    messages = list(state.get("messages", []))

    if round_num == 1:
        result = mediator_propose_initial(
            total_amount=state["total_amount"],
            itemized_costs=state.get("itemized_costs"),
            people=state["people"],
        )
    else:
        # Gather objections from the previous round only
        prev_round = round_num - 1
        recent_objections = [
            o for o in state.get("objections", []) if o["round"] == prev_round
        ]
        result = mediator_propose_revision(
            current_proposal=state["current_proposal"],
            previous_reasoning=state.get("mediator_reasoning", ""),
            objections=recent_objections,
            total_amount=state["total_amount"],
            people=state["people"],
        )

    proposal = result["proposal"]
    reasoning = result.get("reasoning", "")

    # Validate and normalize the proposal sum
    proposal = validate_proposal_sum(proposal, state["total_amount"])

    # Track history
    history = list(state.get("proposal_history", []))
    history.append(proposal)

    # Add mediator chat message
    messages.append({
        "role": "mediator",
        "round": round_num,
        "type": "proposal",
        "proposal": proposal,
        "reasoning": reasoning,
    })

    updated = {
        "current_proposal": proposal,
        "proposal_history": history,
        "mediator_reasoning": reasoning,
        "messages": messages,
        # Clear objections for this new round
        "objections": [o for o in state.get("objections", []) if o["round"] < round_num],
    }

    _log_state({**state, **updated}, f"mediator_propose_round_{round_num}")
    return updated


def validate_hard_constraints(state: NegotiationState) -> Dict[str, Any]:
    """
    Node 2: CODE-LEVEL check — auto-generate Objections for hard budget violations.
    No LLM call. Must run before soft-preference evaluation.
    """
    auto_objections, auto_objected_names = check_hard_budget_limits(
        proposal=state["current_proposal"],
        people=state["people"],
        round_number=state["round_number"],
    )

    objections = list(state.get("objections", []))
    messages = list(state.get("messages", []))

    for obj in auto_objections:
        objections.append(obj)
        messages.append({
            "role": obj["person"],
            "round": state["round_number"],
            "type": "decision",
            "decision": "OBJECT",
            "reason": obj["reason"],
            "auto": True,
        })
        logger.info(
            "Auto-objection for %s: %s", obj["person"], obj["reason"]
        )

    updated = {
        "objections": objections,
        "messages": messages,
        "_auto_objected_names": auto_objected_names,  # Transient, used by next node
    }

    _log_state({**state, **updated}, f"validate_hard_round_{state['round_number']}")
    return updated


def collect_objections(state: NegotiationState) -> Dict[str, Any]:
    """
    Node 3: For each person NOT already auto-objected, call their person agent
    to ACCEPT or OBJECT. Append any objections to state.
    """
    auto_objected = state.get("_auto_objected_names", [])
    objections = list(state.get("objections", []))
    messages = list(state.get("messages", []))
    round_num = state["round_number"]

    for person in state["people"]:
        name = person["name"]
        if name in auto_objected:
            continue

        result = evaluate_proposal(
            name=name,
            preferences=person["preferences"],
            hard_max_budget=person.get("hard_max_budget"),
            current_proposal=state["current_proposal"],
        )

        decision = result["decision"]
        reason = result["reason"]

        messages.append({
            "role": name,
            "round": round_num,
            "type": "decision",
            "decision": decision,
            "reason": reason,
            "auto": False,
        })

        if decision == "OBJECT":
            objections.append(
                Objection(person=name, reason=reason, round=round_num)
            )

    updated = {
        "objections": objections,
        "messages": messages,
    }

    _log_state({**state, **updated}, f"collect_objections_round_{round_num}")
    return updated


def check_convergence(state: NegotiationState) -> Dict[str, Any]:
    """
    Node 4: Routing logic (no LLM call).
    - No objections this round → "converged"
    - round_number >= max_rounds → "unresolved"
    - Else → increment round, "negotiating", loop back
    """
    round_num = state["round_number"]
    max_rounds = state["max_rounds"]
    current_round_objections = [
        o for o in state.get("objections", []) if o["round"] == round_num
    ]

    if len(current_round_objections) == 0:
        updated = {"status": "converged"}
    elif round_num >= max_rounds:
        updated = {"status": "unresolved"}
    else:
        updated = {
            "round_number": round_num + 1,
            "status": "negotiating",
        }

    _log_state({**state, **updated}, f"check_convergence_round_{round_num}")
    return updated


def explain_final(state: NegotiationState) -> Dict[str, Any]:
    """
    Node 5: Call explainer agent to summarize the outcome in plain English.
    """
    explanation = explain_outcome(
        final_proposal=state["current_proposal"],
        status=state["status"],
        proposal_history=state.get("proposal_history", []),
        objections=state.get("objections", []),
    )

    messages = list(state.get("messages", []))
    messages.append({
        "role": "explainer",
        "round": state["round_number"],
        "type": "explanation",
        "text": explanation,
    })

    updated = {
        "final_explanation": explanation,
        "messages": messages,
    }

    _log_state({**state, **updated}, "explain_final")
    return updated
