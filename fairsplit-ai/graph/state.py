"""
FairSplit AI — State Schema

Defines the TypedDict schemas used throughout the LangGraph negotiation graph.
"""

from typing import TypedDict, List, Dict, Optional


class PersonConstraint(TypedDict):
    name: str
    preferences: str
    hard_max_budget: Optional[float]
    hard_min_budget: Optional[float]


class Objection(TypedDict):
    person: str
    reason: str
    round: int


class NegotiationState(TypedDict):
    total_amount: float
    itemized_costs: Optional[Dict[str, float]]
    people: List[PersonConstraint]
    current_proposal: Dict[str, float]
    proposal_history: List[Dict[str, float]]
    objections: List[Objection]
    round_number: int
    max_rounds: int
    status: str  # "negotiating" | "converged" | "unresolved"
    final_explanation: Optional[str]
    mediator_reasoning: Optional[str]
    messages: List[dict]  # Chat messages for the UI to render
