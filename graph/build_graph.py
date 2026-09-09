"""
FairSplit AI — Graph Builder

Wires the LangGraph StateGraph with conditional edges for the negotiation loop.
"""

from langgraph.graph import StateGraph, END

from graph.state import NegotiationState
from graph.nodes import (
    mediator_propose,
    validate_hard_constraints,
    collect_objections,
    check_convergence,
    explain_final,
)


def _route_after_convergence(state: NegotiationState) -> str:
    """
    Conditional edge: after check_convergence, route to either
    mediator_propose (loop) or explain_final (exit).
    """
    status = state.get("status", "negotiating")
    if status == "negotiating":
        return "mediator_propose"
    else:
        # "converged" or "unresolved" → explain and finish
        return "explain_final"


def build_negotiation_graph() -> StateGraph:
    """
    Build and compile the negotiation LangGraph.

    Flow:
      mediator_propose → validate_hard_constraints → collect_objections
        → check_convergence → (loop to mediator_propose OR explain_final → END)
    """
    graph = StateGraph(NegotiationState)

    # Add nodes
    graph.add_node("mediator_propose", mediator_propose)
    graph.add_node("validate_hard_constraints", validate_hard_constraints)
    graph.add_node("collect_objections", collect_objections)
    graph.add_node("check_convergence", check_convergence)
    graph.add_node("explain_final", explain_final)

    # Set entry point
    graph.set_entry_point("mediator_propose")

    # Linear edges
    graph.add_edge("mediator_propose", "validate_hard_constraints")
    graph.add_edge("validate_hard_constraints", "collect_objections")
    graph.add_edge("collect_objections", "check_convergence")

    # Conditional edge from check_convergence
    graph.add_conditional_edges(
        "check_convergence",
        _route_after_convergence,
        {
            "mediator_propose": "mediator_propose",
            "explain_final": "explain_final",
        },
    )

    # explain_final → END
    graph.add_edge("explain_final", END)

    return graph.compile()
