"""
FairSplit AI — Streamlit App

Multi-agent expense negotiation UI with chat-style transcript rendering.
Run with: streamlit run app.py
"""

import html
import json
import os
import sys
import logging
import streamlit as st
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from graph.build_graph import build_negotiation_graph
from graph.state import NegotiationState

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FairSplit AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Global font & smoothing */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* Header styling */
.fairsplit-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 50%, #4a90d9 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(30, 58, 95, 0.3);
    position: relative;
    overflow: hidden;
}
.fairsplit-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.fairsplit-header h1 {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.02em;
}
.fairsplit-header p {
    color: rgba(255, 255, 255, 0.85);
    font-size: 1.05rem;
    margin: 0;
    font-weight: 400;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    border-radius: 14px !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.6rem !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
    border: 1px solid rgba(0, 0, 0, 0.06) !important;
    transition: box-shadow 0.2s ease;
}
[data-testid="stChatMessage"]:hover {
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1) !important;
}

/* Accept styling */
.accept-badge {
    display: inline-block;
    background: linear-gradient(135deg, #059669, #10b981);
    color: white;
    padding: 3px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.8rem;
    letter-spacing: 0.03em;
    box-shadow: 0 2px 6px rgba(5, 150, 105, 0.3);
}
.object-badge {
    display: inline-block;
    background: linear-gradient(135deg, #dc2626, #f87171);
    color: white;
    padding: 3px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.8rem;
    letter-spacing: 0.03em;
    box-shadow: 0 2px 6px rgba(220, 38, 38, 0.3);
}
.auto-tag {
    display: inline-block;
    background: rgba(245, 158, 11, 0.15);
    color: #b45309;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 6px;
}

/* Round divider */
.round-divider {
    text-align: center;
    padding: 0.8rem 0;
    color: #6b7280;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.round-divider span {
    background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
    color: white;
    padding: 4px 18px;
    border-radius: 20px;
    font-size: 0.78rem;
}

/* Containers */
[data-testid="stContainer"] {
    border-radius: 14px !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1e3a5f, #2d6a9f) !important;
    border: none !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(30, 58, 95, 0.04);
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid rgba(30, 58, 95, 0.08);
}

/* Inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    border-radius: 10px !important;
    border: 1.5px solid #d1d5db !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s ease !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #2d6a9f !important;
    box-shadow: 0 0 0 3px rgba(45, 106, 159, 0.12) !important;
}

/* Proposal table */
.proposal-table {
    border-collapse: collapse;
    width: 100%;
    border-radius: 10px;
    overflow: hidden;
    font-size: 0.92rem;
}
.proposal-table th {
    background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
    color: white;
    padding: 10px 16px;
    text-align: left;
    font-weight: 600;
}
.proposal-table td {
    padding: 8px 16px;
    border-bottom: 1px solid #e5e7eb;
}
.proposal-table tr:last-child td {
    border-bottom: none;
}
.proposal-table tr:nth-child(even) {
    background: rgba(30, 58, 95, 0.03);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="fairsplit-header">
    <h1>⚖️ FairSplit AI</h1>
    <p>Multi-agent expense negotiation — AI agents advocate for each person and a mediator finds the fair split.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Avatars
# ---------------------------------------------------------------------------
PERSON_AVATARS = ["🧍", "🧍‍♀️", "🧑‍🦱", "🧑‍🦰", "👩‍🦳", "🧑‍🦲", "👩‍🔬", "🧑‍💼"]
MEDIATOR_AVATAR = "🧑‍⚖️"
EXPLAINER_AVATAR = "📝"


def get_avatar(name: str, index: int) -> str:
    return PERSON_AVATARS[index % len(PERSON_AVATARS)]


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "people_count" not in st.session_state:
    st.session_state.people_count = 2
if "people_data" not in st.session_state:
    st.session_state.people_data = []
if "negotiation_result" not in st.session_state:
    st.session_state.negotiation_result = None
if "loaded_example" not in st.session_state:
    st.session_state.loaded_example = False

# ---------------------------------------------------------------------------
# Sidebar — Load Example
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📁 Quick Start")
    if st.button("🌴 Load Goa Trip Example", use_container_width=True):
        example_path = os.path.join(
            os.path.dirname(__file__), "examples", "goa_trip_example.json"
        )
        with open(example_path) as f:
            example = json.load(f)
        st.session_state.example_data = example
        st.session_state.loaded_example = True
        st.session_state.negotiation_result = None
        st.session_state.people_count = len(example.get("people", []))

        # Write example values directly into widget session state keys
        # so they appear in the form (Streamlit ignores `value=` if key exists)
        for i, person in enumerate(example.get("people", [])):
            st.session_state[f"person_name_{i}"] = person.get("name", "")
            st.session_state[f"person_prefs_{i}"] = person.get("preferences", "")
            hmax = person.get("hard_max_budget")
            hmin = person.get("hard_min_budget")
            st.session_state[f"person_hmax_{i}"] = float(hmax) if hmax is not None else 0.0
            st.session_state[f"person_hmin_{i}"] = float(hmin) if hmin is not None else 0.0
            st.session_state[f"person_hmax_none_{i}"] = hmax is None
            st.session_state[f"person_hmin_none_{i}"] = hmin is None

        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    max_rounds = st.number_input(
        "Max negotiation rounds",
        min_value=1,
        max_value=20,
        value=5,
        help="Maximum number of rounds before declaring 'unresolved'.",
    )

# ---------------------------------------------------------------------------
# Input Form
# ---------------------------------------------------------------------------
st.markdown("### 💰 Expense Details")

# Pre-fill from example if loaded
example = st.session_state.get("example_data", {}) if st.session_state.loaded_example else {}

col_amount, col_items = st.columns([1, 2])

with col_amount:
    total_amount = st.number_input(
        "Total Amount",
        min_value=0.0,
        value=float(example.get("total_amount", 0)),
        step=100.0,
        format="%.2f",
    )

with col_items:
    itemized_str = st.text_area(
        "Itemized Costs (optional, one per line: name=amount)",
        value="\n".join(
            f"{k}={v}" for k, v in example.get("itemized_costs", {}).items()
        )
        if example.get("itemized_costs")
        else "",
        height=100,
        help='Example:\nstay=15000\nfood=8000\ntravel=7000',
    )

# Parse itemized costs
itemized_costs = {}
if itemized_str.strip():
    for line in itemized_str.strip().split("\n"):
        line = line.strip()
        if "=" in line:
            key, val = line.split("=", 1)
            try:
                itemized_costs[key.strip()] = float(val.strip())
            except ValueError:
                pass

# ---------------------------------------------------------------------------
# People Input
# ---------------------------------------------------------------------------
st.markdown("### 👥 People")

example_people = example.get("people", [])

# Determine number of people
if example_people and st.session_state.loaded_example:
    num_people = len(example_people)
else:
    num_people = st.session_state.people_count

col_add, col_remove, _ = st.columns([1, 1, 4])
with col_add:
    if st.button("➕ Add Person"):
        st.session_state.people_count = num_people + 1
        st.session_state.loaded_example = False
        st.rerun()
with col_remove:
    if num_people > 2 and st.button("➖ Remove Last"):
        st.session_state.people_count = num_people - 1
        st.session_state.loaded_example = False
        st.rerun()

people = []
# Render people inputs in columns (2 per row)
for row_start in range(0, num_people, 2):
    cols = st.columns(min(2, num_people - row_start))
    for col_idx, col in enumerate(cols):
        i = row_start + col_idx
        if i >= num_people:
            break

        ex = example_people[i] if i < len(example_people) else {}

        with col:
            with st.container(border=True):
                st.markdown(f"**{get_avatar('', i)} Person {i + 1}**")
                name = st.text_input(
                    "Name",
                    value=ex.get("name", ""),
                    key=f"person_name_{i}",
                )
                prefs = st.text_area(
                    "Preferences / situation",
                    value=ex.get("preferences", ""),
                    key=f"person_prefs_{i}",
                    height=68,
                )
                c1, c2 = st.columns(2)
                with c1:
                    hard_max_val = ex.get("hard_max_budget")
                    no_max = st.checkbox(
                        "No max limit",
                        value=hard_max_val is None,
                        key=f"person_hmax_none_{i}",
                    )
                    hard_max = st.number_input(
                        "Hard Max Budget",
                        min_value=0.0,
                        value=float(hard_max_val) if hard_max_val is not None else 0.0,
                        step=100.0,
                        key=f"person_hmax_{i}",
                        disabled=no_max,
                    )
                with c2:
                    hard_min_val = ex.get("hard_min_budget")
                    no_min = st.checkbox(
                        "No min limit",
                        value=hard_min_val is None,
                        key=f"person_hmin_none_{i}",
                    )
                    hard_min = st.number_input(
                        "Hard Min Budget",
                        min_value=0.0,
                        value=float(hard_min_val) if hard_min_val is not None else 0.0,
                        step=100.0,
                        key=f"person_hmin_{i}",
                        disabled=no_min,
                    )

                if name.strip():
                    people.append({
                        "name": name.strip(),
                        "preferences": prefs.strip(),
                        "hard_max_budget": None if no_max else hard_max,
                        "hard_min_budget": None if no_min else hard_min,
                    })

# ---------------------------------------------------------------------------
# Run Negotiation
# ---------------------------------------------------------------------------
st.markdown("---")

can_run = total_amount > 0 and len(people) >= 2
if not can_run:
    st.info("Enter a total amount and at least 2 people to start the negotiation.")

run_col, _ = st.columns([1, 3])
with run_col:
    run_button = st.button(
        "🚀 Run Negotiation",
        disabled=not can_run,
        type="primary",
        use_container_width=True,
    )

if run_button and can_run:
    # Clear previous results
    st.session_state.negotiation_result = None

    # Initialize negotiation log
    with open("negotiation_log.json", "w") as f:
        json.dump([], f)

    initial_state = NegotiationState(
        total_amount=total_amount,
        itemized_costs=itemized_costs if itemized_costs else None,
        people=people,
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

    # Build and run graph
    graph = build_negotiation_graph()

    with st.spinner("🤖 Agents are negotiating..."):
        try:
            final_state = graph.invoke(initial_state)
            st.session_state.negotiation_result = final_state
        except Exception as e:
            st.error(f"Negotiation failed: {e}")
            logging.exception("Negotiation graph error")

# ---------------------------------------------------------------------------
# Render Results
# ---------------------------------------------------------------------------
result = st.session_state.negotiation_result
if result:
    st.markdown("---")
    st.markdown("### 💬 Negotiation Transcript")

    messages = result.get("messages", [])
    avatar_map = {}
    for i, p in enumerate(result.get("people", [])):
        avatar_map[p["name"]] = get_avatar(p["name"], i)

    current_round = None

    for msg in messages:
        msg_round = msg.get("round", 0)

        # Round divider
        if msg_round != current_round:
            current_round = msg_round
            st.markdown(
                f'<div class="round-divider"><span>Round {current_round}</span></div>',
                unsafe_allow_html=True,
            )

        if msg["type"] == "proposal":
            # Mediator proposal
            with st.chat_message("Mediator", avatar=MEDIATOR_AVATAR):
                st.markdown(f"**Mediator's Proposal — Round {msg_round}**")

                # Render proposal as HTML table
                proposal = msg["proposal"]
                table_html = '<table class="proposal-table"><tr><th>Person</th><th>Amount</th></tr>'
                for name, amount in proposal.items():
                    table_html += f"<tr><td>{html.escape(str(name))}</td><td>₹{amount:,.2f}</td></tr>"
                table_html += "</table>"
                st.markdown(table_html, unsafe_allow_html=True)

                if msg.get("reasoning"):
                    st.caption(f"💭 {msg['reasoning']}")

        elif msg["type"] == "decision":
            person_name = msg["role"]
            avatar = avatar_map.get(person_name, "👤")
            decision = msg["decision"]

            with st.chat_message(person_name, avatar=avatar):
                if decision == "ACCEPT":
                    badge = '<span class="accept-badge">✓ ACCEPT</span>'
                else:
                    badge = '<span class="object-badge">✗ OBJECT</span>'

                auto_tag = ""
                if msg.get("auto"):
                    auto_tag = '<span class="auto-tag">⚡ AUTO</span>'

                st.markdown(
                    f"**{html.escape(person_name)}** {badge}{auto_tag}",
                    unsafe_allow_html=True,
                )
                st.markdown(f"{msg.get('reason', '')}")

        elif msg["type"] == "explanation":
            with st.chat_message("Explainer", avatar=EXPLAINER_AVATAR):
                st.markdown(msg.get("text", ""))

    # ---------------------------------------------------------------------------
    # Final Summary
    # ---------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📊 Final Result")

    with st.container(border=True):
        status = result.get("status", "unknown")
        proposal = result.get("current_proposal", {})

        # Status metrics
        col_status, col_rounds, col_total = st.columns(3)
        with col_status:
            status_emoji = "✅" if status == "converged" else "⚠️"
            st.metric("Status", f"{status_emoji} {status.capitalize()}")
        with col_rounds:
            st.metric("Rounds Used", result.get("round_number", "?"))
        with col_total:
            st.metric("Total Amount", f"₹{result.get('total_amount', 0):,.2f}")

        st.markdown("")

        # Final split table
        df = pd.DataFrame(
            [{"Person": k, "Amount (₹)": f"{v:,.2f}"} for k, v in proposal.items()]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("")

        # Explainer summary
        explanation = result.get("final_explanation", "")
        if status == "converged":
            st.success(f"🎉 {explanation}")
        else:
            st.warning(f"⚠️ {explanation}")
