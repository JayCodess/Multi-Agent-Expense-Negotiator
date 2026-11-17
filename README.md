# FairSplit AI

Multi-agent expense negotiation powered by **LangGraph**. AI agents advocate for each person while a mediator finds the fair split — rendered as a live chat transcript in Streamlit.

**[Try the live demo →](https://fairsplit-ai.streamlit.app/)**

## Architecture

Built on a **LangGraph StateGraph** with five nodes in a loop:

```
mediator_propose → validate_hard_constraints → collect_objections → check_convergence ─┐
       ↑                                                                                │
       └──────────────── negotiating ───────────────────────────────────────────────────┘
                                                                     converged / unresolved
                                                                                        ↓
                                                                                explain_final → END
```

| Node | Role |
|------|------|
| **mediator_propose** | LLM agent produces or revises a split proposal |
| **validate_hard_constraints** | Code-level check — auto-rejects allocations exceeding hard budget limits (no LLM) |
| **collect_objections** | Each person's LLM agent evaluates the proposal and decides ACCEPT or OBJECT |
| **check_convergence** | Routes: all accept → converged, max rounds hit → unresolved, else → next round |
| **explain_final** | LLM agent summarizes the outcome in plain English |

**Hard constraints** (budget limits) are enforced deterministically in code *before* LLM agents run. **Soft constraints** (preferences) are evaluated by each person's LLM agent. Proposals are auto-normalized if amounts don't sum to the total.

## Project Structure

```
├── app.py                        # Streamlit UI
├── graph/                        # LangGraph state, nodes, prompts, wiring
├── agents/                       # LLM client, mediator, person, explainer agents
├── validators/                   # Hard budget & sum validation
├── examples/                     # Pre-built demo scenario (Goa trip)
└── tests/                        # Unit tests (mocked LLM, no key needed)
```

## Setup

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                               # Add your API key
streamlit run app.py
```

Get a **free** Gemini API key (no credit card): [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

```env
GOOGLE_API_KEY=your-key-here
```

> Anthropic and OpenAI are also supported — set `LLM_PROVIDER=anthropic` or `LLM_PROVIDER=openai` in `.env`.

## Tests

```bash
pytest tests/ -v
```

## Tech Stack

LangGraph · Streamlit · Gemini 3.8 Flash (default) · Claude / GPT-4o (optional)
