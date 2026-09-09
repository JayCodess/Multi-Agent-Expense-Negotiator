# FairSplit AI

**Multi-agent expense negotiation powered by LangGraph.** Multiple AI agents — one per person — negotiate a fair split of a shared expense with a mediator agent, converging over multiple rounds. Ships with a polished Streamlit UI showing the negotiation as a live chat transcript.

---

## 🚀 Setup

### 1. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API key (FREE)

```bash
cp .env.example .env
```

Get a **free** Google Gemini API key (no credit card needed):
1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Click "Create API key"
3. Paste it into your `.env` file:

```env
GOOGLE_API_KEY=your-free-google-key-here
```

That's it! Gemini is the default provider and it's **completely free**.

> **Optional paid providers:** You can also use Anthropic (`ANTHROPIC_API_KEY`) or OpenAI (`OPENAI_API_KEY`). Set `LLM_PROVIDER=anthropic` or `LLM_PROVIDER=openai` in `.env` to switch. If `LLM_PROVIDER` is not set, the app auto-detects which key is available.

### 4. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.


---

## 🏗️ Architecture

### Negotiation Loop

FairSplit AI uses a **LangGraph StateGraph** with 5 nodes connected in a loop:

```
mediator_propose → validate_hard_constraints → collect_objections → check_convergence
      ↑                                                                    |
      |______________ (if still negotiating) ______________________________|
                                                                           |
                                                              (if converged/unresolved)
                                                                           ↓
                                                                    explain_final → END
```

1. **mediator_propose** — An LLM-powered mediator agent produces (or revises) a split proposal.
2. **validate_hard_constraints** — A *code-level* check (no LLM) auto-rejects any allocation that exceeds a person's declared hard budget limits. This runs *before* soft preference evaluation.
3. **collect_objections** — For each person *not* already auto-rejected, their personal LLM agent evaluates the proposal and decides to ACCEPT or OBJECT with a reason.
4. **check_convergence** — Pure routing logic:
   - No objections → `"converged"` → exit
   - Objections + max rounds reached → `"unresolved"` → exit
   - Objections + rounds remaining → increment round → loop back
5. **explain_final** — An explainer LLM agent summarizes the outcome in plain English.

### Hard vs Soft Constraints

- **Hard constraints** (budget limits) are enforced in code *before* LLM agents run. This guarantees budget violations are caught deterministically, not left to LLM judgment.
- **Soft constraints** (preferences like "I skipped dinners") are evaluated by each person's LLM agent, which weighs them against the proposal and decides whether to object.

### Proposal Validation

Every proposal is validated to ensure amounts sum to the total (±₹1 tolerance). If the LLM produces a proposal that doesn't sum correctly, it's automatically re-normalized proportionally.

---

## 📁 Project Structure

```
fairsplit-ai/
├── app.py                       # Streamlit UI
├── graph/
│   ├── state.py                 # TypedDict state schema
│   ├── prompts.py               # LLM prompt templates
│   ├── nodes.py                 # Graph node functions
│   └── build_graph.py           # LangGraph wiring
├── agents/
│   ├── llm_client.py            # Provider-agnostic LLM wrapper
│   ├── mediator.py              # Mediator agent
│   ├── person.py                # Person agent
│   └── explainer.py             # Explainer agent
├── validators/
│   └── hard_constraints.py      # Budget limit & sum validation
├── examples/
│   └── goa_trip_example.json    # Pre-built demo scenario
└── tests/
    ├── test_hard_constraints.py  # Budget validation tests
    ├── test_convergence.py       # Convergence logic tests
    └── test_graph_smoke.py       # Full graph smoke test
```

---

## 🎯 Design Decisions

**LLM Provider:** Defaults to Google Gemini (`gemini-3.6-flash`) rather than Anthropic Claude, since Gemini offers a free API tier with no billing setup — better suited for a portfolio/demo project. Anthropic and OpenAI are fully supported as alternatives via `LLM_PROVIDER` in `.env`.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Tests mock the LLM client, so no API key is needed.

---

## 📝 Assumptions

1. **Currency**: The app displays amounts with `₹` symbol (Indian Rupees) since the example scenario uses INR. This is purely cosmetic — the underlying logic is currency-agnostic.
2. **LLM Model**: Defaults to `claude-sonnet-4-20250514` for Anthropic and `gpt-4o` for OpenAI. These can be changed in `agents/llm_client.py`.
3. **Rounding**: A ±₹1 tolerance is allowed for proposal sums to handle floating-point rounding. Proposals outside this tolerance are auto-corrected via proportional re-normalization.
4. **Concurrency**: Person agents are called sequentially (not in parallel) to keep the chat transcript ordering deterministic and the code simpler for this MVP.
5. **State persistence**: All state is in-memory. The `negotiation_log.json` file is overwritten each run for debugging purposes.
6. **Streaming**: The Streamlit UI renders the full negotiation result after the graph completes, using `st.spinner` to indicate progress. True token-level streaming would require LangGraph's streaming API and is a future enhancement.

---

## 📄 License

MIT
