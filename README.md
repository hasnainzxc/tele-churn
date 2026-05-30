# TeleConnect — Churn Prediction & Retention Agent

AI/ML Engineer Take-Home Assessment.

## What this is

Two connected systems for a mid-size telecom (~5K customers):

1. **Churn Prediction Model** — Jupyter notebook that cleans a deliberately dirty dataset, explores churn patterns, trains two model families, and exports a callable pipeline.
2. **Retention Agent** — LangGraph-powered LLM agent that retention reps can talk to in natural language. It looks up customers, runs the churn model, retrieves relevant offers, and synthesizes a recommendation.

## Architecture

```
user query → Streamlit UI → LangGraph ReAct agent
                                ├─ router (LLM decides what to call)
                                ├─ lookup_customer   ──→ mock DB / CSV
                                ├─ predict_churn     ──→ model_pipeline.pkl
                                ├─ get_retention_offers → offer catalog
                                ├─ log_interaction   ──→ in-memory log
                                ├─ escalate_to_supervisor
                                └─ response (synthesize final answer)
```

The agent uses a state graph: one node per tool, conditional edges for routing, and a dedicated response synthesis node. Adding a 6th tool means adding one node + one edge — no orchestration rewrite.

**Evaluation pipeline** runs the agent against 15 structured test cases, scores tool selection / parameter extraction / response completeness, then sends results through an LLM-as-judge with anchored 1-5 rubrics (factual correctness, tool use, actionability, hallucination).

## Setup

```bash
# Clone
git clone <repo-url>
cd tele-churn

# Install deps (uv)
uv sync

# Add your OpenRouter API key
cp .env.example .env
# Edit .env → add your OPENROUTER_API_KEY

# Train the churn model (generates artifacts/model_pipeline.pkl)
uv run python scripts/rebuild_pipeline.py
```

## Key Commands

```bash
uv run pytest                              # 22 tests
uv run python src/telechurn/eval/run_eval.py  # Full eval (needs API key)
uv run streamlit run apps/streamlit_app.py    # Launch agent UI
uv run jupyter lab                        # Open notebook
uv run ruff check .                       # Lint
```

## Project Structure

```
├── notebooks/
│   └── 01_churn_model.ipynb      # Part 1: data quality → EDA → models → export
├── src/telechurn/
│   ├── predict.py                # predict_churn() — preprocessing + model
│   ├── agent/
│   │   ├── tools.py              # Pydantic tool schemas (5 tools)
│   │   ├── offers.py             # Retention offer catalog by risk × contract
│   │   └── agent.py              # LangGraph state graph + ReAct loop
│   └── eval/
│       ├── test_cases.json       # 15 structured test cases
│       ├── metrics.py            # Automated metrics (3)
│       ├── judge.py              # LLM-as-judge with anchored rubrics
│       └── run_eval.py           # Eval runner
├── apps/
│   └── streamlit_app.py          # Chat UI with tool trace panel
├── tests/
│   ├── test_tools.py
│   ├── test_agent.py
│   └── test_eval.py
├── artifacts/
│   └── model_pipeline.pkl        # Exported sklearn pipeline
└── AGENTS.md                     # Workflow rules for this repo
```

## Design Decisions

### Why LangGraph over raw LangChain chains
Agent needs conditional tool routing + state persistence across multi-turn. LangGraph's state graph model maps naturally: node per tool, edge per decision path. Adding a 6th tool = adding one node + edges — no orchestration rewrite.

### Why Recall as primary metric
Churn is a rare-but-costly event (~36%). Missing a churner (false negative) costs revenue; flagging a non-churner (false positive) costs a coupon. Recall minimises false negatives. Precision-Recall AUC complements it for imbalance-aware ranking.

### Why XGBoost + LogisticRegression
Different structural assumptions. XGBoost captures non-linear interactions + missing data natively. LogisticRegression provides interpretable coefficients. If they agree on a prediction, confidence is higher. If they disagree, flag for review.

### Data Cleaning Philosophy
Never silently drop corrupt rows. Flag, document, choose recovery strategy per column type. Produce before/after summary so reviewer can audit every decision.

### Why OpenRouter
Single API for multiple model providers. Agent uses gpt-4o-mini for routing + synthesis; judge uses same model (can be swapped without code changes).

## Evaluation Results

Run against 15 test cases across 7 categories:

```
Tool Selection Accuracy:  0.800    (12/15 called expected tools)
Parameter Precision:      0.822    (82% correct params)
Response Completeness:    0.900    (strong response quality)
Judge Overall Mean:       3.94/5   (LLM-as-judge average across 4 dims)

By Category (tool selection):
  ambiguous_input:         1.000
  model_disagreement:      1.000
  multi_step_chaining:     1.000
  out_of_scope:            1.000
  adversarial_edge_case:   0.667
  escalation_trigger:      0.500
  single_tool_happy_path:  0.500
```

**Success cases**: Multi-step chaining (TC-003, TC-004) — agent correctly chains lookup → predict → offers and produces actionable recommendations. Ambiguous input (TC-005, TC-006) — agent asks clarifying questions instead of guessing.

**Failure cases**: TC-002 (single-tool offers without customer ID) — system prompt requires lookup first, but this query is hypothetical. Fix: add a "general inquiry" path. TC-008 (complex dispute escalation) — agent looks up customer but doesn't escalate. Root cause: mock customer data doesn't surface billing issues. Fix: inject real data in eval or improve escalation detection heuristics.

**Production roadmap**: Containerise the eval runner, run it in CI on every PR, store results in a structured format (JSONL) with timestamps for trend detection. Replace heuristic completeness scoring with LLM-based evaluation. Add latency/p95 tracking. Calibrate judge against 20+ human-labeled examples to measure inter-rater reliability.

## Known Limitations

- **Mock data fallback**: When DataFrame is not injected, `lookup_customer` returns hardcoded mock profiles. Real integration would use a database.
- **No persistence**: `log_interaction` and `escalate_to_supervisor` are in-memory only. Production needs a database or ticketing API.
- **XGBoost underperforms LR**: In the notebook, XGBoost recall (0.546 CV) lags behind LogisticRegression (0.696 CV). With hyperparameter tuning (Optuna), XGBoost would likely surpass LR. Not tuned due to time budget.
- **Judge reliability**: Single-pass LLM judging has positivity bias. Production should use 3 passes per case with median scoring, and flag cases with variance > 1.0.
- **No streaming**: Agent responses are batch-only. Streaming would improve perceived latency in the UI.
