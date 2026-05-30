# tele-churn

TeleConnect AI/ML Engineer Take-Home Assessment.

## Project

- **Part 1**: Churn prediction model → Jupyter notebook `notebooks/01_churn_model.ipynb`
- **Part 2**: Retention agent → Python project `src/telechurn/agent/` + Streamlit app

## Tech Stack

- **Python** 3.11+, managed with **uv**
- **Models**: XGBoost + LogisticRegression
- **Agent**: LangGraph ReAct loop via OpenRouter
- **UI**: Streamlit (tool trace visible in panel)
- **Eval**: Structured JSON test suite + LLM-as-judge with anchored rubrics

## Setup

```bash
uv sync
cp .env.example .env   # add your OPENROUTER_API_KEY
```

## Workflow Rules

- **One thing at a time.** Finish current task fully (code + verify + commit) before starting next. Never scatter work across multiple phases simultaneously.
- **Incremental commits.** Each logical unit of work gets its own conventional commit. No bulk dumps. Commit after every completed feature/file set.
- **Commit after verify.** Run lint/tests if applicable before committing. Fix issues, then commit.
- **Stop and ask** before switching context if current work is incomplete. Confirmation required.
- **Never push** without explicit user confirmation. Always ask before pushing to remote.

## Key Commands

```bash
uv run pytest                    # run tests
uv run streamlit run app.py      # launch agent UI
uv run jupyter lab               # open notebook
uv run ruff check .              # lint
```

## Architecture

```
src/telechurn/
├── predict.py       # predict_churn() — model + preprocessing pipeline
├── agent/
│   ├── tools.py     # 5 Pydantic tool definitions
│   └── agent.py     # LangGraph state graph
├── eval/
│   ├── test_cases.json  # 12+ structured test cases
│   ├── metrics.py       # Automated evaluation metrics
│   └── judge.py         # LLM-as-judge with anchored rubrics
apps/
└── streamlit_app.py     # Chat UI with tool trace panel
tests/
├── test_tools.py
├── test_agent.py
└── test_eval.py
notebooks/
└── 01_churn_model.ipynb
artifacts/
└── model_pipeline.pkl
```

## Env Vars

- `OPENROUTER_API_KEY` — API key for OpenRouter
- `OPENROUTER_BASE_URL` — defaults to `https://openrouter.ai/api/v1`

## Design Decisions

### Why LangGraph over raw LangChain chains
Agent needs conditional tool routing + state persistence across multi-turn. LangGraph's state graph model maps naturally: node per tool, edge per decision path. Adding a 6th tool = adding one node + edges — no orchestration rewrite.

### Why Recall as primary metric
Churn is a rare-but-costly event (~36%). Missing a churner (FN) costs revenue; flagging a non-churner (FP) costs a coupon. Recall minimises FN. Precision-Recall AUC complements it for imbalance-aware ranking.

### Why XGBoost + LogisticRegression
Different structural assumptions. XGBoost captures non-linear interactions + missing data natively. LogisticRegression provides interpretable coefficients. If they agree on a prediction, confidence is higher. If they disagree, flag for review.

### Data Cleaning Philosophy
Never silently drop corrupt rows. Flag, document, choose recovery strategy per column type. Produce before/after summary so reviewer can audit every decision.
