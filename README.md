# TeleConnect Churn Prediction & Retention Agent

AI/ML Engineer take home assessment: churn prediction model (Jupyter notebook) and retention agent (Streamlit + LangGraph).

**Live demo: [tele-churn.streamlit.app](https://tele-churn.streamlit.app)**

## Overview

TeleConnect has about 5,000 customers and 36% of them churn. This project helps retention reps figure out who is about to leave and what to do about it.

Part 1 is a Jupyter notebook that cleans the messy data, explores churn patterns, trains two models from different families, and exports the better one.

Part 2 is a Streamlit chatbot for retention reps. Type a query like "Customer TC-004711 might leave, what can we offer?" — it looks up the customer, runs the churn model, fetches relevant offers, and returns a recommendation with full tool trace visibility.

## Screenshots

### Data Quality

![Billing consistency check](artifacts/screenshots/billing_consistency.png)

total_charges vs monthly_charges × tenure: 31% of rows off by more than $100. Rounding, mid-cycle changes, prorated charges. Kept as signal.

### EDA

![Churn by contract type](artifacts/screenshots/churn_by_contract.png)

Month to month contracts churn 5x more than annual plans. The simplest retention move is getting people onto longer contracts.

![Churn by tenure](artifacts/screenshots/churn_tenure_kde.png)

Most churners leave in the first 20 months. After that they are invested and unlikely to go anywhere.

![Churn by satisfaction](artifacts/screenshots/churn_satisfaction.png)

Satisfaction 0-2 churns at 3x the rate of 8-10.

### Model Evaluation

![Confusion matrices](artifacts/screenshots/confusion_matrices.png)

XGBoost has higher recall; LogisticRegression has fewer false positives. Optimized for recall.

![ROC and PR curves](artifacts/screenshots/roc_pr_curves.png)

Both models beat random. PR curve is more informative than ROC for imbalanced data.

![Feature importance](artifacts/screenshots/feature_importance.png)

Contract type and tenure dominate both models.

## Setup

```bash
git clone https://github.com/hasnainzxc/tele-churn.git
cd tele-churn
uv sync
cp .env.example .env   # add your OPENROUTER_API_KEY
uv run python scripts/rebuild_pipeline.py   # train model, save artifact
```

## Commands

```bash
uv run pytest                                       # 59 tests
uv run python src/telechurn/eval/run_eval.py        # eval suite (needs API key)
uv run streamlit run apps/streamlit_app.py          # launch agent UI
uv run jupyter lab                                  # open notebook
uv run ruff check .                                 # lint
```

## How it works

The agent uses LangGraph. When you send a message, the LLM decides which tool to call based on what you asked. It might chain multiple tools: look up a customer, predict their churn risk, then fetch offers that match their risk level and contract type.

Each tool is a separate node in a state graph. The router (LLM) picks the next node. The response node synthesizes everything into a readable recommendation. Adding a 6th tool is just adding one node and one edge.

Trained on 5,050 records, 16 features. Recall is the primary metric: false negative (missed churner) costs more than false positive (unnecessary retention offer). After hyperparameter tuning XGBoost leads at recall 0.777 vs LR 0.760.

## Why these choices

**LangGraph over LangChain chains.** The agent needs conditional routing. State graph maps naturally: one node per tool, edges for decision paths. No orchestration rewrite when adding tools.

**Recall as primary metric.** Churn is rare but expensive. False negative means lost revenue. False positive means a coupon. Recall minimizes the expensive mistake.

**XGBoost + Logistic Regression.** Different families, different assumptions. If they agree, confidence is higher. If they disagree, flag for human review. After hyperparameter tuning, XGBoost won (recall 0.777 vs LR 0.760).

**OpenRouter.** One API key, multiple model providers. Swap models without changing code.

**Never silently drop bad data.** Every column gets a before and after summary. Corrupted values get flagged, documented, and recovered with a documented strategy. The reviewer can audit every decision.

## Eval results

15 test cases across 7 categories:

| Metric | Score |
|---|---|
| Tool Selection Accuracy | 80% |
| Parameter Precision | 82% |
| Response Completeness | 90% |
| LLM Judge (overall) | 3.94 / 5 |

Best at: multi step chaining (lookup > predict > offers), ambiguous input (asks clarifying questions instead of guessing).

Needs work: escalation triggers (does not always detect complex disputes), single tool path when no customer ID provided.

## Project layout

```
notebooks/01_churn_model.ipynb     Part 1: data quality > EDA > models > export
src/telechurn/predict.py           predict_churn() function + preprocessing
src/telechurn/agent/tools.py       5 Pydantic tool schemas
src/telechurn/agent/offers.py      Retention offer catalog
src/telechurn/agent/agent.py       LangGraph ReAct agent
src/telechurn/eval/test_cases.json 15 structured test cases
src/telechurn/eval/metrics.py      Automated scoring metrics
src/telechurn/eval/judge.py        LLM as judge with anchored rubrics
src/telechurn/eval/run_eval.py     Eval runner
apps/streamlit_app.py              Chat UI with tool trace
tests/                             Test suite (59 tests)
artifacts/model_pipeline.pkl       Trained XGBoost pipeline
artifacts/screenshots/             Notebook visualizations
```

## Known issues

The lookup tool and escalation are in memory only. Real integration would use a database and ticketing API. The LLM judge uses a single pass which has positivity bias. Production would use 3 passes with median scoring. No streaming in the agent yet so responses are batch only.
