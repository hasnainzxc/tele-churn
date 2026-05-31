"""Run full eval suite: load test cases, run agent, score with metrics + LLM judge."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent.parent.parent))

from telechurn.agent.agent import RetentionAgent
from telechurn.eval.judge import aggregate as judge_aggregate
from telechurn.eval.judge import evaluate as judge_evaluate
from telechurn.eval.metrics import (
    parameter_extraction_precision,
    response_completeness,
    tool_selection_accuracy,
)


def load_test_cases() -> list[dict[str, Any]]:
    path = _here / "test_cases.json"
    with open(path) as f:
        return json.load(f)["cases"]


def run_eval(agent: RetentionAgent, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []

    for i, case in enumerate(cases):
        cid = case["id"]
        print(f"[{i+1}/{len(cases)}] {cid}: {case['name']} ...", end=" ", flush=True)

        try:
            result = agent.invoke(case["user_input"])
        except Exception as e:
            result = {"response": f"ERROR: {e}", "tool_calls_made": []}

        response_text = result["response"]
        actual_tools = result["tool_calls_made"]

        expected_tools = case.get("expected_tools", [])

        tool_score = tool_selection_accuracy(expected_tools, actual_tools)
        param_score = parameter_extraction_precision(expected_tools, actual_tools)
        completeness = response_completeness({}, response_text)

        # LLM-as-judge
        try:
            judge_scores = judge_evaluate(case, response_text, actual_tools)
        except Exception as e:
            judge_scores = {
                "factual_correctness": 0,
                "tool_use_appropriateness": 0,
                "actionability": 0,
                "hallucination": 0,
                "brief_reasoning": f"Judge error: {e}",
            }

        results.append({
            "case_id": cid,
            "case_name": case["name"],
            "category": case["category"],
            "response": response_text[:300],
            "tool_selection_accuracy": tool_score["score"],
            "param_precision": param_score["score"],
            "completeness": completeness["score"],
            "judge": judge_scores,
            "extras_called": tool_score.get("extras", []),
        })

        # Print quick summary
        ts = tool_score["score"]
        ps = param_score["score"]
        cs = completeness["score"]
        jm = judge_scores.get("factual_correctness", 0)
        print(f"tool={ts:.2f} param={ps:.2f} complete={cs:.2f} judge_factual={jm}")

    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    n = len(results)
    ts_avg = sum(r["tool_selection_accuracy"] for r in results) / n
    ps_avg = sum(r["param_precision"] for r in results) / n
    cs_avg = sum(r["completeness"] for r in results) / n

    judge_scores_list = [r["judge"] for r in results]
    judge_agg = judge_aggregate(judge_scores_list)

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Cases:                    {n}")
    print(f"Tool Selection Accuracy:  {ts_avg:.3f}")
    print(f"Parameter Precision:      {ps_avg:.3f}")
    print(f"Response Completeness:    {cs_avg:.3f}")
    print(f"Judge Overall Mean:       {judge_agg.get('overall_mean', 'N/A')}")
    for dim, stats in judge_agg.get("by_dimension", {}).items():
        print(f"  {dim}: mean={stats['mean']:.2f} min={stats['min']} max={stats['max']}")

    by_cat: dict[str, list[float]] = {}
    for r in results:
        cat = r["category"]
        by_cat.setdefault(cat, []).append(r["tool_selection_accuracy"])
    print("\n=== BY CATEGORY (tool selection accuracy) ===")
    for cat, scores in sorted(by_cat.items()):
        print(f"  {cat}: {sum(scores)/len(scores):.3f}")


if __name__ == "__main__":
    import pandas as pd

    from telechurn.predict import predict_churn

    cases = load_test_cases()
    print(f"Loaded {len(cases)} test cases\n")

    DATA_PATH = _here.parents[3] / "test_datafile.csv"
    df = pd.read_csv(DATA_PATH)

    agent = RetentionAgent(
        model="openai/gpt-4o-mini",
        df=df,
        predict_fn=predict_churn,
    )
    results = run_eval(agent, cases)
    print_summary(results)

    # Write full results
    out = _here / "eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results written to {out}")
