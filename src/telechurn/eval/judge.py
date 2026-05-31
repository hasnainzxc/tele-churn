"""LLM-as-Judge evaluator with anchored rubrics.

Uses a separate LLM call to evaluate agent responses against structured criteria.

Anchored rubrics (1-5 scale with behavioral descriptions):

Factual Correctness:
  1 — Multiple factual errors; contradicts customer data or model output.
  3 — Mostly correct; minor errors that don't change the recommendation.
  5 — All facts verified against customer data and model output; no errors.

Tool Use Appropriateness:
  1 — Wrong tools called; missed required tools; called tools in wrong order.
  3 — Mostly correct tool sequence; minor ordering or parameter issues.
  5 — Flawless tool selection, ordering, and parameter passing.

Actionability:
  1 — Vague or generic response; rep would need to ask follow-ups.
  3 — Useful but missing some specifics; rep can work with it but needs context.
  5 — Ready to use immediately; specific offers named, clear next steps, no ambiguity.

Hallucination:
  1 — Fabricated customer data, offers, or facts not from tools.
  3 — Minor embellishments or inferences not strictly from tools but plausible.
  5 — Every claim traceable to tool output; no invented facts.

Judge Reliability Discussion:
- Positivity bias: LLMs tend toward middle-of-scale scores (3-4). Mitigate with
  behavioral anchors that make the 1 and 5 extremes concrete.
- Inter-rater consistency: Running judge twice on same case with different
  prompts can surface sensitivity. Ideally calibrate against 10-20 human-labeled
  examples, though not done here due to scope.
- Prompt sensitivity: Slight changes to rubric wording can shift scores. Lock
  rubric as a constant and never ad-hoc modify per case.
- Production approach: Use a stronger model as judge (e.g., GPT-4o), run 3 passes
  per case and take median, flag cases with variance > 1.0 for human review.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

# Static rubric template — json.dumps handles the test_case/tool_calls insertion.
# Using a constant avoids re-building the prompt string each call, though .format()
# is still called per-evaluation. Fine for this scale.
JUDGE_RUBRIC = """
You are an evaluator for a customer retention AI agent. Score the agent's response
on a 1-5 scale for each criterion below. Use the anchored definitions precisely.

## Rubric

### factual_correctness (1-5)
1 — Multiple factual errors; contradicts customer data or model output.
3 — Mostly correct; minor errors that don't change the recommendation.
5 — All facts verified against customer data and model output; no errors.

### tool_use_appropriateness (1-5)
1 — Wrong tools called; missed required tools; called tools in wrong order.
3 — Mostly correct tool sequence; minor ordering or parameter issues.
5 — Flawless tool selection, ordering, and parameter passing.

### actionability (1-5)
1 — Vague or generic response; rep would need to ask follow-ups.
3 — Useful but missing some specifics; rep can work with it but needs context.
5 — Ready to use immediately; specific offers named, clear next steps, no ambiguity.

### hallucination (1-5)
1 — Fabricated customer data, offers, or facts not from tools.
3 — Minor embellishments or inferences not strictly from tools but plausible.
5 — Every claim traceable to tool output; no invented facts.

## Input
Test Case: {test_case}
Expected Tools: {expected_tools}
Agent Response: {agent_response}
Tool Calls Made: {tool_calls}

## Output Format
Return ONLY a JSON object. No markdown, no explanation.
{{
  "factual_correctness": <int 1-5>,
  "tool_use_appropriateness": <int 1-5>,
  "actionability": <int 1-5>,
  "hallucination": <int 1-5>,
  "brief_reasoning": "<one sentence explaining scores>"
}}
"""


_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Set it in .env or st.secrets."
        )
    _client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    return _client


def evaluate(
    test_case: dict[str, Any],
    agent_response: str,
    tool_calls: list[dict[str, Any]],
    model: str = "openai/gpt-4o-mini",
) -> dict[str, Any]:
    """Run LLM-as-judge evaluation for a single test case."""
    client = get_client()

    # Build the full prompt by stuffing test case, response, and tool calls
    # into the rubric template. json.dumps for structured fields ensures
    # proper escaping inside the JSON output format block.
    prompt = JUDGE_RUBRIC.format(
        test_case=json.dumps(test_case, indent=2),
        expected_tools=json.dumps(test_case.get("expected_tools", []), indent=2),
        agent_response=agent_response,
        tool_calls=json.dumps(tool_calls, indent=2),
    )

    # temperature=0.0 because we want deterministic scoring, not creative fluff.
    # max_tokens=512 is plenty for the 4 scores + one reasoning sentence.
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )

    content = response.choices[0].message.content or "{}"
    content = content.strip()

    # LLMs love wrapping JSON in ``` fences even when told not to. Strip them.
    content = re.sub(r"^```(?:json)?\s*\n?|```\s*$", "", content.strip(), flags=re.DOTALL)

    try:
        scores = json.loads(content)
    except json.JSONDecodeError:
        scores = {}

    if not scores or not isinstance(scores, dict):
        scores = {
            "factual_correctness": 0,
            "tool_use_appropriateness": 0,
            "actionability": 0,
            "hallucination": 0,
            "brief_reasoning": "Judge failed to produce valid JSON",
            "raw_output": content,
        }

    # Attach metadata so downstream consumers can correlate scores back to
    # the specific test case and know which judge model was used.
    scores["_model"] = model
    scores["_test_case_id"] = test_case.get("id", "unknown")
    return scores


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate judge scores across test suite."""
    if not scores:
        return {"error": "No scores to aggregate"}

    # The four dimensions from our rubric — if the judge schema changes,
    # this list needs updating too (fragile, but explicit is fine here).
    dims = ["factual_correctness", "tool_use_appropriateness", "actionability", "hallucination"]
    agg = {"num_cases": len(scores), "by_dimension": {}}

    for dim in dims:
        # Some cases might be missing a dimension if the judge flaked on just one
        vals = [s[dim] for s in scores if dim in s]
        if vals:
            agg["by_dimension"][dim] = {
                "mean": round(sum(vals) / len(vals), 2),
                "min": min(vals),
                "max": max(vals),
            }

    # Grand mean across the four dimensions — rough but useful for a quick
    # "how's the agent doing overall" number.
    agg["overall_mean"] = round(
        sum(d["mean"] for d in agg["by_dimension"].values()) / len(agg["by_dimension"]), 2
    )
    return agg
