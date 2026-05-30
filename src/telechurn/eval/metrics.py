"""Evaluation metrics for agent test suite.

Metrics chosen:
1. Tool selection accuracy — did agent call the right tools in right order?
   Why: Core proxy for correct reasoning. Wrong tool = wrong outcome.
2. Parameter extraction precision — did agent pass correct params to tools?
   Why: Right tool with wrong params still fails. Measures attention to detail.
3. Response completeness — does final response cover all expected elements?
   Why: End-user experience. Even correct tools can produce incomplete answers.
"""

from __future__ import annotations

from typing import Any


def tool_selection_accuracy(
    expected_tools: list[dict[str, Any]], actual_tools: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compute tool selection accuracy: correct tool name match ratio."""
    # Edge: no expected tools specified — if agent also called none, perfect score
    if not expected_tools:
        return {"score": 1.0 if not actual_tools else 0.0, "matched": [], "extras": actual_tools}

    # Extract just the tool names, since we only care about which tools were called
    expected_names = [t["name"] for t in expected_tools]
    actual_names = [t.get("name", "") for t in actual_tools]

    # Check each expected tool name against the actual call list
    matched = []
    for e_name in expected_names:
        found = e_name in actual_names
        matched.append({"expected": e_name, "found": found})

    # Tools the agent called that weren't in the spec — could be hallucinated or wrong
    extras = [n for n in actual_names if n not in expected_names]

    # Simple ratio: correct tools / total expected tools
    score = sum(1 for m in matched if m["found"]) / len(expected_names)

    return {"score": round(score, 3), "matched": matched, "extras": extras}


def parameter_extraction_precision(
    expected_tools: list[dict[str, Any]], actual_tools: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compute parameter precision: fraction of expected key params correctly passed."""
    # Edge: no expected tools means nothing to check params on
    if not expected_tools:
        return {"score": 1.0, "details": []}

    total_params = 0
    matched_params = 0
    details = []

    for e_tool in expected_tools:
        e_name = e_tool["name"]
        # Only the params we explicitly asked to verify — not all of them
        e_params = e_tool.get("key_params", {})

        # Find the matching actual tool call by name, if it exists
        a_tool = next((t for t in actual_tools if t.get("name") == e_name), None)

        for key, expected_val in e_params.items():
            total_params += 1
            # If the tool wasn't called at all, param is None — no match
            # tool_calls_made uses "args" key, not "params"
            actual_val = a_tool.get("args", {}).get(key) if a_tool else None
            # Coerce both to strings for comparison; handles int/float/str mismatches
            is_match = str(actual_val) == str(expected_val) if actual_val is not None else False
            if is_match:
                matched_params += 1
            # Track every param check individually for debugging granularity
            details.append({
                "tool": e_name,
                "param": key,
                "expected": expected_val,
                "actual": actual_val,
                "match": is_match,
            })

    # Avoid division by zero — shouldn't happen since we return early above
    score = matched_params / total_params if total_params > 0 else 1.0
    return {"score": round(score, 3), "details": details}


def response_completeness(expected_criteria: dict[str, Any], actual_response: str) -> dict[str, Any]:
    """Score response completeness using heuristic keyword/rules matching.

    Falls back to structural checks when no criteria specified:
    - Has non-empty text
    - Contains customer ID reference if applicable
    - Contains actionable recommendation
    """
    checks = []
    score = 0.0
    max_score = 3.0

    # Heuristic 1: response isn't a stub — >20 chars means it's a real reply
    # 20 is arbitrary but catches empty strings and one-word shrugs
    if len(actual_response.strip()) > 20:
        score += 1
        checks.append("non_trivial_length")
    else:
        checks.append("too_short")

    # Heuristic 2: mentions a customer identifier — shows it grounded in the task
    # Checks for both the generic word and the TC- pattern from our dataset
    if "customer" in actual_response.lower() or "TC-" in actual_response.upper():
        score += 0.5
        checks.append("mentions_customer")
    else:
        checks.append("no_customer_reference")

    # Heuristic 3: has action verbs — means it's giving the rep something to do
    # Keyword list is tuned for retention scenarios (recommend, escalate, offer, etc.)
    action_words = ["recommend", "suggest", "offer", "escalate", "call", "contact", "advise"]
    if any(w in actual_response.lower() for w in action_words):
        score += 1
        checks.append("actionable")
    else:
        checks.append("not_actionable")

    # Heuristic 4: acknowledges when it can't do something — honesty check
    # Not always present, so it's bonus 0.5 rather than required
    if "error" in actual_response.lower() or "unable" in actual_response.lower():
        score += 0.5
        checks.append("acknowledges_errors")

    # Clamp to [0, 1.0] — score can technically exceed max_score with error bonus
    normalized = round(score / max_score, 3)
    return {"score": min(normalized, 1.0), "checks": checks}
