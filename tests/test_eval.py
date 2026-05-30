"""Tests for evaluation pipeline."""

import json
from pathlib import Path

from telechurn.eval.metrics import (
    parameter_extraction_precision,
    response_completeness,
    tool_selection_accuracy,
)


class TestToolSelectionAccuracy:
    def test_perfect_match(self):
        expected = [{"name": "lookup_customer", "key_params": {"customer_id": "TC-001"}}]
        actual = [{"name": "lookup_customer", "params": {"customer_id": "TC-001"}}]
        result = tool_selection_accuracy(expected, actual)
        assert result["score"] == 1.0

    def test_no_match(self):
        expected = [{"name": "lookup_customer"}]
        actual = [{"name": "get_retention_offers"}]
        result = tool_selection_accuracy(expected, actual)
        assert result["score"] == 0.0

    def test_extra_tools(self):
        expected = [{"name": "lookup_customer"}]
        actual = [{"name": "lookup_customer"}, {"name": "log_interaction"}]
        result = tool_selection_accuracy(expected, actual)
        assert result["score"] == 1.0
        assert len(result["extras"]) == 1

    def test_empty_expected(self):
        result = tool_selection_accuracy([], [])
        assert result["score"] == 1.0


class TestParameterExtraction:
    def test_exact_match(self):
        expected = [{"name": "lookup", "key_params": {"id": "TC-001"}}]
        actual = [{"name": "lookup", "params": {"id": "TC-001"}}]
        result = parameter_extraction_precision(expected, actual)
        assert result["score"] == 1.0

    def test_mismatch(self):
        expected = [{"name": "lookup", "key_params": {"id": "TC-001"}}]
        actual = [{"name": "lookup", "params": {"id": "TC-002"}}]
        result = parameter_extraction_precision(expected, actual)
        assert result["score"] == 0.0

    def test_empty(self):
        result = parameter_extraction_precision([], [])
        assert result["score"] == 1.0


class TestResponseCompleteness:
    def test_good_response(self):
        result = response_completeness({}, "I recommend offering the 15% discount to customer TC-004711.")
        assert result["score"] > 0.5

    def test_short_response(self):
        result = response_completeness({}, "ok")
        assert result["score"] < 0.5

    def test_actionable_response(self):
        result = response_completeness({}, "I suggest escalating this case to a supervisor immediately.")
        assert result["score"] > 0.5


class TestTestSuiteLoading:
    def test_cases_load(self):
        path = Path(__file__).parent.parent / "src" / "telechurn" / "eval" / "test_cases.json"
        with open(path) as f:
            data = json.load(f)
        assert len(data["cases"]) >= 12
        categories = {c["category"] for c in data["cases"]}
        assert "single_tool_happy_path" in categories
        assert "multi_step_chaining" in categories
        assert "ambiguous_input" in categories
        assert "out_of_scope" in categories
        assert "escalation_trigger" in categories
        assert "model_disagreement" in categories
        assert "adversarial_edge_case" in categories
