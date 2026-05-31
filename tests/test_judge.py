"""Tests for LLM-as-judge evaluation pipeline."""

import os
from unittest import mock

import pytest

from telechurn.eval.judge import aggregate, evaluate, get_client


class TestGetClient:
    def test_requires_api_key(self):
        """get_client raises clear error when API key is missing."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                get_client()

    def test_caches_client(self):
        """Second call returns same client instance (singleton pattern)."""
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            c1 = get_client()
            c2 = get_client()
            assert c1 is c2


class TestEvaluate:
    def test_returns_expected_keys(self):
        """evaluate() returns all four rubric dimensions plus metadata."""
        tc = {"id": "TC-TEST", "messages": [{"role": "user", "content": "test"}],
              "expected_tools": [], "category": "test"}
        mock_response = mock.Mock()
        mock_response.choices = [mock.Mock()]
        mock_response.choices[0].message.content = (
            '{"factual_correctness": 4, "tool_use_appropriateness": 3, '
            '"actionability": 4, "hallucination": 5, '
            '"brief_reasoning": "Good response"}'
        )
        mock_client = mock.Mock()
        mock_client.chat.completions.create.return_value = mock_response

        with mock.patch("telechurn.eval.judge.get_client", return_value=mock_client):
            scores = evaluate(tc, "Agent response here", [])

        assert "factual_correctness" in scores
        assert scores["factual_correctness"] == 4
        assert scores["_model"] == "openai/gpt-4o-mini"
        assert scores["_test_case_id"] == "TC-TEST"

    def test_handles_json_fallback(self):
        """When judge returns invalid JSON, scores default to 0."""
        tc = {"id": "TC-TEST", "messages": [], "expected_tools": [], "category": "test"}
        mock_response = mock.Mock()
        mock_response.choices = [mock.Mock()]
        mock_response.choices[0].message.content = "not valid json {{{"

        mock_client = mock.Mock()
        mock_client.chat.completions.create.return_value = mock_response

        with mock.patch("telechurn.eval.judge.get_client", return_value=mock_client):
            scores = evaluate(tc, "Agent response", [])

        assert scores["factual_correctness"] == 0
        assert scores["brief_reasoning"] == "Judge failed to produce valid JSON"

    def test_handles_empty_response(self):
        """None/empty response handled gracefully — returns zeroed scores."""
        tc = {"id": "TC-TEST", "messages": [], "expected_tools": [], "category": "test"}
        mock_response = mock.Mock()
        mock_response.choices = [mock.Mock()]
        mock_response.choices[0].message.content = None

        mock_client = mock.Mock()
        mock_client.chat.completions.create.return_value = mock_response

        with mock.patch("telechurn.eval.judge.get_client", return_value=mock_client):
            scores = evaluate(tc, "Agent response", [])

        assert scores["factual_correctness"] == 0
        assert scores["brief_reasoning"] == "Judge failed to produce valid JSON"


class TestAggregate:
    def test_computes_stats(self):
        scores = [
            {"factual_correctness": 4, "tool_use_appropriateness": 3,
             "actionability": 4, "hallucination": 5},
            {"factual_correctness": 3, "tool_use_appropriateness": 4,
             "actionability": 3, "hallucination": 4},
        ]
        result = aggregate(scores)
        assert result["num_cases"] == 2
        dims = result["by_dimension"]
        assert dims["factual_correctness"]["mean"] == 3.5
        assert dims["factual_correctness"]["min"] == 3
        assert dims["factual_correctness"]["max"] == 4
        assert "overall_mean" in result

    def test_empty_scores(self):
        result = aggregate([])
        assert result == {"error": "No scores to aggregate"}

    def test_partial_dimensions(self):
        """Missing dimensions are skipped in aggregation."""
        scores = [
            {"factual_correctness": 4, "actionability": 4},
            {"factual_correctness": 3, "tool_use_appropriateness": 3, "hallucination": 5},
        ]
        result = aggregate(scores)
        assert result["num_cases"] == 2
        # tool_use_appropriateness: only in second, mean=3.0
        assert result["by_dimension"]["tool_use_appropriateness"]["mean"] == 3.0
        assert result["by_dimension"]["factual_correctness"]["mean"] == 3.5
