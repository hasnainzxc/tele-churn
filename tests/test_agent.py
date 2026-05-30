"""Tests for agent orchestration."""

import pandas as pd
import pytest

from telechurn.agent.agent import RetentionAgent


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customer_id": ["TC-004711", "TC-000692"],
        "age": [32.0, 59.0],
        "gender": ["Male", "Female"],
        "tenure_months": [10.0, 3.0],
        "contract_type": ["Month-to-month", "Month-to-month"],
        "monthly_charges": [69.24, 98.48],
        "total_charges": [656.42, 251.15],
        "internet_service": ["DSL", "DSL"],
        "phone_service": ["Yes", "no"],
        "avg_monthly_gb_used": [11.7, 9.46],
        "num_support_tickets": [4.0, 1.0],
        "avg_monthly_minutes": [324.0, 306.8],
        "satisfaction_score": [7.8, 6.0],
        "payment_method": ["bank transfer", "Electronic check"],
        "num_additional_services": [2, 5],
        "last_interaction_date": ["2024-06-14", "2024-06-23"],
        "churned": [1, 1],
    })


class TestRetentionAgent:
    def test_build_graph(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        graph = agent.build_graph()
        assert graph is not None

    def test_invoke_with_customer_id(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        result = agent.invoke("Look up customer TC-004711")
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 10

    def test_invoke_no_customer_id(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        result = agent.invoke("I have an unhappy customer. Help me.")
        assert "response" in result
        assert len(result["response"]) > 10
        # Should not blindly call tools without ID
        calls = result.get("tool_calls_made", [])
        lookup_calls = [c for c in calls if c["name"] == "lookup_customer"]
        # Agent should either ask for ID or make limited calls
        assert len(lookup_calls) <= 1

    def test_escalation_legal_threat(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        result = agent.invoke("Customer TC-004711 is threatening to sue us! Help!")
        assert "response" in result
        # Should contain escalation language
        assert len(result["response"]) > 10

    def test_nonexistent_customer(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        result = agent.invoke("Look up customer TC-999999")
        assert "response" in result

    def test_initial_state(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        result = agent.invoke("Hello")
        assert "response" in result
        assert "tool_calls_made" in result
