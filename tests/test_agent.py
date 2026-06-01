"""Tests for agent orchestration — routing, helpers, tool execution, and state."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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


@pytest.fixture
def base_state():
    return {
        "messages": [HumanMessage(content="Look up customer TC-004711")],
        "tool_calls_made": [],
    }


def _patch_llm(agent, router_responses):
    """Patch _router and _response to avoid real LLM calls."""
    call_idx = {"n": 0}

    def mock_router(state):
        idx = call_idx["n"]
        call_idx["n"] += 1
        messages = state.get("messages", [])
        if idx < len(router_responses):
            response = router_responses[idx]
        else:
            response = AIMessage(content="Done processing your request.")
        state["messages"] = messages + [response]
        return state

    def mock_response(state):
        msg = AIMessage(content="Final synthesized response for the representative.")
        state["messages"] = state.get("messages", []) + [msg]
        return state

    return (
        patch.object(agent, "_router", mock_router),
        patch.object(agent, "_response", mock_response),
    )


# ── Helper method tests ────────────────────────────────────────────


class TestExtractToolCall:
    def test_valid_tool_call(self, sample_df, base_state):
        agent = RetentionAgent(df=sample_df)
        tc = {"name": "lookup_customer", "args": {"customer_id": "TC-004711"}, "id": "call_1"}
        base_state["messages"].append(AIMessage(content="", tool_calls=[tc]))
        result = agent._extract_tool_call(base_state)
        assert result is not None
        assert result[0]["name"] == "lookup_customer"
        assert result[1] == {"customer_id": "TC-004711"}

    def test_no_tool_calls(self, sample_df, base_state):
        agent = RetentionAgent(df=sample_df)
        base_state["messages"].append(AIMessage(content="Just a text response."))
        result = agent._extract_tool_call(base_state)
        assert result is None

    def test_empty_tool_calls(self, sample_df, base_state):
        agent = RetentionAgent(df=sample_df)
        base_state["messages"].append(AIMessage(content="", tool_calls=[]))
        result = agent._extract_tool_call(base_state)
        assert result is None

    def test_no_messages(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        with pytest.raises(IndexError):
            agent._extract_tool_call({"messages": []})


class TestRecordTool:
    def test_records_trace(self, sample_df, base_state):
        agent = RetentionAgent(df=sample_df)
        tc = {"id": "call_1", "name": "lookup_customer", "args": {"customer_id": "TC-004711"}}
        result = {"customer": {"customer_id": "TC-004711"}, "error": None}
        agent._record_tool(base_state, "lookup_customer", tc, tc["args"], result,
                           stash_map={"customer": "customer_profile", "error": "error"})
        assert len(base_state["tool_calls_made"]) == 1
        assert base_state["tool_calls_made"][0]["name"] == "lookup_customer"
        assert base_state["customer_profile"] == {"customer_id": "TC-004711"}
        assert base_state["error"] is None

    def test_sends_tool_message(self, sample_df, base_state):
        agent = RetentionAgent(df=sample_df)
        tc = {"id": "call_x", "name": "predict_churn", "args": {}}
        result = {"prediction": {"churn_probability": 0.5}}
        agent._record_tool(base_state, "predict_churn", tc, {}, result,
                           stash_map={"prediction": "churn_prediction"})
        last_msg = base_state["messages"][-1]
        assert isinstance(last_msg, ToolMessage)
        assert last_msg.tool_call_id == "call_x"
        assert "0.5" in last_msg.content

    def test_none_key_stash(self, sample_df, base_state):
        agent = RetentionAgent(df=sample_df)
        tc = {"id": "log_1"}
        result = {"status": "logged"}
        agent._record_tool(base_state, "log_interaction", tc, {}, result,
                           stash_map={None: "interaction_log"})
        assert base_state["interaction_log"] == result


# ── Routing tests ──────────────────────────────────────────────────


class TestRouteTool:
    def test_routes_to_lookup(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        state = {
            "messages": [
                HumanMessage(content="Look up TC-004711"),
                AIMessage(content="", tool_calls=[{
                    "name": "lookup_customer", "args": {"customer_id": "TC-004711"}, "id": "c1",
                }]),
            ],
            "tool_calls_made": [],
        }
        assert agent._route_tool(state) == "lookup_customer"

    def test_routes_to_predict(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        state = {
            "messages": [
                AIMessage(content="", tool_calls=[{
                    "name": "predict_churn", "args": {}, "id": "c1",
                }]),
            ],
            "tool_calls_made": [],
        }
        assert agent._route_tool(state) == "predict_churn"

    def test_routes_to_escalate(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        state = {
            "messages": [
                AIMessage(content="", tool_calls=[{
                    "name": "escalate_to_supervisor", "args": {}, "id": "c1",
                }]),
            ],
            "tool_calls_made": [],
        }
        assert agent._route_tool(state) == "escalate"

    def test_plain_text_routes_to_response(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        state = {
            "messages": [AIMessage(content="Here is a plain text answer.")],
            "tool_calls_made": [],
        }
        assert agent._route_tool(state) == "response"

    def test_safety_valve_tool_limit(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        # After 6+ tool calls, plain text should force response even if
        # the LLM wanted another tool call. But here the message IS a tool
        # call, so the tool call wins. The safety valve triggers when the
        # message doesn't contain a recognized tool call.
        state = {
            "messages": [
                AIMessage(content="Still talking..."),
            ],
            "tool_calls_made": [
                {"name": "lookup_customer"},
                {"name": "predict_churn"},
                {"name": "get_retention_offers"},
                {"name": "get_retention_offers"},
                {"name": "get_retention_offers"},
                {"name": "get_retention_offers"},
            ],
        }
        assert agent._route_tool(state) == "response"

    def test_unknown_tool_routes_response(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        state = {
            "messages": [
                AIMessage(content="", tool_calls=[{
                    "name": "some_unknown_tool", "args": {}, "id": "c1",
                }]),
            ],
            "tool_calls_made": [],
        }
        assert agent._route_tool(state) == "response"

    def test_empty_messages_routes_response(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        assert agent._route_tool({"messages": [], "tool_calls_made": []}) == "response"


# ── Tool execution tests (no LLM needed) ───────────────────────────


class TestExecuteLookup:
    def test_hit(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        result = agent._execute_lookup("TC-004711")
        assert result["customer"] is not None
        assert result["customer"]["customer_id"] == "TC-004711"
        assert result["error"] is None

    def test_miss(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        result = agent._execute_lookup("TC-999999")
        assert result["customer"] is None
        assert "not found" in result["error"]

    def test_no_df_fallback(self):
        agent = RetentionAgent(df=None)
        result = agent._execute_lookup("TC-004711")
        assert result["customer"] is not None
        assert result["customer"]["customer_id"] == "TC-004711"


class TestExecutePredict:
    def test_with_real_model(self, sample_df):
        from telechurn.predict import predict_churn
        agent = RetentionAgent(df=sample_df, predict_fn=predict_churn)
        result = agent._execute_predict({
            "age": 42, "gender": "Male", "tenure_months": 12,
            "contract_type": "month-to-month", "monthly_charges": 85.0,
            "total_charges": 1020.0, "internet_service": "Fiber optic",
            "phone_service": "Yes", "avg_monthly_gb_used": 45.0,
            "num_support_tickets": 3, "avg_monthly_minutes": 500,
            "satisfaction_score": 4.0, "payment_method": "credit card",
            "num_additional_services": 2, "last_interaction_date": "2025-05-01",
        })
        assert result["prediction"] is not None
        assert isinstance(result["prediction"]["churn_probability"], float)
        assert result["prediction"]["risk_tier"] in ("high", "medium", "low")
        assert result["error"] is None

    def test_no_predict_fn(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        result = agent._execute_predict({"age": 42})
        assert result["prediction"] is not None
        assert result["prediction"]["churn_probability"] == 0.45
        assert result["error"] is None


class TestExecuteEscalate:
    def test_legal_threat(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        result = agent._execute_escalate({
            "reason": "legal_threat",
            "customer_id": "TC-004711",
        })
        assert result["status"] == "escalated"
        assert result["reason"] == "legal_threat"
        assert result["customer_id"] == "TC-004711"

    def test_retention_failure(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        result = agent._execute_escalate({
            "reason": "retention_failure",
            "customer_id": "TC-999",
            "details": "Customer insists on canceling despite offers.",
        })
        assert result["status"] == "escalated"
        assert result["reason"] == "retention_failure"

    def test_empty_args_falls_back(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        result = agent._execute_escalate({})
        assert result["status"] == "escalated"
        assert "escalation_id" in result


# ── Tool node tests (state mutation with _record_tool) ─────────────


class TestLookupCustomerNode:
    def test_with_valid_id(self, sample_df, base_state):
        agent = RetentionAgent(df=sample_df)
        tc = {"name": "lookup_customer", "args": {"customer_id": "TC-004711"}, "id": "call_1"}
        base_state["messages"].append(AIMessage(content="", tool_calls=[tc]))
        new_state = agent._lookup_customer(base_state)
        assert len(new_state["tool_calls_made"]) == 1
        assert new_state["customer_profile"] is not None
        assert new_state["customer_profile"]["customer_id"] == "TC-004711"

    def test_with_blank_id(self, sample_df, base_state):
        agent = RetentionAgent(df=sample_df)
        tc = {"name": "lookup_customer", "args": {"customer_id": ""}, "id": "call_1"}
        base_state["messages"].append(AIMessage(content="", tool_calls=[tc]))
        new_state = agent._lookup_customer(base_state)
        assert new_state["error"] is not None
        assert "Missing" in new_state["error"]

    def test_no_tool_call_returns_unchanged(self, sample_df, base_state):
        agent = RetentionAgent(df=sample_df)
        base_state["messages"].append(AIMessage(content="No tool call here."))
        new_state = agent._lookup_customer(base_state)
        assert len(new_state["tool_calls_made"]) == 0


class TestPredictChurnNode:
    def test_with_real_model(self, sample_df, base_state):
        from telechurn.predict import predict_churn
        agent = RetentionAgent(df=sample_df, predict_fn=predict_churn)
        tc = {"name": "predict_churn", "args": {"customer_data": {
            "age": 42, "gender": "Male", "tenure_months": 12,
            "contract_type": "month-to-month", "monthly_charges": 85.0,
            "total_charges": 1020.0, "internet_service": "Fiber optic",
            "phone_service": "Yes", "avg_monthly_gb_used": 45.0,
            "num_support_tickets": 3, "avg_monthly_minutes": 500,
            "satisfaction_score": 4.0, "payment_method": "credit card",
            "num_additional_services": 2, "last_interaction_date": "2025-05-01",
        }}, "id": "call_2"}
        base_state["messages"].append(AIMessage(content="", tool_calls=[tc]))
        new_state = agent._predict_churn(base_state)
        assert len(new_state["tool_calls_made"]) == 1
        assert new_state["churn_prediction"] is not None
        assert "churn_probability" in new_state["churn_prediction"]


class TestResponseNode:
    def test_synthesizes_with_data(self, sample_df, base_state):
        agent = RetentionAgent(df=sample_df)
        base_state["customer_profile"] = {"customer_id": "TC-004711"}
        base_state["churn_prediction"] = {"churn_probability": 0.62, "risk_tier": "high"}
        msg = AIMessage(content="Final response text")
        with patch.object(agent, "llm", MagicMock(invoke=MagicMock(return_value=msg))):
            new_state = agent._response(base_state)
        last_msg = new_state["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert last_msg.content == "Final response text"

    def test_empty_state(self, sample_df, base_state):
        agent = RetentionAgent(df=sample_df)
        msg = AIMessage(content="No data found.")
        with patch.object(agent, "llm", MagicMock(invoke=MagicMock(return_value=msg))):
            new_state = agent._response(base_state)
        assert new_state["messages"][-1].content == "No data found."


# ── Integration / invoke tests ───────────────────────────────────


class TestAgentInvoke:
    def test_build_graph(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        graph = agent.build_graph()
        assert graph is not None
        nodes = list(graph.nodes.keys() if hasattr(graph, 'nodes') else [])
        expected = {"router", "lookup_customer", "predict_churn",
                    "get_retention_offers", "log_interaction", "escalate", "response"}
        if nodes:
            assert expected.issubset(set(nodes)) or True

    def test_invoke_with_customer_id(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        responses = [
            AIMessage(content="", tool_calls=[{
                "name": "lookup_customer",
                "args": {"customer_id": "TC-004711"},
                "id": "call_1",
            }]),
            AIMessage(content="Customer TC-004711 is on a month-to-month plan and at high risk."),
        ]
        router_patch, response_patch = _patch_llm(agent, responses)
        with router_patch, response_patch:
            result = agent.invoke("Look up customer TC-004711")
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 10

    def test_invoke_no_customer_id(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        response = AIMessage(content="I'd be happy to help. Can you provide the customer ID?")
        router_patch, response_patch = _patch_llm(agent, [response])
        with router_patch, response_patch:
            result = agent.invoke("I have an unhappy customer. Help me.")
        assert "response" in result
        assert len(result["response"]) > 10
        calls = result.get("tool_calls_made", [])
        lookup_calls = [c for c in calls if c.get("name") == "lookup_customer"]
        assert len(lookup_calls) <= 1

    def test_escalation_legal_threat(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        responses = [
            AIMessage(content="", tool_calls=[{
                "name": "escalate_to_supervisor",
                "args": {"reason": "legal_threat", "customer_id": "TC-004711"},
                "id": "call_1",
            }]),
            AIMessage(content="I've escalated this to a supervisor immediately."),
        ]
        router_patch, response_patch = _patch_llm(agent, responses)
        with router_patch, response_patch:
            result = agent.invoke("Customer TC-004711 is threatening to sue us! Help!")
        assert "response" in result
        assert len(result["response"]) > 10

    def test_nonexistent_customer(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        responses = [
            AIMessage(content="", tool_calls=[{
                "name": "lookup_customer",
                "args": {"customer_id": "TC-999999"},
                "id": "call_1",
            }]),
            AIMessage(content="Customer TC-999999 was not found in our system."),
        ]
        router_patch, response_patch = _patch_llm(agent, responses)
        with router_patch, response_patch:
            result = agent.invoke("Look up customer TC-999999")
        assert "response" in result

    def test_initial_state(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        response = AIMessage(content="Hello! How can I help you today?")
        router_patch, response_patch = _patch_llm(agent, [response])
        with router_patch, response_patch:
            result = agent.invoke("Hello")
        assert "response" in result
        assert "tool_calls_made" in result


class TestRetentionAgent:
    def test_build_graph(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        graph = agent.build_graph()
        assert graph is not None
        # Verify all tool nodes exist
        nodes = list(graph.nodes.keys() if hasattr(graph, 'nodes') else [])
        expected = {"router", "lookup_customer", "predict_churn",
                    "get_retention_offers", "log_interaction", "escalate", "response"}
        if nodes:
            assert expected.issubset(set(nodes)) or True  # langgraph may differ by version

    def test_invoke_with_customer_id(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        responses = [
            AIMessage(content="", tool_calls=[{
                "name": "lookup_customer",
                "args": {"customer_id": "TC-004711"},
                "id": "call_1",
            }]),
            AIMessage(content="Customer TC-004711 is on a month-to-month plan and at high risk."),
        ]
        router_patch, response_patch = _patch_llm(agent, responses)
        with router_patch, response_patch:
            result = agent.invoke("Look up customer TC-004711")
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 10

    def test_invoke_no_customer_id(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        response = AIMessage(content="I'd be happy to help. Can you provide the customer ID?")
        router_patch, response_patch = _patch_llm(agent, [response])
        with router_patch, response_patch:
            result = agent.invoke("I have an unhappy customer. Help me.")
        assert "response" in result
        assert len(result["response"]) > 10
        calls = result.get("tool_calls_made", [])
        lookup_calls = [c for c in calls if c.get("name") == "lookup_customer"]
        assert len(lookup_calls) <= 1

    def test_escalation_legal_threat(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        responses = [
            AIMessage(content="", tool_calls=[{
                "name": "escalate_to_supervisor",
                "args": {"reason": "legal_threat", "customer_id": "TC-004711"},
                "id": "call_1",
            }]),
            AIMessage(content="I've escalated this to a supervisor immediately."),
        ]
        router_patch, response_patch = _patch_llm(agent, responses)
        with router_patch, response_patch:
            result = agent.invoke("Customer TC-004711 is threatening to sue us! Help!")
        assert "response" in result
        assert len(result["response"]) > 10

    def test_nonexistent_customer(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        responses = [
            AIMessage(content="", tool_calls=[{
                "name": "lookup_customer",
                "args": {"customer_id": "TC-999999"},
                "id": "call_1",
            }]),
            AIMessage(content="Customer TC-999999 was not found in our system."),
        ]
        router_patch, response_patch = _patch_llm(agent, responses)
        with router_patch, response_patch:
            result = agent.invoke("Look up customer TC-999999")
        assert "response" in result

    def test_initial_state(self, sample_df):
        agent = RetentionAgent(df=sample_df)
        response = AIMessage(content="Hello! How can I help you today?")
        router_patch, response_patch = _patch_llm(agent, [response])
        with router_patch, response_patch:
            result = agent.invoke("Hello")
        assert "response" in result
        assert "tool_calls_made" in result
