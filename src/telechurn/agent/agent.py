"""LangGraph ReAct agent for retention.

State graph architecture:
- One node per tool (5 nodes)
- Router node: LLM decides which tool(s) to call
- Response node: LLM synthesizes final response from tool outputs
- Conditional edges: router -> tool1/tool2/... or router -> response

Adding a 6th tool = add one node definition + one tool description + one conditional edge.
No orchestration layer changes needed.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from .offers import get_offers
from .tools import (
    ChurnPrediction,
    CustomerProfile,
    EscalateToSupervisorInput,
    EscalateToSupervisorOutput,
    GetRetentionOffersInput,
    GetRetentionOffersOutput,
    LogInteractionInput,
    LogInteractionOutput,
    LookupCustomerInput,
    LookupCustomerOutput,
    PredictChurnInput,
    PredictChurnOutput,
    RetentionOffer,
)

SYSTEM_PROMPT = """You are a retention agent for TeleConnect, a telecommunications company.
Your job is to help retention representatives save at-risk customers.

You have access to these tools:
1. lookup_customer(customer_id) — Retrieve customer profile by ID
2. predict_churn(customer_data) — Run churn prediction model on customer features
3. get_retention_offers(risk_tier, contract_type) — Get retention offers for a risk tier
4. log_interaction(customer_id, outcome, offers_presented, notes) — Log conversation outcome
5. escalate_to_supervisor(customer_id, reason, context_summary, priority) — Escalate to human

Rules:
- Always look up a customer before predicting churn or getting offers for them.
- If no customer_id is provided, ASK for it. Do not guess.
- If customer threatens legal action, escalate immediately. Do not try retention offers.
- If the situation is outside retention scope (sales, tech support, billing disputes), say so.
- Synthesize a useful recommendation. Do not dump raw data.
- If the model prediction conflicts with profile warning signs, flag it and recommend caution.
- If lookup fails (customer not found), tell the rep clearly and stop.
- Log interactions when the rep confirms an outcome was reached.
"""

TOOL_DESCRIPTIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_customer",
            "description": "Retrieve a customer's full profile by their customer ID (format TC-XXXXXX).",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer ID in format TC-XXXXXX",
                    }
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_churn",
            "description": "Run churn prediction model on customer features. Returns churn probability, risk tier, and top risk factors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_data": {
                        "type": "object",
                        "description": "Full customer feature dictionary from lookup",
                    }
                },
                "required": ["customer_data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_retention_offers",
            "description": "Get retention offers filtered by risk tier (high/medium/low) and contract type (Month-to-month/One year/Two year).",
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_tier": {
                        "type": "string",
                        "description": "Risk tier: high, medium, or low",
                    },
                    "contract_type": {
                        "type": "string",
                        "description": "Contract type: Month-to-month, One year, or Two year",
                    },
                },
                "required": ["risk_tier", "contract_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_interaction",
            "description": "Record the outcome of a retention conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Customer ID"},
                    "outcome": {
                        "type": "string",
                        "description": "Outcome summary: accepted_offer, declined, escalated, no_action, follow_up",
                    },
                    "offers_presented": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of offer IDs presented",
                    },
                    "notes": {"type": "string", "description": "Additional notes"},
                },
                "required": ["customer_id", "outcome"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_supervisor",
            "description": "Escalate case to human supervisor. Use when customer threatens legal action, has complex dispute, or situation is beyond agent scope.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Customer ID"},
                    "reason": {
                        "type": "string",
                        "description": "Reason: legal_threat, complex_dispute, out_of_scope, data_quality_issue, high_risk_override",
                    },
                    "context_summary": {
                        "type": "string",
                        "description": "Brief summary of situation and relevant facts",
                    },
                    "priority": {
                        "type": "string",
                        "description": "Priority: low, normal, high, critical",
                    },
                },
                "required": ["customer_id", "reason", "context_summary"],
            },
        },
    },
]


class AgentState(TypedDict):
    messages: list[Any]
    tool_calls_made: list[dict[str, Any]]
    customer_profile: dict[str, Any] | None
    churn_prediction: dict[str, Any] | None
    retention_offers: list[dict[str, Any]] | None
    escalation: dict[str, Any] | None
    interaction_log: dict[str, Any] | None
    error: str | None


class RetentionAgent:
    def __init__(
        self,
        model: str = "anthropic/claude-3.5-sonnet",
        temperature: float = 0.1,
        df: pd.DataFrame | None = None,
        predict_fn: Any = None,
    ):
        self.model = model
        self.temperature = temperature
        self.df = df
        self.predict_fn = predict_fn
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
        self.llm_with_tools = self.llm.bind_tools(TOOL_DESCRIPTIONS)

    def build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(AgentState)

        builder.add_node("router", self._router)
        builder.add_node("lookup_customer", self._lookup_customer)
        builder.add_node("predict_churn", self._predict_churn)
        builder.add_node("get_retention_offers", self._get_retention_offers)
        builder.add_node("log_interaction", self._log_interaction)
        builder.add_node("escalate", self._escalate)
        builder.add_node("response", self._response)

        builder.add_edge(START, "router")
        builder.add_conditional_edges("router", self._route_tool, {
            "lookup_customer": "lookup_customer",
            "predict_churn": "predict_churn",
            "get_retention_offers": "get_retention_offers",
            "log_interaction": "log_interaction",
            "escalate": "escalate",
            "response": "response",
            END: END,
        })
        builder.add_edge("lookup_customer", "router")
        builder.add_edge("predict_churn", "router")
        builder.add_edge("get_retention_offers", "router")
        builder.add_edge("log_interaction", "router")
        builder.add_edge("escalate", "router")

        return builder.compile()

    def _router(self, state: AgentState) -> AgentState:
        messages = state.get("messages", [])
        if not messages:
            return state

        system_msg = SystemMessage(content=SYSTEM_PROMPT)
        response = self.llm_with_tools.invoke([system_msg] + messages)

        state["messages"] = messages + [response]
        return state

    def _route_tool(self, state: AgentState) -> str:
        messages = state["messages"]
        if not messages:
            return "response"

        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            tool_call = last_msg.tool_calls[0]
            name = tool_call["name"]
            if name in {"lookup_customer", "predict_churn", "get_retention_offers",
                        "log_interaction", "escalate_to_supervisor"}:
                return {
                    "lookup_customer": "lookup_customer",
                    "predict_churn": "predict_churn",
                    "get_retention_offers": "get_retention_offers",
                    "log_interaction": "log_interaction",
                    "escalate_to_supervisor": "escalate",
                }[name]

        # Check if we've exceeded max tool calls (safety valve)
        tool_count = sum(1 for t in state.get("tool_calls_made", []) if t.get("name"))
        if tool_count >= 6:
            return "response"

        return "response"

    def _lookup_customer(self, state: AgentState) -> AgentState:
        tool_calls = state["messages"][-1].tool_calls
        tc = tool_calls[0]
        args = tc["args"]
        customer_id = args.get("customer_id", "")

        result = self._execute_lookup(customer_id)
        state["tool_calls_made"].append({
            "name": "lookup_customer",
            "args": args,
            "result": result,
        })
        if result.get("customer"):
            state["customer_profile"] = result["customer"]
            state["error"] = None
        else:
            state["error"] = result.get("error")

        content = json.dumps(result)
        state["messages"].append(ToolMessage(content=content, tool_call_id=tc["id"]))
        return state

    def _predict_churn(self, state: AgentState) -> AgentState:
        tool_calls = state["messages"][-1].tool_calls
        tc = tool_calls[0]
        args = tc["args"]
        customer_data = args.get("customer_data", state.get("customer_profile", {}))

        result = self._execute_predict(customer_data)
        state["tool_calls_made"].append({
            "name": "predict_churn",
            "args": args,
            "result": result,
        })
        if result.get("prediction"):
            state["churn_prediction"] = result["prediction"]

        content = json.dumps(result)
        state["messages"].append(ToolMessage(content=content, tool_call_id=tc["id"]))
        return state

    def _get_retention_offers(self, state: AgentState) -> AgentState:
        tool_calls = state["messages"][-1].tool_calls
        tc = tool_calls[0]
        args = tc["args"]
        risk_tier = args.get("risk_tier", "medium")
        contract_type = args.get("contract_type", "Month-to-month")

        offers = get_offers(risk_tier, contract_type)
        state["tool_calls_made"].append({
            "name": "get_retention_offers",
            "args": args,
            "result": {"offers": offers[:5], "risk_tier": risk_tier, "contract_type": contract_type},
        })
        state["retention_offers"] = offers[:5]

        content = json.dumps({"offers": offers[:5], "risk_tier": risk_tier, "contract_type": contract_type})
        state["messages"].append(ToolMessage(content=content, tool_call_id=tc["id"]))
        return state

    def _log_interaction(self, state: AgentState) -> AgentState:
        tool_calls = state["messages"][-1].tool_calls
        tc = tool_calls[0]
        args = tc["args"]
        log_entry = self._execute_log(args)
        state["tool_calls_made"].append({
            "name": "log_interaction",
            "args": args,
            "result": log_entry,
        })
        state["interaction_log"] = log_entry
        content = json.dumps(log_entry)
        state["messages"].append(ToolMessage(content=content, tool_call_id=tc["id"]))
        return state

    def _escalate(self, state: AgentState) -> AgentState:
        tool_calls = state["messages"][-1].tool_calls
        tc = tool_calls[0]
        args = tc["args"]
        result = self._execute_escalate(args)
        state["tool_calls_made"].append({
            "name": "escalate_to_supervisor",
            "args": args,
            "result": result,
        })
        state["escalation"] = result
        content = json.dumps(result)
        state["messages"].append(ToolMessage(content=content, tool_call_id=tc["id"]))
        return state

    def _response(self, state: AgentState) -> AgentState:
        messages = state.get("messages", [])
        tool_data = {
            "customer_profile": state.get("customer_profile"),
            "churn_prediction": state.get("churn_prediction"),
            "retention_offers": state.get("retention_offers"),
            "escalation": state.get("escalation"),
            "interaction_log": state.get("interaction_log"),
            "error": state.get("error"),
        }

        summary = f"""
Tool outputs collected:
{json.dumps(tool_data, indent=2, default=str)}

Based on all tool outputs above, synthesize a clear, actionable recommendation for the retention representative.
Be specific: name offers, cite risk factors, suggest exact next steps.
If there were errors or conflicting signals, explain them.
Keep it focused on what the rep should do right now.
"""
        system = SystemMessage(content=SYSTEM_PROMPT)
        response = self.llm.invoke([system] + messages + [HumanMessage(content=summary)])
        state["messages"].append(response)
        return state

    def invoke(self, user_message: str) -> dict[str, Any]:
        """Run agent on a single user message and return structured result."""
        graph = self.build_graph()
        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_message)],
            "tool_calls_made": [],
            "customer_profile": None,
            "churn_prediction": None,
            "retention_offers": None,
            "escalation": None,
            "interaction_log": None,
            "error": None,
        }
        result = graph.invoke(initial_state, config={"recursion_limit": 20})

        messages = result.get("messages", [])
        final_response = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not hasattr(msg, "tool_calls"):
                final_response = msg.content
                break
            elif isinstance(msg, AIMessage) and msg.content:
                final_response = msg.content
                break

        return {
            "response": final_response,
            "tool_calls_made": result.get("tool_calls_made", []),
            "customer_profile": result.get("customer_profile"),
            "churn_prediction": result.get("churn_prediction"),
            "retention_offers": result.get("retention_offers"),
            "escalation": result.get("escalation"),
            "interaction_log": result.get("interaction_log"),
            "error": result.get("error"),
        }

    def _execute_lookup(self, customer_id: str) -> dict[str, Any]:
        if self.df is None:
            return {"customer": {"customer_id": customer_id, "age": 42, "gender": "Male",
                    "tenure_months": 12.0, "contract_type": "Month-to-month",
                    "monthly_charges": 75.0, "total_charges": 900.0,
                    "internet_service": "Fiber optic", "phone_service": "Yes",
                    "avg_monthly_gb_used": 25.0, "num_support_tickets": 2.0,
                    "avg_monthly_minutes": 300.0, "satisfaction_score": 6.5,
                    "payment_method": "Credit card", "num_additional_services": 3,
                    "last_interaction_date": "2024-06-15"}, "error": None}

        row = self.df[self.df["customer_id"] == customer_id]
        if row.empty:
            return {"customer": None, "error": f"Customer {customer_id} not found"}

        r = row.iloc[0].to_dict()
        return {"customer": r, "error": None}

    def _execute_predict(self, customer_data: dict[str, Any]) -> dict[str, Any]:
        if self.predict_fn is None:
            return {
                "prediction": {
                    "churn_probability": 0.45,
                    "risk_tier": "medium",
                    "top_risk_factors": [
                        {"feature": "contract_type", "contribution": "Month-to-month increases churn risk"},
                        {"feature": "tenure_months", "contribution": "Low tenure correlates with churn"},
                        {"feature": "monthly_charges", "contribution": "High monthly charges signal price sensitivity"},
                    ],
                },
                "error": None,
            }

        try:
            result = self.predict_fn(customer_data)
            return {"prediction": result, "error": None}
        except Exception as e:
            return {"prediction": None, "error": str(e)}

    def _execute_log(self, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "log_id": f"LOG-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "logged",
            **args,
        }

    def _execute_escalate(self, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "escalation_id": f"ESC-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "escalated",
            "message": f"Case escalated — {args.get('reason', 'unspecified')}. Supervisor notified.",
            **args,
        }
