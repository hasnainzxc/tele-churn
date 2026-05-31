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
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

_load_dotenv = load_dotenv(Path(__file__).resolve().parents[3] / ".env")

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from .offers import get_offers

# This prompt is the whole steering wheel. The LLM gets it as a SystemMessage
# every single turn (router + response nodes). Small wording changes here
# can drastically shift behaviour, so test after any edits.
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
- If no customer_id is provided, DO NOT call any tools. Respond asking for the customer ID (format TC-XXXXXX). Never call lookup_customer with an empty or guessed ID.
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
    # messages: the full conversation transcript (Human, AI, System, Tool messages).
    # LangGraph uses this as a rolling context; we append to it in every node.
    messages: list[Any]
    # Track every tool invocation so the UI can render a trace panel.
    tool_calls_made: list[dict[str, Any]]
    # Accumulated results from tools — populated lazily as the graph runs.
    customer_profile: dict[str, Any] | None
    churn_prediction: dict[str, Any] | None
    retention_offers: list[dict[str, Any]] | None
    escalation: dict[str, Any] | None
    interaction_log: dict[str, Any] | None
    error: str | None


class RetentionAgent:
    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.3,
        df: pd.DataFrame | None = None,
        predict_fn: Callable | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.df = df
        self.predict_fn = predict_fn
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Set it in .env or st.secrets."
            )
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
        # bind_tools turns TOOL_DESCRIPTIONS into OpenAI-compatible function-calling format.
        # This is what lets the LLM emit structured tool_call blocks instead of free text.
        self.llm_with_tools = self.llm.bind_tools(TOOL_DESCRIPTIONS)

    def build_graph(self) -> CompiledStateGraph:
        """Build and compile the ReAct-style state graph. Called once per invoke()."""
        builder = StateGraph(AgentState)

        # Register nodes — one per tool + router + response synth.
        builder.add_node("router", self._router)
        builder.add_node("lookup_customer", self._lookup_customer)
        builder.add_node("predict_churn", self._predict_churn)
        builder.add_node("get_retention_offers", self._get_retention_offers)
        builder.add_node("log_interaction", self._log_interaction)
        builder.add_node("escalate", self._escalate)
        builder.add_node("response", self._response)

        # Entry point — router always fires first.
        builder.add_edge(START, "router")

        # Conditional edges: router decides which tool node (or response/END) to jump to.
        builder.add_conditional_edges("router", self._route_tool, {
            "lookup_customer": "lookup_customer",
            "predict_churn": "predict_churn",
            "get_retention_offers": "get_retention_offers",
            "log_interaction": "log_interaction",
            "escalate": "escalate",
            "response": "response",
            END: END,
        })

        # Every tool node loops back to the router after execution.
        # This is the ReAct loop: think → act → observe → think again.
        builder.add_edge("lookup_customer", "router")
        builder.add_edge("predict_churn", "router")
        builder.add_edge("get_retention_offers", "router")
        builder.add_edge("log_interaction", "router")
        builder.add_edge("escalate", "router")

        return builder.compile()

    def _router(self, state: AgentState) -> AgentState:
        """Feed the LLM the full message history + system prompt, get back a decision.

        The LLM either emits tool_calls or a text response. We append whatever it says
        to the message list and let _route_tool decide what to do with it next.
        """
        messages = state.get("messages", [])
        if not messages:
            return state

        # Always re-attach the system prompt. LangChain doesn't persist it across
        # invocations, and we need it fresh every trip through the router.
        system_msg = SystemMessage(content=SYSTEM_PROMPT)
        response = self.llm_with_tools.invoke([system_msg] + messages)

        state["messages"] = messages + [response]
        return state

    def _route_tool(self, state: AgentState) -> str:
        """Inspect the LLM's last message and return the next node name.

        If the LLM emitted a tool_call → route to the matching tool node.
        If the LLM emitted plain text → route to "response" to synthesize final answer.
        """
        messages = state["messages"]
        if not messages:
            return "response"

        last_msg = messages[-1]
        # LLM asked to call a tool — map the tool name to our internal node name.
        if isinstance(last_msg, AIMessage) and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            tool_call = last_msg.tool_calls[0]
            name = tool_call["name"]
            if name in {"lookup_customer", "predict_churn", "get_retention_offers",
                        "log_interaction", "escalate_to_supervisor"}:
                # escalate_to_supervisor → "escalate" is a mismatch between the LLM-facing
                # tool name and the internal graph node name. This mapping bridges it.
                return {
                    "lookup_customer": "lookup_customer",
                    "predict_churn": "predict_churn",
                    "get_retention_offers": "get_retention_offers",
                    "log_interaction": "log_interaction",
                    "escalate_to_supervisor": "escalate",
                }[name]

        # Safety valve: if the LLM gets stuck in a tool-calling loop, force it to
        # synthesize a response after 6 tool invocations. Prevents infinite ReAct loops.
        tool_count = sum(1 for t in state.get("tool_calls_made", []) if t.get("name"))
        if tool_count >= 6:
            return "response"

        return "response"

    def _extract_tool_call(self, state: AgentState) -> tuple[dict, dict] | None:
        """Pull tool_call metadata from the last AIMessage. Returns (tc, args) or None."""
        last_msg = state["messages"][-1]
        if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
            return None
        tc = last_msg.tool_calls[0]
        return tc, tc["args"]

    def _record_tool(
        self, state: AgentState, tool_name: str,
        tc: dict, args: dict, result: dict,
        stash_map: dict[str | None, str] | None = None,
    ):
        """Record trace, stash result keys into state, send ToolMessage back to LLM."""
        state["tool_calls_made"].append({
            "name": tool_name, "args": args, "result": result,
        })
        if stash_map:
            for result_key, state_key in stash_map.items():
                if result_key is None:
                    state[state_key] = result
                elif result_key in result:
                    state[state_key] = result[result_key]
        state["messages"].append(ToolMessage(
            content=json.dumps(result), tool_call_id=tc["id"],
        ))

    def _lookup_customer(self, state: AgentState) -> AgentState:
        extracted = self._extract_tool_call(state)
        if extracted is None:
            return state
        tc, args = extracted

        customer_id = (args.get("customer_id", "") or "").strip()
        if not customer_id:
            result = {
                "customer": None,
                "error": "Missing customer ID. Ask for a valid ID (format TC-XXXXXX).",
            }
        else:
            result = self._execute_lookup(customer_id)

        self._record_tool(state, "lookup_customer", tc, args, result,
                          stash_map={"customer": "customer_profile", "error": "error"})
        return state

    def _predict_churn(self, state: AgentState) -> AgentState:
        extracted = self._extract_tool_call(state)
        if extracted is None:
            return state
        tc, args = extracted

        customer_data = args.get("customer_data", state.get("customer_profile", {}))
        result = self._execute_predict(customer_data)
        self._record_tool(state, "predict_churn", tc, args, result,
                          stash_map={"prediction": "churn_prediction"})
        return state

    def _get_retention_offers(self, state: AgentState) -> AgentState:
        extracted = self._extract_tool_call(state)
        if extracted is None:
            return state
        tc, args = extracted

        risk_tier = args.get("risk_tier", "medium")
        contract_type = args.get("contract_type", "Month-to-month")
        offers = get_offers(risk_tier, contract_type)
        result = {"offers": offers[:5], "risk_tier": risk_tier, "contract_type": contract_type}
        self._record_tool(state, "get_retention_offers", tc, args, result,
                          stash_map={"offers": "retention_offers"})
        return state

    def _log_interaction(self, state: AgentState) -> AgentState:
        extracted = self._extract_tool_call(state)
        if extracted is None:
            return state
        tc, args = extracted
        result = self._execute_log(args)
        self._record_tool(state, "log_interaction", tc, args, result,
                          stash_map={None: "interaction_log"})
        return state

    def _escalate(self, state: AgentState) -> AgentState:
        extracted = self._extract_tool_call(state)
        if extracted is None:
            return state
        tc, args = extracted
        result = self._execute_escalate(args)
        self._record_tool(state, "escalate_to_supervisor", tc, args, result,
                          stash_map={None: "escalation"})
        return state

    def _response(self, state: AgentState) -> AgentState:
        """Final synthesis node. Dumps all accumulated tool results into a prompt
        and asks the LLM to produce a cohesive recommendation for the rep."""
        messages = state.get("messages", [])

        # Gather everything the tools produced across the conversation.
        tool_data = {
            "customer_profile": state.get("customer_profile"),
            "churn_prediction": state.get("churn_prediction"),
            "retention_offers": state.get("retention_offers"),
            "escalation": state.get("escalation"),
            "interaction_log": state.get("interaction_log"),
            "error": state.get("error"),
        }

        # The summary template is deliberately detailed — it acts as a "here's what
        # you know, now tell the rep what to do" instruction. Without this the LLM
        # sometimes forgets tool results and goes off-script.
        summary = f"""
Tool outputs collected:
{json.dumps(tool_data, indent=2, default=str)}

Based on all tool outputs above, synthesize a clear, actionable recommendation for the retention representative.
Be specific: name offers, cite risk factors, suggest exact next steps.
If there were errors or conflicting signals, explain them.
Keep it focused on what the rep should do right now.
"""
        system = SystemMessage(content=SYSTEM_PROMPT)
        # Use raw llm (not llm_with_tools) here — we don't want the LLM emitting
        # tool calls in the final response; it should just produce text.
        response = self.llm.invoke([system] + messages + [HumanMessage(content=summary)])
        state["messages"].append(response)
        return state

    def invoke(self, user_message: str) -> dict[str, Any]:
        """Run agent on a single user message and return structured result."""
        graph = self.build_graph()
        # Initial state — empty except for the user's message.
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
        # recursion_limit=20 is generous but safe. Standard ReAct loop is 1-4 tools.
        result = graph.invoke(initial_state, config={"recursion_limit": 20})

        # Extract the final assistant response from the message list.
        # Walk backwards to find the last AIMessage with content (skipping tool-only ones).
        messages = result.get("messages", [])
        final_response = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not hasattr(msg, "tool_calls"):
                final_response = msg.content
                break
            elif isinstance(msg, AIMessage) and msg.content:
                # Fallback: an AIMessage that has both content and tool_calls.
                # Use the content portion — the LLM sometimes writes a partial response.
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
        """Look up customer in the DataFrame or return a hardcoded mock if no df supplied."""
        # No DataFrame injected → return a canned mock profile for demo/eval.
        # This is hacky but lets the agent run without a real data pipeline.
        if self.df is None:
            return {"customer": {"customer_id": customer_id, "age": 42, "gender": "Male",
                    "tenure_months": 12.0, "contract_type": "Month-to-month",
                    "monthly_charges": 75.0, "total_charges": 900.0,
                    "internet_service": "Fiber optic", "phone_service": "Yes",
                    "avg_monthly_gb_used": 25.0, "num_support_tickets": 2.0,
                    "avg_monthly_minutes": 300.0, "satisfaction_score": 6.5,
                    "payment_method": "Credit card", "num_additional_services": 3,
                    "last_interaction_date": "2024-06-15"}, "error": None}

        # Real lookup path: filter the DataFrame by customer_id.
        row = self.df[self.df["customer_id"] == customer_id]
        if row.empty:
            return {"customer": None, "error": f"Customer {customer_id} not found"}

        # iloc[0] grabs the first match — assumes customer_id is unique.
        r = row.iloc[0].to_dict()
        return {"customer": r, "error": None}

    def _execute_predict(self, customer_data: dict[str, Any]) -> dict[str, Any]:
        """Run churn prediction. Falls back to a static mock if no predict_fn injected."""
        # No model → return synthetic prediction. Risk tier and factors are hardcoded
        # to "medium" and generic reasons. Real predict_fn would return actual scores.
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

        # Real model path: delegate to the injected predict_fn.
        try:
            result = self.predict_fn(customer_data)
            return {"prediction": result, "error": None}
        except Exception as e:
            return {"prediction": None, "error": str(e)}

    def _execute_log(self, args: dict[str, Any]) -> dict[str, Any]:
        """Generate a log entry with a UUID and UTC timestamp. No persistence — ephemeral."""
        return {
            "log_id": f"LOG-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "logged",
            **args,  # spread the LLM-provided args (outcome, notes, etc.) into the entry
        }

    def _execute_escalate(self, args: dict[str, Any]) -> dict[str, Any]:
        """Generate an escalation ticket. Also ephemeral — real version hits a ticketing system."""
        return {
            "escalation_id": f"ESC-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "escalated",
            "message": f"Case escalated — {args.get('reason', 'unspecified')}. Supervisor notified.",
            **args,
        }
