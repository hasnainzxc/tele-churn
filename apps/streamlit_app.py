"""Streamlit UI for TeleConnect Retention Agent.

Modern chat interface with:
- Branded dark/teal theme
- Risk gauge visualization
- Tool cards with icons
- Message bubble styling
- Integrated tool trace inspector
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()
from telechurn.agent.agent import RetentionAgent  # noqa: E402
from telechurn.predict import predict_churn  # noqa: E402

# ── Page config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TeleConnect Retention Agent",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme CSS ────────────────────────────────────────────────────────────

_STYLE_PATH = Path(__file__).parent / "style.css"
_CSS = _STYLE_PATH.read_text() if _STYLE_PATH.exists() else ""

# ── Tool metadata ────────────────────────────────────────────────────────

_TOOL_META: dict[str, dict[str, str]] = {
    "lookup_customer":        {"icon": "🔍", "label": "Customer Lookup",       "color": "#3b82f6"},
    "predict_churn":          {"icon": "📊", "label": "Churn Prediction",      "color": "#8b5cf6"},
    "get_retention_offers":   {"icon": "🎁", "label": "Retention Offers",      "color": "#f59e0b"},
    "log_interaction":        {"icon": "📝", "label": "Interaction Log",       "color": "#10b981"},
    "escalate_to_supervisor": {"icon": "🚨", "label": "Escalation",            "color": "#ef4444"},
}


def _risk_color(tier: str) -> tuple[str, str, str]:
    if tier == "high":
        return "#fca5a5", "high", "tier-high"
    if tier == "medium":
        return "#fde68a", "medium", "tier-medium"
    return "#86efac", "low", "tier-low"


def _render_risk_gauge(prediction: dict) -> None:
    proba = prediction.get("churn_probability", 0)
    tier = prediction.get("risk_tier", "unknown")
    factors = prediction.get("top_risk_factors", [])
    pct_color, _, tier_class = _risk_color(tier)

    html = f"""
    <div class="risk-gauge">
        <div class="label">Churn Probability</div>
        <div class="value-row">
            <span class="pct" style="color:{pct_color}">{proba * 100:.1f}%</span>
            <span class="tier-badge {tier_class}">{html.escape(tier.upper())} RISK</span>
        </div>
    """
    bar = (
        '<div style="background:#0f172a;border-radius:8px;'
        'height:8px;margin:0.5rem 0;overflow:hidden">'
    )
    html += bar
    fill = (
        '<div style="width:{}%;height:100%;'
        'background:linear-gradient(90deg,#22c55e,#fbbf24,#ef4444);'
        'border-radius:8px;transition:width 0.6s ease"></div>'
    ).format(proba * 100)
    html += fill
    html += "</div>"

    if factors:
        html += '<div class="factors">'
        for f in factors:
            html += f'<span class="factor-chip">{html.escape(f["feature"])}</span>'
        html += "</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _render_tool_card(call: dict, idx: int) -> None:
    name = call.get("name", "unknown_tool")
    meta = _TOOL_META.get(name, {"icon": "🔧", "label": name, "color": "#64748b"})

    html = f"""
    <div class="tool-card">
        <div class="tool-header">
            <span class="tool-icon">{meta["icon"]}</span>
            <span class="tool-name" style="color:{meta["color"]}">{meta["label"]}</span>
        </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Input")
        st.json(call.get("args", {}))
    with col_b:
        st.caption("Output")
        st.json(call.get("result", {}))


def _render_chat_message(role: str, content: str, tool_trace: list | None = None,
                          prediction: dict | None = None) -> None:
    bubble_class = "user" if role == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(
            f'<div class="chat-bubble {bubble_class}">{html.escape(content)}</div>',
            unsafe_allow_html=True,
        )

    if role == "assistant":
        if prediction and prediction.get("churn_probability", 0) > 0:
            _render_risk_gauge(prediction)

        if tool_trace:
            with st.expander("🔍 Tool Trace", expanded=bool(
                any(c["name"] == "escalate_to_supervisor" for c in tool_trace)
            )):
                for i, call in enumerate(tool_trace):
                    _render_tool_card(call, i)


def _extract_prediction(tool_trace: list[dict]) -> dict | None:
    for call in reversed(tool_trace):
        if call["name"] == "predict_churn":
            return call.get("result", {}).get("prediction") or call.get("result", {})
    return None


# ── Data loading ─────────────────────────────────────────────────────────

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "test_datafile.csv")


def _load_data() -> pd.DataFrame:
    if os.path.exists(_DATA_PATH):
        return pd.read_csv(_DATA_PATH)
    if os.path.exists("test_datafile.csv"):
        return pd.read_csv("test_datafile.csv")
    raise FileNotFoundError(
        f"Data file not found at {_DATA_PATH} or ./test_datafile.csv."
    )


# ── Welcome screen ───────────────────────────────────────────────────────

_WELCOME_PROMPTS = [
    ("👤 Look up", "Customer TC-004711"),
    ("📊 Full check", "Retention check for TC-004711"),
    ("⚠️ Risk review", "I have a high-risk customer — what offers?"),
    ("🚨 Escalate", "Customer TC-003427 threatens to sue"),
]


def _render_welcome() -> None:
    st.markdown("""
    <div class="welcome-screen">
        <div class="icon">📡</div>
        <h2>TeleConnect Retention Assistant</h2>
        <div class="desc">
            AI-powered churn prevention. Look up customers, predict churn risk,
            and get actionable retention offers — all in real time.
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    for i, (label, prompt) in enumerate(_WELCOME_PROMPTS):
        with cols[i % 2]:
            if st.button(f"{label}\n\n_{prompt}_", key=f"welcome_{i}", use_container_width=True):
                st.session_state.messages.append({
                    "role": "user", "content": prompt, "tool_trace": [],
                })
                st.rerun()


# ── Sidebar ──────────────────────────────────────────────────────────────

def _render_sidebar(model: str) -> str:
    with st.sidebar:
        st.markdown("### ⚙️ Model")
        model = st.selectbox(
            "LLM",
            ["openai/gpt-4o-mini", "openai/gpt-4o", "google/gemini-2.5-flash"],
            index=0,
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("### 📡 About")
        st.markdown("""
        **TeleConnect Retention Agent** uses a real XGBoost churn model
        trained on 5,050 customer records to predict churn risk and
        recommend retention offers.

        Built with LangGraph + OpenRouter.
        """)
        st.caption("Tool calls trace shown inline with each response.")
    return model


# ── Main ─────────────────────────────────────────────────────────────────

def _get_api_key() -> str | None:
    key = os.environ.get("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", "")
    if key:
        os.environ["OPENROUTER_API_KEY"] = key
        return key
    return None


def main() -> None:
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

    # ── Header ──
    st.markdown("""
    <div class="tele-header">
        <h1>📡 TeleConnect Retention Agent</h1>
        <div class="subtitle">AI-powered churn prevention &amp; retention</div>
    </div>
    """, unsafe_allow_html=True)

    api_key = _get_api_key()
    if not api_key:
        st.error(
            "**OPENROUTER_API_KEY not set.** "
            "Create a `.env` file or add it to Streamlit secrets."
        )
        st.code("OPENROUTER_API_KEY=sk-or-v1-...")
        return

    try:
        df = _load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    model = _render_sidebar("openai/gpt-4o-mini")

    # Session init — recreate agent when model changes.
    if "agent" not in st.session_state or st.session_state.get("_model") != model:
        st.session_state.agent = RetentionAgent(model=model, df=df, predict_fn=predict_churn)
        st.session_state._model = model

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Chat history ──
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            _render_welcome()
        else:
            for msg in st.session_state.messages:
                pred = _extract_prediction(msg.get("tool_trace", []))
                _render_chat_message(
                    msg["role"], msg["content"],
                    msg.get("tool_trace"), pred,
                )

    # ── Chat input ──
    if prompt := st.chat_input("Ask about a customer or retention strategy..."):
        st.session_state.messages.append({
            "role": "user", "content": prompt, "tool_trace": [],
        })

        with st.spinner(""):
            try:
                result = st.session_state.agent.invoke(prompt)
            except Exception as e:
                st.error(f"Agent error: {e}")
                result = {
                    "response": "Sorry, an error occurred. Please try again.",
                    "tool_calls_made": [],
                }

        tool_trace = result.get("tool_calls_made", [])
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["response"],
            "tool_trace": tool_trace,
        })

        st.rerun()


if __name__ == "__main__":
    main()
