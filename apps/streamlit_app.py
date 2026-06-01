"""Streamlit UI for TeleConnect Retention Agent.

Modern chat interface with:
- Model status indicator (green dot = live)
- Staggered tool trace animation
- Risk gauge visualization
- Suggested follow-up prompts
- Tool cards with step tracking
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

# ── CSS ──────────────────────────────────────────────────────────────────

_STYLE_PATH = Path(__file__).parent / "style.css"
_CSS = _STYLE_PATH.read_text() if _STYLE_PATH.exists() else ""

# ── Tool metadata ────────────────────────────────────────────────────────

_TOOL_META: dict[str, dict[str, str]] = {
    "lookup_customer":        {"icon": "🔍", "label": "Lookup Customer",      "color": "#3b82f6"},
    "predict_churn":          {"icon": "📊", "label": "Predict Churn",        "color": "#8b5cf6"},
    "get_retention_offers":   {"icon": "🎁", "label": "Retention Offers",     "color": "#f59e0b"},
    "log_interaction":        {"icon": "📝", "label": "Log Interaction",      "color": "#10b981"},
    "escalate_to_supervisor": {"icon": "🚨", "label": "Escalate",             "color": "#ef4444"},
}

_TOOL_ICONS = {k: v["icon"] for k, v in _TOOL_META.items()}

# ── Welcome prompts ──────────────────────────────────────────────────────

_WELCOME_PROMPTS = [
    ("👤 Look up customer", "Customer TC-004711"),
    ("📊 Full retention check", "Retention check for TC-003011"),
    ("⚠️ High risk review", "Customer TC-002385 is at risk, what can we offer?"),
    ("🚨 Escalate", "Customer TC-003427 threatens to sue"),
    ("📋 Batch review", "Review all high-risk customers"),
    ("❓ Unhappy customer", "I have an unhappy customer, help"),
    ("🔄 Contract upgrade", "Check TC-000527 for a contract upgrade offer"),
    ("📝 Log interaction", "Log a call with TC-001836, offered 10% discount"),
]

# ── Helpers ──────────────────────────────────────────────────────────────

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

    parts = [
        '<div class="risk-gauge">',
        '<div class="label">Churn Probability</div>',
        '<div class="value-row">',
        f'<span class="pct" style="color:{pct_color}">{proba * 100:.1f}%</span>',
        f'<span class="tier-badge {tier_class}">{html.escape(tier.upper())} RISK</span>',
        '</div>',
        ('<div style="background:#0f172a;border-radius:8px;'
         'height:8px;margin:0.5rem 0;overflow:hidden">'),
        ('<div style="width:{}%;height:100%;'
         'background:linear-gradient(90deg,#22c55e,#fbbf24,#ef4444);'
         'border-radius:8px;transition:width 0.6s ease"></div>'
         ).format(proba * 100),
        '</div>',
    ]
    if factors:
        parts.append('<div class="factors">')
        for f in factors:
            parts.append(
                f'<span class="factor-chip">{html.escape(f["feature"])}</span>'
            )
        parts.append('</div>')
    parts.append('</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def _render_tool_card(call: dict) -> None:
    name = call.get("name", "unknown_tool")
    meta = _TOOL_META.get(name, {"icon": "🔧", "label": name, "color": "#64748b"})

    st.markdown(
        f'<div class="tool-card">'
        f'<div class="tool-header">'
        f'<span class="tool-icon">{meta["icon"]}</span>'
        f'<span class="tool-name" style="color:{meta["color"]}">{meta["label"]}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Input")
        st.json(call.get("args", {}))
    with col_b:
        st.caption("Output")
        st.json(call.get("result", {}))


def _render_processing_steps(tool_trace: list[dict]) -> None:
    if not tool_trace:
        return
    icons = [_TOOL_ICONS.get(c["name"], "🔧") for c in tool_trace]
    labels = [_TOOL_META.get(c["name"], {}).get("label", c["name"]) for c in tool_trace]

    parts = ['<div class="processing-steps">']
    parts.append('<span class="step-label">Ran</span>')
    for i, (icon, label) in enumerate(zip(icons, labels)):
        if i > 0:
            parts.append('<span class="step-arrow">→</span>')
        parts.append(f'<span class="step-item">{icon} {label}</span>')
    parts.append('</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def _render_suggested_followups(tool_trace: list[dict]) -> None:
    tool_names = {c["name"] for c in tool_trace}
    suggestions = []

    if "predict_churn" in tool_names:
        suggestions.append("What offers are available for this customer?")
    if "lookup_customer" in tool_names and "predict_churn" not in tool_names:
        suggestions.append("Run a churn check on this customer")
    if "escalate_to_supervisor" in tool_names:
        suggestions.append("What happened with the escalation?")
    if "get_retention_offers" in tool_names:
        suggestions.append("Log this interaction for the record")
    if not suggestions:
        suggestions.append("Check another customer")

    st.markdown(
        '<div class="suggested-actions">'
        '<div class="label">Try next</div>'
        '<div class="chips">'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(len(suggestions))
    for i, s in enumerate(suggestions):
        with cols[i]:
            if st.button(s, key=f"followup_{s[:20]}_{len(st.session_state.messages)}"):
                st.session_state._pending = s
                st.rerun()


def _render_chat_message(
    role: str, content: str, tool_trace: list | None = None,
    prediction: dict | None = None,
) -> None:
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
            _render_processing_steps(tool_trace)
            with st.expander("🔍 Tool Trace", expanded=False):
                for call in tool_trace:
                    _render_tool_card(call)

            _render_suggested_followups(tool_trace)


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

def _render_welcome() -> None:
    st.markdown("""
    <div class="welcome-screen">
        <div class="icon">📡</div>
        <h2>TeleConnect Retention Assistant</h2>
        <div class="desc">
            AI-powered churn prevention. Look up customers, predict churn risk,
            and get actionable retention offers in real time.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="welcome-prompt-grid">', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (label, prompt) in enumerate(_WELCOME_PROMPTS):
        with cols[i % 2]:
            if st.button(label, key=f"welcome_{i}", help=prompt):
                st.session_state._pending = prompt
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


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
        trained on 5,050 customer records. It chains lookup, prediction,
        and retention offers into one conversation.

        Built with LangGraph + OpenRouter.
        """)
        st.caption("Each response shows what tools ran and in what order.")
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
        <div>
            <h1>📡 TeleConnect Retention Agent</h1>
            <div class="subtitle">AI-powered churn prevention &amp; retention</div>
        </div>
        <div class="header-right">
            <div class="status-dot">
                <span class="dot green"></span> Model active
            </div>
            <span class="header-badge">XGBoost 0.777 recall</span>
            <span class="header-badge">5,050 customers</span>
        </div>
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

    if "agent" not in st.session_state or st.session_state.get("_model") != model:
        st.session_state.agent = RetentionAgent(model=model, df=df, predict_fn=predict_churn)
        st.session_state._model = model

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "_pending" not in st.session_state:
        st.session_state._pending = None

    # ── Process pending prompt (from welcome buttons or chat input) ──
    pending = st.session_state._pending
    chat_prompt = st.chat_input("Ask about a customer or retention strategy...")
    prompt = pending or chat_prompt

    if prompt:
        if pending:
            st.session_state._pending = None
        else:
            st.session_state.messages.append({
                "role": "user", "content": prompt, "tool_trace": [],
            })

        with st.chat_message("user"):
            st.markdown(
                f'<div class="chat-bubble user">{html.escape(prompt)}</div>',
                unsafe_allow_html=True,
            )

        with st.status("Processing your request...", expanded=False) as status:
            try:
                st.write("Running agent pipeline...")
                result = st.session_state.agent.invoke(prompt)
                tool_trace = result.get("tool_calls_made", [])
                for i, call in enumerate(tool_trace):
                    meta = _TOOL_META.get(call["name"], {})
                    st.write(f"{meta.get('icon', '')} {meta.get('label', call['name'])} — done")
                step_count = "1 step" if len(tool_trace) == 1 else f"{len(tool_trace)} steps"
                status.update(label=f"Completed — {step_count}", state="complete")
            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"Agent error: {e}")
                result = {
                    "response": "Sorry, an error occurred. Please try again.",
                    "tool_calls_made": [],
                }
                tool_trace = []

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["response"],
            "tool_trace": tool_trace,
        })

        st.rerun()

    # ── Chat history ──
    if not st.session_state.messages:
        _render_welcome()
    else:
        for msg in st.session_state.messages:
            pred = _extract_prediction(msg.get("tool_trace", []))
            _render_chat_message(
                msg["role"], msg["content"],
                msg.get("tool_trace"), pred,
            )


if __name__ == "__main__":
    main()
