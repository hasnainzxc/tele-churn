"""Streamlit UI for TeleConnect Retention Agent.

Modern chat interface with:
- Branded dark/teal theme
- Risk gauge visualization
- Tool cards with icons
- Message bubble styling
- Integrated tool trace inspector
"""

from __future__ import annotations

import os
import sys

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

_CSS = """
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
}

/* ── Header ── */
.tele-header {
    padding: 1.5rem 2rem;
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border-bottom: 1px solid rgba(56, 189, 248, 0.15);
    margin-bottom: 1.5rem;
}
.tele-header h1 {
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    margin: 0;
}
.tele-header .subtitle {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 0.25rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a2332 0%, #15202b 100%);
    border-right: 1px solid rgba(56, 189, 248, 0.08);
}
section[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] h3 {
    color: #38bdf8 !important;
    font-weight: 600;
}

/* ── Chat messages ── */
.stChatMessage {
    background: transparent !important;
    padding: 0 !important;
}
.chat-bubble {
    max-width: 85%;
    padding: 1rem 1.25rem;
    border-radius: 16px;
    line-height: 1.6;
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    animation: fadeIn 0.3s ease;
}
.chat-bubble.user {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    color: #f1f5f9;
    margin-left: auto;
    margin-right: 1rem;
    border-bottom-right-radius: 4px;
}
.chat-bubble.assistant {
    background: #1e293b;
    color: #e2e8f0;
    border: 1px solid rgba(56, 189, 248, 0.15);
    margin-right: auto;
    margin-left: 1rem;
    border-bottom-left-radius: 4px;
}

/* ── Risk gauge ── */
.risk-gauge {
    background: #1e293b;
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 14px;
    padding: 1.25rem;
    margin: 0.75rem 1rem;
}
.risk-gauge .label {
    color: #94a3b8;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.5rem;
}
.risk-gauge .value-row {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}
.risk-gauge .pct {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'Inter', sans-serif;
}
.risk-gauge .tier-badge {
    display: inline-block;
    padding: 0.3rem 0.85rem;
    border-radius: 99px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.risk-gauge .tier-high {
    background: rgba(239, 68, 68, 0.15);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.25);
}
.risk-gauge .tier-medium {
    background: rgba(251, 191, 36, 0.15);
    color: #fde68a;
    border: 1px solid rgba(251, 191, 36, 0.25);
}
.risk-gauge .tier-low {
    background: rgba(34, 197, 94, 0.15);
    color: #86efac;
    border: 1px solid rgba(34, 197, 94, 0.25);
}
.risk-gauge .factors {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.6rem;
}
.risk-gauge .factor-chip {
    background: rgba(56, 189, 248, 0.08);
    color: #7dd3fc;
    border: 1px solid rgba(56, 189, 248, 0.15);
    padding: 0.25rem 0.65rem;
    border-radius: 99px;
    font-size: 0.78rem;
}

/* ── Tool cards ── */
.tool-card {
    background: #1a2332;
    border: 1px solid rgba(56, 189, 248, 0.1);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0;
    transition: border-color 0.2s;
}
.tool-card:hover {
    border-color: rgba(56, 189, 248, 0.3);
}
.tool-card .tool-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.65rem;
}
.tool-card .tool-icon {
    font-size: 1.1rem;
}
.tool-card .tool-name {
    font-weight: 600;
    font-size: 0.85rem;
    color: #38bdf8;
}
.tool-card .tool-detail {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.65rem;
}
.tool-card .tool-section-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    margin-bottom: 0.25rem;
}
.tool-card pre {
    background: #0f172a;
    border-radius: 6px;
    padding: 0.55rem 0.7rem;
    color: #94a3b8;
    font-size: 0.75rem;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
    margin: 0;
    max-height: 180px;
    overflow-y: auto;
}

/* ── Welcome screen ── */
.welcome-screen {
    text-align: center;
    padding: 4rem 2rem;
    max-width: 560px;
    margin: 0 auto;
}
.welcome-screen .icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
}
.welcome-screen h2 {
    color: #e2e8f0;
    font-weight: 700;
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}
.welcome-screen .desc {
    color: #94a3b8;
    font-size: 0.9rem;
    line-height: 1.6;
    margin-bottom: 2rem;
}
.welcome-prompt-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.65rem;
    max-width: 480px;
    margin: 0 auto;
}
.welcome-prompt-btn {
    background: #1e293b;
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    text-align: left;
    color: #cbd5e1;
    font-size: 0.82rem;
    cursor: pointer;
    transition: all 0.2s;
}
.welcome-prompt-btn:hover {
    border-color: #38bdf8;
    background: #233044;
}

/* ── Overrides ── */
.stChatInput textarea {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    border-radius: 12px !important;
}
.stChatInput textarea::placeholder {
    color: #475569 !important;
}
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
    gap: 0 !important;
}
.st-emotion-cache-1j04bjo, .st-emotion-cache-1gulkj5 {
    background: transparent !important;
}
.stMarkdown p, .stMarkdown li {
    color: #cbd5e1;
}
hr {
    border-color: rgba(56, 189, 248, 0.08) !important;
}

/* ── Animations ── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.5; }
}
</style>
"""

# ── Tool metadata ────────────────────────────────────────────────────────

_TOOL_META: dict[str, dict[str, str]] = {
    "lookup_customer":        {"icon": "🔍", "label": "Customer Lookup",       "color": "#3b82f6"},
    "predict_churn":          {"icon": "📊", "label": "Churn Prediction",      "color": "#8b5cf6"},
    "get_retention_offers":   {"icon": "🎁", "label": "Retention Offers",      "color": "#f59e0b"},
    "log_interaction":        {"icon": "📝", "label": "Interaction Log",       "color": "#10b981"},
    "escalate_to_supervisor": {"icon": "🚨", "label": "Escalation",            "color": "#ef4444"},
}


def _risk_color(proba: float) -> tuple[str, str, str]:
    if proba >= 0.6:
        return "#fca5a5", "high", "tier-high"
    if proba >= 0.3:
        return "#fde68a", "medium", "tier-medium"
    return "#86efac", "low", "tier-low"


def _render_risk_gauge(prediction: dict) -> None:
    proba = prediction.get("churn_probability", 0)
    tier = prediction.get("risk_tier", "unknown")
    factors = prediction.get("top_risk_factors", [])
    pct_color, _, tier_class = _risk_color(proba)

    html = f"""
    <div class="risk-gauge">
        <div class="label">Churn Probability</div>
        <div class="value-row">
            <span class="pct" style="color:{pct_color}">{proba * 100:.1f}%</span>
            <span class="tier-badge {tier_class}">{tier.upper()} RISK</span>
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
            html += f'<span class="factor-chip">{f["feature"]}</span>'
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
            f'<div class="chat-bubble {bubble_class}">{content}</div>',
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
    st.markdown(_CSS, unsafe_allow_html=True)

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
