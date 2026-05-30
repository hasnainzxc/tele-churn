"""Streamlit UI for TeleConnect Retention Agent.

Shows:
- Chat interface for retention reps
- Tool call trace panel (what was called, in what order, what was returned)
- Full agent response
"""

import os
import sys

import pandas as pd
import streamlit as st

# Inject src/ into path so we can import the telechurn package directly.
# Q: why not install the package? A: keeps dev iteration loop fast.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

# load_dotenv() runs before telechurn imports — agent module reads env at import time
load_dotenv()
from telechurn.agent.agent import RetentionAgent  # noqa: E402
from telechurn.predict import predict_churn  # noqa: E402

st.set_page_config(page_title="TeleConnect Retention Agent", page_icon="📞", layout="wide")
st.title("📞 TeleConnect Retention Agent")
st.caption("AI-powered retention assistant for customer representatives")

with st.sidebar:
    st.header("⚙️ Configuration")
    # Model dropdown — could add more from OpenRouter but these three cover
    # the common quality/cost tiers well enough.
    model = st.selectbox("Model", ["openai/gpt-4o-mini", "openai/gpt-4o", "google/gemini-2.5-flash"], index=0)
    st.divider()
    st.header("📋 Instructions")
    st.markdown("""
    **Try these prompts:**
    - "Look up customer TC-004711"
    - "Full retention check for TC-004711"
    - "I have a high-risk customer — what offers?"
    - "Customer TC-003427 threatens to sue"
    - "Check churn for TC-000034"
    """)
    st.divider()
    st.caption("Tool calls are traced in the right panel.")

# Resolves from apps/streamlit_app.py up to project root where test_datafile.csv lives.
# ../test_datafile.csv = /home/.../tele-churn/test_datafile.csv
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "test_datafile.csv")


def load_data() -> pd.DataFrame:
    # Try the resolved DATA_PATH first, fall back to CWD for flexibility
    # (e.g. if someone runs it from the project root directly).
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.read_csv("test_datafile.csv")


def main() -> None:
    # Fail early if no API key — no point rendering the full UI without it.
    # Check .env first, then Streamlit Cloud secrets (st.secrets).
    # Set os.environ so agent/judge modules can find it too.
    api_key = os.environ.get("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", "")
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
    else:
        st.error("OPENROUTER_API_KEY not set. Create a .env file with your API key.")
        st.code("OPENROUTER_API_KEY=sk-or-v1-...")
        return

    df = load_data()

    # Session state: keep the agent and chat history alive across reruns.
    # Streamlit re-runs the entire script on every interaction, so we stash
    # stateful objects here to persist them.

    # Lazy-init agent — only create it once, reusing the DataFrame.
    # Note: changing the model dropdown DOESN'T update an existing agent.
    # That's a known quirk — you need to clear cache or add a model-changed
    # check if it matters.
    if "agent" not in st.session_state:
        st.session_state.agent = RetentionAgent(model=model, df=df, predict_fn=predict_churn)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Two-column layout: chat on the left, tool trace on the right (3:2 ratio)
    col_chat, col_trace = st.columns([3, 2])

    # ── Left column: Chat ──────────────────────────────────────────────
    with col_chat:
        st.subheader("💬 Chat")

        # Render existing messages from history — messages already have their
        # tool traces embedded, so we show those inline too.
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("tool_trace"):
                    with st.expander("🔧 Tool Calls", expanded=False):
                        for call in msg["tool_trace"]:
                            st.json(call)

        # Chat input at the bottom — returns the prompt string or None if empty
        if prompt := st.chat_input("Type your message..."):
            # Append user message to session history immediately
            st.session_state.messages.append({"role": "user", "content": prompt, "tool_trace": []})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.agent.invoke(prompt)

                # Show the agent's final text response
                st.markdown(result["response"])

                # Show tool call trace in a collapsed expander within the message
                # (also displayed in the right panel, but keeping it inline is
                # convenient for scrolling context)
                tool_trace = result.get("tool_calls_made", [])
                if tool_trace:
                    with st.expander("🔧 Tool Calls", expanded=True):
                        for i, call in enumerate(tool_trace):
                            st.caption(f"Step {i + 1}: **{call['name']}**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.caption("Args")
                                st.json(call.get("args", {}))
                            with col2:
                                st.caption("Result")
                                st.json(call.get("result", {}))

            # Append assistant message (with tool trace) to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["response"],
                "tool_trace": tool_trace,
            })

    # ── Right column: Tool Trace Panel ──────────────────────────────────
    with col_trace:
        st.subheader("🔍 Tool Trace")

        # Show trace for the most recent assistant message only.
        # If the last interaction had no tool calls, show a hint.
        if st.session_state.messages:
            last = st.session_state.messages[-1]
            if last["role"] == "assistant" and last.get("tool_trace"):
                for i, call in enumerate(last["tool_trace"]):
                    # border=True gives each step a visual card — easier to scan
                    with st.container(border=True):
                        st.markdown(f"**Step {i + 1}: `{call['name']}`**")
                        st.caption("Input")
                        st.json(call.get("args", {}))
                        st.caption("Output")
                        st.json(call.get("result", {}))
            else:
                st.info("No tool calls in last interaction. Send a message to see the trace.")
        else:
            st.info("Send a message to see the tool call trace here.")


if __name__ == "__main__":
    main()
