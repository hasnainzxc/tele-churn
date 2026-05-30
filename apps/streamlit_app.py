"""Streamlit UI for TeleConnect Retention Agent.

Shows:
- Chat interface for retention reps
- Tool call trace panel (what was called, in what order, what was returned)
- Full agent response
"""

import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()
from telechurn.agent.agent import RetentionAgent

st.set_page_config(page_title="TeleConnect Retention Agent", page_icon="📞", layout="wide")
st.title("📞 TeleConnect Retention Agent")
st.caption("AI-powered retention assistant for customer representatives")

with st.sidebar:
    st.header("⚙️ Configuration")
    model = st.selectbox("Model", ["anthropic/claude-3.5-sonnet", "openai/gpt-4o", "openai/gpt-4o-mini"], index=0)
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

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "test_datafile.csv")

def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.read_csv("test_datafile.csv")


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        st.error("OPENROUTER_API_KEY not set. Create a .env file with your API key.")
        st.code("OPENROUTER_API_KEY=sk-or-v1-...")
        return

    df = load_data()

    if "agent" not in st.session_state:
        st.session_state.agent = RetentionAgent(model=model, df=df)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    col_chat, col_trace = st.columns([3, 2])

    with col_chat:
        st.subheader("💬 Chat")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("tool_trace"):
                    with st.expander("🔧 Tool Calls", expanded=False):
                        for call in msg["tool_trace"]:
                            st.json(call)

        if prompt := st.chat_input("Type your message..."):
            st.session_state.messages.append({"role": "user", "content": prompt, "tool_trace": []})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.agent.invoke(prompt)

                st.markdown(result["response"])

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

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["response"],
                "tool_trace": tool_trace,
            })

    with col_trace:
        st.subheader("🔍 Tool Trace")

        if st.session_state.messages:
            last = st.session_state.messages[-1]
            if last["role"] == "assistant" and last.get("tool_trace"):
                for i, call in enumerate(last["tool_trace"]):
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
