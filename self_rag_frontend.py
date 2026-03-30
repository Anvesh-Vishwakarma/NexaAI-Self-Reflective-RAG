import streamlit as st
import time

# Import your backend
from self_rag_backend import app

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Self RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# ----------------------------
# Custom CSS (Professional UI)
# ----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
    .answer-box {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    show_context = st.toggle("Show Retrieved Context", value=False)
    show_evidence = st.toggle("Show Evidence", value=True)
    st.markdown("---")
    st.info("Self-Reflection RAG System\n\n- Retrieval + Verification\n- Answer Revision Loop")

# ----------------------------
# Session State
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Title
# ----------------------------
st.title("🤖 Self Fact-Check RAG Assistant")
st.caption("Ask questions about company documents with verified answers")

# ----------------------------
# Display Chat History
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# User Input
# ----------------------------
if prompt := st.chat_input("Ask your question..."):

    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Simulate typing animation
        full_response = ""
        
        # Call backend
        result = app.invoke({
            "question": prompt,
            "docs": [],
            "relevant_docs": [],
            "context": "",
            "answer": "",
            "retries": 0
        })

        answer = result.get("answer", "No response generated.")
        context = result.get("context", "")
        evidence = result.get("evidence", [])

        # Streaming effect
        for chunk in answer.split():
            full_response += chunk + " "
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.02)

        message_placeholder.markdown(full_response)

        # Optional Context
        if show_context and context:
            with st.expander("📄 Retrieved Context"):
                st.write(context)

        # Optional Evidence
        if show_evidence and evidence:
            with st.expander("✅ Supporting Evidence"):
                for ev in evidence:
                    st.markdown(f"- {ev}")

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })