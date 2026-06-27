import streamlit as st
import os
from dotenv import load_dotenv

# ─── Load Environment Variables ───────────────────────────────────────────────
load_dotenv()

# Pre-import verification to prevent start-up crash if API key is missing
api_key = os.getenv("HuggingFace_API_KEY")
if not api_key:
    st.set_page_config(page_title="Nexa Verify Configuration Error", page_icon="🚨")
    st.error("🚨 **Configuration Error**: `HuggingFace_API_KEY` is missing from your `.env` file.")
    st.info("Please set your Hugging Face API key inside the `.env` file in the project folder to start the application.")
    st.stop()

# Safe to import backend after verified env variables
import self_rag_backend

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nexa Verify · RAG Agent",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS Theme ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

/* Main Theme Colors and Background */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #070a13 !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] > .main {
    background-color: #070a13 !important;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #03050a !important;
    border-right: 1px solid #1e293b;
}

/* Glassmorphism visual cards */
.glass-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
}

.brand-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    background: linear-gradient(90deg, #00e5ff 0%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.brand-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #64748b;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* Custom Chat Bubbles */
.chat-msg {
    padding: 1rem 1.25rem;
    border-radius: 8px;
    margin-bottom: 0.85rem;
    font-size: 0.95rem;
    line-height: 1.6;
    border: 1px solid rgba(255, 255, 255, 0.03);
}

.chat-msg-user {
    background-color: #1e293b;
    border-left: 4px solid #00e5ff;
}

.chat-msg-agent {
    background-color: #0f172a;
    border-left: 4px solid #10b981;
}

/* Step Badges */
.badge-active {
    display: inline-block;
    background-color: rgba(6, 182, 212, 0.15);
    color: #00e5ff;
    border: 1px solid rgba(6, 182, 212, 0.3);
    padding: 0.25rem 0.6rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    border-radius: 4px;
    font-weight: bold;
}

.badge-success {
    display: inline-block;
    background-color: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 0.25rem 0.6rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    border-radius: 4px;
    font-weight: bold;
}

.badge-error {
    display: inline-block;
    background-color: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
    padding: 0.25rem 0.6rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    border-radius: 4px;
    font-weight: bold;
}

/* Pulsing Status Animation */
@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}
</style>
""", unsafe_allow_html=True)

# ─── Initialize Session States ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "active_state" not in st.session_state:
    st.session_state["active_state"] = None
if "processed_uploads" not in st.session_state:
    st.session_state["processed_uploads"] = None
if "using_defaults" not in st.session_state:
    st.session_state["using_defaults"] = True

# ─── Sidebar Configuration & File Uploader ────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand-title">⬡ NEXA VERIFY</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Smart RAG Agent</div>', unsafe_allow_html=True)

    # Agent Online status indicator
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:2rem;">
        <div style="width:9px;height:9px;border-radius:50%;background:#10b981;
            box-shadow:0 0 10px #10b981;animation:pulse 2s infinite;"></div>
        <span style="font-family:'Space Mono',monospace;font-size:0.75rem;
            color:#10b981;letter-spacing:0.1em;font-weight:700;">AGENT ONLINE</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📄 Document Uploader")
    uploaded_files = st.file_uploader(
        "Upload Custom PDFs to Query",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files. The vector database will re-initialize automatically."
    )

    # Process uploads
    if uploaded_files:
        upload_dir = os.path.join(os.path.dirname(os.path.abspath(self_rag_backend.__file__)), "uploaded_docs")
        os.makedirs(upload_dir, exist_ok=True)
        
        uploaded_names = [f.name for f in uploaded_files]
        if st.session_state["processed_uploads"] != uploaded_names:
            # Clear old uploads first
            for filename in os.listdir(upload_dir):
                try:
                    os.remove(os.path.join(upload_dir, filename))
                except Exception:
                    pass

            uploaded_paths = []
            for f in uploaded_files:
                p = os.path.join(upload_dir, f.name)
                with open(p, "wb") as out_f:
                    out_f.write(f.read())
                uploaded_paths.append(p)
                
            with st.spinner("⚡ Re-indexing knowledge database..."):
                self_rag_backend.build_vector_store(uploaded_paths)
                st.session_state["processed_uploads"] = uploaded_names
                st.session_state["using_defaults"] = False
            st.success("Database Re-indexed!")
    else:
        # Revert back to default PDFs if uploader was cleared
        if st.session_state["processed_uploads"] is not None or st.session_state["using_defaults"] is False:
            with st.spinner("🔄 Restoring default system PDFs..."):
                self_rag_backend.build_vector_store(self_rag_backend.DEFAULT_PDFS)
                st.session_state["processed_uploads"] = None
                st.session_state["using_defaults"] = True
            st.info("Reverted to default documents.")

    # Show Loaded Files in Sidebar
    st.markdown("---")
    st.markdown("#### 📂 Current Agent Knowledge")
    if st.session_state["processed_uploads"]:
        for name in st.session_state["processed_uploads"]:
            st.caption(f"✓ 📄 {name}")
    else:
        st.caption("✓ 📄 Company_Policies.pdf")
        st.caption("✓ 📄 Company_Profile.pdf")
        st.caption("✓ 📄 Product_and_Pricing.pdf")
        
    st.markdown("---")
    # Reset conversation button
    if st.button("✕ Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["active_state"] = None
        st.rerun()

# ─── Main Panel Layout ─────────────────────────────────────────────────────────
col_chat, col_mind = st.columns([5, 4])

# Column 1: Chat Workspace
with col_chat:
    st.markdown("### 💬 Chat Workspace")
    
    # Message Display
    for msg in st.session_state["messages"]:
        role_class = "chat-msg-user" if msg["role"] == "user" else "chat-msg-agent"
        role_label = "👤 YOU" if msg["role"] == "user" else "🤖 NEXA AGENT"
        st.markdown(f"""
        <div class="chat-msg {role_class}">
            <strong style="font-family:'Space Mono',monospace; font-size:0.75rem; letter-spacing:0.05em; display:block; margin-bottom:0.4rem;">{role_label}</strong>
            {msg["content"]}
        </div>
        """, unsafe_allow_html=True)

    # Chat Input
    if prompt := st.chat_input("Ask a question about the active documents..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.rerun()

# Execute query if last message is from user
if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
    user_prompt = st.session_state["messages"][-1]["content"]
    
    with col_chat:
        with st.spinner("🤖 Processing query through RAG agent..."):
            try:
                # Call compiled LangGraph workflow from backend
                result = self_rag_backend.app.invoke({
                    "question": user_prompt,
                    "docs": [],
                    "relevant_docs": [],
                    "context": "",
                    "answer": "",
                    "retries": 0
                })
                
                answer = result.get("answer", "No response generated.")
                st.session_state["active_state"] = result
                st.session_state["messages"].append({"role": "assistant", "content": answer})
                st.rerun()
            except Exception as e:
                st.error(f"Agent Execution Failure: {e}")
                st.session_state["messages"].append({"role": "assistant", "content": f"Sorry, I failed to complete the request: {e}"})
                st.rerun()

# Column 2: Agent Mind / Pipeline Monitor
with col_mind:
    st.markdown("### 🧠 Agent Decision Monitor")
    
    if st.session_state["active_state"]:
        state = st.session_state["active_state"]
        
        need_retrieval = state.get("need_retrieval", False)
        issup = state.get("issup", "no_support")
        isuse = state.get("isuse", "not_useful")
        retries = state.get("retries", 0)
        context = state.get("context", "")
        evidence = state.get("evidence", [])
        
        # 1. Routing Decision Card
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h4 style="margin-top:0; color:#00e5ff;">🎯 Pipeline Entry Decision</h4>', unsafe_allow_html=True)
            if need_retrieval:
                st.markdown('<span class="badge-active">RETRIEVAL MODE</span>', unsafe_allow_html=True)
                st.write("Agent analyzed query and decided that knowledge base retrieval is required.")
            else:
                st.markdown('<span class="badge-success">DIRECT MODE</span>', unsafe_allow_html=True)
                st.write("Agent resolved question directly using general knowledge without database retrieval.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # 2. Fact-Checking Grounding and Revisions Card
        if need_retrieval:
            with st.container():
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<h4 style="margin-top:0; color:#10b981;">⚖️ Corrective Verification loops</h4>', unsafe_allow_html=True)
                
                # Grounding support status
                sup_labels = {
                    "fully_supported": ("Fully Supported", "badge-success"),
                    "partially_supported": ("Partially Grounded", "badge-active"),
                    "no_support": ("Ungrounded / Hallucinated", "badge-error")
                }
                label, badge_class = sup_labels.get(issup, ("Unknown", "badge-error"))
                
                st.markdown(f"**Grounding Quality:** <span class='{badge_class}'>{label}</span>", unsafe_allow_html=True)
                st.write(f"**Self-Correction Revisions:** `{retries}` revision loops executed.")
                
                # Usefulness check status
                use_badge = "badge-success" if isuse == "useful" else "badge-error"
                st.markdown(f"**Relevance to Question:** <span class='{use_badge}'>{isuse.upper()}</span>", unsafe_allow_html=True)
                if state.get("use_reason"):
                    st.caption(f"*Evaluation Reason:* {state['use_reason']}")
                st.markdown('</div>', unsafe_allow_html=True)

            # 3. Grounding Quotes Evidence
            if evidence:
                with st.expander("🔍 Supporting Quotes (Evidence Chunks)", expanded=True):
                    for ev in evidence:
                        st.info(f"“ {ev} ”")
                        
            # 4. Context Chunks
            if context:
                with st.expander("📄 Raw Chunks Retrieved"):
                    chunks = context.split("\n\n---\n\n")
                    for i, ch in enumerate(chunks):
                        st.markdown(f"**Chunk {i+1:02d}**")
                        st.code(ch.strip(), language="text")
    else:
        st.info("Submit a question in the workspace to monitor the agent's decision-making flow in real time.")
