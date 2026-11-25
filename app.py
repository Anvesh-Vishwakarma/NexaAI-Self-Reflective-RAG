import html
import os
import time
import uuid
from collections import Counter
from typing import Any

import streamlit as st
from dotenv import load_dotenv

# ─── Load Environment Variables ───────────────────────────────────────────────
load_dotenv()

# Pre-import verification to prevent start-up crash if API key is missing
api_key = os.getenv("GROQ_API_KEY") or os.getenv("HuggingFace_API_KEY")
if not api_key:
    st.set_page_config(page_title="Nexa Verify Configuration Error", page_icon="🚨")
    st.error("🚨 **Configuration Error**: `GROQ_API_KEY` (or `HuggingFace_API_KEY`) is missing from your `.env` file.")
    st.info("Please set your API key inside the `.env` file in the project folder to start the application.")
    st.stop()

# Safe to import backend after verified env variables
import self_rag_backend

BASE_DIR = os.path.dirname(os.path.abspath(self_rag_backend.__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_docs")

NODE_LABELS = {
    "decide_retrieval": ("🧭", "Decide Retrieval", "Evaluating whether external PDFs are required"),
    "generate_direct": ("⚡", "Direct Generation", "Answering directly from model general knowledge"),
    "retrieve": ("🔎", "Vector Retrieval", "Fetching top-k candidate chunks from FAISS"),
    "is_relevant": ("🧹", "Relevance Grading", "Filtering retrieved chunks for question relevance"),
    "generate_from_context": ("✍️", "Grounded Generation", "Drafting answer strictly from relevant context"),
    "no_relevant_docs": ("📭", "No Relevant Docs", "No supporting context found in indexed PDFs"),
    "is_sup": ("🛡️", "Support Verification (IsSUP)", "Checking answer grounding against context"),
    "revise_answer": ("🔁", "Strict Revision", "Rewriting answer to quote-only supported claims"),
    "is_use": ("🎯", "Usefulness Check (IsUSE)", "Verifying whether the answer addresses the question"),
    "rewrite_question": ("✏️", "Query Rewrite", "Optimizing retrieval query for another search pass"),
}

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nexa Verify · Self-Corrective RAG",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS Theme ──────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #070a13 !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] > .main {
    background-color: #070a13 !important;
}

section[data-testid="stSidebar"] {
    background-color: #03050a !important;
    border-right: 1px solid #1e293b;
}

.glass-card {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 10px;
    padding: 1.15rem 1.25rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35);
}

.metric-tile {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.18);
    border-radius: 8px;
    padding: 0.65rem 0.5rem;
    text-align: center;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    color: #00e5ff;
    line-height: 1.2;
}

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

.brand-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    background: linear-gradient(90deg, #00e5ff 0%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.15rem;
}

.brand-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #64748b;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.1rem;
}

.badge-active, .badge-success, .badge-warn, .badge-error {
    display: inline-block;
    padding: 0.22rem 0.58rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    border-radius: 4px;
    font-weight: 700;
    letter-spacing: 0.03em;
}

.badge-active {
    background-color: rgba(6, 182, 212, 0.15);
    color: #00e5ff;
    border: 1px solid rgba(6, 182, 212, 0.35);
}

.badge-success {
    background-color: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

.badge-warn {
    background-color: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}

.badge-error {
    background-color: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
}

.trace-step {
    background: rgba(15, 23, 42, 0.55);
    border-left: 3px solid #00e5ff;
    border-radius: 6px;
    padding: 0.55rem 0.8rem;
    margin-bottom: 0.45rem;
    font-size: 0.84rem;
}

.trace-meta {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #94a3b8;
}

@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}
</style>
""",
    unsafe_allow_html=True,
)


# ─── Session State Initialization ─────────────────────────────────────────────
def init_session_state() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "active_state": None,
        "active_trace": [],
        "processed_signature": None,
        "indexed_files": {},  # {filename: page_count}
        "total_pages": 0,
        "total_chunks": 0,
        "top_k": 4,
        "uploader_key": 0,
        "pending_prompt": None,
        "sandbox_results": None,
        "thread_id": str(uuid.uuid4()),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ─── Backend Sync & Helper Functions ──────────────────────────────────────────
def extract_answer_text(raw_answer: Any) -> str:
    """Normalize backend answer (str or AIMessage) into displayable text."""
    if raw_answer is None:
        return "No response generated."
    if hasattr(raw_answer, "content"):
        raw_answer = raw_answer.content
    if isinstance(raw_answer, list):
        parts = []
        for item in raw_answer:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(raw_answer)


def apply_retriever_k(k: int) -> None:
    """Update top-k on the active backend retriever if initialized."""
    if self_rag_backend.retriever is not None:
        self_rag_backend.retriever.search_kwargs["k"] = k


def _update_index_stats(docs: list[Any], fallback_names: list[str]) -> None:
    counts = Counter()
    for doc in docs:
        src = doc.metadata.get("source", "Document")
        counts[os.path.basename(str(src))] += 1

    if not counts and fallback_names:
        for name in fallback_names:
            counts[os.path.basename(name)] = 0

    st.session_state["indexed_files"] = dict(counts)
    st.session_state["total_pages"] = len(docs)
    vs = self_rag_backend.vector_store
    st.session_state["total_chunks"] = int(vs.index.ntotal) if vs is not None else 0
    apply_retriever_k(st.session_state["top_k"])


def index_uploaded_files(uploaded_files: list[Any]) -> bool:
    """Persist uploaded PDFs, load them via backend, and build the FAISS store."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

    saved_paths: list[str] = []
    for uploaded in uploaded_files:
        dest_path = os.path.join(UPLOAD_DIR, uploaded.name)
        with open(dest_path, "wb") as out_f:
            out_f.write(uploaded.getvalue())
        saved_paths.append(dest_path)

    docs = self_rag_backend.load_documents(saved_paths)
    if not docs:
        self_rag_backend.vector_store = None
        self_rag_backend.retriever = None
        st.session_state["indexed_files"] = {}
        st.session_state["total_pages"] = 0
        st.session_state["total_chunks"] = 0
        return False

    self_rag_backend.build_vector_store(docs)
    _update_index_stats(docs, [f.name for f in uploaded_files])
    return True


def index_local_paths(file_paths: list[str]) -> bool:
    """Load user-selected local PDF paths via backend and build the FAISS store."""
    docs = self_rag_backend.load_documents(file_paths)
    if not docs:
        return False
    self_rag_backend.build_vector_store(docs)
    _update_index_stats(docs, file_paths)
    return True


def clear_knowledge_base() -> None:
    """Clear uploaded files, reset backend vector store/retriever, and reset UI state."""
    if os.path.isdir(UPLOAD_DIR):
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass

    self_rag_backend.vector_store = None
    self_rag_backend.retriever = None
    st.session_state["processed_signature"] = None
    st.session_state["indexed_files"] = {}
    st.session_state["total_pages"] = 0
    st.session_state["total_chunks"] = 0
    st.session_state["sandbox_results"] = None
    st.session_state["uploader_key"] += 1


def discover_local_pdfs() -> list[str]:
    """Find PDF files present in the workspace root so the user can optionally load them."""
    pdfs = [
        os.path.join(BASE_DIR, name)
        for name in sorted(os.listdir(BASE_DIR))
        if name.lower().endswith(".pdf") and os.path.isfile(os.path.join(BASE_DIR, name))
    ]
    return pdfs


def build_markdown_report(
    messages: list[dict[str, Any]],
    active_state: dict[str, Any] | None,
    trace: list[dict[str, Any]],
) -> str:
    """Export the conversation and verification telemetry as Markdown."""
    lines = ["# Nexa Verify — Self-Corrective RAG Report", ""]
    lines.append("## Conversation")
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Nexa Agent"
        lines.append(f"### {role}")
        lines.append(msg["content"])
        lines.append("")

    if active_state:
        lines.extend(
            [
                "## Latest Verification Telemetry",
                f"- **Mode:** {'Retrieval RAG' if active_state.get('need_retrieval') else 'Direct LLM'}",
                f"- **Grounding (`issup`):** `{active_state.get('issup', 'n/a')}`",
                f"- **Usefulness (`isuse`):** `{active_state.get('isuse', 'n/a')}`",
                f"- **Retries:** `{active_state.get('retries', 0)}`",
            ]
        )
        if active_state.get("use_reason"):
            lines.append(f"- **Usefulness Reason:** {active_state['use_reason']}")
        if active_state.get("evidence"):
            lines.append("\n### Supporting Quotes")
            for quote in active_state["evidence"]:
                lines.append(f"> {quote}")

    if trace:
        lines.append("\n## Pipeline Execution Trace")
        for idx, step in enumerate(trace, start=1):
            lines.append(f"{idx}. `{step['node']}` ({step['elapsed']:.2f}s)")

    return "\n".join(lines)


def run_agent_query(prompt_text: str) -> None:
    """Execute the LangGraph Self-RAG pipeline with live step-by-step streaming updates."""
    initial_state = {
        "question": prompt_text,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "retries": 0,
        "retrieval_query": prompt_text,
        "evidence": [],
    }

    merged_state: dict[str, Any] = dict(initial_state)
    trace: list[dict[str, Any]] = []
    start_total = time.perf_counter()

    with st.status("🤖 Running Self-Corrective RAG Pipeline...", expanded=True) as status:
        last_step_t = time.perf_counter()
        config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
        try:
            for chunk in self_rag_backend.app.stream(initial_state, config=config, stream_mode="updates"):
                if not isinstance(chunk, dict):
                    continue
                now = time.perf_counter()
                elapsed = now - last_step_t
                last_step_t = now

                for node_name, node_update in chunk.items():
                    if isinstance(node_update, dict):
                        merged_state.update(node_update)
                    icon, title, desc = NODE_LABELS.get(node_name, ("⚙️", node_name, "Executing node"))
                    detail = ""
                    if node_name == "decide_retrieval" and isinstance(node_update, dict):
                        detail = " → Retrieval needed" if node_update.get("need_retrieval") else " → Direct mode"
                    elif node_name == "retrieve" and isinstance(node_update, dict):
                        detail = f" → {len(node_update.get('docs', []))} chunks"
                    elif node_name == "is_relevant" and isinstance(node_update, dict):
                        detail = f" → {len(node_update.get('relevant_docs', []))} relevant"
                    elif node_name == "is_sup" and isinstance(node_update, dict):
                        detail = f" → {node_update.get('issup', '')}"
                    elif node_name == "is_use" and isinstance(node_update, dict):
                        detail = f" → {node_update.get('isuse', '')}"
                    elif node_name == "rewrite_question" and isinstance(node_update, dict):
                        detail = f" → `{node_update.get('retrieval_query', '')}`"

                    st.write(f"{icon} **{title}**{detail} *({elapsed:.2f}s)*")
                    trace.append(
                        {
                            "node": node_name,
                            "title": title,
                            "desc": desc,
                            "detail": detail.lstrip(" →"),
                            "elapsed": elapsed,
                        }
                    )

            total_elapsed = time.perf_counter() - start_total
            status.update(label=f"✅ Pipeline completed in {total_elapsed:.2f}s", state="complete", expanded=False)
        except KeyError as ke:
            if "no_relevan_docs" in str(ke):
                merged_state["answer"] = "No relevant document found."
                merged_state["context"] = ""
                status.update(label="⚠️ Stopped at rewrite limit (no relevant docs)", state="complete", expanded=False)
            else:
                status.update(label="❌ Pipeline error", state="error", expanded=True)
                raise
        except Exception:
            status.update(label="❌ Pipeline execution failed", state="error", expanded=True)
            raise

    answer_text = extract_answer_text(merged_state.get("answer"))
    merged_state["answer"] = answer_text

    st.session_state["active_state"] = merged_state
    st.session_state["active_trace"] = trace
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer_text,
            "meta": {
                "need_retrieval": merged_state.get("need_retrieval", False),
                "issup": merged_state.get("issup"),
                "isuse": merged_state.get("isuse"),
                "retries": merged_state.get("retries", 0),
                "trace_len": len(trace),
            },
        }
    )


# ─── Sidebar: Knowledge Base & Controls ───────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand-title">⬡ NEXA VERIFY</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Self-Corrective RAG Agent</div>', unsafe_allow_html=True)

    has_index = self_rag_backend.retriever is not None and st.session_state["total_pages"] > 0
    status_color = "#10b981" if has_index else "#fbbf24"
    status_text = "INDEX READY · AGENT ONLINE" if has_index else "NO PDFS INDEXED · DIRECT MODE"
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1.1rem;">
            <div style="width:9px;height:9px;border-radius:50%;background:{status_color};
                box-shadow:0 0 10px {status_color};animation:pulse 2s infinite;"></div>
            <span style="font-family:'Space Mono',monospace;font-size:0.72rem;
                color:{status_text and status_color};letter-spacing:0.08em;font-weight:700;">{status_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Live Knowledge Base Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f'<div class="metric-tile"><div class="metric-value">{len(st.session_state["indexed_files"])}</div><div class="metric-label">Files</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-tile"><div class="metric-value">{st.session_state["total_pages"]}</div><div class="metric-label">Pages</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-tile"><div class="metric-value">{st.session_state["total_chunks"]}</div><div class="metric-label">Chunks</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 📄 Document Uploader")
    uploaded_files = st.file_uploader(
        "Upload PDFs to Index",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state['uploader_key']}",
        help="Upload one or more PDF documents. Loaded via self_rag_backend.load_documents() and indexed in FAISS.",
    )

    # Auto-index when new files are uploaded
    if uploaded_files:
        signature = tuple((f.name, f.size) for f in uploaded_files)
        if st.session_state["processed_signature"] != signature:
            with st.spinner("⚡ Loading & indexing uploaded PDFs..."):
                ok = index_uploaded_files(uploaded_files)
            if ok:
                st.session_state["processed_signature"] = signature
                st.toast(f"Indexed {len(uploaded_files)} PDF(s)!", icon="✅")
            else:
                st.error("Could not extract text from the uploaded PDFs.")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🔄 Re-index", width="stretch", disabled=not bool(uploaded_files)):
            with st.spinner("⚡ Re-indexing uploaded PDFs..."):
                if index_uploaded_files(uploaded_files):
                    st.session_state["processed_signature"] = tuple((f.name, f.size) for f in uploaded_files)
                    st.toast("Vector store rebuilt!", icon="🔄")
    with btn_col2:
        if st.button("🗑️ Clear KB", width="stretch", disabled=not has_index and not uploaded_files):
            clear_knowledge_base()
            st.toast("Knowledge base cleared.", icon="🗑️")
            st.rerun()

    # Optional: load PDFs already present in workspace
    local_pdfs = discover_local_pdfs()
    if local_pdfs:
        with st.expander("📁 Load Workspace PDFs", expanded=not has_index):
            selected_local: list[str] = []
            for pdf_path in local_pdfs:
                pdf_name = os.path.basename(pdf_path)
                if st.checkbox(pdf_name, key=f"local_pdf_{pdf_name}"):
                    selected_local.append(pdf_path)
            if st.button(
                "📥 Index Selected PDFs",
                width="stretch",
                disabled=not selected_local,
                type="primary",
            ):
                with st.spinner("⚡ Loading selected PDFs into FAISS..."):
                    if index_local_paths(selected_local):
                        st.session_state["processed_signature"] = None
                        st.toast(f"Indexed {len(selected_local)} local PDF(s)!", icon="✅")
                        st.rerun()
                    else:
                        st.error("Failed to load selected PDFs.")

    # Active Documents List
    st.markdown("---")
    st.markdown("#### 📂 Active Knowledge Base")
    if st.session_state["indexed_files"]:
        for fname, pages in st.session_state["indexed_files"].items():
            st.caption(f"✓ `{fname}` ({pages} pages)")
    else:
        st.caption("No PDFs indexed yet. Upload PDFs above or select workspace PDFs.")

    # Retrieval Controls
    st.markdown("---")
    st.markdown("#### ⚙️ Retriever Settings")
    new_k = st.slider(
        "Top-K Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=st.session_state["top_k"],
        help="Number of chunks fetched per retrieval step from FAISS.",
    )
    if new_k != st.session_state["top_k"]:
        st.session_state["top_k"] = new_k
        apply_retriever_k(new_k)
        st.toast(f"Retriever Top-K set to {new_k}", icon="⚙️")

    st.markdown("---")
    st.caption(f"🧵 Memory Thread: `{st.session_state['thread_id'][:8]}`")
    act1, act2 = st.columns(2)
    with act1:
        if st.button("✕ Clear Chat", width="stretch", disabled=not bool(st.session_state["messages"])):
            st.session_state["messages"] = []
            st.session_state["active_state"] = None
            st.session_state["active_trace"] = []
            st.session_state["thread_id"] = str(uuid.uuid4())
            st.rerun()
    with act2:
        report_md = build_markdown_report(
            st.session_state["messages"],
            st.session_state["active_state"],
            st.session_state["active_trace"],
        )
        st.download_button(
            "⬇️ Export",
            data=report_md,
            file_name="nexa_verify_report.md",
            mime="text/markdown",
            width="stretch",
            disabled=not bool(st.session_state["messages"]),
        )


# ─── Main Panel Layout ─────────────────────────────────────────────────────────
col_chat, col_mind = st.columns([5, 4], gap="large")

# ── Column 1: Interactive Chat Workspace ──────────────────────────────────────
with col_chat:
    st.markdown("### 💬 Chat Workspace")

    if not has_index:
        st.info(
            "📄 **No PDFs are currently indexed.** Upload PDFs in the sidebar (or load local workspace PDFs) "
            "to enable grounded RAG queries. General-knowledge questions still work in **Direct Mode**."
        )

    # Starter Prompts when conversation is empty
    if not st.session_state["messages"]:
        st.markdown("##### ✨ Try a Quick Prompt")
        q_cols = st.columns(3)
        sample_prompts = [
            ("📘 Summarize PDFs", "Give a concise summary of the key topics covered in the uploaded documents."),
            ("📋 Policies & Terms", "What key policies, plans, or rules are mentioned in the documents?"),
            ("💡 General Knowledge", "What is Retrieval-Augmented Generation (RAG) and why is it useful?"),
        ]
        for idx, (label, sample_q) in enumerate(sample_prompts):
            with q_cols[idx]:
                if st.button(label, width="stretch", key=f"starter_{idx}"):
                    st.session_state["pending_prompt"] = sample_q
                    st.rerun()

    # Render Chat History
    for msg in st.session_state["messages"]:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "assistant" and msg.get("meta"):
                meta = msg["meta"]
                if meta.get("need_retrieval"):
                    issup = meta.get("issup") or "unknown"
                    sup_badge = {
                        "fully_supported": ("Fully Supported", "badge-success"),
                        "partially_supported": ("Partially Supported", "badge-warn"),
                        "no_support": ("No Support", "badge-error"),
                    }.get(issup, (issup, "badge-active"))
                    st.markdown(
                        f"<span class='badge-active'>RAG VERIFIED</span> "
                        f"<span class='{sup_badge[1]}'>{html.escape(sup_badge[0])}</span> "
                        f"<span class='trace-meta'>retries: {meta.get('retries', 0)} · steps: {meta.get('trace_len', 0)}</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<span class='badge-success'>DIRECT MODE</span> "
                        "<span class='trace-meta'>general knowledge</span>",
                        unsafe_allow_html=True,
                    )
            st.markdown(msg["content"])

    # Quick follow-up controls when messages exist
    if st.session_state["messages"]:
        last_user_msgs = [m["content"] for m in st.session_state["messages"] if m["role"] == "user"]
        ctrl1, ctrl2, _ = st.columns([1.2, 1.2, 2.6])
        with ctrl1:
            if last_user_msgs and st.button("🔄 Re-run Last", width="stretch"):
                st.session_state["pending_prompt"] = last_user_msgs[-1]
                st.rerun()
        with ctrl2:
            if st.button("🧹 Reset Chat", width="stretch"):
                st.session_state["messages"] = []
                st.session_state["active_state"] = None
                st.session_state["active_trace"] = []
                st.session_state["thread_id"] = str(uuid.uuid4())
                st.rerun()

    # Chat Input
    user_input = st.chat_input("Ask a question about your uploaded PDFs or general knowledge...")
    if user_input:
        st.session_state["pending_prompt"] = user_input

    # Process Pending Prompt
    if st.session_state["pending_prompt"]:
        prompt_to_run = st.session_state["pending_prompt"]
        st.session_state["pending_prompt"] = None
        st.session_state["messages"].append({"role": "user", "content": prompt_to_run})

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt_to_run)

        with st.chat_message("assistant", avatar="🤖"):
            try:
                run_agent_query(prompt_to_run)
                st.rerun()
            except Exception as e:
                err_msg = f"Agent Execution Failure: {e}"
                st.error(err_msg)
                st.session_state["messages"].append({"role": "assistant", "content": err_msg})
                st.rerun()


# ── Column 2: Agent Mind & Verification Inspector ─────────────────────────────
with col_mind:
    st.markdown("### 🧠 Agent Verification & Inspector")

    tab_trace, tab_evidence, tab_sandbox = st.tabs(
        ["🧭 Pipeline Trace", "🔍 Evidence & Chunks", "🧪 Vector Search"]
    )

    state = st.session_state["active_state"]
    trace = st.session_state["active_trace"]

    with tab_trace:
        if state:
            need_retrieval = bool(state.get("need_retrieval", False))
            issup = state.get("issup", "n/a")
            isuse = state.get("isuse", "n/a")
            retries = state.get("retries", 0)
            retrieved_count = len(state.get("docs", []) or [])
            relevant_count = len(state.get("relevant_docs", []) or [])

            # 1. Pipeline Entry Card
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h4 style="margin-top:0; color:#00e5ff;">🎯 Pipeline Entry Decision</h4>', unsafe_allow_html=True)
            if need_retrieval:
                st.markdown('<span class="badge-active">RETRIEVAL MODE</span>', unsafe_allow_html=True)
                st.write(
                    f"Query routed through FAISS retrieval (**{retrieved_count}** fetched → "
                    f"**{relevant_count}** passed relevance filter)."
                )
            else:
                st.markdown('<span class="badge-success">DIRECT MODE</span>', unsafe_allow_html=True)
                st.write("Answered directly from general knowledge without vector retrieval.")
            st.markdown("</div>", unsafe_allow_html=True)

            # 2. Verification & Self-Correction Card
            if need_retrieval:
                sup_labels = {
                    "fully_supported": ("Fully Supported", "badge-success"),
                    "partially_supported": ("Partially Grounded", "badge-warn"),
                    "no_support": ("Ungrounded / No Support", "badge-error"),
                }
                sup_text, sup_cls = sup_labels.get(str(issup), (str(issup), "badge-active"))
                use_cls = "badge-success" if isuse == "useful" else "badge-error"

                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<h4 style="margin-top:0; color:#10b981;">⚖️ Verification Loops</h4>', unsafe_allow_html=True)
                st.markdown(
                    f"**Grounding (`IsSUP`):** <span class='{sup_cls}'>{html.escape(sup_text)}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"**Usefulness (`IsUSE`):** <span class='{use_cls}'>{html.escape(str(isuse).upper())}</span>",
                    unsafe_allow_html=True,
                )
                st.write(f"**Revision / Rewrite Retries:** `{retries}`")
                if state.get("use_reason"):
                    st.caption(f"*Usefulness Reason:* {state['use_reason']}")
                if state.get("retrieval_query") and state.get("retrieval_query") != state.get("question"):
                    st.caption(f"*Rewritten Retrieval Query:* `{state['retrieval_query']}`")
                st.markdown("</div>", unsafe_allow_html=True)

            # 3. Node-by-Node Trace Timeline
            if trace:
                st.markdown("##### ⏱️ Execution Timeline")
                for idx, step in enumerate(trace, start=1):
                    detail_html = (
                        f" — <code>{html.escape(step['detail'])}</code>" if step.get("detail") else ""
                    )
                    st.markdown(
                        f"<div class='trace-step'>"
                        f"<strong>{idx}. {html.escape(step['title'])}</strong>{detail_html}<br/>"
                        f"<span class='trace-meta'>{html.escape(step['desc'])} · {step['elapsed']:.2f}s</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Submit a question in the Chat Workspace to inspect the live Self-RAG decision trace.")

    with tab_evidence:
        if state:
            evidence = state.get("evidence", []) or []
            relevant_docs = state.get("relevant_docs", []) or []
            all_docs = state.get("docs", []) or []

            st.markdown("##### 📌 Supporting Quotes (`IsSUP` Evidence)")
            if evidence:
                for ev in evidence:
                    st.info(f"“ {ev} ”")
            else:
                st.caption("No supporting quotes recorded for this response.")

            st.markdown(f"##### ✅ Relevant Chunks Used ({len(relevant_docs)})")
            if relevant_docs:
                for i, doc in enumerate(relevant_docs, start=1):
                    src = os.path.basename(str(doc.metadata.get("source", "Document")))
                    page = doc.metadata.get("page")
                    page_label = f" · Page {int(page) + 1}" if page is not None else ""
                    with st.expander(f"Relevant Chunk {i:02d} — {src}{page_label}", expanded=(i == 1)):
                        st.code(doc.page_content.strip(), language="text")
            else:
                st.caption("No chunks passed relevance filtering.")

            relevant_contents = {d.page_content for d in relevant_docs}
            rejected_docs = [d for d in all_docs if d.page_content not in relevant_contents]
            if rejected_docs:
                with st.expander(f"🚫 Filtered-Out Chunks ({len(rejected_docs)})"):
                    for i, doc in enumerate(rejected_docs, start=1):
                        src = os.path.basename(str(doc.metadata.get("source", "Document")))
                        page = doc.metadata.get("page")
                        page_label = f" · Page {int(page) + 1}" if page is not None else ""
                        st.caption(f"**Rejected {i:02d}** — {src}{page_label}")
                        st.code(doc.page_content.strip(), language="text")
        else:
            st.info("Evidence quotes and retrieved PDF chunks will appear here after a RAG query.")

    with tab_sandbox:
        st.markdown("##### 🧪 Direct FAISS Vector Search")
        st.caption("Test vector retrieval directly against your indexed PDFs without invoking the LLM.")
        sandbox_query = st.text_input(
            "Search Query",
            placeholder="e.g., refund policy, pricing tiers, leave entitlement...",
            key="sandbox_query_input",
        )
        sb1, sb2 = st.columns([2, 1])
        with sb1:
            run_search = st.button(
                "🔎 Search Vector Store",
                width="stretch",
                type="primary",
                disabled=not has_index or not sandbox_query.strip(),
            )
        with sb2:
            if st.button("Clear Results", width="stretch", disabled=st.session_state["sandbox_results"] is None):
                st.session_state["sandbox_results"] = None
                st.rerun()

        if run_search and self_rag_backend.retriever is not None:
            st.session_state["sandbox_results"] = self_rag_backend.retriever.invoke(sandbox_query.strip())

        if not has_index:
            st.warning("Upload or select at least one PDF in the sidebar to test vector search.")
        elif st.session_state["sandbox_results"] is not None:
            results = st.session_state["sandbox_results"]
            st.success(f"Found {len(results)} chunk(s) (Top-K = {st.session_state['top_k']})")
            for i, doc in enumerate(results, start=1):
                src = os.path.basename(str(doc.metadata.get("source", "Document")))
                page = doc.metadata.get("page")
                page_label = f" · Page {int(page) + 1}" if page is not None else ""
                with st.expander(f"Match {i:02d} — {src}{page_label}", expanded=(i == 1)):
                    st.code(doc.page_content.strip(), language="text")
