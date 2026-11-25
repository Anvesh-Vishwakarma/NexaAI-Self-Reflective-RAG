from typing import Annotated, Any, List, Literal, Sequence, TypedDict, cast
from langchain_core.runnables import Runnable

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain_groq import ChatGroq

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

# DEFAULT_PDFS = ["Company_Policies.pdf", "Company_Profile.pdf", "Product_and_Pricing.pdf"]


def _load_single_pdf(pdf_path: str) -> List[Document]:
    """Load a single PDF file from disk into LangChain Documents."""
    try:
        return PyPDFLoader(pdf_path).load()
    except Exception as e:
        print(f"Error loading {pdf_path}: {e}")
        return []


def load_documents(
    pdf_sources: str | os.PathLike[str] | Sequence[str | os.PathLike[str] | Any] | Any,
) -> List[Document]:
    """Load user-provided PDF paths, directories, or file-like objects into Documents."""
    if not pdf_sources:
        return []

    if isinstance(pdf_sources, (str, os.PathLike)) or hasattr(pdf_sources, "read"):
        sources: List[Any] = [pdf_sources]
    else:
        sources = list(pdf_sources)

    loaded: List[Document] = []
    for source in sources:
        if isinstance(source, (str, os.PathLike)):
            path_str = os.fspath(source)
            if os.path.isdir(path_str):
                for name in sorted(os.listdir(path_str)):
                    if name.lower().endswith(".pdf"):
                        loaded.extend(_load_single_pdf(os.path.join(path_str, name)))
            elif os.path.exists(path_str):
                loaded.extend(_load_single_pdf(path_str))
            else:
                print(f"File not found: {path_str}")
        elif hasattr(source, "read"):
            tmp_path: str | None = None
            try:
                if hasattr(source, "seek"):
                    source.seek(0)
                data = source.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(data)
                    tmp_path = tmp_file.name
                docs = _load_single_pdf(tmp_path)
                source_name = getattr(source, "name", None)
                if source_name:
                    for doc in docs:
                        doc.metadata["source"] = source_name
                loaded.extend(docs)
            except Exception as e:
                print(f"Error loading uploaded PDF: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            print(f"Unsupported PDF source: {source!r}")

    return loaded

# Initialize global embeddings and splitter
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)

# Global variables for vector store and retriever
vector_store = None
retriever = None


def build_vector_store(
    documents_or_sources: Sequence[Document] | str | os.PathLike[str] | Sequence[Any] | Any = None,
) -> FAISS:
    """Chunk loaded Documents (or load user-provided PDFs first) and build the FAISS vector store."""
    global vector_store, retriever

    if documents_or_sources is None:
        loaded_docs: List[Document] = []
    elif isinstance(documents_or_sources, Document):
        loaded_docs = [documents_or_sources]
    elif isinstance(documents_or_sources, (list, tuple)) and all(
        isinstance(item, Document) for item in documents_or_sources
    ):
        loaded_docs = list(documents_or_sources)
    else:
        loaded_docs = load_documents(documents_or_sources)

    if not loaded_docs:
        # Fallback to avoid error when no docs are found/uploaded
        loaded_docs = [
            Document(
                page_content="No documents uploaded yet. Please upload PDF files.",
                metadata={"source": "placeholder"},
            )
        ]

    chunks = splitter.split_documents(loaded_docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return vector_store


# Build retriever initially using default files
# build_vector_store(DEFAULT_PDFS)

#llm = HuggingFaceEndpoint(repo_id = "meta-llama/Meta-Llama-3-8B-Instruct",huggingfacehub_api_token = os.getenv("HuggingFace_API_KEY"))
#model = ChatHuggingFace(llm=llm)

model = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))

# --------------------------------------------------
# Graph State
# --------------------------------------------------
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    question: str
    need_retrieval: bool

    docs: List[Document]
    relevant_docs: List[Document]
    
    context: str
    answer: str
    
    # Post-generation verification
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str]
    
    retries: int
    retrieval_query: str
    
    isuse: Literal["useful", "not_useful"]
    use_reason: str


def _format_chat_history(messages: Sequence[AnyMessage], max_messages: int = 12) -> str:
    """Format recent checkpointed conversation turns for prompt context."""
    if not messages:
        return "No previous conversation."
    lines: List[str] = []
    for msg in messages[-max_messages:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
    
    
class RetrieveDecision(TypedDict):
    should_retrieve: Annotated[
        bool,
        ...,
        "True if external documents are needed to answer reliably, else False.",
    ]

decide_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether retrieval is needed.\n"
            "Return JSON that matches this schema:\n"
            "{{'should_retrieve': boolean}}\n\n"
            "Guidelines:\n"
            "- should_retrieve=True if answering requires specific facts, citations, or info likely not in the model (including follow-ups about uploaded documents).\n"
            "- should_retrieve=False for general explanations, definitions, reasoning, or questions solely about prior conversation turns that don't need new document lookups.\n"
            "- If unsure, choose True."
        ),
        (
            "human",
            "Conversation History:\n{chat_history}\n\nQuestion: {question}",
        ),
    ]
)

retrieve_llm_chain = cast(
    Runnable[dict[str, Any], RetrieveDecision],
    decide_retrieval_prompt | model.with_structured_output(cast(Any, RetrieveDecision)),
)

def decide_retrieval(state: State):
    question = state["question"]
    chat_history = _format_chat_history(state.get("messages", []))

    decision = retrieve_llm_chain.invoke({"question": question, "chat_history": chat_history})

    return {
        "need_retrieval": decision["should_retrieve"],
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "evidence": [],
        "retries": 0,
        "retrieval_query": question,
    }


direct_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only your general knowledge and the conversation history.\n"
            "Do NOT assume access to external documents.\n"
            "If you are unsure or the answer requires specific sources, say:\n"
            "'I don't know based on my general knowledge.'"
        ),
        (
            "human",
            "Conversation History:\n{chat_history}\n\nQuestion:\n{question}",
        ),
    ]
)

def generate_direct(state: State):
    question = state["question"]
    chat_history = _format_chat_history(state.get("messages", []))

    response = model.invoke(
        direct_generation_prompt.format_messages(question=question, chat_history=chat_history)
    )
    answer_text = response.content if isinstance(response.content, str) else str(response.content)

    return {
        "answer": answer_text,
        "messages": [HumanMessage(content=question), AIMessage(content=answer_text)],
    }

def retrieve(state: State):
    if retriever is None:
        return {"docs": []}
    query = state.get("retrieval_query") or state["question"]
    return {"docs": retriever.invoke(query)}


class RelevanceDecision(TypedDict):
    is_relevant: Annotated[
        bool,
        ...,
        "True if the document helps answer the question, else False.",
    ]

is_relevant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging document relevance.\n"
            "Return JSON that matches this schema:\n"
            "{{'is_relevant': boolean}}\n\n"
            "A document is relevant if it contains information useful for answering the question."
        ),
        (
            "human",
            "Question:\n{question}\n\nDocument:\n{document}"
        ),
    ]
)

relevance_llm_chain = cast(
    Runnable[dict[str, Any], RelevanceDecision],
    is_relevant_prompt | model.with_structured_output(cast(Any, RelevanceDecision)),
)

def is_relevant(state: State):
    
    relevant_docs: List[Document] = []
    
    for doc in state["docs"]:
        decision = relevance_llm_chain.invoke({"question": state["question"], "document": doc.page_content})
        
        if decision["is_relevant"]:
            relevant_docs.append(doc)
    
    return {"relevant_docs": relevant_docs}


# New
rag_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a business RAG assistant.\n"
            "Answer the user's question using ONLY the provided context.\n"
            "You may use the conversation history solely to understand follow-up references.\n"
            "If the context does not contain enough information, say:\n"
            "'No relevant document found.'\n"
            "Do not use outside knowledge.\n"
        ),
        (
            "human",
            "Conversation History:\n{chat_history}\n\n"
            "Question:\n{question}\n\n"
            "Context:\n{context}\n"
        ),
    ]
)


def generate_from_context(state: State):
    
    context = "\n\n---\n\n".join([doc.page_content for doc in state.get("relevant_docs", [])]).strip()
    
    if not context:
        return {"answer": "No relevant document found."}
    
    chat_history = _format_chat_history(state.get("messages", []))
    out = model.invoke(
        rag_generation_prompt.format_messages(
            question=state["question"],
            context=context,
            chat_history=chat_history,
        )
    )
    
    return {"answer": out.content, "context": context}


# -----------------------------
# 6) IsSUP verify + revise loop
# -----------------------------
class IsSUPDecision(TypedDict):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str]

issup_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are verifying whether the ANSWER is supported by the CONTEXT.\n"
            "Return JSON with keys: issup, evidence.\n"
            "issup must be one of: fully_supported, partially_supported, no_support.\n\n"
            "How to decide issup:\n"
            "- fully_supported:\n"
            "  Every meaningful claim is explicitly supported by CONTEXT, and the ANSWER does NOT introduce\n"
            "  any qualitative/interpretive words that are not present in CONTEXT.\n"
            "  (Examples of disallowed words unless present in CONTEXT: culture, generous, robust, designed to,\n"
            "  supports professional development, best-in-class, employee-first, etc.)\n\n"
            "- partially_supported:\n"
            "  The core facts are supported, BUT the ANSWER includes ANY abstraction, interpretation, or qualitative\n"
            "  phrasing not explicitly stated in CONTEXT (e.g., calling policies 'culture', saying leave is 'generous',\n"
            "  or inferring outcomes like 'supports professional development').\n\n"
            "- no_support:\n"
            "  The key claims are not supported by CONTEXT.\n\n"
            "Rules:\n"
            "- Be strict: if you see ANY unsupported qualitative/interpretive phrasing, choose partially_supported.\n"
            "- If the answer is mostly unrelated to the question or unsupported, choose no_support.\n"
            "- Evidence: include up to 3 short direct quotes from CONTEXT that support the supported parts.\n"
            "- Do not use outside knowledge."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Answer:\n{answer}\n\n"
            "Context:\n{context}\n"
        ),
    ]
)

issup_llm_chain = cast(
    Runnable[dict[str, Any], IsSUPDecision],
    issup_prompt | model.with_structured_output(cast(Any, IsSUPDecision)),
)

def is_sup(state: State):
    
    question = state["question"]
    answer = state.get("answer", "")
    context = state.get("context", "")
    
    decision = issup_llm_chain.invoke({"question": question, "answer": answer, "context": context})
    
    return {"issup": decision["issup"], "evidence": decision["evidence"]}

MAX_RETRIES = 8

def route_after_issup(state: State) -> Literal["accept_answer", "revise_answer"]:
    
    if state["issup"] == "fully_supported":
        return "accept_answer"
    
     # stop if we've already tried enough
    if state.get("retries", 0) >= MAX_RETRIES:
        return "accept_answer" # or return a "give_up" node if you want
    
    # otherwise revise again
    return "revise_answer"

def accept_answer(state: State):
    return {}  # keep answer as-is


revise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a STRICT reviser.\n\n"
            "You must output based on the following format:\n\n"
            "FORMAT (quote-only answer):\n"
            "- <direct quote from the CONTEXT>\n"
            "- <direct quote from the CONTEXT>\n\n"
            "Rules:\n"
            "- Use ONLY the CONTEXT.\n"
            "- Do NOT add any new words besides bullet dashes and the quotes themselves.\n"
            "- Do NOT explain anything.\n"
            "- Do NOT say 'context', 'not mentioned', 'does not mention', 'not provided', etc.\n"
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Current Answer:\n{answer}\n\n"
            "CONTEXT:\n{context}"
        ),
    ]
)

def revise_answer(state: State):
    
    question = state["question"]
    answer = state["answer"]
    context = state.get("context", "")
    
    revised_answer = model.invoke(revise_prompt.format_messages(question=question, answer=answer, context=context))
    
    return {"answer": revised_answer.content, "retries": state.get("retries", 0) + 1}


class IsUSEDecision(TypedDict):
    isuse: Literal["useful", "not_useful"]
    reason: Annotated[str, ..., "Short reason in 1 line."]

isuse_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are judging USEFULNESS of the ANSWER for the QUESTION.\n\n"
            "Goal:\n"
            "- Decide if the answer actually addresses what the user asked.\n\n"
            "Return JSON with keys: isuse, reason.\n"
            "isuse must be one of: useful, not_useful.\n\n"
            "Rules:\n"
            "- useful: The answer directly answers the question or provides the requested specific info.\n"
            "- not_useful: The answer is generic, off-topic, or only gives related background without answering.\n"
            "- Do NOT use outside knowledge.\n"
            "- Do NOT re-check grounding (IsSUP already did that). Only check: 'Did we answer the question?'\n"
            "- Keep reason to 1 short line."
        ),
        (
            "human",
            "Question:\n{question}\n\nAnswer:\n{answer}"
        ),
    ]
)

isuse_llm_chain = cast(
    Runnable[dict[str, Any], IsUSEDecision],
    isuse_prompt | model.with_structured_output(cast(Any, IsUSEDecision)),
)

def is_use(state: State):
    
    question = state["question"]
    answer = state.get("answer", "")
    
    decision = isuse_llm_chain.invoke({"question": question, "answer": answer})
    
    if decision["isuse"] == "useful":
        return {
            "isuse": decision["isuse"],
            "use_reason": decision["reason"],
            "messages": [HumanMessage(content=question), AIMessage(content=answer)],
        }
    return {"isuse": decision["isuse"], "use_reason": decision["reason"]}

MAX_REWRITE_TRIES = 3

def route_after_isuse(state: State) -> Literal["END", "rewrite_question", "no_relevant_docs"]:
    if state.get("isuse") == "useful":
        return "END"
    
    if state.get("retries", 0) >= MAX_REWRITE_TRIES:
        return "no_relevant_docs"
    
    return "rewrite_question"


class RewriteDecision(TypedDict):
    retrieval_query: Annotated[
        str,
        ...,
        "Rewritten query optimized for vector retrieval against internal company PDFs.",
    ]

rewrite_for_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's QUESTION into a query optimized for vector retrieval over INTERNAL company PDFs.\n\n"
            "Rules:\n"
            "- Keep it short (6–16 words).\n"
            "- Preserve key entities (e.g., NexaAI, plan names).\n"
            "- Add 2–5 high-signal keywords that likely appear in policy/pricing docs.\n"
            "- Remove filler words.\n"
            "- Do NOT answer the question.\n"
            "- Output JSON with key: retrieval_query\n\n"
            "Examples:\n"
            "Q: 'Do NexaAI plans include a free trial?'\n"
            "-> {{'retrieval_query': 'NexaAI free trial duration trial period plans'}}\n\n"
            "Q: 'What is NexaAI refund policy?'\n"
            "-> {{'retrieval_query': 'NexaAI refund policy cancellation refund timeline charges'}}"
        ),
        (
            "human",
            "Conversation History:\n{chat_history}\n\n"
            "QUESTION:\n{question}\n\n"
            "Previous retrieval query:\n{retrieval_query}\n\n"
            "Answer (if any):\n{answer}"
        ),
    ]
)

rewrite_llm_chain = cast(
    Runnable[dict[str, Any], RewriteDecision],
    rewrite_for_retrieval_prompt | model.with_structured_output(cast(Any, RewriteDecision)),
)

def rewrite_question(state: State):
    
    question = state["question"]
    answer = state.get("answer", "")
    retrieval_query = state.get("retrieval_query", question)  # start with original question for first rewrite
    chat_history = _format_chat_history(state.get("messages", []))
    
    decision = rewrite_llm_chain.invoke(
        {
            "question": question,
            "answer": answer,
            "retrieval_query": retrieval_query,
            "chat_history": chat_history,
        }
    )
    
    return {"retrieval_query": decision["retrieval_query"], "retries": state.get("retries", 0) + 1}

def no_relevant_docs(state: State):
    msg = "No relevant document found."
    return {
        "answer": msg,
        "context": "",
        "messages": [HumanMessage(content=state["question"]), AIMessage(content=msg)],
    }

def route_after_decide(state: State) -> Literal["generate_direct", "retrieve"]:
    if state["need_retrieval"]:
        return "retrieve"
    else:
        return "generate_direct"
    
def route_after_relevance(state: State) -> Literal["generate_from_context", "no_relevant_docs"]:
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    return "no_relevant_docs"

g = StateGraph(State)

g.add_node("decide_retrieval", decide_retrieval)
g.add_node("generate_direct", generate_direct)
g.add_node("retrieve", retrieve)

g.add_node("is_relevant", is_relevant)
g.add_node("generate_from_context", generate_from_context)
g.add_node("no_relevant_docs", no_relevant_docs)

g.add_node("is_sup", is_sup)

#g.add_node("accept_answer", accept_answer)
g.add_node("revise_answer", revise_answer)

g.add_node("is_use", is_use)

g.add_node("rewrite_question", rewrite_question)

# --------------------
# Edges
# --------------------

g.add_edge(START, "decide_retrieval")
g.add_conditional_edges("decide_retrieval", route_after_decide, {"generate_direct": "generate_direct", "retrieve": "retrieve"})
g.add_edge("generate_direct", END)
g.add_edge("retrieve", "is_relevant")
g.add_edge("no_relevant_docs", END)

g.add_conditional_edges("is_relevant", route_after_relevance, {"generate_from_context": "generate_from_context", "no_relevant_docs": "no_relevant_docs"})
g.add_edge("generate_from_context", "is_sup")
g.add_conditional_edges("is_sup", route_after_issup, {"accept_answer": "is_use", "revise_answer": "revise_answer"})
g.add_edge("revise_answer", "is_sup")  # loop back to verification after revision
# --------------------
# IsUSE routing
# --------------------
g.add_conditional_edges(
    "is_use",
    route_after_isuse,
    {
        "END": END,
        "rewrite_question": "rewrite_question",
        "no_relevant_docs": "no_relevant_docs",
    },
)

g.add_edge("rewrite_question", "retrieve")  # after rewriting, go back to retrieval with new query

checkpointer = InMemorySaver()

app = g.compile(checkpointer=checkpointer)

app