# 🛡️ Nexa-Verify: Self-Correction Agentic RAG

Nexa-Verify is an enterprise-grade, agentic RAG system designed to eliminate LLM hallucinations in internal business environments. Built with LangGraph and Qwen-2.5, the system implements a multi-stage "Self-Reflection" loop that retrieves, validates, and auto-corrects answers against internal company documentation.

# 🖼️ System Interface & Workflow
![image alt](https://github.com/Anvesh-Vishwakarma/NexaAI-Self-Reflective-RAG/blob/main/Screenshot%20(19).png?raw=true)

![image](https://github.com/Anvesh-Vishwakarma/NexaAI-Self-Reflective-RAG/blob/main/Screenshot%20(20).png?raw=true)

![image](https://github.com/Anvesh-Vishwakarma/NexaAI-Self-Reflective-RAG/blob/main/Screenshot%20(21).png?raw=true)

![image](https://github.com/Anvesh-Vishwakarma/NexaAI-Self-Reflective-RAG/blob/main/Screenshot%20(22).png?raw=true)

# WorkFlow
![image](https://github.com/Anvesh-Vishwakarma/NexaAI-Self-Reflective-RAG/blob/main/app.png?raw=true)

# 🚀 Key Features
* Agentic Decision Making: Uses a decide_retrieval node to determine if a query requires external facts or can be answered via general knowledge.

* Multi-Stage Verification (Self-RAG):
   * Relevance Filtering: Dynamically grades retrieved documents to prune noise.
   * Hallucination Guard (IsSUP): Validates if the generated answer is strictly supported by the context, preventing qualitative "drifting".
   * Utility Check (IsUSE): Ensures the final output directly addresses the user's intent.

* Autonomous Revision Loop: If an answer is unsupported, the agent enters a revise_answer state, stripping away hallucinations and strictly adhering to context quotes (Max 8 retries).

* Automated Query Rewriting: Failed retrievals trigger a rewrite_question node to optimize keywords for vector search (FAISS).

# 🏗️ Technical Architecture
The system follows a cyclic directed acyclic graph (DAG) structure:

1. Ingestion: Processes PDF-based company policies and product data using RecursiveCharacterTextSplitter.
2. Vector Store: Utilizes FAISS-CPU with all-MiniLM-L6-v2 embeddings for high-speed similarity search.
3. LLM Backend: Powered by Qwen/Qwen2.5-7B-Instruct via Hugging Face Inference Endpoints.
4. State Management: Orchestrated by LangGraph to manage complex branching and loops between retrieval and generation.

# 🛠️ Tech Stack

* Orchestration: LangGraph, LangChain 
* LLM: Qwen-2.5-7B (Hugging Face)
* Vector DB: FAISS 
* Frontend: Streamlit 
* Environment: Python-dotenv, PyPDF

# 📂 Dataset Overview
The system is pre-configured with the NexaAI Solutions corporate data suite:
* Company Policies: HR, leave, and workplace conduct.
* Product & Pricing: Modular AI suite (NexaChat, NexaInsight) and subscription tiers (Starter, Growth, Enterprise).
* Company Profile: Organizational mission and leadership hierarchy.

# 🧠 Agentic Self-Reflection Logic
Unlike standard RAG, which is a linear "Retrieve -> Generate" pipeline, Nexa-Verify uses a cyclic graph to ensure factual integrity.
1. Dynamic Routing (decide_retrieval)
The agent first analyzes the query.
   * If the user asks for a general greeting or non-factual information, it bypasses the vector store to save latency and compute.
   * If the query requires company-specific data (e.g., "What is the Growth Plan price?"), it triggers the retrieve node.
2. The Grading Loop (is_relevant)
Retrieved chunks are passed through a Relevance Grader. If the retrieved documents do not contain the answer, the system does not attempt to "guess." Instead, it triggers a rewrite_question node to optimize the search query and tries again.
3. Hallucination Guard (is_sup)
Once an answer is generated, the is_sup (Is Supported) node performs a binary cross-check:
    * Condition: Does every claim in the generated answer have a direct reference in the retrieved context?
    * Action: If "No," the answer is sent to the revise_answer node to be stripped of hallucinations and re-generated using only confirmed facts.
4. Utility Validation (is_use)
The final check ensures the answer isn't just factually correct but also useful. If the LLM provides a technically true but irrelevant response, the system loops back to re-address the user's intent.

# 📊 Data & Knowledge Base
The system is built to ingest and understand complex organizational structures. The current knowledge base includes:

* Hierarchical HR Policies: Detailed leave structures (18 Annual, 10 Sick, 6 Casual days) and career progression frameworks.
* Product Specifications: Technical details for NexaChat, NexaInsight, and NexaSupport.
* Financial Tiers: Subscription models ranging from ₹2,499 (Starter) to Custom Enterprise pricing.

# 🔧 Performance Optimizations
* Recursive Splitting: Uses RecursiveCharacterTextSplitter with a chunk size of 600 and 150-character overlap to preserve semantic context across chunk boundaries.
* Embedding Efficiency: Utilizes all-MiniLM-L6-v2, a lightweight yet powerful sentence transformer, making the system suitable for low-latency production environments.
* Transparency: The frontend includes an "Evidence" toggle, allowing users to see the exact document snippets used to form the response, which is critical for enterprise trust.

# ⚡ Quick Start
1. Clone the repository:
```
git clone https://github.com/Anvesh-Vishwakarma/Nexa-Verify.git
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Set up Environment Variables: 
Create a .env file and add your HuggingFace_API_KEY.
4. Run the Application:
```
streamlit run self_rag_frontend.py
```
