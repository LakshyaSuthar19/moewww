import os
import re
import numpy as np
import streamlit as st

# Handle PyMuPDF import cleanly
try:
    import pymupdf as fitz
except ImportError:
    import fitz

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from openai import OpenAI

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Lumina RAG • Universal Document Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM MODERN CSS
# =========================================================

st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Gradient Header Accent */
    .rag-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(168, 85, 247, 0.12) 50%, rgba(236, 72, 153, 0.08) 100%);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
    }
    .rag-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(120deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .rag-subtitle {
        color: #94a3b8;
        font-size: 0.98rem;
        margin: 0;
        line-height: 1.5;
    }
    .rag-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 4px 10px;
        border-radius: 20px;
        background: rgba(99, 102, 241, 0.18);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.35);
        margin-bottom: 8px;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: left;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(139, 92, 246, 0.4);
        transform: translateY(-2px);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .metric-lbl {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }

    /* Hero Empty State */
    .hero-card {
        border: 1px dashed rgba(148, 163, 184, 0.25);
        background: rgba(255, 255, 255, 0.015);
        border-radius: 18px;
        padding: 45px 35px;
        text-align: center;
        margin: 20px 0;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-top: 25px;
    }
    .feature-item {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 18px;
        text-align: left;
    }
    .feature-icon {
        font-size: 1.5rem;
        margin-bottom: 8px;
    }
    .feature-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #e2e8f0;
        margin-bottom: 4px;
    }
    .feature-desc {
        font-size: 0.82rem;
        color: #94a3b8;
        line-height: 1.4;
    }

    /* Citation Badges */
    .source-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .source-meta {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .badge-page {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-score {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Prompt Suggestion Chips */
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 18px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Clean Tab styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "doc_meta" not in st.session_state:
    st.session_state.doc_meta = {}

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "index" not in st.session_state:
    st.session_state.index = None

if "bm25" not in st.session_state:
    st.session_state.bm25 = None

if "chip_prompt" not in st.session_state:
    st.session_state.chip_prompt = None

# =========================================================
# OPENROUTER & ENVIRONMENT SETUP
# =========================================================

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Check Streamlit Cloud secrets if not in environment
if not api_key and hasattr(st, "secrets") and "OPENROUTER_API_KEY" in st.secrets:
    api_key = st.secrets["OPENROUTER_API_KEY"]

if not api_key:
    with st.sidebar:
        st.warning("⚠️ OpenRouter API Key required")
        user_key = st.text_input("Enter OpenRouter API Key:", type="password", key="user_api_key")
        if user_key:
            api_key = user_key
            st.rerun()

if not api_key:
    st.info("👋 Welcome! Please enter your **OpenRouter API Key** in the sidebar to get started.")
    st.stop()

try:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
except Exception as e:
    st.error(f"❌ Could not initialize OpenRouter client:\n\n{e}")
    st.stop()

# =========================================================
# MODEL CACHING
# =========================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Tokenizer for BM25
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def ask_llm(prompt, model_name="openrouter/free", temperature=0.2):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content

# =========================================================
# SIDEBAR: CONTROL CENTER & DOCUMENT HUB
# =========================================================

with st.sidebar:
    # Sidebar Header
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
            <div style="background: linear-gradient(135deg, #6366f1, #a855f7); width: 34px; height: 34px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);">
                ⚡
            </div>
            <div>
                <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #f8fafc;">Lumina RAG</h3>
                <span style="font-size: 0.72rem; color: #94a3b8; letter-spacing: 0.05em; text-transform: uppercase;">Universal Doc Engine</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    # Document Uploader
    st.markdown("#### 📄 Document Hub")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF documents to build a semantic index."
    )

    # Retrieval & Model Settings
    with st.expander("⚙️ RAG Engine Parameters", expanded=False):
        model_options = [
            "openrouter/free",
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "deepseek/deepseek-chat:free",
            "qwen/qwen-2.5-72b-instruct:free"
        ]
        selected_model = st.selectbox(
            "Generation Model",
            options=model_options,
            index=0,
            help="Choose the LLM to generate responses."
        )

        top_k = st.slider(
            "Top Chunks to Retrieve (K)",
            min_value=3,
            max_value=12,
            value=5,
            help="Number of most relevant reranked chunks provided to the LLM."
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Lower values yield more factual, deterministic answers."
        )

        semantic_weight = st.slider(
            "Hybrid Semantic Weight",
            min_value=0.1,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="Balance between Dense Semantic (FAISS) and Sparse Lexical (BM25) search."
        )

    # Document Library & Diagnostics
    if st.session_state.documents:
        st.markdown("#### 📚 Active Document Index")
        for doc_name in st.session_state.documents:
            meta = st.session_state.doc_meta.get(doc_name, {})
            pages = meta.get("pages", "?")
            chunks_count = meta.get("chunks", "?")
            size_kb = meta.get("size_kb", "?")
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;">
                    <div style="font-weight: 600; font-size: 0.88rem; color: #e2e8f0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{doc_name}">
                        📄 {doc_name}
                    </div>
                    <div style="font-size: 0.74rem; color: #94a3b8; display: flex; gap: 12px; margin-top: 4px;">
                        <span>📖 {pages} pages</span>
                        <span>🧩 {chunks_count} chunks</span>
                        <span>💾 {size_kb} KB</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        st.caption(f"Total Chunks: **{len(st.session_state.chunks)}** • Index: **FAISS L2 + BM25**")

    # Action Buttons
    st.markdown("---")
    col_act1, col_act2 = st.columns(2)
    with col_act1:
        if st.button("🗑️ Clear Chat", use_container_width=True, disabled=len(st.session_state.messages) == 0):
            st.session_state.messages = []
            st.rerun()
            
    with col_act2:
        if st.button("🔄 Reset App", use_container_width=True):
            st.session_state.messages = []
            st.session_state.documents = []
            st.session_state.chunks = []
            st.session_state.doc_meta = {}
            st.session_state.processed_files = []
            st.session_state.index = None
            st.session_state.bm25 = None
            st.rerun()

# =========================================================
# LOAD MODELS IN BACKGROUND
# =========================================================

embedding_model = load_embedding_model()
reranker = load_reranker()

# =========================================================
# DOCUMENT PROCESSING PIPELINE
# =========================================================

if uploaded_files:
    current_files = [file.name for file in uploaded_files]
    previous_files = st.session_state.processed_files

    if current_files != previous_files:
        all_chunks = []
        doc_meta = {}
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        with st.status("⚡ Indexing uploaded documents...", expanded=True) as status_box:
            for uploaded_file in uploaded_files:
                status_box.write(f"📖 Parsing PDF: **{uploaded_file.name}**")
                try:
                    pdf_bytes = uploaded_file.read()
                    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                    page_count = len(doc)
                    file_chunks_count = 0

                    for page_number, page in enumerate(doc):
                        text = page.get_text().strip()
                        if not text:
                            continue

                        page_chunks = splitter.split_text(text)
                        for chunk in page_chunks:
                            if not chunk.strip():
                                continue
                            all_chunks.append({
                                "text": chunk,
                                "page": page_number + 1,
                                "filename": uploaded_file.name
                            })
                            file_chunks_count += 1

                    doc.close()
                    doc_meta[uploaded_file.name] = {
                        "pages": page_count,
                        "chunks": file_chunks_count,
                        "size_kb": round(len(pdf_bytes) / 1024, 1)
                    }
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {e}")

            if len(all_chunks) == 0:
                status_box.update(label="❌ No extractable text found", state="error")
                st.error("No readable text found in the uploaded PDF(s). Please verify the PDFs contain selectable text.")
                st.stop()

            # Create Embeddings
            status_box.write(f"🧠 Generating embeddings for {len(all_chunks)} chunks...")
            texts = [chunk["text"] for chunk in all_chunks]
            embeddings = embedding_model.encode(texts, show_progress_bar=False)
            embeddings = np.array(embeddings).astype("float32")

            # FAISS Index
            status_box.write("⚡ Building FAISS L2 vector index...")
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)

            # BM25 Index
            status_box.write("📚 Building BM25 keyword index...")
            tokenized_chunks = [tokenize(chunk["text"]) for chunk in all_chunks]
            bm25 = BM25Okapi(tokenized_chunks)

            # Update Session State
            st.session_state.index = index
            st.session_state.chunks = all_chunks
            st.session_state.bm25 = bm25
            st.session_state.documents = current_files
            st.session_state.doc_meta = doc_meta
            st.session_state.processed_files = current_files
            st.session_state.messages = []

            status_box.update(label=f"✅ Ready! Indexed {len(all_chunks)} chunks across {len(current_files)} document(s).", state="complete")
            st.rerun()

# =========================================================
# MAIN INTERFACE HEADER
# =========================================================

st.markdown("""
<div class="rag-header">
    <div class="rag-badge">⚡ Advanced RAG Architecture</div>
    <div class="rag-title">Lumina Universal PDF Assistant</div>
    <p class="rag-subtitle">
        Deep multi-document research assistant powered by Dense Semantic Search (FAISS), Sparse Lexical Search (BM25), Cross-Encoder Reranking, and contextual LLM synthesis.
    </p>
</div>
""", unsafe_allow_html=True)

# Overview Metric Cards
if st.session_state.documents:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{len(st.session_state.documents)}</div>
                <div class="metric-lbl">Active Documents</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{len(st.session_state.chunks)}</div>
                <div class="metric-lbl">Indexed Chunks</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{selected_model.split('/')[-1]}</div>
                <div class="metric-lbl">Active LLM</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">Hybrid + Rerank</div>
                <div class="metric-lbl">Pipeline Engine</div>
            </div>
        """, unsafe_allow_html=True)
    st.write("")

# =========================================================
# WORKSPACE TABS
# =========================================================

tab_chat, tab_explorer, tab_pipeline = st.tabs([
    "💬 AI Assistant",
    "📑 Document & Chunk Explorer",
    "📊 Pipeline & Metrics"
])

# ---------------------------------------------------------
# TAB 1: AI CHAT ASSISTANT
# ---------------------------------------------------------

with tab_chat:
    if st.session_state.index is None:
        # Empty State Hero
        st.markdown("""
            <div class="hero-card">
                <div style="font-size: 2.8rem; margin-bottom: 12px;">📚</div>
                <h3 style="font-size: 1.45rem; font-weight: 700; margin-bottom: 8px; color: #f8fafc;">
                    Upload Your PDF Documents to Begin
                </h3>
                <p style="color: #94a3b8; max-width: 580px; margin: 0 auto; font-size: 0.94rem; line-height: 1.6;">
                    Upload one or multiple PDF documents in the left sidebar. Lumina will extract text, segment it into overlapping chunks, construct dual vector & keyword indices, and answer your complex questions with precise citations.
                </p>
                <div class="feature-grid">
                    <div class="feature-item">
                        <div class="feature-icon">⚡</div>
                        <div class="feature-title">Hybrid Retrieval</div>
                        <div class="feature-desc">Combines semantic vector embeddings (FAISS) with keyword frequency (BM25) for high recall.</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon">🎯</div>
                        <div class="feature-title">Cross-Encoder Reranker</div>
                        <div class="feature-desc">Deep transformer scoring re-evaluates top candidates to eliminate hallucinations.</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon">📍</div>
                        <div class="feature-title">Page-Level Citations</div>
                        <div class="feature-desc">Every claim is grounded with document names, exact page numbers, and previewable passages.</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Quick Prompt Suggestions Chips
        st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #94a3b8; margin-bottom: 6px;'>💡 QUICK INQUIRIES</p>", unsafe_allow_html=True)
        chip_cols = st.columns(4)
        sample_prompts = [
            ("📌 Executive Summary", "Provide a comprehensive executive summary of this document highlighting key points and conclusions."),
            ("📋 Key Definitions", "What are the primary definitions, terms, and concepts introduced in the text?"),
            ("📅 Timeline & Dates", "Extract and summarize all important dates, timelines, and milestones mentioned."),
            ("⚖️ Key Takeaways", "What are the main arguments, findings, and takeaways outlined in the document?")
        ]

        for i, (label, prompt_text) in enumerate(sample_prompts):
            with chip_cols[i % 4]:
                if st.button(label, use_container_width=True, key=f"chip_{i}"):
                    st.session_state.chip_prompt = prompt_text
                    st.rerun()

        st.markdown("---")

        # Render Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # If assistant message has attached sources, render citation cards
                if message["role"] == "assistant" and "sources" in message and message["sources"]:
                    with st.expander(f"📚 Retrieved Sources ({len(message['sources'])} citations)", expanded=False):
                        if "search_query" in message:
                            st.caption(f"🔍 **Optimized Query:** `{message['search_query']}`")
                        for idx, src in enumerate(message["sources"]):
                            st.markdown(f"""
                                <div class="source-card">
                                    <div class="source-meta">
                                        <div style="font-weight: 600; color: #e2e8f0; font-size: 0.88rem;">
                                            📄 {src['filename']}
                                        </div>
                                        <div>
                                            <span class="badge-page">Page {src['page']}</span>
                                            <span class="badge-score">Score: {src['score']:.4f}</span>
                                        </div>
                                    </div>
                                    <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.5; font-style: italic; background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px;">
                                        "{src['text'][:320]}..."
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

        # Handle Chat Input (via text input or clicked chip)
        user_query = st.chat_input("💬 Ask anything about your uploaded documents...")
        if st.session_state.chip_prompt:
            user_query = st.session_state.chip_prompt
            st.session_state.chip_prompt = None

        if user_query:
            # 1. Append User Message
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # 2. Process Assistant Response
            with st.chat_message("assistant"):
                with st.status("🧠 Lumina Research Engine Active...", expanded=True) as status:
                    # Build Conversation History
                    chat_history = ""
                    for msg in st.session_state.messages[:-1]:
                        chat_history += f"{msg['role'].upper()}: {msg['content']}\n"

                    # Step 1: Query Rewriting
                    status.write("🔄 Reformulating contextual search query...")
                    rewrite_prompt = f"""
You are an intelligent document search query rewriting assistant.
Rewrite the user's latest question into a clear standalone search query that can be used to search PDF documents.
Use previous conversation context to resolve pronouns like it, its, this, that, they, the previous section.

IMPORTANT:
1. Preserve the meaning of the user's question.
2. Do not answer the question.
3. Return ONLY the rewritten search query with no additional commentary.

PREVIOUS CONVERSATION:
{chat_history}

LATEST USER QUESTION:
{user_query}
"""
                    try:
                        search_query = ask_llm(rewrite_prompt, model_name=selected_model, temperature=0.1).strip()
                    except Exception:
                        search_query = user_query

                    # Step 2: Semantic (FAISS) + Lexical (BM25) Hybrid Retrieval
                    status.write(f"🔍 Running Hybrid Search for: *'{search_query}'*")
                    query_embedding = embedding_model.encode([search_query])
                    query_embedding = np.array(query_embedding).astype("float32")

                    total_chunks = len(st.session_state.chunks)
                    faiss_k = min(20, total_chunks)
                    faiss_distances, faiss_indices = st.session_state.index.search(query_embedding, faiss_k)

                    tokenized_query = tokenize(search_query)
                    bm25_scores = st.session_state.bm25.get_scores(tokenized_query)
                    bm25_k = min(20, len(bm25_scores))
                    bm25_indices = np.argsort(bm25_scores)[::-1][:bm25_k]

                    # Combine Scores
                    combined_scores = {}
                    dense_w = semantic_weight
                    sparse_w = 1.0 - dense_w

                    for position, idx in enumerate(faiss_indices[0]):
                        distance = faiss_distances[0][position]
                        semantic_score = 1.0 / (1.0 + distance)
                        combined_scores[idx] = combined_scores.get(idx, 0) + dense_w * semantic_score

                    max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 0
                    if max_bm25 > 0:
                        for idx in bm25_indices:
                            keyword_score = bm25_scores[idx] / max_bm25
                            combined_scores[idx] = combined_scores.get(idx, 0) + sparse_w * keyword_score

                    candidate_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:20]

                    # Step 3: Cross-Encoder Reranking
                    status.write("🎯 Cross-Encoder Reranking top candidate passages...")
                    rerank_pairs = []
                    for chunk_idx, _ in candidate_results:
                        chunk = st.session_state.chunks[chunk_idx]
                        rerank_pairs.append([search_query, chunk["text"]])

                    if rerank_pairs:
                        rerank_scores = reranker.predict(rerank_pairs)
                    else:
                        rerank_scores = []

                    reranked_results = []
                    for i, (chunk_idx, _) in enumerate(candidate_results):
                        reranked_results.append((chunk_idx, float(rerank_scores[i])))

                    reranked_results.sort(key=lambda x: x[1], reverse=True)
                    top_candidates = reranked_results[:top_k]

                    # Step 4: Construct Grounding Context
                    context = ""
                    sources_data = []
                    for pos, (chunk_idx, r_score) in enumerate(top_candidates):
                        chunk = st.session_state.chunks[chunk_idx]
                        context += f"\n--- SOURCE {pos + 1} ---\nDocument: {chunk['filename']}\nPage: {chunk['page']}\nText:\n{chunk['text']}\n"
                        sources_data.append({
                            "filename": chunk["filename"],
                            "page": chunk["page"],
                            "score": r_score,
                            "text": chunk["text"]
                        })

                    # Step 5: Synthesize Answer
                    status.write("✍️ Synthesizing grounded response with citations...")
                    final_prompt = f"""
You are an intelligent document research assistant.
Answer the user's question accurately using ONLY the retrieved content from the uploaded PDF documents.

RULES:
1. Use the retrieved context as the primary factual source.
2. Give a clear, structured, and insightful answer. Do not simply copy-paste raw text.
3. Preserve key names, dates, numbers, technical terms, definitions, and facts.
4. If the retrieved context does not contain enough information, state: "The uploaded documents do not contain enough information to answer this question."
5. Do not hallucinate or extrapolate beyond the provided text.
6. Clearly mention the relevant document name and page number when stating facts.
7. Use bullet points or numbered steps whenever describing causes, features, steps, or comparisons.

FORMAT:
## Summary Answer
1-3 concise sentences giving the direct answer.

## Detailed Explanation
Comprehensive explanation with logical paragraphs and bullet points.

## Key Takeaways
- Key point 1
- Key point 2

RETRIEVED DOCUMENT CONTEXT:
{context}

ORIGINAL QUESTION:
{user_query}
"""
                    try:
                        answer_text = ask_llm(final_prompt, model_name=selected_model, temperature=temperature)
                        status.update(label="✅ Answer generated successfully!", state="complete")
                    except Exception as e:
                        answer_text = f"❌ Generation error: {e}"
                        status.update(label="❌ Failed to generate response", state="error")

                # Display Answer in Chat
                st.markdown(answer_text)

                # Render Sources under this new answer
                with st.expander(f"📚 Retrieved Sources ({len(sources_data)} citations)", expanded=True):
                    st.caption(f"🔍 **Optimized Query:** `{search_query}`")
                    for src in sources_data:
                        st.markdown(f"""
                            <div class="source-card">
                                <div class="source-meta">
                                    <div style="font-weight: 600; color: #e2e8f0; font-size: 0.88rem;">
                                        📄 {src['filename']}
                                    </div>
                                    <div>
                                        <span class="badge-page">Page {src['page']}</span>
                                        <span class="badge-score">Score: {src['score']:.4f}</span>
                                    </div>
                                </div>
                                <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.5; font-style: italic; background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px;">
                                    "{src['text'][:320]}..."
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                # Save to History
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources_data,
                    "search_query": search_query
                })

# ---------------------------------------------------------
# TAB 2: DOCUMENT & CHUNK EXPLORER
# ---------------------------------------------------------

with tab_explorer:
    st.markdown("### 📑 Document Passage Explorer")
    st.caption("Inspect and test the search index directly across your documents without generating LLM responses.")

    if not st.session_state.chunks:
        st.info("💡 Upload PDF documents in the sidebar to inspect chunks and test retrieval.")
    else:
        test_search = st.text_input("🔎 Test Vector & Keyword Search across indexed chunks", placeholder="Enter any keyword or question to see matching chunks...")
        
        if test_search.strip():
            # Run quick search
            q_emb = embedding_model.encode([test_search])
            q_emb = np.array(q_emb).astype("float32")
            dists, idxs = st.session_state.index.search(q_emb, min(10, len(st.session_state.chunks)))
            
            st.markdown(f"#### Top Matches for: *'{test_search}'*")
            for rank, chunk_idx in enumerate(idxs[0]):
                c = st.session_state.chunks[chunk_idx]
                dist = dists[0][rank]
                similarity = 1.0 / (1.0 + dist)
                
                with st.expander(f"#{rank + 1} — 📄 {c['filename']} (Page {c['page']}) • Match: {similarity * 100:.1f}%"):
                    st.markdown(f"**Document:** `{c['filename']}` | **Page:** `{c['page']}` | **L2 Distance:** `{dist:.4f}`")
                    st.text_area("Passage Text", c["text"], height=120, key=f"explorer_res_{rank}")
        else:
            # Filter chunks by document
            selected_doc = st.selectbox("Filter chunks by document:", options=["All Documents"] + st.session_state.documents)
            filtered_chunks = [c for c in st.session_state.chunks if selected_doc == "All Documents" or c["filename"] == selected_doc]
            
            st.caption(f"Displaying {len(filtered_chunks)} chunks:")
            for i, chunk in enumerate(filtered_chunks[:30]):
                with st.expander(f"Chunk #{i + 1} — {chunk['filename']} (Page {chunk['page']})"):
                    st.write(chunk["text"])

# ---------------------------------------------------------
# TAB 3: PIPELINE ARCHITECTURE & METRICS
# ---------------------------------------------------------

with tab_pipeline:
    st.markdown("### 📊 System Architecture & Index Analytics")
    st.caption("Deep visibility into the multi-stage RAG retrieval and reranking pipeline.")

    col_pipe1, col_pipe2 = st.columns([1, 1])

    with col_pipe1:
        st.markdown("#### 🔄 Multi-Stage Retrieval Flow")
        steps = [
            ("1. Contextual Query Rewriting", "LLM resolves conversational pronouns and produces a targeted standalone query.", "#6366f1", "🔄"),
            ("2. Dense Semantic Search (FAISS)", "Encodes query via all-MiniLM-L6-v2 and extracts top semantic vectors via L2 distance.", "#8b5cf6", "🧠"),
            ("3. Sparse Lexical Search (BM25)", "Tokenizes query to extract keyword-frequency matches across document passages.", "#a855f7", "📚"),
            ("4. Hybrid Rank Fusion", f"Blends {int(semantic_weight*100)}% Dense + {int((1-semantic_weight)*100)}% Sparse scores to retrieve top 20 candidate passages.", "#ec4899", "⚡"),
            ("5. Cross-Encoder Reranker", "ms-marco-MiniLM-L-6-v2 scores (Query, Passage) pairs with deep cross-attention.", "#f43f5e", "🎯"),
            ("6. Grounded Synthesis & Citations", f"Top {top_k} reranked passages injected into OpenRouter prompt with page-level citations.", "#10b981", "✍️")
        ]
        for title, desc, color, icon in steps:
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.02); border-left: 4px solid {color}; border-radius: 8px; padding: 10px 14px; margin-bottom: 10px; border-top: 1px solid rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <div style="font-weight: 600; font-size: 0.9rem; color: #f8fafc; display: flex; align-items: center; gap: 8px;">
                        <span>{icon}</span> {title}
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 3px; line-height: 1.4;">
                        {desc}
                    </div>
                </div>
            """, unsafe_allow_html=True)

    with col_pipe2:
        st.markdown("#### ⚙️ Pipeline Specifications")
        st.markdown(f"""
        | Component | Configuration | Status |
        | :--- | :--- | :--- |
        | **Embedding Model** | `all-MiniLM-L6-v2` (384-dim) | 🟢 Active |
        | **Vector Database** | `FAISS IndexFlatL2` | 🟢 {len(st.session_state.chunks)} vectors |
        | **Keyword Index** | `BM25Okapi (regex tokenized)` | 🟢 Active |
        | **Cross-Encoder** | `ms-marco-MiniLM-L-6-v2` | 🟢 Active |
        | **Text Splitter** | Recursive (Size: 1000, Overlap: 150) | 🟢 Ready |
        | **LLM Provider** | OpenRouter (`{selected_model}`) | 🟢 Connected |
        """)

        if st.session_state.chunks:
            chunk_lengths = [len(c["text"]) for c in st.session_state.chunks]
            avg_len = sum(chunk_lengths) / len(chunk_lengths)
            st.markdown(f"""
            #### 📈 Corpus Statistics
            - **Total Chunks:** {len(st.session_state.chunks)}
            - **Average Chunk Length:** {avg_len:.0f} characters
            - **Max Chunk Length:** {max(chunk_lengths)} characters
            - **Min Chunk Length:** {min(chunk_lengths)} characters
            """)