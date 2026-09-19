---
title: Lumina RAG Universal Assistant
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.42.0
app_file: app.py
pinned: false
license: mit
short_description: Multi-PDF RAG with FAISS, BM25, and Cross-Encoder Reranking
---

# ⚡ Lumina RAG • Universal Document Intelligence

A state-of-the-art multi-document research assistant with Hybrid Retrieval, Cross-Encoder Reranking, and contextual LLM synthesis.

## 🚀 Features
- **Hybrid Retrieval**: Combines Dense Semantic Search (FAISS) with Sparse Lexical Search (BM25).
- **Cross-Encoder Reranking**: Re-evaluates top candidates with `ms-marco-MiniLM-L-6-v2` to suppress false positives.
- **Exact Page-Level Citations**: Every claim is cited with the original document name, page number, and previewable text snippet.
- **Contextual Query Rewriter**: Resolves conversational pronouns across chat history.
- **Multi-Document Support**: Ingest and index multiple PDF documents simultaneously.

## ⚙️ Environment Variables & Secrets
Under your Hugging Face Space **Settings** -> **Variables and secrets**, add:

| Secret Name | Description |
| :--- | :--- |
| `OPENROUTER_API_KEY` | Your OpenRouter API Key (sk-or-v1-...) |

## 🛠️ Local Development
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```
