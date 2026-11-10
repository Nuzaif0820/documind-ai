"""
DocuMind AI - Intelligent PDF Assistant
Streamlit Frontend Application
"""

import os
import json
import time
from typing import List, Dict, Any
import streamlit as st

from config import (
    DEFAULT_GROQ_MODEL,
    SUPPORTED_GROQ_MODELS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    MIN_TOP_K,
    MAX_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    MAX_FILE_SIZE_MB,
    get_groq_api_key
)
from document_processor import DocumentProcessor, PDFProcessingError
from embeddings import get_embedding_model
from vector_store import VectorStoreManager
from rag_chain import RAGPipeline


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="DocuMind AI — Intelligent PDF Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Global styles */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    /* Citation chip */
    .citation-badge {
        display: inline-flex;
        align-items: center;
        background-color: #EEF2FF;
        color: #3730A3;
        border: 1px solid #C7D2FE;
        border-radius: 6px;
        padding: 4px 10px;
        margin-right: 8px;
        margin-top: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    /* Score indicator */
    .score-badge {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    /* Stat cards */
    .stat-card {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .stat-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1F2937;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #6B7280;
        text-transform: uppercase;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    # Map: filename -> stats dict
    st.session_state.indexed_files = {}

@st.cache_resource(show_spinner="Loading Sentence Transformer Embeddings...")
def load_cached_embedding_model():
    """Cache the embedding model in RAM across user interactions."""
    return get_embedding_model()

# Initialize Vector Store Manager
if "vector_store_manager" not in st.session_state:
    embedding_model = load_cached_embedding_model()
    st.session_state.vector_store_manager = VectorStoreManager(embedding_model=embedding_model)

# ---------------------------------------------------------
# Sidebar: Configuration, File Ingestion & Index Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://api.iconify.design/heroicons:academic-cap-solid.svg", width=42)
    st.title("DocuMind AI")
    st.caption("Verifiable RAG-Powered PDF Assistant")

    st.markdown("---")

    # 1. API Key Configuration
    st.subheader("🔑 LLM Configuration")
    env_groq_key = get_groq_api_key()
    user_api_key = st.text_input(
        "Groq API Key",
        value=env_groq_key,
        type="password",
        placeholder="gsk_...",
        help="Obtain a free API key at https://console.groq.com"
    )

    if user_api_key:
        st.success("API Key detected", icon="✅")
    else:
        st.warning("Please supply a Groq API key to query documents.", icon="⚠️")

    selected_model = st.selectbox(
        "Model",
        options=SUPPORTED_GROQ_MODELS,
        index=0,
        help="Select the Groq-hosted LLM model."
    )

    st.markdown("---")

    # 2. PDF Upload & Ingestion
    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Max size {MAX_FILE_SIZE_MB}MB per PDF."
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)}** file(s) selected:")
        for uf in uploaded_files:
            file_size_kb = uf.size / 1024
            st.text(f"• {uf.name} ({file_size_kb:.1f} KB)")

        process_button = st.button("⚡ Process & Index Documents", type="primary", use_container_width=True)

        if process_button:
            processor = DocumentProcessor()
            success_count = 0
            total_chunks_added = 0

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, file_obj in enumerate(uploaded_files):
                status_text.text(f"Extracting & chunking '{file_obj.name}'...")
                try:
                    chunks, stats = processor.process_and_chunk(file_obj, file_obj.name)
                    st.session_state.vector_store_manager.add_documents(chunks, stats)
                    st.session_state.indexed_files[file_obj.name] = stats
                    total_chunks_added += len(chunks)
                    success_count += 1
                except PDFProcessingError as pe:
                    st.error(f"Error in '{file_obj.name}': {str(pe)}")
                except Exception as e:
                    st.error(f"Unexpected error processing '{file_obj.name}': {str(e)}")

                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.empty()
            progress_bar.empty()

            if success_count > 0:
                st.success(
                    f"Successfully indexed {success_count} document(s) into {total_chunks_added} searchable chunks!"
                )

    st.markdown("---")

    # 3. Document Scope & Retrieval Controls
    st.subheader("🎯 Retrieval Scope & Filters")
    indexed_filenames = list(st.session_state.indexed_files.keys())

    if indexed_filenames:
        target_doc_filter = st.selectbox(
            "Filter Search Scope",
            options=["All Documents"] + indexed_filenames,
            index=0,
            help="Direct queries to a specific PDF or across all indexed documents."
        )
    else:
        target_doc_filter = "All Documents"
        st.info("No documents indexed yet.")

    with st.expander("⚙️ Advanced Retrieval Tuning", expanded=False):
        top_k = st.slider("Top-K Chunks to Retrieve", min_value=MIN_TOP_K, max_value=MAX_TOP_K, value=DEFAULT_TOP_K)
        sim_threshold = st.slider(
            "Similarity Score Threshold",
            min_value=0.0,
            max_value=0.8,
            value=DEFAULT_SIMILARITY_THRESHOLD,
            step=0.05,
            help="Chunks with similarity below this cutoff are rejected as irrelevant."
        )
        temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.05)

    st.markdown("---")

    # 4. Session Actions
    st.subheader("🧹 Session Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("Reset Index", use_container_width=True):
            st.session_state.vector_store_manager.clear()
            st.session_state.indexed_files = {}
            st.session_state.messages = []
            st.success("Vector store cleared.")
            st.rerun()


# ---------------------------------------------------------
# Main Interface: Metrics & Header
# ---------------------------------------------------------
st.markdown('<div class="main-title">DocuMind AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Intelligent, citation-grounded PDF Question Answering with transparent source verification.</div>',
    unsafe_allow_html=True
)

# Ingestion Summary Dashboard
doc_stats_col1, doc_stats_col2, doc_stats_col3 = st.columns(3)
total_indexed_docs = len(st.session_state.indexed_files)
total_indexed_chunks = sum(s.get("total_chunks", 0) for s in st.session_state.indexed_files.values())
total_pages = sum(s.get("total_pages", 0) for s in st.session_state.indexed_files.values())

with doc_stats_col1:
    st.markdown(
        f'<div class="stat-card"><div class="stat-number">{total_indexed_docs}</div><div class="stat-label">Indexed Documents</div></div>',
        unsafe_allow_html=True
    )
with doc_stats_col2:
    st.markdown(
        f'<div class="stat-card"><div class="stat-number">{total_pages}</div><div class="stat-label">Total Document Pages</div></div>',
        unsafe_allow_html=True
    )
with doc_stats_col3:
    st.markdown(
        f'<div class="stat-card"><div class="stat-number">{total_indexed_chunks}</div><div class="stat-label">Vector Chunks in FAISS</div></div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Chat History Display
# ---------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display verifiable citations if present
        if msg.get("citations"):
            st.markdown("**Verifiable Source Citations:**")
            badge_html = " ".join([
                f'<span class="citation-badge">📄 {c["source"]} &nbsp;|&nbsp; Page {c["page"]}</span>'
                for c in msg["citations"]
            ])
            st.markdown(badge_html, unsafe_allow_html=True)

        # Display expandable source chunks panel
        if msg.get("retrieved_chunks"):
            with st.expander("🔍 Inspect Retrieved Context & Similarity Scores"):
                for idx, chunk in enumerate(msg["retrieved_chunks"]):
                    score_pct = int(chunk["score"] * 100)
                    st.markdown(
                        f"**Passage {idx+1}** — *{chunk['source']}* (Page {chunk['page']}) "
                        f"&nbsp; <span class=\"score-badge\">Similarity: {score_pct}%</span>",
                        unsafe_allow_html=True
                    )
                    st.text_area(
                        label=f"Content for passage {idx+1}",
                        value=chunk["content"],
                        height=100,
                        disabled=True,
                        key=f"hist_{msg.get('id', '')}_chunk_{idx}"
                    )


# ---------------------------------------------------------
# Chat Input & RAG Execution
# ---------------------------------------------------------
user_query = st.chat_input("Ask a question about your uploaded documents...")

if user_query:
    # 1. Append and display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Check prerequisites
    if not st.session_state.vector_store_manager.is_indexed:
        warning_msg = "Please upload and process at least one PDF in the sidebar before asking questions."
        with st.chat_message("assistant"):
            st.warning(warning_msg)
        st.session_state.messages.append({"role": "assistant", "content": warning_msg})
    elif not user_api_key:
        api_warning = "Please provide a valid Groq API Key in the sidebar to enable answer generation."
        with st.chat_message("assistant"):
            st.warning(api_warning)
        st.session_state.messages.append({"role": "assistant", "content": api_warning})
    else:
        # 3. Instantiate and run RAG Pipeline
        rag_pipeline = RAGPipeline(
            vector_store_manager=st.session_state.vector_store_manager,
            groq_api_key=user_api_key,
            model_name=selected_model,
            temperature=temperature
        )

        with st.chat_message("assistant"):
            with st.spinner("Analyzing document context and synthesizing grounded answer..."):
                start_time = time.time()
                result = rag_pipeline.answer_question(
                    question=user_query,
                    chat_history=st.session_state.messages[:-1],  # Exclude current question
                    top_k=top_k,
                    score_threshold=sim_threshold,
                    filter_source=target_doc_filter if target_doc_filter != "All Documents" else None
                )
                elapsed_time = round(time.time() - start_time, 2)

                # Render Answer
                st.markdown(result["answer"])

                # Render Citations
                if result["citations"]:
                    st.markdown("**Verifiable Source Citations:**")
                    badge_html = " ".join([
                        f'<span class="citation-badge">📄 {c["source"]} &nbsp;|&nbsp; Page {c["page"]}</span>'
                        for c in result["citations"]
                    ])
                    st.markdown(badge_html, unsafe_allow_html=True)

                # Render Retrieved Context Inspector
                if result["retrieved_chunks"]:
                    with st.expander(f"🔍 Inspect Retrieved Context ({len(result['retrieved_chunks'])} chunks, {elapsed_time}s)"):
                        if result.get("standalone_query") and result["standalone_query"] != user_query:
                            st.caption(f"**Reformulated Search Query:** *{result['standalone_query']}*")

                        for idx, chunk in enumerate(result["retrieved_chunks"]):
                            score_pct = int(chunk["score"] * 100)
                            st.markdown(
                                f"**Passage {idx+1}** — *{chunk['source']}* (Page {chunk['page']}) "
                                f"&nbsp; <span class=\"score-badge\">Similarity: {score_pct}%</span>",
                                unsafe_allow_html=True
                            )
                            st.text_area(
                                label=f"Excerpt {idx+1}",
                                value=chunk["content"],
                                height=110,
                                disabled=True,
                                key=f"current_chunk_{idx}"
                            )

        # 4. Save to session state
        st.session_state.messages.append({
            "id": str(time.time()),
            "role": "assistant",
            "content": result["answer"],
            "citations": result["citations"],
            "retrieved_chunks": result["retrieved_chunks"]
        })


# ---------------------------------------------------------
# Chat History Export
# ---------------------------------------------------------
if st.session_state.messages:
    st.markdown("---")
    export_col1, export_col2 = st.columns([1, 4])
    with export_col1:
        # Export as JSON
        chat_export = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            label="📥 Export Chat (JSON)",
            data=chat_export,
            file_name="documind_chat_history.json",
            mime="application/json"
        )
    with export_col2:
        # Export as Markdown
        md_export = "# DocuMind AI - Chat Transcript\n\n"
        for m in st.session_state.messages:
            speaker = "User" if m["role"] == "user" else "DocuMind AI"
            md_export += f"### {speaker}:\n{m['content']}\n\n"
            if m.get("citations"):
                md_export += "**Citations:** " + ", ".join([f"{c['source']} (p. {c['page']})" for c in m["citations"]]) + "\n\n"
        st.download_button(
            label="📝 Export Chat (Markdown)",
            data=md_export,
            file_name="documind_chat_transcript.md",
            mime="text/markdown"
        )
