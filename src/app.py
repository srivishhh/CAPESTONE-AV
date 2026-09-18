"""
Streamlit Application: Research Paper Answer Bot
GenAI Pinnacle Plus Capstone — Advanced Option 2
Author: Antigravity AI / Pinnacle Plus
"""
import os
import sys
import time
import html
from typing import List, Dict, Any
import streamlit as st
import pandas as pd
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.ingestion import load_research_papers, PAPER_METADATA_REGISTRY
from src.chunking import chunk_recursive
from src.embeddings import HuggingFaceEmbedder
from src.vector_store import VectorStoreManager
from src.retrieval import RAGRetriever
from src.rag_chain import GroundedRAGChain

# Page setup
st.set_page_config(
    page_title="Research Paper Answer Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling with robust theme compatibility
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #3B82F6;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .source-card {
        background-color: #1E293B;
        color: #F8FAFC !important;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        margin-bottom: 12px;
        border-radius: 6px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.25);
    }
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .source-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #60A5FA !important;
    }
    .source-meta {
        font-size: 0.84rem;
        color: #CBD5E1 !important;
        margin-bottom: 8px;
        line-height: 1.5;
    }
    .source-meta b {
        color: #94A3B8 !important;
    }
    .source-meta code {
        background-color: #0F172A !important;
        color: #38BDF8 !important;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.80rem;
    }
    .source-content {
        font-size: 0.87rem;
        line-height: 1.5;
        color: #F1F5F9 !important;
        background-color: #0F172A;
        padding: 10px 14px;
        border-radius: 4px;
        border: 1px solid #334155;
    }
    .metric-badge {
        background-color: rgba(59, 130, 246, 0.2);
        color: #93C5FD !important;
        border: 1px solid rgba(59, 130, 246, 0.4);
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Initializing RAG Knowledge Base & Models...")
def init_rag_system():
    p_dir = os.path.join(project_dir, "data", "papers")
    chroma_dir = os.path.join(project_dir, "data", "chroma_db")

    docs = load_research_papers(p_dir)
    chunks = chunk_recursive(docs)
    embedder = HuggingFaceEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    vm = VectorStoreManager(chroma_dir, embedder)
    
    # Check if index exists, index if empty
    if vm.collection.count() == 0:
        vm.index_chunks(chunks)

    retriever = RAGRetriever(vm, chunks)
    rag_chain = GroundedRAGChain(retriever)
    return docs, chunks, retriever, rag_chain

docs, chunks, retriever, rag_chain = init_rag_system()

def render_source_cards(sources: List[Dict[str, Any]]):
    """Render Top-K supporting sources with high-contrast formatting and defensive text fallbacks."""
    for src in sources:
        rank = src.get("rank", 1)
        paper_title = html.escape(str(src.get("paper_title") or "Unknown Paper"))
        page = src.get("page_number")
        page_str = str(page) if page is not None and str(page) != "-1" else "N/A"
        chunk_id = html.escape(str(src.get("chunk_id") or "N/A"))
        score = src.get("score", 0.0)
        score_str = f"{float(score):.4f}" if isinstance(score, (int, float)) else str(score)
        passage = src.get("passage") or src.get("text", "")

        if not passage or not passage.strip():
            content_html = "<span style='color: #EF4444; font-style: italic;'>Source text unavailable</span>"
        else:
            clean_p = html.escape(passage.strip())
            content_html = f'"{clean_p}"'

        card_html = f"""
        <div class='source-card'>
            <div class='source-header'>
                <span class='source-title'>SOURCE {rank}</span>
                <span class='metric-badge'>Score: {score_str}</span>
            </div>
            <div class='source-meta'>
                <b>Paper:</b> {paper_title}<br/>
                <b>Page:</b> {page_str} &nbsp;|&nbsp; <b>Chunk:</b> <code>{chunk_id}</code>
            </div>
            <div class='source-content'>
                <b>Content:</b><br/>
                {content_html}
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

def render_debug_inspector(debug_info: Dict[str, Any], prompt_context: str = ""):
    """Render comprehensive developer diagnostics."""
    with st.expander("🛠️ Show RAG Debug Information", expanded=True):
        st.markdown(f"**QUERY:** `{debug_info.get('query', '')}`")
        if debug_info.get("matched_papers"):
            st.markdown(f"**GENERIC PAPER TITLE MATCHES:** `{', '.join(debug_info['matched_papers'])}`")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 🔹 Dense Retrieval (Top Candidates)")
            for i, r in enumerate(debug_info.get("dense_results", []), 1):
                cid = r.get("chunk_id") or r.get("metadata", {}).get("chunk_id", "N/A")
                st.markdown(
                    f"**{i}. {r.get('paper_title', 'Unknown')}** (Page {r.get('page_number', 'N/A')})\n\n"
                    f"Score: `{r.get('score', 0.0)}` | Chunk: `{cid}`\n\n"
                    f"> *{(r.get('text', ''))[:160]}...*"
                )

        with col2:
            st.markdown("##### 🔸 BM25 Retrieval (Top Candidates)")
            for i, r in enumerate(debug_info.get("bm25_results", []), 1):
                cid = r.get("chunk_id") or "N/A"
                st.markdown(
                    f"**{i}. {r.get('paper_title', 'Unknown')}** (Page {r.get('page_number', 'N/A')})\n\n"
                    f"Score: `{r.get('score', 0.0)}` | Chunk: `{cid}`\n\n"
                    f"> *{(r.get('text', ''))[:160]}...*"
                )

        st.markdown("##### ⚡ Hybrid / Fused Retrieval")
        for i, r in enumerate(debug_info.get("hybrid_results", []), 1):
            cid = r.get("chunk_id") or "N/A"
            st.markdown(
                f"**{i}. {r.get('paper_title', 'Unknown')}** (Page {r.get('page_number', 'N/A')})\n\n"
                f"Fused Score: `{r.get('score', 0.0)}` | Chunk: `{cid}`\n\n"
                f"> *{(r.get('text', ''))[:200]}...*"
            )

        st.markdown("##### 📄 Final Context Passed to LLM")
        st.text_area("Exact Context Text", value=prompt_context, height=180, disabled=True)

        st.markdown("##### 🏷️ Source Metadata (Complete Schema)")
        for i, r in enumerate(debug_info.get("final_hits", []), 1):
            with st.expander(f"Metadata — [{i}] {r.get('paper_title')} (p.{r.get('page_number')})"):
                st.json({
                    "chunk_id": r.get("chunk_id"),
                    "paper_id": r.get("paper_id") or r.get("source_file"),
                    "paper_title": r.get("paper_title"),
                    "authors": r.get("authors"),
                    "year": r.get("year"),
                    "page_number": r.get("page_number"),
                    "source_file": r.get("source_file"),
                    "score": r.get("score"),
                    "strategy": r.get("strategy")
                })

@st.cache_data
def get_benchmark_dataframe(csv_path: str):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_data
def get_eda_plots(eda_dir: str):
    if os.path.exists(eda_dir):
        return sorted([f for f in os.listdir(eda_dir) if f.endswith(".png")])
    return []

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=64)
    st.markdown("### ⚙️ RAG Configuration")
    
    selected_strategy = st.selectbox(
        "Retrieval Strategy",
        options=[
            "Hybrid (BM25 + Dense RRF) (⚡ Ultra-Fast ~30ms)",
            "Cross-Encoder Reranker (🎯 High Accuracy)",
            "Dense Cosine Similarity",
            "Maximal Marginal Relevance (MMR)",
            "Multi-Query Expansion"
        ],
        index=0,
        help="Select the retrieval algorithm. Hybrid Search provides sub-second query latency with high accuracy."
    )

    top_k = st.slider("Top Supporting Passages (k)", min_value=1, max_value=5, value=3)

    st.markdown("---")
    st.markdown("### 🛠️ Developer Diagnostics")
    show_debug = st.checkbox("Show RAG Debug Information", value=False, help="Inspect raw Dense, BM25, Hybrid retrieval, context, and LLM prompt.")

    st.markdown("---")
    st.markdown("### 📑 Indexed Paper Library")
    st.markdown(f"**Total Papers:** `{len(PAPER_METADATA_REGISTRY)}` | **Pages:** `{len(docs)}` | **Chunks:** `{len(chunks)}`")
    
    with st.expander("View Included Papers"):
        for fname, info in PAPER_METADATA_REGISTRY.items():
            st.markdown(f"- **{info['title']}**\n  *{info['authors'][:35]}... ({info['year']})*")

    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        rag_chain.clear_history()
        st.rerun()

# -------------------------------------------------------------
# TABS INTERFACE
# -------------------------------------------------------------
tab_chat, tab_benchmark, tab_eda = st.tabs(["💬 Q&A Chatbot", "📊 Retrieval Benchmark", "📈 Dataset EDA"])

# TAB 1: Chatbot Interface
with tab_chat:
    st.markdown("<div class='main-header'>Research Paper Answer Bot</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Grounded, academic question-answering over seminal Generative AI & Transformer research papers with exact Top-3 source citations.</div>",
        unsafe_allow_html=True
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your Research Paper Answer Bot. Ask me any question about **Transformers**, **BERT**, **RAG**, **LoRA**, or **LLaMA**, and I will provide factually grounded answers with exact paper and page citations.",
                "sources": [],
                "debug_info": None,
                "context_prompt": ""
            }
        ]

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Top-{len(msg['sources'])} Supporting Sources", expanded=False):
                    render_source_cards(msg["sources"])
            if show_debug and msg.get("debug_info"):
                render_debug_inspector(msg["debug_info"], msg.get("context_prompt", ""))

    # Sample queries shortcuts
    st.markdown("**Try a sample question:**")
    col1, col2, col3 = st.columns(3)
    sample_q = None
    if col1.button("Attention Scaling Formula"):
        sample_q = "What is the scaled dot-product attention equation and why is scaling applied?"
    if col2.button("BERT MLM Objective"):
        sample_q = "What are the two pre-training objectives used for BERT, and how does Masked LM work?"
    if col3.button("LoRA Weight Decomposition"):
        sample_q = "How does LoRA decompose the weight update matrix W into low-rank matrices?"

    user_input = st.chat_input("Ask a question about the research papers...") or sample_q

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response with real-time streaming
        with st.chat_message("assistant"):
            clean_strategy = selected_strategy.split("(")[0].strip().lower()
            with st.spinner(f"Retrieving context passages via {selected_strategy.split('(')[0].strip()}..."):
                t_start = time.time()
                debug_info = retriever.retrieve_with_debug(user_input, strategy=clean_strategy, top_k=top_k)
                hits = debug_info["final_hits"]
                context_prompt = rag_chain.build_prompt(user_input, hits)
            
            # Stream the generated answer directly into the UI
            stream_gen = rag_chain.stream_answer(user_input, hits)
            answer_text = st.write_stream(stream_gen)
            latency = round(time.time() - t_start, 2)

            st.caption(f"⚡ Generated in {latency}s via **{selected_strategy.split('(')[0].strip()}**")

            top_sources = rag_chain.format_citations(hits, clean_strategy)
            if top_sources:
                with st.expander(f"📚 Top-{len(top_sources)} Supporting Sources", expanded=True):
                    render_source_cards(top_sources)

            if show_debug:
                render_debug_inspector(debug_info, context_prompt)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": top_sources,
            "debug_info": debug_info,
            "context_prompt": context_prompt
        })
        rag_chain.chat_history.append({"user": user_input, "assistant": answer_text})

# TAB 2: Retrieval Benchmark
with tab_benchmark:
    st.header("📊 Multi-Strategy Retrieval Benchmark")
    st.markdown("Comparison across **5 distinct retrieval strategies** on the 12-query benchmark suite:")

    summary_csv = os.path.join(project_dir, "data", "eval_results", "evaluation_summary.csv")
    df_bench = get_benchmark_dataframe(summary_csv)
    if df_bench is not None:
        st.dataframe(df_bench, use_container_width=True)

        colA, colB = st.columns(2)
        with colA:
            st.subheader("Hit Rate Comparison")
            st.bar_chart(data=df_bench.set_index("Retrieval Strategy")[["Hit Rate @ 1", "Hit Rate @ 3"]])
        with colB:
            st.subheader("Mean Reciprocal Rank (MRR)")
            st.bar_chart(data=df_bench.set_index("Retrieval Strategy")["MRR"])
    else:
        st.info("Run `python -m src.evaluation` to generate benchmark data.")

# TAB 3: Dataset EDA
with tab_eda:
    st.header("📈 Research Paper Dataset Exploratory Data Analysis")
    eda_dir = os.path.join(project_dir, "data", "eda_plots")
    plot_files = get_eda_plots(eda_dir)
    if plot_files:
        selected_plot = st.selectbox("Select EDA Visualization", options=plot_files)
        st.image(os.path.join(eda_dir, selected_plot), use_container_width=True)
    else:
        st.info("No EDA plots found. Generate via `python -m src.eda`.")
