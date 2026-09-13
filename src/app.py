"""
Streamlit Application: Research Paper Answer Bot
GenAI Pinnacle Plus Capstone — Advanced Option 2
Author: Antigravity AI / Pinnacle Plus
"""
import os
import sys
import time
import streamlit as st
import pandas as pd
from PIL import Image

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

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .source-card {
        background-color: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-radius: 4px;
    }
    .metric-badge {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 3px 8px;
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

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=64)
    st.markdown("### ⚙️ RAG Configuration")
    
    selected_strategy = st.selectbox(
        "Retrieval Strategy",
        options=[
            "Cross-Encoder Reranker (Recommended)",
            "Hybrid (BM25 + Dense RRF)",
            "Dense Cosine Similarity",
            "Maximal Marginal Relevance (MMR)",
            "Multi-Query Expansion"
        ],
        index=0,
        help="Select the retrieval algorithm used to select context passages."
    )

    top_k = st.slider("Top Supporting Passages (k)", min_value=1, max_value=5, value=3)

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
                "sources": []
            }
        ]

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Top-{len(msg['sources'])} Supporting Sources", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(f"""
                        <div class='source-card'>
                            <b>[{src['rank']}] {src['paper_title']}</b> — Page {src['page_number']} <span class='metric-badge'>Score: {src['score']}</span><br/>
                            <i>"{src['passage']}"</i>
                        </div>
                        """, unsafe_allow_html=True)

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

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner(f"Retrieving passages via {selected_strategy} & synthesizing grounded answer..."):
                t_start = time.time()
                strategy_key = selected_strategy.lower()
                result = rag_chain.generate_answer(user_input, strategy=strategy_key, top_k=top_k)
                latency = round(time.time() - t_start, 2)

                st.markdown(result["answer"])
                st.caption(f"⚡ Generated in {latency}s via **{selected_strategy}**")

                if result["top_sources"]:
                    with st.expander(f"📚 Top-{len(result['top_sources'])} Supporting Sources", expanded=True):
                        for src in result["top_sources"]:
                            st.markdown(f"""
                            <div class='source-card'>
                                <b>[{src['rank']}] {src['paper_title']}</b> — Page {src['page_number']} <span class='metric-badge'>Score: {src['score']}</span><br/>
                                <i>"{src['passage']}"</i>
                            </div>
                            """, unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["top_sources"]
        })

# TAB 2: Retrieval Benchmark
with tab_benchmark:
    st.header("📊 Multi-Strategy Retrieval Benchmark")
    st.markdown("Comparison across **5 distinct retrieval strategies** on the 12-query benchmark suite:")

    summary_csv = os.path.join(project_dir, "data", "eval_results", "evaluation_summary.csv")
    if os.path.exists(summary_csv):
        df_bench = pd.read_csv(summary_csv)
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
    
    if os.path.exists(eda_dir):
        plot_files = sorted([f for f in os.listdir(eda_dir) if f.endswith(".png")])
        for p in plot_files:
            p_path = os.path.join(eda_dir, p)
            st.image(p_path, caption=p.replace(".png", "").replace("_", " ").title(), use_container_width=True)
    else:
        st.info("Run `python -m src.eda` to generate EDA visual plots.")
