# 📚 Research Paper Answer Bot — GenAI Pinnacle Plus Capstone
> **Analytics Vidhya GenAI Pinnacle Plus Capstone Project**  
> **Track**: Advanced Option 2 — Streamlit Interactive Application  
> **Author**: Sri Vishnu  
> **Technologies**: Python 3.12, LangChain, ChromaDB, SentenceTransformers, Cross-Encoder, Rank-BM25, Google Gemini API, Streamlit

---

## 🎯 Project Overview
The **Research Paper Answer Bot** is a production-grade, academically grounded Retrieval-Augmented Generation (RAG) system built over foundational research papers in Generative AI and Large Language Models.

The bot allows researchers, engineers, and students to query complex academic concepts and receive **factually grounded answers** accompanied by **exact Top-3 supporting citations** (Paper Title, Page Number, and verbatim passage excerpt) with **0% hallucination** on out-of-domain queries.

---

## 📑 Curated Research Paper Dataset
The system indexes 5 seminal GenAI research papers spanning **103 pages** and **305,258 characters**:
1. **Attention Is All You Need** (*Vaswani et al., 2017*) — 15 pages
2. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** (*Devlin et al., 2018*) — 16 pages
3. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (*Lewis et al., 2020*) — 19 pages
4. **LoRA: Low-Rank Adaptation of Large Language Models** (*Hu et al., 2021*) — 26 pages
5. **LLaMA: Open and Efficient Foundation Language Models** (*Touvron et al., 2023*) — 27 pages

---

## 🏗️ Architecture Pipeline
```text
User Question
      ↓
Streamlit UI
      ↓
Retrieval Strategy Selection
(Dense Cosine / MMR / Hybrid BM25+Dense / Cross-Encoder Reranker / Multi-Query)
      ↓
Persistent ChromaDB Vector Store (630 Chunks with full metadata)
      ↓
Top-K Context Passages Assembly
      ↓
Grounded RAG Prompt Template (Strict Anti-Hallucination Guardrail)
      ↓
Pretrained LLM (Gemini Flash)
      ↓
Grounded Synthesis + Top-3 Supporting Citations (Title, Page #, Excerpt)
      ↓
Streamlit Interactive UI / Chatbot
```

---

## 📊 Experimental Benchmarks & Evaluation

### 1. Embedding Models Comparison
| Embedding Model | Type | Dimension | Latency (ms/chunk) | Relevant Cosine Sim | Unrelated Cosine Sim | Discrimination Margin |
|---|---|---|---|---|---|---|
| **all-MiniLM-L6-v2** | Open-Source | 384 | **13.80 ms** | 0.6466 | 0.2772 | **0.3694** |
| **gemini-embedding-001** | Commercial | 3072 | 736.31 ms | 0.8155 | 0.6198 | 0.1957 |

*Selection*: `all-MiniLM-L6-v2` executes 53x faster on local CPU with zero network overhead and a wide discrimination margin (0.3694).

### 2. Multi-Strategy Retrieval Benchmark (12 Evaluation Queries)
| Retrieval Strategy | Hit Rate @ 1 | Hit Rate @ 3 | Hit Rate @ 5 | MRR | Avg Latency (ms) |
|---|---|---|---|---|---|
| **Dense Cosine** | 90.9% | 100.0% | 100.0% | 0.9545 | 47.5 ms |
| **MMR (Max Marginal Relevance)** | 90.9% | 100.0% | 100.0% | 0.9545 | 1289.7 ms |
| **Hybrid (BM25 + Dense RRF)** | 90.9% | 100.0% | 100.0% | 0.9545 | **30.4 ms** |
| **Cross-Encoder Reranker** | **100.0%** | **100.0%** | **100.0%** | **1.0000** | 1438.6 ms |
| **Multi-Query Expansion** | 90.9% | 100.0% | 100.0% | 0.9545 | 784.3 ms |

*Key Findings*:
- **Cross-Encoder Reranker** delivers **100% Hit Rate @ 1** and a perfect **1.0000 MRR**, eliminating false positives via deep cross-attention.
- **Hybrid Search (BM25 + Dense RRF)** achieves the lowest latency (**30.4 ms**), capturing rare keywords like *SwiGLU*, *SentencePiece*, and *DPR*.
- **Anti-Hallucination Verification**: Negative control queries strictly return *"I do not have sufficient information in the provided research papers to answer this question."* (0% hallucination).

---

## 📂 Repository Structure
```text
CAPESTONE-AV/
├── data/
│   ├── papers/                 # Seminal research paper PDFs
│   ├── eda_plots/              # 5 Publication-grade EDA figures
│   └── eval_results/           # CSV summaries & benchmark reports
├── notebooks/
│   └── research_paper_answer_bot.ipynb  # End-to-end reproducible notebook
├── presentation/
│   └── presentation_slides.md  # 12-Slide capstone defense presentation
├── src/
│   ├── __init__.py
│   ├── ingestion.py            # PDF loading & metadata extraction
│   ├── chunking.py             # Recursive, Fixed, and Semantic chunking
│   ├── eda.py                  # 5 In-depth EDA visualizations
│   ├── embeddings.py           # Open-source & Commercial embedding benchmarks
│   ├── vector_store.py         # Persistent ChromaDB indexing
│   ├── retrieval.py            # 5 Retrieval strategies (Dense, MMR, Hybrid, Reranker, Multi-Q)
│   ├── rag_chain.py            # Grounded RAG chain & citation extractor
│   ├── evaluation.py           # 12-Query quantitative benchmark suite
│   └── app.py                  # Streamlit web application
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Quick Start Guide

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/srivishhh/CAPESTONE-AV.git
cd CAPESTONE-AV

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Keys
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Launch the Streamlit Application (Stretch Goal)
```bash
streamlit run src/app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run the Jupyter Notebook
```bash
jupyter notebook notebooks/research_paper_answer_bot.ipynb
```

---

## 🎓 Capstone Rubric Alignment
- **Problem Understanding & Data Preparation (15 Marks)**: Clean ingestion of 103 pages with complete metadata retention and OCR diagnosis.
- **Vector Database & Embedding Strategy (15 Marks)**: Rigorous comparison of Open-Source vs Commercial embeddings; ChromaDB indexing.
- **Retrieval Strategy (15 Marks)**: 5 strategies implemented with quantitative metrics (Hit Rate, MRR, Latency).
- **RAG Pipeline & Stretch Goal (15 Marks)**: Factually grounded RAG chain with Top-3 citations and full Streamlit web application.
- **Technical Understanding & Viva (40 Marks)**: Structured code, clean documentation, and slide deck for expert evaluation.
