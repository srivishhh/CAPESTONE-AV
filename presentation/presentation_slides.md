# GenAI Pinnacle Plus Capstone Presentation Deck
## Project: Research Paper Answer Bot
**Track**: Advanced Option 2 — Streamlit Interactive Application  
**Student Name**: Sri Vishnu  
**Program**: GenAI Pinnacle Plus Program (Analytics Vidhya)  
**Date**: September 2026  

---

### Slide 1: Title Slide
- **Title**: Research Paper Answer Bot — Production-Grade Grounded RAG System
- **Subtitle**: Factually Grounded Question Answering over Seminal Generative AI Literature
- **Author**: Sri Vishnu
- **Program**: GenAI Pinnacle Plus Capstone
- **Specialization**: RAG Architecture, Vector DBs, Multi-Strategy Retrieval, Streamlit

---

### Slide 2: Business Case & Problem Statement
- **Context**: ArXiv and enterprise research repositories house thousands of dense, technical AI papers. Researchers, engineers, and students struggle with keyword search limitations and context fragmentation.
- **Problem**: Traditional keyword search fails on semantic nuance; standard LLMs suffer from hallucinations, lack verifiable citations, and cut off at their training date.
- **Objective**: Build a production-grade RAG pipeline that indexes seminal AI papers, retrieves precise context across multiple strategies, and synthesizes answers strictly grounded in literature with verbatim page-level citations.

---

### Slide 3: End-to-End System Architecture
- **Ingestion**: PDF Parser (`pypdf`) with page-level metadata extraction.
- **Chunking**: RecursiveCharacterTextSplitter (700 chars, 120 overlap) retaining title, author, year, and page.
- **Indexing**: Persistent ChromaDB vector store paired with Hugging Face `all-MiniLM-L6-v2`.
- **Multi-Strategy Retrieval**: Dense Cosine, MMR, BM25+Dense Hybrid (RRF), Cross-Encoder Reranker, and Multi-Query.
- **Generation**: Grounded RAG prompt template with strict anti-hallucination guardrails via Google Gemini LLM.
- **Delivery**: Streamlit interactive UI featuring dynamic strategy toggles and collapsible Top-3 citation inspector.

---

### Slide 4: Dataset Ingestion & Characterization
- **Authoritative Dataset**: 5 seminal AI research papers spanning 103 total pages:
  1. *Attention Is All You Need* (Vaswani et al., 2017) — 15 pages
  2. *BERT: Pre-training of Deep Bidirectional Transformers* (Devlin et al., 2018) — 16 pages
  3. *Retrieval-Augmented Generation for Knowledge-Intensive NLP* (Lewis et al., 2020) — 19 pages
  4. *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021) — 26 pages
  5. *LLaMA: Open and Efficient Foundation Language Models* (Touvron et al., 2023) — 27 pages
- **Data Quality & OCR Diagnosis**: All papers were verified as digital vector PDFs with high text extraction density (>2,800 chars/page); OCR was not required.

---

### Slide 5: Chunking Strategy & EDA Findings
- **Chunking Benchmark**:
  - *Recursive Character* (Winner): 630 chunks, Mean 613.9 chars, Std 120.9 chars. Preserves sentence and paragraph boundaries.
  - *Fixed Window*: 718 chunks, Mean 562.3 chars, Std 106.1 chars. Suffers from arbitrary boundary cuts.
  - *Semantic Paragraph*: 103 chunks, Mean 3,324.8 chars. Exceeds optimal embedding window.
- **EDA Insights**:
  - Vocabulary richness (Type-Token Ratio) ranges from 12.4% to 18.2%, highlighting dense technical terminology.
  - Domain concept heatmap confirms strong distinct clustering around specific architectural paradigms (e.g., "SwiGLU" in LLaMA, "Low-Rank" in LoRA).

---

### Slide 6: Embedding Model Benchmark (Open-Source vs Commercial)
- **Experimental Comparison**:
  | Model | Type | Dim | Latency (ms) | Relevant Sim | Unrelated Sim | Discrimination Margin |
  |---|---|---|---|---|---|---|
  | **all-MiniLM-L6-v2** | Open-Source | 384 | **13.80** | 0.6466 | 0.2772 | **0.3694** |
  | **gemini-embedding-001** | Commercial | 3072 | 736.31 | 0.8155 | 0.6198 | 0.1957 |
- **Justification**: Open-source `all-MiniLM-L6-v2` was selected for production indexing due to 53x lower latency, zero API rate-limit bottlenecks, and a wide semantic discrimination margin (0.3694).

---

### Slide 7: Retrieval Strategy Comparison & Vector DB
- **Persistent Storage**: ChromaDB collection storing 630 chunks with full metadata.
- **Benchmark Across 5 Strategies (12 Queries)**:
  | Retrieval Strategy | Hit Rate @ 1 | Hit Rate @ 3 | Hit Rate @ 5 | MRR | Avg Latency (ms) |
  |---|---|---|---|---|---|
  | Dense Cosine | 90.9% | 100.0% | 100.0% | 0.9545 | 47.5 |
  | MMR (Diversity) | 90.9% | 100.0% | 100.0% | 0.9545 | 1289.7 |
  | Hybrid (BM25+Dense) | 90.9% | 100.0% | 100.0% | 0.9545 | **30.4** |
  | **Cross-Encoder Reranker** | **100.0%** | **100.0%** | **100.0%** | **1.0000** | 1438.6 |
  | Multi-Query | 90.9% | 100.0% | 100.0% | 0.9545 | 784.3 |
- **Production Choice**: **Cross-Encoder Reranker** is selected for primary answering due to 100% Top-1 precision, with Hybrid BM25 as high-speed fallback (30.4 ms).

---

### Slide 8: RAG Chain Construction & Anti-Hallucination Guardrails
- **Prompt Engineering**:
  - Explicit instruction: Answer strictly from context passages.
  - Negative abstention rule: If information is missing, output: *"I do not have sufficient information in the provided research papers to answer this question."*
- **Structured Context Injection**: Passages injected with explicit `--- PASSAGE i ---` headers specifying Paper Title and Page Number.
- **Memory**: Multi-turn conversation buffer for contextual follow-up questions.

---

### Slide 9: Results & Grounded Q&A Demo
- **Sample Query 1**: *"What are the two unsupervised pre-training tasks for BERT, and how does Masked LM work?"*
  - **Answer**: Factually synthesized: Masked Language Model (MLM) and Next Sentence Prediction (NSP).
  - **Top-3 Citations**:
    1. *BERT Paper, Page 8* (Confidence: 0.9973)
    2. *BERT Paper, Page 2* (Confidence: 0.9968)
    3. *BERT Paper, Page 4* (Confidence: 0.9953)
- **Sample Query 2**: *"What is the key advantage of Low-Rank Adaptation (LoRA)?"*
  - **Answer**: Injects trainable low-rank decomposition matrices while freezing pretrained weights, reducing VRAM by 3x and parameters by 10,000x.
  - **Top Citation**: *LoRA Paper, Page 2* (Confidence: 0.9997).

---

### Slide 10: Stretch Goal — Streamlit Web Application (Option 2)
- **Interactive UI**:
  - Strategy switcher: Cosine, MMR, Hybrid BM25, Reranker, Multi-Query.
  - Top-k slider (1–5) and dynamic chat history.
  - Collapsible source inspector with verbatim excerpts and similarity badges.
  - Embedded dashboards for live benchmark metrics and dataset EDA plots.
- **Deployment**: Launched locally via `streamlit run src/app.py`.

---

### Slide 11: Challenges & Technical Solutions
1. **API Rate Quotas & Fallbacks**: Free-tier Gemini endpoints occasionally hit per-minute spikes (429/503). *Solution*: Implemented automated exponential backoff, rate-limit pacing, and local rule-based query expansion fallbacks.
2. **Special Characters & Ligatures**: Mathematical PDF notations (`\u221a`, `\ufb01`) caused Windows cp1252 print exceptions. *Solution*: Enforced UTF-8 stream reformatting and normalization across ingestion and retrieval.
3. **Chunk Boundary Truncation**: Arbitrary chunk cuts disrupted mathematical formulas. *Solution*: Optimized Recursive Character Splitting with tailored separators (`\n\n`, `\n`, `. `, ` `).

---

### Slide 12: Conclusion & Viva Preparation
- **Summary**: Delivered an end-to-end, reproducible, production-grade RAG solution exceeding all compulsory and stretch goals.
- **Key Metrics Achieved**:
  - 100% Hit Rate @ 1 and 1.0000 MRR on Cross-Encoder Reranker.
  - 0% Hallucination on out-of-domain negative control queries.
  - 30.4 ms retrieval latency on Hybrid BM25+Dense search.
  - Verified Top-3 verbatim citations with paper title and exact page numbers.
- **Future Enhancements**: Integration of multi-modal vision models for architectural figures and automated agentic Corrective RAG (CRAG) web verification.
