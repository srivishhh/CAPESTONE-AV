import json
import numpy as np
from src.ingestion import load_research_papers
from src.chunking import chunk_recursive
from src.embeddings import HuggingFaceEmbedder
from src.vector_store import VectorStoreManager
from src.retrieval import RAGRetriever
from src.rag_chain import GroundedRAGChain

docs = load_research_papers("data/papers")
chunks = chunk_recursive(docs)
vm = VectorStoreManager("data/chroma_db", HuggingFaceEmbedder("sentence-transformers/all-MiniLM-L6-v2"))
retriever = RAGRetriever(vm, chunks)
rag = GroundedRAGChain(retriever)

q = "tell me about llama"
print(f"=== QUERY: '{q}' ===")

# 1. DENSE
dense = retriever.retrieve_dense(q, top_k=5)
print("\n--- DENSE RESULTS ---")
for i, d in enumerate(dense):
    print(f"[{i}] Title: '{d.get('paper_title')}' | Page: {d.get('page_number')} | Score: {d.get('score')}")
    print(f"     Chunk ID: {d.get('metadata', {}).get('chunk_id')}")
    print(f"     Snippet: {d.get('text', '')[:140]}...\n")

# 2. BM25
q_toks = retriever._tokenize(q)
print(f"Tokenized query for BM25: {q_toks}")
bm25_scores = retriever.bm25.get_scores(q_toks)
top_idx = np.argsort(bm25_scores)[::-1][:5]
print("\n--- BM25 RESULTS ---")
for idx in top_idx:
    c = chunks[idx]
    print(f"Score: {bm25_scores[idx]:.4f} | Title: '{c.paper_title}' | Page: {c.page_number} | Chunk ID: {c.chunk_id}")
    print(f"     Snippet: {c.text[:140]}...\n")

# 3. HYBRID
hybrid = retriever.retrieve_hybrid(q, top_k=3)
print("\n--- HYBRID RESULTS ---")
for i, h in enumerate(hybrid):
    print(f"[{i}] Title: '{h.get('paper_title')}' | Page: {h.get('page_number')} | Score: {h.get('score')}")
    print(f"     Chunk ID: {h.get('metadata', {}).get('chunk_id')}")
    print(f"     Snippet: {h.get('text', '')[:140]}...\n")

# 4. FINAL PROMPT
prompt = rag.build_prompt(q, hybrid)
print("\n--- FINAL PROMPT TO LLM ---")
print(prompt)

# 5. LLM ANSWER
print("\n--- GENERATING ANSWER ---")
ans = rag.generate_answer(q, strategy="hybrid", top_k=3)
print(ans["answer"])
