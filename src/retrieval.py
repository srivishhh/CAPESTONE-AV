"""
Retrieval Strategies Module
Implements 5 distinct retrieval strategies:
1. Dense Cosine Similarity
2. Maximal Marginal Relevance (MMR)
3. Hybrid Search (BM25 + Dense with Reciprocal Rank Fusion - RRF)
4. Cross-Encoder Reranker (Two-stage: Dense candidate recall -> Cross-Encoder precision)
5. Multi-Query Expansion via LLM
"""
import os
import sys
import re
from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from google import genai

from src.vector_store import VectorStoreManager
from src.chunking import TextChunk
from src.embeddings import cosine_similarity

sys.stdout.reconfigure(encoding="utf-8")

class RAGRetriever:
    def __init__(self, vector_manager: VectorStoreManager, all_chunks: List[TextChunk]):
        self.vm = vector_manager
        self.all_chunks = all_chunks
        self.embedder = vector_manager.embedder

        # Initialize BM25 index over all chunks for Hybrid search
        print("Initializing BM25 index over chunks...")
        self.corpus_tokens = [self._tokenize(c.text) for c in all_chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

        # Initialize Cross-Encoder model for Reranker strategy
        print("Initializing Cross-Encoder model (ms-marco-MiniLM-L-6-v2)...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Initialize Gemini client for Multi-Query expansion
        api_key = os.environ.get("GOOGLE_API_KEY")
        self.gemini_client = genai.Client(api_key=api_key) if api_key else None

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    # -------------------------------------------------------------
    # STRATEGY 1: Dense Retrieval (Cosine Similarity)
    # -------------------------------------------------------------
    def retrieve_dense(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.vm.similarity_search(query, top_k=top_k)
        for r in results:
            r["strategy"] = "Dense (Cosine)"
        return results

    # -------------------------------------------------------------
    # STRATEGY 2: Maximal Marginal Relevance (MMR)
    # -------------------------------------------------------------
    def retrieve_mmr(self, query: str, top_k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.65) -> List[Dict[str, Any]]:
        """
        Balances query relevance and diversity among retrieved chunks to eliminate redundant passages.
        MMR = argmax_{d_i} [ lambda * Sim(query, d_i) - (1-lambda) * max_{d_j in Selected} Sim(d_i, d_j) ]
        """
        candidates = self.vm.similarity_search(query, top_k=fetch_k)
        if not candidates or len(candidates) <= top_k:
            for r in candidates:
                r["strategy"] = "MMR"
            return candidates

        q_emb = self.embedder.embed_query(query)
        cand_embs = self.embedder.embed_documents([c["text"] for c in candidates])

        selected_indices = [0]
        selected_embs = [cand_embs[0]]
        remaining_indices = list(range(1, len(candidates)))

        while len(selected_indices) < top_k and remaining_indices:
            best_score = -float("inf")
            best_idx = remaining_indices[0]

            for idx in remaining_indices:
                cand_emb = cand_embs[idx]
                sim_to_query = cosine_similarity(q_emb, cand_emb)
                max_sim_to_selected = max(cosine_similarity(cand_emb, s_emb) for s_emb in selected_embs)
                mmr_score = lambda_mult * sim_to_query - (1 - lambda_mult) * max_sim_to_selected

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected_indices.append(best_idx)
            selected_embs.append(cand_embs[best_idx])
            remaining_indices.remove(best_idx)

        final_results = []
        for rank, idx in enumerate(selected_indices):
            hit = dict(candidates[idx])
            hit["strategy"] = "MMR"
            hit["mmr_rank"] = rank + 1
            final_results.append(hit)
        return final_results

    # -------------------------------------------------------------
    # STRATEGY 3: Hybrid Search (BM25 + Dense with Reciprocal Rank Fusion)
    # -------------------------------------------------------------
    def retrieve_hybrid(self, query: str, top_k: int = 5, alpha: float = 0.5, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Combines BM25 exact keyword match with dense semantic similarity using Reciprocal Rank Fusion (RRF).
        RRF Score = 1 / (rrf_k + rank_dense) + 1 / (rrf_k + rank_bm25)
        """
        # 1. Dense retrieval
        dense_results = self.vm.similarity_search(query, top_k=top_k * 3)

        # 2. BM25 retrieval
        q_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:top_k * 3]

        # 3. Reciprocal Rank Fusion
        fused_scores = {}
        item_lookup = {}

        # Dense ranks
        for rank, item in enumerate(dense_results):
            cid = item["metadata"]["chunk_id"]
            item_lookup[cid] = item
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1)) * (1.0 - alpha)

        # BM25 ranks
        for rank, idx in enumerate(top_bm25_idx):
            chunk = self.all_chunks[idx]
            cid = chunk.chunk_id
            if cid not in item_lookup:
                item_lookup[cid] = {
                    "text": chunk.text,
                    "paper_title": chunk.paper_title,
                    "source": chunk.file_name,
                    "page_number": chunk.page_number,
                    "authors": chunk.metadata.get("authors", "Unknown"),
                    "year": chunk.metadata.get("year", 2024),
                    "score": round(float(bm25_scores[idx]), 4),
                    "metadata": chunk.metadata
                }
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1)) * alpha

        # Sort by fused score
        sorted_cids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)[:top_k]
        results = []
        for cid in sorted_cids:
            hit = dict(item_lookup[cid])
            hit["score"] = round(fused_scores[cid] * 100, 4)  # Normalized score
            hit["strategy"] = "Hybrid (BM25 + Dense RRF)"
            results.append(hit)
        return results

    # -------------------------------------------------------------
    # STRATEGY 4: Cross-Encoder Reranker
    # -------------------------------------------------------------
    def retrieve_reranker(self, query: str, top_k: int = 5, candidate_pool: int = 20) -> List[Dict[str, Any]]:
        """
        Two-stage retrieval:
        Stage 1: High recall dense retrieval (top-20).
        Stage 2: Cross-Encoder (ms-marco-MiniLM-L-6-v2) computes joint query-document relevance logits for high precision.
        """
        candidates = self.vm.similarity_search(query, top_k=candidate_pool)
        if not candidates:
            return []

        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        for c, score in zip(candidates, rerank_scores):
            # Sigmoid normalization for human-readable probability score
            c["rerank_logit"] = float(score)
            c["score"] = round(float(1.0 / (1.0 + np.exp(-score))), 4)
            c["strategy"] = "Cross-Encoder Reranker"

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    # -------------------------------------------------------------
    # STRATEGY 5: Multi-Query Retrieval
    # -------------------------------------------------------------
    def retrieve_multi_query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Expands input query into multiple perspectives using LLM to maximize recall across phrasing variations,
        then aggregates and deduplicates results.
        """
        expanded_queries = [query]
        if self.gemini_client:
            try:
                prompt = (
                    f"You are an AI research assistant. Generate 3 diverse search query variants from different angles "
                    f"to find relevant passages in academic AI papers for this user question: '{query}'. "
                    f"Output ONLY the 3 queries, one per line without numbers or bullets."
                )
                resp = self.gemini_client.models.generate_content(
                    model="gemini-flash-latest",
                    contents=prompt
                )
                lines = [l.strip() for l in resp.text.strip().split("\n") if l.strip()]
                for l in lines[:3]:
                    clean_l = re.sub(r"^\d+[\.\)]\s*", "", l).strip()
                    if clean_l and clean_l != query:
                        expanded_queries.append(clean_l)
            except Exception as e:
                print(f"Warning: Multi-query generation fallback ({e})")

        all_hits = {}
        for q in expanded_queries:
            hits = self.vm.similarity_search(q, top_k=top_k)
            for h in hits:
                cid = h["metadata"]["chunk_id"]
                if cid not in all_hits or h["score"] > all_hits[cid]["score"]:
                    h["strategy"] = "Multi-Query"
                    all_hits[cid] = h

        sorted_hits = sorted(all_hits.values(), key=lambda x: x["score"], reverse=True)
        return sorted_hits[:top_k]

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    p_dir = os.path.join(project_dir, "data", "papers")
    chroma_dir = os.path.join(project_dir, "data", "chroma_db")

    from src.ingestion import load_research_papers
    from src.chunking import chunk_recursive
    from src.embeddings import HuggingFaceEmbedder

    docs = load_research_papers(p_dir)
    chunks = chunk_recursive(docs)
    embedder = HuggingFaceEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    vm = VectorStoreManager(chroma_dir, embedder)
    
    retriever = RAGRetriever(vm, chunks)

    q = "What is the key advantage of Low-Rank Adaptation (LoRA)?"
    print(f"\nEvaluating query: '{q}' across 5 strategies...")
    for strat, fn in [
        ("1. Dense Cosine", lambda: retriever.retrieve_dense(q, top_k=3)),
        ("2. MMR", lambda: retriever.retrieve_mmr(q, top_k=3)),
        ("3. Hybrid (BM25+Dense)", lambda: retriever.retrieve_hybrid(q, top_k=3)),
        ("4. Cross-Encoder Reranker", lambda: retriever.retrieve_reranker(q, top_k=3)),
        ("5. Multi-Query", lambda: retriever.retrieve_multi_query(q, top_k=3))
    ]:
        res = fn()
        top_hit = res[0] if res else None
        print(f"\n--- {strat} ---")
        if top_hit:
            print(f"Top Hit: {top_hit['paper_title']} (Page {top_hit['page_number']}) | Score: {top_hit['score']}")
            print(f"Snippet: {repr(top_hit['text'][:100])}")
