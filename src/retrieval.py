"""
Retrieval Strategies Module
Implements 5 distinct retrieval strategies:
1. Dense Cosine Similarity
2. Maximal Marginal Relevance (MMR)
3. Hybrid Search (BM25 + Dense with Reciprocal Rank Fusion - RRF)
4. Cross-Encoder Reranker (Two-stage: Hybrid candidate recall -> Cross-Encoder precision)
5. Multi-Query Expansion via LLM
"""
import os
import sys
import re
from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from google import genai

load_dotenv()

from src.vector_store import VectorStoreManager
from src.chunking import TextChunk
from src.embeddings import cosine_similarity

sys.stdout.reconfigure(encoding="utf-8")

# Common conversational stopwords that dilute keyword search in queries like "tell me about llama"
CONVERSATIONAL_STOPWORDS = {
    "tell", "me", "about", "what", "is", "are", "the", "a", "an", "and", "or", "how", "does",
    "do", "can", "you", "explain", "describe", "give", "details", "of", "in", "on", "for",
    "with", "to", "from", "by", "at", "it", "this", "that", "these", "those", "please"
}

class RAGRetriever:
    def __init__(self, vector_manager: VectorStoreManager, all_chunks: List[TextChunk]):
        self.vm = vector_manager
        self.all_chunks = all_chunks
        self.embedder = vector_manager.embedder

        # Build dynamic registry of all indexed paper titles and entities
        self.paper_entity_index = self._build_paper_entity_index(all_chunks)

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

    def _tokenize(self, text: str, remove_stopwords: bool = False) -> List[str]:
        tokens = re.findall(r"\w+", text.lower())
        if remove_stopwords:
            filtered = [t for t in tokens if t not in CONVERSATIONAL_STOPWORDS and len(t) > 1]
            return filtered if filtered else tokens
        return tokens

    def _build_paper_entity_index(self, all_chunks: List[TextChunk]) -> Dict[str, Dict[str, Any]]:
        """
        Dynamically extracts paper titles, acronyms, and keywords from indexed metadata
        to enable generic title/entity matching without hardcoding any specific paper.
        """
        registry = {}
        for c in all_chunks:
            title = c.paper_title
            source = c.file_name
            if title not in registry:
                distinctive_terms = set()
                # 1. Prefix before colon if present (e.g. "LLaMA: ...", "BERT: ...", "LoRA: ...")
                if ":" in title:
                    prefix = title.split(":")[0].strip().lower()
                    distinctive_terms.add(prefix)
                    for w in re.findall(r"\w+", prefix):
                        if len(w) >= 3 and w not in CONVERSATIONAL_STOPWORDS:
                            distinctive_terms.add(w)

                # 2. Filename parts
                base_fname = os.path.splitext(source)[0].lower()
                for part in base_fname.split("_"):
                    if len(part) >= 3 and part not in {"open", "deep", "language", "models", "paper"}:
                        distinctive_terms.add(part)

                # 3. Full title words
                title_words = re.findall(r"\w+", title.lower())
                for w in title_words:
                    if len(w) >= 4 and w not in CONVERSATIONAL_STOPWORDS:
                        distinctive_terms.add(w)

                clean_title_phrase = " ".join(title_words)

                registry[title] = {
                    "title": title,
                    "source": source,
                    "terms": distinctive_terms,
                    "full_phrase": clean_title_phrase,
                    "first_page_chunks": []
                }

            # Collect introductory / abstract chunks (page 1)
            if c.page_number == 1 and len(registry[title]["first_page_chunks"]) < 4:
                registry[title]["first_page_chunks"].append(c)

        return registry

    def _detect_matching_papers(self, query: str) -> List[Dict[str, Any]]:
        """Identify which indexed papers are referenced in the query."""
        q_lower = query.lower()
        q_tokens = set(re.findall(r"\w+", q_lower))
        matched = []

        for title, p_info in self.paper_entity_index.items():
            # Check full phrase match
            if p_info["full_phrase"] in q_lower:
                matched.append(p_info)
                continue
            # Check term intersection
            common = q_tokens.intersection(p_info["terms"])
            if common:
                matched.append(p_info)

        return matched

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
        Includes generic paper title matching to guarantee introduction retrieval for paper-entity queries.
        RRF Score = (1 - alpha) / (rrf_k + rank_dense) + alpha / (rrf_k + rank_bm25)
        """
        # 1. Dense retrieval
        dense_results = self.vm.similarity_search(query, top_k=max(top_k * 4, 15))

        # 2. BM25 retrieval with stopword filtering for conversational queries
        filtered_tokens = self._tokenize(query, remove_stopwords=True)
        if not filtered_tokens:
            filtered_tokens = self._tokenize(query, remove_stopwords=False)
        
        bm25_scores = self.bm25.get_scores(filtered_tokens)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:max(top_k * 4, 15)]

        # 3. Generic Paper Entity / Title Detection
        matched_paper_cids = set()
        matched_papers = self._detect_matching_papers(query)
        for p_info in matched_papers:
            for c in p_info["first_page_chunks"]:
                matched_paper_cids.add(c.chunk_id)

        # 4. Reciprocal Rank Fusion (RRF)
        fused_scores = {}
        item_lookup = {}

        # Dense ranks
        for rank, item in enumerate(dense_results):
            cid = item.get("chunk_id") or item["metadata"].get("chunk_id")
            item_lookup[cid] = item
            fused_scores[cid] = fused_scores.get(cid, 0.0) + ((1.0 - alpha) / (rrf_k + rank + 1))

        # BM25 ranks
        for rank, idx in enumerate(top_bm25_idx):
            chunk = self.all_chunks[idx]
            cid = chunk.chunk_id
            if cid not in item_lookup:
                item_lookup[cid] = {
                    "chunk_id": cid,
                    "paper_id": chunk.file_name,
                    "paper_title": chunk.paper_title,
                    "authors": chunk.metadata.get("authors", "Unknown"),
                    "year": chunk.metadata.get("year", 2024),
                    "page_number": chunk.page_number,
                    "source_file": chunk.file_name,
                    "source": chunk.file_name,
                    "text": chunk.text,
                    "score": round(float(bm25_scores[idx]), 4),
                    "metadata": chunk.to_dict()
                }
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (alpha / (rrf_k + rank + 1))

        # Apply entity boost to early chunks of title-matched papers
        for cid in matched_paper_cids:
            if cid in item_lookup:
                fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (rrf_k + 1))
            else:
                for c in self.all_chunks:
                    if c.chunk_id == cid:
                        item_lookup[cid] = {
                            "chunk_id": cid,
                            "paper_id": c.file_name,
                            "paper_title": c.paper_title,
                            "authors": c.metadata.get("authors", "Unknown"),
                            "year": c.metadata.get("year", 2024),
                            "page_number": c.page_number,
                            "source_file": c.file_name,
                            "source": c.file_name,
                            "text": c.text,
                            "score": 0.0,
                            "metadata": c.to_dict()
                        }
                        fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (rrf_k + 1))
                        break

        # Sort by fused score descending
        sorted_cids = sorted(fused_scores.keys(), key=lambda c: fused_scores[c], reverse=True)[:top_k]
        results = []
        for cid in sorted_cids:
            hit = dict(item_lookup[cid])
            hit["score"] = round(fused_scores[cid] * 100, 4)
            hit["strategy"] = "Hybrid (BM25 + Dense RRF)"
            results.append(hit)
        return results

    # -------------------------------------------------------------
    # STRATEGY 4: Cross-Encoder Reranker
    # -------------------------------------------------------------
    def retrieve_reranker(self, query: str, top_k: int = 5, candidate_pool: int = 10) -> List[Dict[str, Any]]:
        """
        Two-stage retrieval:
        Stage 1: High recall hybrid candidate recall (dense + BM25 + entity matching).
        Stage 2: Cross-Encoder (ms-marco-MiniLM-L-6-v2) computes joint query-document relevance logits.
        """
        # Fetch candidate pool via Hybrid to combine keyword, semantic, and title recall
        candidates = self.retrieve_hybrid(query, top_k=candidate_pool)
        if not candidates:
            return []

        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs, batch_size=16, show_progress_bar=False)

        for c, score in zip(candidates, rerank_scores):
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
            prompt = (
                f"You are an AI research assistant. Generate 3 diverse search query variants from different angles "
                f"to find relevant passages in academic AI papers for this user question: '{query}'. "
                f"Output ONLY the 3 queries, one per line without numbers or bullets."
            )
            for model_id in ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-3.6-flash"]:
                try:
                    resp = self.gemini_client.models.generate_content(
                        model=model_id,
                        contents=prompt
                    )
                    lines = [l.strip() for l in resp.text.strip().split("\n") if l.strip()]
                    for l in lines[:3]:
                        clean_l = re.sub(r"^\d+[\.\)]\s*", "", l).strip()
                        if clean_l and clean_l != query:
                            expanded_queries.append(clean_l)
                    break
                except Exception as e:
                    continue

        all_hits = {}
        for q in expanded_queries:
            hits = self.retrieve_hybrid(q, top_k=top_k)
            for h in hits:
                cid = h.get("chunk_id") or h["metadata"].get("chunk_id")
                if cid not in all_hits or h["score"] > all_hits[cid]["score"]:
                    h["strategy"] = "Multi-Query"
                    all_hits[cid] = h

        sorted_hits = sorted(all_hits.values(), key=lambda x: x["score"], reverse=True)
        return sorted_hits[:top_k]

    # -------------------------------------------------------------
    # DIAGNOSTICS & DEBUG RETRIEVAL
    # -------------------------------------------------------------
    def retrieve_with_debug(self, query: str, strategy: str = "hybrid", top_k: int = 3) -> Dict[str, Any]:
        """
        Executes dense, BM25, and hybrid pipelines, returning intermediate step diagnostics
        for Developer / Debug Inspection.
        """
        dense_hits = self.retrieve_dense(query, top_k=top_k)

        # BM25 debug
        filtered_tokens = self._tokenize(query, remove_stopwords=True)
        if not filtered_tokens:
            filtered_tokens = self._tokenize(query, remove_stopwords=False)
        bm25_scores = self.bm25.get_scores(filtered_tokens)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:top_k]
        bm25_hits = []
        for idx in top_bm25_idx:
            chunk = self.all_chunks[idx]
            bm25_hits.append({
                "chunk_id": chunk.chunk_id,
                "paper_title": chunk.paper_title,
                "page_number": chunk.page_number,
                "score": round(float(bm25_scores[idx]), 4),
                "text": chunk.text,
                "metadata": chunk.to_dict()
            })

        hybrid_hits = self.retrieve_hybrid(query, top_k=top_k)

        s = strategy.lower()
        if "dense" in s or "cosine" in s:
            final_hits = dense_hits
        elif "mmr" in s:
            final_hits = self.retrieve_mmr(query, top_k=top_k)
        elif "hybrid" in s or "bm25" in s:
            final_hits = hybrid_hits
        elif "multi" in s:
            final_hits = self.retrieve_multi_query(query, top_k=top_k)
        else:
            final_hits = self.retrieve_reranker(query, top_k=top_k)

        matched_papers = [p["title"] for p in self._detect_matching_papers(query)]

        return {
            "query": query,
            "tokens": filtered_tokens,
            "matched_papers": matched_papers,
            "dense_results": dense_hits,
            "bm25_results": bm25_hits,
            "hybrid_results": hybrid_hits,
            "final_hits": final_hits
        }

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

    q = "tell me about llama"
    print(f"\nEvaluating query: '{q}' across strategies...")
    res = retriever.retrieve_hybrid(q, top_k=3)
    for i, h in enumerate(res, 1):
        print(f"Rank {i}: {h['paper_title']} | Page {h['page_number']} | Chunk: {h['chunk_id']} | Score: {h['score']}")
