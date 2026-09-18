import re
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from src.embeddings import HuggingFaceEmbedder

# 1. Inspect chunks directly from Chroma
client = chromadb.PersistentClient('data/chroma_db')
col = client.get_collection('papers_minilm_l6_v2')
all_data = col.get(include=['documents', 'metadatas'])
num_chunks = len(all_data['ids'])
print(f"Loaded {num_chunks} chunks from ChromaDB.")

# Build chunks list
chunks = []
for cid, doc, meta in zip(all_data['ids'], all_data['documents'], all_data['metadatas']):
    chunks.append({
        "chunk_id": cid,
        "text": doc,
        "paper_title": meta["paper_title"],
        "page_number": meta["page_number"],
        "source": meta["source"],
        "source_file": meta["source"],
        "authors": meta.get("authors", "Unknown"),
        "year": meta.get("year", 2024),
        "metadata": meta
    })

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "can", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", 
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", 
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", 
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", 
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", 
    "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them", 
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", 
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", 
    "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves",
    # Conversational RAG queries stopwords
    "tell", "explain", "describe", "give", "show", "please", "paper", "papers", "model", "models"
}

def tokenize_for_bm25(text, filter_stops=False):
    tokens = re.findall(r"\b[a-zA-Z0-9_\u0370-\u03ff\u2200-\u22ff-]+\b", text.lower())
    if filter_stops:
        content = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        return content if content else tokens
    return tokens

# Build BM25 index with content tokens
corpus_tokens = [tokenize_for_bm25(c["text"], filter_stops=False) for c in chunks]
bm25 = BM25Okapi(corpus_tokens)

# Generic paper registry for title matching
paper_titles = sorted(list(set(c["paper_title"] for c in chunks)))
paper_intro_chunks = {}
for p in paper_titles:
    # Find chunks on page 1 for each paper
    p_chunks = [c for c in chunks if c["paper_title"] == p and c["page_number"] == 1]
    paper_intro_chunks[p] = p_chunks

print(f"Indexed {len(paper_titles)} paper titles for generic title detection.")

embedder = HuggingFaceEmbedder('sentence-transformers/all-MiniLM-L6-v2')

def hybrid_search(query, top_k=3, rrf_k=60):
    # 1. Dense Search
    q_emb = embedder.embed_query(query)
    dense_res = col.query(query_embeddings=[q_emb], n_results=top_k * 5, include=['documents', 'metadatas', 'distances'])
    
    dense_ranked = []
    if dense_res and dense_res['ids'] and dense_res['ids'][0]:
        for cid, doc, meta, dist in zip(dense_res['ids'][0], dense_res['documents'][0], dense_res['metadatas'][0], dense_res['distances'][0]):
            cos_score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
            dense_ranked.append({
                "chunk_id": cid,
                "text": doc,
                "paper_title": meta["paper_title"],
                "page_number": meta["page_number"],
                "source": meta["source"],
                "source_file": meta["source"],
                "authors": meta.get("authors", "Unknown"),
                "year": meta.get("year", 2024),
                "dense_score": cos_score,
                "metadata": meta
            })

    # 2. BM25 Search with query stopword filtering
    q_tokens = tokenize_for_bm25(query, filter_stops=True)
    bm25_scores = bm25.get_scores(q_tokens)
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:top_k * 5]
    
    bm25_ranked = []
    for idx in top_bm25_idx:
        if bm25_scores[idx] > 0.0:  # ONLY consider chunks with actual term matches
            c = dict(chunks[idx])
            c["bm25_score"] = float(bm25_scores[idx])
            bm25_ranked.append(c)

    # 3. Generic Paper Title Matching (Requirement #8)
    q_lower = query.lower()
    title_matched_chunks = []
    for title in paper_titles:
        # Check if title or distinctive title words appear in query
        title_tokens = [w for w in re.findall(r"\b[a-zA-Z0-9]+\b", title.lower()) if w not in STOPWORDS and len(w) > 2]
        # Match if significant title words match
        matched_words = [w for w in title_tokens if w in q_lower]
        if matched_words and (len(matched_words) >= 1):
            # Check if it's an exact word match (e.g. 'llama' in 'tell me about llama')
            for w in matched_words:
                if re.search(r"\b" + re.escape(w) + r"\b", q_lower):
                    for intro_c in paper_intro_chunks.get(title, []):
                        title_matched_chunks.append(intro_c)
                    break

    # 4. RRF Score Fusion
    fused_scores = {}
    lookup = {}

    # Dense contributions
    for rank, item in enumerate(dense_ranked):
        cid = item["chunk_id"]
        lookup[cid] = item
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1)) * 0.5

    # BM25 contributions
    for rank, item in enumerate(bm25_ranked):
        cid = item["chunk_id"]
        if cid not in lookup:
            lookup[cid] = item
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1)) * 0.5

    # Title match bonus (ensures page 1 introduction is prioritized for "tell me about X")
    for rank, item in enumerate(title_matched_chunks):
        cid = item["chunk_id"]
        if cid not in lookup:
            lookup[cid] = item
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1)) * 0.4

    sorted_cids = sorted(fused_scores.keys(), key=lambda c: fused_scores[c], reverse=True)[:top_k]
    results = []
    for cid in sorted_cids:
        item = dict(lookup[cid])
        item["score"] = round(fused_scores[cid] * 100, 4)
        item["strategy"] = "Hybrid (BM25 + Dense RRF)"
        results.append(item)
    return results

print("\n" + "="*60)
print("TESTING HYBRID RETRIEVAL ON 'tell me about llama'")
print("="*60)

res = hybrid_search("tell me about llama", top_k=3)
for i, r in enumerate(res, 1):
    print(f"Rank {i}: {r['paper_title']} | Page: {r['page_number']} | Chunk ID: {r['chunk_id']} | Score: {r['score']}")
    print(f"Content: {r['text'][:160]}...\n")
