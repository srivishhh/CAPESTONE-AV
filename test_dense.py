import chromadb
import numpy as np
from src.embeddings import HuggingFaceEmbedder

client = chromadb.PersistentClient('data/chroma_db')
col = client.get_collection('papers_minilm_l6_v2')
embedder = HuggingFaceEmbedder('sentence-transformers/all-MiniLM-L6-v2')

queries = [
    'tell me about llama',
    'llama',
    'What is LLaMA?',
    'What is the LLaMA model?',
    'What is the attention scaling formula?'
]

for q in queries:
    q_emb = embedder.embed_query(q)
    res = col.query(query_embeddings=[q_emb], n_results=5, include=['documents', 'metadatas', 'distances'])
    print(f"\n=== QUERY: '{q}' ===")
    for i in range(5):
        m = res['metadatas'][0][i]
        d = res['distances'][0][i]
        cos_sim = 1.0 - (d / 2.0)
        print(f"[{i}] {m['paper_title']} | Page {m['page_number']} | L2 dist: {d:.4f} | Cosine: {cos_sim:.4f} | Chunk: {m['chunk_id']}")
        print(f"    Snippet: {res['documents'][0][i][:100]}...")
