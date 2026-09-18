"""
Vector Store Module (ChromaDB)
Indexes all document chunks with rich citation metadata into persistent ChromaDB collections.
"""
import os
import shutil
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings
from src.ingestion import load_research_papers
from src.chunking import chunk_recursive, TextChunk
from src.embeddings import HuggingFaceEmbedder, BaseEmbeddingModel

class VectorStoreManager:
    def __init__(self, db_dir: str, embedder: BaseEmbeddingModel, collection_name: str = "papers_minilm_l6_v2"):
        self.db_dir = db_dir
        self.embedder = embedder
        self.collection_name = collection_name
        os.makedirs(db_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "GenAI Research Papers Chunked Index", "embedding_model": embedder.name}
        )

    def index_chunks(self, chunks: List[TextChunk], batch_size: int = 100, force_reindex: bool = False):
        """Index chunks with embeddings and full citation metadata."""
        count = self.collection.count()
        if count > 0 and not force_reindex:
            print(f"Collection '{self.collection_name}' already contains {count} indexed chunks. Skipping indexing.")
            return

        if force_reindex and count > 0:
            print(f"Clearing {count} existing chunks for fresh index...")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)

        print(f"Indexing {len(chunks)} chunks into ChromaDB '{self.collection_name}'...")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            embeddings = self.embedder.embed_documents(texts)
            
            # Prepare metadata dicts (must be str, int, float, or bool for Chroma)
            metas = []
            for c in batch:
                m = {
                    "chunk_id": c.chunk_id,
                    "paper_title": str(c.paper_title),
                    "source": str(c.file_name),
                    "page_number": int(c.page_number),
                    "authors": str(c.metadata.get("authors", "Unknown")),
                    "year": int(c.metadata.get("year", 2024)),
                    "arxiv_id": str(c.metadata.get("arxiv_id", "N/A")),
                    "char_count": int(len(c.text))
                }
                metas.append(m)

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metas
            )
            print(f"  Indexed batch {i+1} to {min(i+batch_size, len(chunks))} of {len(chunks)} chunks.")

        print(f"Indexing complete. Total indexed chunks: {self.collection.count()}")

    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query ChromaDB using dense cosine similarity with complete metadata preservation."""
        q_emb = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        hits = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]
            ids = results.get("ids", [[]])[0]
            for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
                cid = ids[i] if i < len(ids) else meta.get("chunk_id", f"chunk_{i}")
                # Chroma collection space is L2 by default: squared Euclidean distance dist in [0, 4] for normalized embs.
                # Cosine similarity = 1 - (dist / 2.0)
                sim_score = max(0.0, min(1.0, 1.0 - (float(dist) / 2.0)))
                hit_dict = {
                    "chunk_id": cid,
                    "paper_id": meta.get("source", ""),
                    "paper_title": meta.get("paper_title", ""),
                    "authors": meta.get("authors", "Unknown"),
                    "year": meta.get("year", 2024),
                    "page_number": meta.get("page_number", -1),
                    "source_file": meta.get("source", ""),
                    "source": meta.get("source", ""),
                    "text": doc,
                    "score": round(sim_score, 4),
                    "metadata": meta
                }
                hits.append(hit_dict)
        return hits

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    p_dir = os.path.join(project_dir, "data", "papers")
    chroma_dir = os.path.join(project_dir, "data", "chroma_db")

    docs = load_research_papers(p_dir)
    chunks = chunk_recursive(docs)
    embedder = HuggingFaceEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    manager = VectorStoreManager(chroma_dir, embedder)
    manager.index_chunks(chunks, force_reindex=True)

    # Test query
    test_q = "What is the scaled dot-product attention formula?"
    hits = manager.similarity_search(test_q, top_k=3)
    print(f"\nTest Query: '{test_q}'")
    for i, h in enumerate(hits, 1):
        print(f"Hit {i}: {h['paper_title']} | Page {h['page_number']} | Score: {h['score']}")
        print(f"Snippet: {h['text'][:120]}...\n")
