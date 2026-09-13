"""
RAG Chain & Grounded Generation Module
Connects retrieval strategies to LLM generation with strict anti-hallucination guardrails,
conversational memory, and exact Top-3 citation extraction.
"""
import os
import sys
from typing import List, Dict, Any, Optional
from google import genai

from src.retrieval import RAGRetriever

sys.stdout.reconfigure(encoding="utf-8")

SYSTEM_PROMPT = """You are the Research Paper Answer Bot, an authoritative and precise academic AI assistant.
Your job is to answer the user's question with absolute factual grounding based ONLY on the provided research paper passages.

STRICT GROUNDING RULES:
1. Base your answer EXCLUSIVELY on the facts explicitly mentioned in the provided Context passages.
2. DO NOT hallucinate, extrapolate, or use outside knowledge.
3. If the provided context DOES NOT contain sufficient facts to answer the question, state clearly:
   "I do not have sufficient information in the provided research papers to answer this question."
4. When citing specific claims, reference the paper title and page number indicated in the context headers (e.g. [Attention Is All You Need, Page 3]).
5. Be concise, academically precise, and well-structured. Use markdown formatting.
"""

class GroundedRAGChain:
    def __init__(self, retriever: RAGRetriever, model_name: str = "gemini-flash-latest"):
        self.retriever = retriever
        self.model_name = model_name
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required.")
        self.client = genai.Client(api_key=api_key)
        self.chat_history: List[Dict[str, str]] = []

    def retrieve_context(self, query: str, strategy: str = "reranker", top_k: int = 3) -> List[Dict[str, Any]]:
        """Fetch supporting chunks using the selected strategy."""
        s = strategy.lower()
        if "dense" in s or "cosine" in s:
            hits = self.retriever.retrieve_dense(query, top_k=top_k)
        elif "mmr" in s:
            hits = self.retriever.retrieve_mmr(query, top_k=top_k)
        elif "hybrid" in s or "bm25" in s:
            hits = self.retriever.retrieve_hybrid(query, top_k=top_k)
        elif "multi" in s:
            hits = self.retriever.retrieve_multi_query(query, top_k=top_k)
        else:  # Default to high-precision reranker
            hits = self.retriever.retrieve_reranker(query, top_k=top_k)
        return hits

    def generate_answer(self, query: str, strategy: str = "reranker", top_k: int = 3) -> Dict[str, Any]:
        """
        Execute full RAG pipeline:
        1. Retrieve top-k context passages.
        2. Construct prompt injecting retrieved passages with metadata headers.
        3. Invoke LLM for grounded synthesis.
        4. Format Top-3 supporting citations.
        """
        hits = self.retrieve_context(query, strategy=strategy, top_k=top_k)

        # Context assembly
        context_blocks = []
        for i, hit in enumerate(hits, 1):
            block = (
                f"--- PASSAGE {i} ---\n"
                f"Paper: {hit['paper_title']}\n"
                f"Page: {hit['page_number']}\n"
                f"Text: {hit['text']}\n"
            )
            context_blocks.append(block)

        context_str = "\n".join(context_blocks)

        # History representation
        history_str = ""
        if self.chat_history:
            recent = self.chat_history[-4:]
            formatted = [f"User: {h['user']}\nAssistant: {h['assistant']}" for h in recent]
            history_str = "Conversation History:\n" + "\n".join(formatted) + "\n\n"

        prompt = (
            f"{history_str}Provided Research Context:\n{context_str}\n\n"
            f"User Question: {query}\n\n"
            f"Answer:"
        )

        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=[SYSTEM_PROMPT, prompt]
            )
            answer_text = resp.text.strip()
        except Exception as e:
            # Fallback model attempt if primary is busy
            try:
                resp = self.client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=[SYSTEM_PROMPT, prompt]
                )
                answer_text = resp.text.strip()
            except Exception as e2:
                answer_text = f"Error during generation: {e2}"

        # Record conversation
        self.chat_history.append({"user": query, "assistant": answer_text})

        # Format top-3 citations explicitly
        top_citations = []
        for rank, h in enumerate(hits[:3], 1):
            top_citations.append({
                "rank": rank,
                "paper_title": h["paper_title"],
                "source": h["source"],
                "page_number": h["page_number"],
                "score": h["score"],
                "strategy": h.get("strategy", strategy),
                "passage": h["text"]
            })

        return {
            "query": query,
            "strategy": strategy,
            "answer": answer_text,
            "top_sources": top_citations
        }

    def clear_history(self):
        self.chat_history = []

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    p_dir = os.path.join(project_dir, "data", "papers")
    chroma_dir = os.path.join(project_dir, "data", "chroma_db")

    from src.ingestion import load_research_papers
    from src.chunking import chunk_recursive
    from src.embeddings import HuggingFaceEmbedder
    from src.vector_store import VectorStoreManager

    docs = load_research_papers(p_dir)
    chunks = chunk_recursive(docs)
    embedder = HuggingFaceEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    vm = VectorStoreManager(chroma_dir, embedder)
    retriever = RAGRetriever(vm, chunks)

    rag = GroundedRAGChain(retriever)

    test_query = "What are the two pre-training objectives used for BERT, and how does Masked LM work?"
    print(f"\nUser Question: {test_query}\n")
    res = rag.generate_answer(test_query, strategy="reranker", top_k=3)

    print("=== GROUNDED ANSWER ===")
    print(res["answer"])

    print("\n=== TOP-3 SUPPORTING CITATIONS ===")
    for src in res["top_sources"]:
        print(f"[{src['rank']}] {src['paper_title']} (Page {src['page_number']}) | Score: {src['score']}")
        print(f"Passage: {src['passage'][:140]}...\n")
