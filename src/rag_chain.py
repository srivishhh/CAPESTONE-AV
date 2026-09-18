"""
RAG Chain & Grounded Generation Module
Connects retrieval strategies to LLM generation with strict anti-hallucination guardrails,
conversational memory, and exact Top-3 citation extraction.
"""
import os
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from google import genai

load_dotenv()

from src.retrieval import RAGRetriever

sys.stdout.reconfigure(encoding="utf-8")

SYSTEM_PROMPT = """You are the Research Paper Answer Bot, an authoritative and precise academic AI assistant.
Your job is to answer the user's question with absolute factual grounding based ONLY on the supplied research-paper context.

GROUNDING & CITATION RULES:
1. Base your answer EXCLUSIVELY on the facts explicitly mentioned in the provided Research Context.
2. If the context contains relevant information, synthesize a clear, comprehensive, and accurate answer to the user's question.
3. If the provided context DOES NOT contain sufficient facts to answer the question, state clearly:
   "I do not have sufficient information in the provided research papers to answer this question."
4. When citing specific claims, reference the paper title and page number indicated in the context headers (e.g. [Attention Is All You Need, Page 3]).
5. Be concise, academically precise, and well-structured using markdown formatting.
"""

class GroundedRAGChain:
    FALLBACK_MODELS = [
        "gemini-flash-lite-latest",
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-flash-latest"
    ]

    def __init__(self, retriever: RAGRetriever, model_name: str = "gemini-flash-lite-latest"):
        self.retriever = retriever
        self.model_name = model_name
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required.")
        self.client = genai.Client(api_key=api_key)
        self.chat_history: List[Dict[str, str]] = []

    def retrieve_context(self, query: str, strategy: str = "hybrid", top_k: int = 3) -> List[Dict[str, Any]]:
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

    def build_prompt(self, query: str, hits: List[Dict[str, Any]]) -> str:
        """Construct prompt injecting retrieved passages with structured metadata headers."""
        context_blocks = []
        for i, hit in enumerate(hits, 1):
            paper = hit.get("paper_title") or "Unknown Paper"
            page = hit.get("page_number")
            page_str = str(page) if page is not None and str(page) != "-1" else "N/A"
            cid = hit.get("chunk_id") or hit.get("metadata", {}).get("chunk_id", f"chunk_{i}")
            text = hit.get("text", "").strip()
            block = (
                f"SOURCE {i}\n"
                f"Paper: {paper}\n"
                f"Page: {page_str}\n"
                f"Chunk ID: {cid}\n\n"
                f"{text}\n"
            )
            context_blocks.append(block)

        context_str = "\n---\n".join(context_blocks) if context_blocks else "[No supporting passages found in index]"

        history_str = ""
        if self.chat_history:
            recent = self.chat_history[-4:]
            formatted = [f"User: {h['user']}\nAssistant: {h['assistant']}" for h in recent]
            history_str = "Conversation History:\n" + "\n".join(formatted) + "\n\n"

        return (
            f"{history_str}Provided Research Context:\n{context_str}\n\n"
            f"User Question: {query}\n\n"
            f"Answer:"
        )

    def _call_llm_with_fallback(self, prompt: str) -> str:
        models_to_try = [self.model_name] + [m for m in self.FALLBACK_MODELS if m != self.model_name]
        last_err = None
        for model in models_to_try:
            for attempt in range(2):
                try:
                    resp = self.client.models.generate_content(
                        model=model,
                        contents=[SYSTEM_PROMPT, prompt]
                    )
                    return resp.text.strip()
                except Exception as e:
                    last_err = e
                    err_msg = str(e)
                    if "503" in err_msg or "UNAVAILABLE" in err_msg or "ResourceExhausted" in err_msg:
                        import time
                        time.sleep(0.5)
                        continue
                    else:
                        break
        return f"Error during generation: {last_err}"

    def stream_answer(self, query: str, hits: List[Dict[str, Any]]):
        """Generator that yields text tokens one by one with transparent model fallback."""
        prompt = self.build_prompt(query, hits)
        models_to_try = [self.model_name] + [m for m in self.FALLBACK_MODELS if m != self.model_name]
        last_err = None

        for model in models_to_try:
            try:
                stream = self.client.models.generate_content_stream(
                    model=model,
                    contents=[SYSTEM_PROMPT, prompt]
                )
                has_yielded = False
                for chunk in stream:
                    if chunk.text:
                        has_yielded = True
                        yield chunk.text
                if has_yielded:
                    return
            except Exception as e:
                last_err = e
                err_msg = str(e)
                if "503" in err_msg or "UNAVAILABLE" in err_msg or "ResourceExhausted" in err_msg or "404" in err_msg:
                    continue
                else:
                    break

        yield f"Error during generation: {last_err}"

    def format_citations(self, hits: List[Dict[str, Any]], strategy: str = "Hybrid") -> List[Dict[str, Any]]:
        """Extract exact Top-3 citation records preserving complete metadata."""
        top_citations = []
        for rank, h in enumerate(hits[:3], 1):
            page = h.get("page_number")
            page_val = page if page is not None and str(page) != "-1" else "N/A"
            cid = h.get("chunk_id") or h.get("metadata", {}).get("chunk_id", f"chunk_{rank}")
            source = h.get("source_file") or h.get("source", "")
            paper_title = h.get("paper_title") or "Unknown Paper"
            score = h.get("score", 0.0)
            passage = h.get("text", "")
            top_citations.append({
                "rank": rank,
                "paper_title": paper_title,
                "paper_id": source,
                "source": source,
                "source_file": source,
                "page_number": page_val,
                "chunk_id": cid,
                "score": score,
                "strategy": h.get("strategy", strategy),
                "passage": passage,
                "text": passage,
                "metadata": h.get("metadata", {})
            })
        return top_citations

    def generate_answer(self, query: str, strategy: str = "hybrid", top_k: int = 3) -> Dict[str, Any]:
        """Execute full RAG pipeline returning answer and citations."""
        hits = self.retrieve_context(query, strategy=strategy, top_k=top_k)
        prompt = self.build_prompt(query, hits)
        answer_text = self._call_llm_with_fallback(prompt)

        # Record conversation
        self.chat_history.append({"user": query, "assistant": answer_text})

        return {
            "query": query,
            "strategy": strategy,
            "answer": answer_text,
            "top_sources": self.format_citations(hits, strategy)
        }

    def clear_history(self):
        self.chat_history = []
