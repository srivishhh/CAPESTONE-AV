"""
Automated Retrieval & RAG Pipeline Verification Test Suite
Evaluates Top-3 retrieval accuracy, metadata preservation, and context sufficiency
across benchmark queries including LLaMA, BERT, LoRA, Transformers, and RAG.
"""
import os
import sys
from typing import List, Dict, Any

# Ensure project root is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.ingestion import load_research_papers
from src.chunking import chunk_recursive
from src.embeddings import HuggingFaceEmbedder
from src.vector_store import VectorStoreManager
from src.retrieval import RAGRetriever
from src.rag_chain import GroundedRAGChain

BENCHMARK_TESTS = [
    {
        "id": "Q1",
        "query": "What is the attention scaling formula?",
        "expected_keyword": "Attention Is All You Need",
        "expected_paper_name": "Attention Is All You Need"
    },
    {
        "id": "Q2",
        "query": "What is the BERT masked language modeling objective?",
        "expected_keyword": "BERT",
        "expected_paper_name": "BERT: Pre-training of Deep Bidirectional Transformers"
    },
    {
        "id": "Q3",
        "query": "How does LoRA decompose the weight update?",
        "expected_keyword": "LoRA",
        "expected_paper_name": "LoRA: Low-Rank Adaptation"
    },
    {
        "id": "Q4",
        "query": "What is LLaMA?",
        "expected_keyword": "LLaMA",
        "expected_paper_name": "LLaMA: Open and Efficient Foundation Language Models"
    },
    {
        "id": "Q5",
        "query": "What is retrieval augmented generation?",
        "expected_keyword": "Retrieval-Augmented",
        "expected_paper_name": "Retrieval-Augmented Generation"
    },
    {
        "id": "Q_LLAMA_TARGET",
        "query": "tell me about llama",
        "expected_keyword": "LLaMA",
        "expected_paper_name": "LLaMA: Open and Efficient Foundation Language Models"
    }
]

def run_retrieval_test_suite():
    print("=" * 80)
    print("INITIALIZING RAG RETRIEVAL TEST SUITE")
    print("=" * 80)

    p_dir = os.path.join(project_dir, "data", "papers")
    chroma_dir = os.path.join(project_dir, "data", "chroma_db")

    docs = load_research_papers(p_dir)
    chunks = chunk_recursive(docs)
    print(f"Loaded {len(docs)} document pages, {len(chunks)} chunks.")

    embedder = HuggingFaceEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    vm = VectorStoreManager(chroma_dir, embedder)
    retriever = RAGRetriever(vm, chunks)
    chain = GroundedRAGChain(retriever)

    print("\nRunning Retrieval Evaluation across Benchmark Queries (Strategy: Hybrid RRF)...\n")

    results_table = []
    all_passed = True

    for test in BENCHMARK_TESTS:
        q = test["query"]
        expected_kw = test["expected_keyword"].lower()
        expected_name = test["expected_paper_name"]

        # Run retrieval via Hybrid (primary production strategy)
        hits = retriever.retrieve_hybrid(q, top_k=3)

        retrieved_titles = [h.get("paper_title", "Unknown") for h in hits]
        retrieved_short = [t[:25] + "..." if len(t) > 25 else t for t in retrieved_titles]

        # Check Hit@3
        hit_at_3 = any(expected_kw in t.lower() for t in retrieved_titles)

        # Check Text Present
        text_present = all(bool(h.get("text") and len(h["text"].strip()) > 20) for h in hits)

        # Check Metadata Present
        metadata_present = all(
            bool(h.get("chunk_id")) and
            h.get("page_number") is not None and
            bool(h.get("source_file") or h.get("source")) and
            bool(h.get("paper_title"))
            for h in hits
        )

        # Check Final Context
        prompt = chain.build_prompt(q, hits)
        context_non_empty = ("SOURCE 1" in prompt) and len(prompt) > 200

        test_passed = hit_at_3 and text_present and metadata_present and context_non_empty
        if not test_passed:
            all_passed = False

        results_table.append({
            "Query": q,
            "Expected Paper": expected_name[:30],
            "Retrieved Top-3": " | ".join(retrieved_short),
            "Hit@3": "PASS" if hit_at_3 else "FAIL",
            "Text Present": "PASS" if text_present else "FAIL",
            "Metadata Present": "PASS" if metadata_present else "FAIL",
            "Context Ready": "PASS" if context_non_empty else "FAIL"
        })

    # Print compact markdown evaluation table
    print("\n" + "=" * 120)
    print("COMPACT EVALUATION TABLE (Hit@3 & Metadata Retention)")
    print("=" * 120)
    header = f"{'Query':<35} | {'Expected Paper':<25} | {'Hit@3':<6} | {'Text':<6} | {'Meta':<6} | {'Context':<8}"
    print(header)
    print("-" * 120)
    for r in results_table:
        line = f"{r['Query'][:35]:<35} | {r['Expected Paper']:<25} | {r['Hit@3']:<6} | {r['Text Present']:<6} | {r['Metadata Present']:<6} | {r['Context Ready']:<8}"
        print(line)
    print("-" * 120)

    # Detailed inspection of target query: 'tell me about llama'
    print("\n" + "=" * 80)
    print("DETAILED VERIFICATION: 'tell me about llama'")
    print("=" * 80)
    target_hits = retriever.retrieve_hybrid("tell me about llama", top_k=3)
    for i, h in enumerate(target_hits, 1):
        print(f"\nSOURCE {i}:")
        print(f"  Paper:    {h.get('paper_title')}")
        print(f"  Page:     {h.get('page_number')}")
        print(f"  Chunk ID: {h.get('chunk_id')}")
        print(f"  Score:    {h.get('score')}")
        print(f"  Content:  {h.get('text', '')[:200]}...")

    # Test LLM Generation on target query
    print("\n" + "=" * 80)
    print("TESTING LLM GENERATION: 'tell me about llama'")
    print("=" * 80)
    gen_result = chain.generate_answer("tell me about llama", strategy="hybrid", top_k=3)
    print("ANSWER GENERATED:")
    print(gen_result["answer"])
    print("\nTOP SOURCES ATTACHED:")
    for src in gen_result["top_sources"]:
        print(f"  - [{src['rank']}] {src['paper_title']} (Page {src['page_number']}) | Chunk: {src['chunk_id']}")

    # Negative Control Test:
    print("\n" + "=" * 80)
    print("NEGATIVE CONTROL TEST: Query definitely not present in research papers")
    print("=" * 80)
    neg_q = "Who won the FIFA World Cup in 1994 and who scored the winning goal?"
    neg_result = chain.generate_answer(neg_q, strategy="hybrid", top_k=3)
    print(f"Query: {neg_q}")
    print("Answer:")
    print(neg_result["answer"])

    print("\n" + "=" * 80)
    cond1 = all_passed
    cond2 = "insufficient" not in gen_result["answer"].lower()
    cond3 = "insufficient" in neg_result["answer"].lower()
    print(f"CHECK RESULTS: all_passed={cond1}, llama_sufficient={cond2}, negative_control_insufficient={cond3}")
    if cond1 and cond2 and cond3:
        print(">>> ALL BENCHMARK & PIPELINE TESTS PASSED PERFECTLY! <<<")
    else:
        print(">>> SOME VERIFICATION CHECKS REQUIRE ATTENTION <<<")
    print("=" * 80)

if __name__ == "__main__":
    run_retrieval_test_suite()
