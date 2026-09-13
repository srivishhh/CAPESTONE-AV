"""
Embedding Models & Benchmark Comparison Module
Compares Open-Source (all-MiniLM-L6-v2) vs Commercial (Gemini-embedding-001 / OpenAI).
Computes exact dimensions, latency (ms/chunk), and cosine similarity metrics.
"""
import os
import time
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

load_dotenv()

class BaseEmbeddingModel:
    name: str
    dimension: int
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError
    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError

class HuggingFaceEmbedder(BaseEmbeddingModel):
    """Open-source dense embedding model using SentenceTransformers."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        emb = self.model.encode(text, show_progress_bar=False, normalize_embeddings=True)
        return emb.tolist()

class GeminiCommercialEmbedder(BaseEmbeddingModel):
    """Commercial embedding model using Google Gemini Embedding API."""
    def __init__(self, model_name: str = "gemini-embedding-001"):
        self.name = model_name
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required.")
        self.client = genai.Client(api_key=api_key)
        self.dimension = 3072

    def embed_documents(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                resp = self.client.models.embed_content(model=self.name, contents=text)
                emb = resp.embeddings[0].values
                all_embeddings.append(emb)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        resp = self.client.models.embed_content(model=self.name, contents=text)
        return resp.embeddings[0].values

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def benchmark_embedding_models() -> pd.DataFrame:
    """
    Rigorously compare Open-Source (all-MiniLM-L6-v2) and Commercial (gemini-embedding-001)
    on real semantic relevance pairs and negative discrimination.
    """
    print("Initializing embedding models for rigorous comparison...")
    hf_embedder = HuggingFaceEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    commercial_embedder = GeminiCommercialEmbedder("gemini-embedding-001")

    test_pairs = [
        {
            "query": "How does multi-head attention work in Transformer architecture?",
            "relevant": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.",
            "unrelated": "LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer."
        },
        {
            "query": "What are the pretraining objectives of BERT?",
            "relevant": "We pre-train BERT using two unsupervised tasks: Masked Language Model (MLM) and Next Sentence Prediction (NSP).",
            "unrelated": "Retrieval-Augmented Generation models combine pre-trained parametric and non-parametric memory for language generation."
        },
        {
            "query": "How does LoRA achieve parameter-efficient fine-tuning?",
            "relevant": "LoRA decomposes the weight update into two low-rank matrices W0 + BA, greatly reducing trainable parameters while maintaining performance.",
            "unrelated": "LLaMA is an auto-regressive language model based on the transformer architecture with pre-normalization and SwiGLU."
        }
    ]

    models = [
        ("Open-Source: all-MiniLM-L6-v2", hf_embedder),
        ("Commercial: Gemini-embedding-001", commercial_embedder)
    ]
    results = []

    for label, embedder in models:
        t0 = time.time()
        _ = embedder.embed_documents(["Warmup query sentence for latency benchmarking."] * 5)
        latency_ms = (time.time() - t0) / 5.0 * 1000

        rel_sims = []
        unrel_sims = []
        for pair in test_pairs:
            q_emb = embedder.embed_query(pair["query"])
            r_emb = embedder.embed_query(pair["relevant"])
            u_emb = embedder.embed_query(pair["unrelated"])
            rel_sims.append(cosine_similarity(q_emb, r_emb))
            unrel_sims.append(cosine_similarity(q_emb, u_emb))

        mean_rel = float(np.mean(rel_sims))
        mean_unrel = float(np.mean(unrel_sims))
        margin = mean_rel - mean_unrel

        results.append({
            "Embedding Model": label,
            "Type": "Open-Source" if "Open-Source" in label else "Commercial",
            "Dimension": embedder.dimension,
            "Latency (ms/chunk)": round(latency_ms, 2),
            "Mean Relevant Cosine Sim": round(mean_rel, 4),
            "Mean Unrelated Cosine Sim": round(mean_unrel, 4),
            "Discrimination Margin": round(margin, 4)
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = benchmark_embedding_models()
    print("\n=== Embedding Model Benchmark Results ===")
    print(df.to_string(index=False))
