"""
Chunking Strategies & Metadata Retention Module
Compares Fixed-Size, Recursive Character, and Paragraph-Semantic chunking.
Ensures rigorous metadata preservation for citation traceability.
"""
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion import DocumentPage, load_research_papers
import pandas as pd
import numpy as np

class TextChunk:
    def __init__(self, chunk_id: str, text: str, paper_title: str, file_name: str, page_number: int, strategy: str, metadata: Dict[str, Any]):
        self.chunk_id = chunk_id
        self.text = text
        self.paper_title = paper_title
        self.file_name = file_name
        self.page_number = page_number
        self.strategy = strategy
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.metadata)
        d.update({
            "chunk_id": self.chunk_id,
            "text": self.text,
            "paper_title": self.paper_title,
            "file_name": self.file_name,
            "page_number": self.page_number,
            "strategy": self.strategy,
            "char_count": len(self.text)
        })
        return d

def chunk_recursive(pages: List[DocumentPage], chunk_size: int = 700, chunk_overlap: int = 120) -> List[TextChunk]:
    """
    Split text using RecursiveCharacterTextSplitter targeting natural boundaries
    (paragraphs, sentences, words) while strictly preserving document metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks: List[TextChunk] = []
    chunk_counter = 0

    for page in pages:
        if not page.text.strip():
            continue
        splits = splitter.split_text(page.text)
        for idx, split_text in enumerate(splits):
            if len(split_text.strip()) < 40:
                continue
            chunk_counter += 1
            cid = f"{page.file_name}_p{page.page_number}_c{idx+1}"
            meta = dict(page.metadata)
            meta["chunk_index_in_page"] = idx + 1
            chunks.append(TextChunk(
                chunk_id=cid,
                text=split_text.strip(),
                paper_title=page.paper_title,
                file_name=page.file_name,
                page_number=page.page_number,
                strategy="recursive",
                metadata=meta
            ))
    return chunks

def chunk_fixed(pages: List[DocumentPage], chunk_size: int = 600, chunk_overlap: int = 100) -> List[TextChunk]:
    """Fixed-window character chunking baseline."""
    chunks: List[TextChunk] = []
    step = chunk_size - chunk_overlap

    for page in pages:
        text = page.text.strip()
        if len(text) < 40:
            continue
        splits = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            splits.append(text[start:end])
            if end == len(text):
                break
            start += step

        for idx, split_text in enumerate(splits):
            cid = f"fixed_{page.file_name}_p{page.page_number}_c{idx+1}"
            meta = dict(page.metadata)
            meta["chunk_index_in_page"] = idx + 1
            chunks.append(TextChunk(
                chunk_id=cid,
                text=split_text.strip(),
                paper_title=page.paper_title,
                file_name=page.file_name,
                page_number=page.page_number,
                strategy="fixed",
                metadata=meta
            ))
    return chunks

def chunk_paragraph_semantic(pages: List[DocumentPage], max_size: int = 800) -> List[TextChunk]:
    """Paragraph-based semantic chunking grouping cohesive paragraphs."""
    chunks: List[TextChunk] = []
    for page in pages:
        paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
        current_chunk = ""
        idx = 0
        for p in paragraphs:
            if len(current_chunk) + len(p) + 2 <= max_size:
                current_chunk = f"{current_chunk}\n\n{p}" if current_chunk else p
            else:
                if current_chunk:
                    idx += 1
                    cid = f"para_{page.file_name}_p{page.page_number}_c{idx}"
                    chunks.append(TextChunk(
                        chunk_id=cid,
                        text=current_chunk.strip(),
                        paper_title=page.paper_title,
                        file_name=page.file_name,
                        page_number=page.page_number,
                        strategy="semantic_paragraph",
                        metadata=page.metadata
                    ))
                current_chunk = p
        if current_chunk:
            idx += 1
            cid = f"para_{page.file_name}_p{page.page_number}_c{idx}"
            chunks.append(TextChunk(
                chunk_id=cid,
                text=current_chunk.strip(),
                paper_title=page.paper_title,
                file_name=page.file_name,
                page_number=page.page_number,
                strategy="semantic_paragraph",
                metadata=page.metadata
            ))
    return chunks

def compare_chunking_strategies(pages: List[DocumentPage]) -> pd.DataFrame:
    """Generate comparative quantitative metrics across chunking strategies."""
    rec_chunks = chunk_recursive(pages)
    fix_chunks = chunk_fixed(pages)
    sem_chunks = chunk_paragraph_semantic(pages)

    strategies = {
        "Recursive Character": [len(c.text) for c in rec_chunks],
        "Fixed Window": [len(c.text) for c in fix_chunks],
        "Semantic Paragraph": [len(c.text) for c in sem_chunks]
    }

    metrics = []
    for name, lengths in strategies.items():
        arr = np.array(lengths)
        metrics.append({
            "Strategy": name,
            "Total Chunks": len(arr),
            "Mean Length (chars)": round(float(np.mean(arr)), 1),
            "Std Dev (chars)": round(float(np.std(arr)), 1),
            "Min Length": int(np.min(arr)),
            "Median Length": round(float(np.median(arr)), 1),
            "Max Length": int(np.max(arr))
        })
    return pd.DataFrame(metrics)

if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    p_dir = os.path.join(os.path.dirname(current_dir), "data", "papers")
    docs = load_research_papers(p_dir)
    df = compare_chunking_strategies(docs)
    print("=== Chunking Strategy Comparison ===")
    print(df.to_string(index=False))
