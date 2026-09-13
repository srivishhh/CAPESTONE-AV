"""
Document Ingestion & Metadata Extraction Module
Handles loading research paper PDFs, extracting text per page, 
normalizing metadata, and diagnosing scanned/OCR requirements.
"""
import os
import re
from typing import List, Dict, Any
from pypdf import PdfReader

# Authoritative paper metadata mapping for clean citations
PAPER_METADATA_REGISTRY = {
    "attention_is_all_you_need.pdf": {
        "title": "Attention Is All You Need",
        "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, et al.",
        "year": 2017,
        "venue": "NeurIPS 2017",
        "arxiv_id": "1706.03762"
    },
    "bert_pretraining_deep_bidirectional_transformers.pdf": {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova",
        "year": 2018,
        "venue": "NAACL-HLT 2019",
        "arxiv_id": "1810.04805"
    },
    "retrieval_augmented_generation_nlp.pdf": {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": "Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, et al.",
        "year": 2020,
        "venue": "NeurIPS 2020",
        "arxiv_id": "2005.11401"
    },
    "lora_low_rank_adaptation.pdf": {
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": "Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, et al.",
        "year": 2021,
        "venue": "ICLR 2022",
        "arxiv_id": "2106.09685"
    },
    "llama_open_foundation_models.pdf": {
        "title": "LLaMA: Open and Efficient Foundation Language Models",
        "authors": "Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, et al.",
        "year": 2023,
        "venue": "Meta AI Research",
        "arxiv_id": "2302.13971"
    }
}

class DocumentPage:
    def __init__(self, text: str, paper_title: str, file_name: str, page_number: int, metadata: Dict[str, Any]):
        self.text = text
        self.paper_title = paper_title
        self.file_name = file_name
        self.page_number = page_number
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "paper_title": self.paper_title,
            "file_name": self.file_name,
            "page_number": self.page_number,
            "metadata": self.metadata
        }

def clean_pdf_text(text: str) -> str:
    """Clean common PDF extraction artifacts while preserving semantic structure."""
    if not text:
        return ""
    # Fix broken hyphenations at line ends (e.g., 'trans- \nformer' -> 'transformer')
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # Normalize excess vertical whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Fix ligature characters
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    return text.strip()

def load_research_papers(papers_dir: str) -> List[DocumentPage]:
    """
    Ingest all research paper PDFs from directory, extract text per page,
    and attach comprehensive metadata for retrieval citations.
    """
    if not os.path.exists(papers_dir):
        raise FileNotFoundError(f"Papers directory '{papers_dir}' does not exist.")

    pdf_files = sorted([f for f in os.listdir(papers_dir) if f.endswith(".pdf")])
    if not pdf_files:
        raise ValueError(f"No PDF documents found in '{papers_dir}'.")

    documents: List[DocumentPage] = []

    for fname in pdf_files:
        full_path = os.path.join(papers_dir, fname)
        reader = PdfReader(full_path)
        paper_info = PAPER_METADATA_REGISTRY.get(fname, {
            "title": fname.replace("_", " ").replace(".pdf", "").title(),
            "authors": "Unknown",
            "year": 2024,
            "venue": "Research Archive",
            "arxiv_id": "N/A"
        })

        for page_idx, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            cleaned = clean_pdf_text(raw_text)
            
            # Diagnose scanned vs digital text
            is_scanned = len(cleaned.strip()) < 30
            
            meta = {
                "source": fname,
                "file_path": full_path,
                "paper_title": paper_info["title"],
                "authors": paper_info["authors"],
                "year": paper_info["year"],
                "venue": paper_info["venue"],
                "arxiv_id": paper_info["arxiv_id"],
                "page_number": page_idx + 1,
                "total_pages": len(reader.pages),
                "char_count": len(cleaned),
                "is_scanned": is_scanned,
                "ocr_required": is_scanned
            }
            
            documents.append(DocumentPage(
                text=cleaned,
                paper_title=paper_info["title"],
                file_name=fname,
                page_number=page_idx + 1,
                metadata=meta
            ))

    return documents

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    p_dir = os.path.join(project_dir, "data", "papers")
    docs = load_research_papers(p_dir)
    print(f"Loaded {len(docs)} pages across {len(set(d.file_name for d in docs))} papers.")
    for d in docs[:3]:
        print(f"- {d.paper_title} | Page {d.page_number} | Chars: {len(d.text)}")
