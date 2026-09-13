"""
Exploratory Data Analysis (EDA) Module
Generates 5 rigorous, publication-grade EDA visualizations on the research paper dataset.
"""
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter

from src.ingestion import load_research_papers
from src.chunking import chunk_recursive, chunk_fixed, chunk_paragraph_semantic

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.sans-serif": "Arial", "axes.edgecolor": "#cccccc"})

def run_eda(papers_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    docs = load_research_papers(papers_dir)
    print(f"Running EDA on {len(docs)} document pages...")

    # Data collection for documents
    doc_data = []
    for d in docs:
        doc_data.append({
            "paper_title": d.paper_title,
            "short_title": d.paper_title.split(":")[0][:25],
            "file_name": d.file_name,
            "page_number": d.page_number,
            "char_count": d.metadata["char_count"],
            "word_count": len(d.text.split())
        })
    df_pages = pd.DataFrame(doc_data)

    # -------------------------------------------------------------
    # PLOT 1: Page Count and Character Volume Distribution per Paper
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 5))
    agg_df = df_pages.groupby("short_title").agg(
        total_pages=("page_number", "max"),
        total_chars=("char_count", "sum")
    ).reset_index()

    x = np.arange(len(agg_df))
    width = 0.35

    color1 = "#2b5c8f"
    color2 = "#d95f02"

    rects1 = ax1.bar(x - width/2, agg_df["total_pages"], width, label="Page Count", color=color1)
    ax1.set_ylabel("Total Pages", color=color1, fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(agg_df["short_title"], rotation=15, ha="right", fontsize=10)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, agg_df["total_chars"] / 1000, width, label="Chars (in thousands)", color=color2, alpha=0.85)
    ax2.set_ylabel("Total Characters (k)", color=color2, fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.grid(False)

    plt.title("EDA Plot 1: Document Volume (Page Count & Character Scale)", fontsize=14, pad=15, fontweight="bold")
    fig.tight_layout()
    plot1_path = os.path.join(output_dir, "eda_1_document_volume.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path}")

    # -------------------------------------------------------------
    # PLOT 2: Chunk Size & Distribution across Chunking Strategies
    # -------------------------------------------------------------
    rec_chunks = chunk_recursive(docs)
    fix_chunks = chunk_fixed(docs)
    
    chunk_rows = []
    for c in rec_chunks:
        chunk_rows.append({"Strategy": "Recursive (700 char)", "Length": len(c.text)})
    for c in fix_chunks:
        chunk_rows.append({"Strategy": "Fixed Window (600 char)", "Length": len(c.text)})
        
    df_chunks = pd.DataFrame(chunk_rows)

    plt.figure(figsize=(10, 5))
    sns.histplot(data=df_chunks, x="Length", hue="Strategy", kde=True, bins=35, palette=["#1b9e77", "#7570b3"], alpha=0.6)
    plt.title("EDA Plot 2: Chunk Character Length Distribution Comparison", fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("Chunk Length (Characters)", fontsize=11)
    plt.ylabel("Chunk Frequency", fontsize=11)
    plot2_path = os.path.join(output_dir, "eda_2_chunk_distribution.png")
    plt.tight_layout()
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path}")

    # -------------------------------------------------------------
    # PLOT 3: Lexical Diversity & Vocabulary Richness (TTR)
    # -------------------------------------------------------------
    paper_vocab = {}
    for p_title in df_pages["short_title"].unique():
        p_docs = [d.text.lower() for d in docs if d.paper_title.startswith(p_title.split(":")[0][:10])]
        full_text = " ".join(p_docs)
        words = re.findall(r"\b[a-z]{3,}\b", full_text)
        unique_words = set(words)
        ttr = len(unique_words) / len(words) if words else 0
        paper_vocab[p_title] = {
            "Total Words": len(words),
            "Unique Vocab": len(unique_words),
            "Type-Token Ratio (%)": round(ttr * 100, 2)
        }
    df_vocab = pd.DataFrame.from_dict(paper_vocab, orient="index").reset_index()
    df_vocab.rename(columns={"index": "Paper"}, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = sns.barplot(data=df_vocab, x="Paper", y="Type-Token Ratio (%)", palette="Blues_d", ax=ax)
    for p in bars.patches:
        ax.annotate(f"{p.get_height():.1f}%",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                    textcoords='offset points', fontweight='bold')
    plt.title("EDA Plot 3: Lexical Diversity (Type-Token Ratio / Unique Vocab Ratio)", fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("Research Paper", fontsize=11)
    plt.ylabel("Lexical Richness (%)", fontsize=11)
    plt.xticks(rotation=15, ha="right")
    plt.ylim(0, max(df_vocab["Type-Token Ratio (%)"]) * 1.25)
    plot3_path = os.path.join(output_dir, "eda_3_lexical_diversity.png")
    plt.tight_layout()
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved: {plot3_path}")

    # -------------------------------------------------------------
    # PLOT 4: Key GenAI Concept Frequency Heatmap Across Papers
    # -------------------------------------------------------------
    concepts = [
        "attention", "transformer", "encoder", "decoder", "embedding", 
        "retrieval", "grounding", "fine-tuning", "lora", "llama", "bert"
    ]
    matrix = []
    papers_list = list(agg_df["short_title"])
    for p_title in papers_list:
        p_docs = [d.text.lower() for d in docs if d.paper_title.startswith(p_title.split(":")[0][:10])]
        combined = " ".join(p_docs)
        counts = [len(re.findall(rf"\b{term}\b", combined)) for term in concepts]
        matrix.append(counts)

    df_heatmap = pd.DataFrame(matrix, index=papers_list, columns=[c.capitalize() for c in concepts])

    plt.figure(figsize=(11, 5.5))
    sns.heatmap(df_heatmap, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={"label": "Term Occurrence Count"})
    plt.title("EDA Plot 4: Domain Concept Frequency Heatmap Across Papers", fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("Core Architectural & Algorithmic Concepts", fontsize=11)
    plt.ylabel("Research Paper", fontsize=11)
    plot4_path = os.path.join(output_dir, "eda_4_concept_heatmap.png")
    plt.tight_layout()
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"Saved: {plot4_path}")

    # -------------------------------------------------------------
    # PLOT 5: Page Text Density Heatmap
    # -------------------------------------------------------------
    max_pages = df_pages["page_number"].max()
    density_matrix = np.zeros((len(papers_list), max_pages))

    for row_idx, p_title in enumerate(papers_list):
        p_df = df_pages[df_pages["short_title"] == p_title]
        for _, row in p_df.iterrows():
            col_idx = int(row["page_number"]) - 1
            density_matrix[row_idx, col_idx] = row["char_count"]

    plt.figure(figsize=(12, 5))
    sns.heatmap(density_matrix, cmap="mako", cbar_kws={"label": "Characters per Page"},
                yticklabels=papers_list, xticklabels=range(1, max_pages + 1))
    plt.title("EDA Plot 5: Page-by-Page Character Density Heatmap", fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("Page Number", fontsize=11)
    plt.ylabel("Research Paper", fontsize=11)
    plot5_path = os.path.join(output_dir, "eda_5_page_density_heatmap.png")
    plt.tight_layout()
    plt.savefig(plot5_path, dpi=300)
    plt.close()
    print(f"Saved: {plot5_path}")

    print("All 5 EDA visualizations generated successfully!")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    papers = os.path.join(project_dir, "data", "papers")
    out = os.path.join(project_dir, "data", "eda_plots")
    run_eda(papers, out)
