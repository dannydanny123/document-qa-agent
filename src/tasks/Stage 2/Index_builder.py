import json
import time
from pathlib import Path
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import pickle
import re

from typing import List, Dict


# --- Configuration ---
SRC_DIR = Path("src")
TASKS_DIR = SRC_DIR / "tasks"
DATA_DIR = SRC_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

INDEX_DIR.mkdir(parents=True, exist_ok=True)
MAPPING_FILE = INDEX_DIR / "indexed_documents.json"

# --- UPGRADE: Enhanced text cleaning and normalization ---
def normalize_text(text: str) -> str:
    """Collapses multiple whitespace characters and strips leading/trailing space."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_arxiv_header(text: str) -> str:
    """Removes the common, noisy arXiv header from the first chunk of text."""
    arxiv_pattern = re.compile(r'(\d{2}\d{2}\.\d{4,5}v\d+|\[[\w\.\-]+\]|arXiv:.*?)', re.IGNORECASE)
    lines = text.split('\n')
    last_header_line_idx = -1
    for i, line in enumerate(lines):
        if arxiv_pattern.search(line) or "v\ni\nX\nr\na" in line:
            last_header_line_idx = i
    if last_header_line_idx != -1:
        return "\n".join(lines[last_header_line_idx + 1:])
    return text

def table_to_text(csv_path: str, title: str, page: int) -> str:
    """Reads a table from a CSV and converts it to a readable text format."""
    try:
        df = pd.read_csv(csv_path)
        # Filter out empty or nonsensical rows before creating the text representation
        df.dropna(how='all', inplace=True)
        if df.empty: return ""
        return f"Table from document '{title}' on page {page}:\n" + df.to_string(index=False)
    except Exception:
        return ""

# --- UPGRADE: This function is now metadata-agnostic and weights content ---
def load_and_prepare_documents(processed_dir: Path) -> List[Dict]:
    """
    Scans processed data and creates a unified list of "indexable documents"
    from all available metadata keys, applying cleaning and importance weighting.
    """
    all_docs_for_indexing = []
    # --- UPGRADE: Importance weighting for different content types ---
    weight_map = {"summary": 3.0, "equation": 1.5, "figure": 1.2, "table": 1.1, "chunk": 1.0}

    for doc_folder in processed_dir.iterdir():
        if not doc_folder.is_dir(): continue
        metadata_path = doc_folder / "metadata.json"
        if not metadata_path.exists(): continue
        
        with open(metadata_path, 'r', encoding='utf-8') as f: data = json.load(f)
        
        doc_id = data["doc_id"]
        title = data.get("title", "Unknown Title")
        
        # --- UPGRADE: Dynamic, key-based processing loop (future-proof) ---
        # Process title/abstract first as a high-importance summary
        abstract = normalize_text(data.get("abstract", ""))
        summary_content = normalize_text(f"Title: {title}. Abstract: {abstract}")
        all_docs_for_indexing.append({
            "content": summary_content,
            "metadata": {"doc_id": doc_id, "page": 1, "type": "summary", "weight": weight_map["summary"]}
        })

        # Dynamically process all other list-based fields in the metadata
        for key, items in data.items():
            if not isinstance(items, list): continue # Skip simple fields like doc_id, title

            for i, item in enumerate(items):
                content = ""
                # Use a dispatcher to handle different types cleanly
                if key == "chunks":
                    text = item["text"]
                    # Clean only the first few chunks on page 1 for arXiv headers
                    if item["metadata"]["page"] == 1 and i < 2: text = clean_arxiv_header(text)
                    content = f"Section: {item['metadata'].get('section', '')}. Text: {text}"
                elif key == "tables":
                    content = table_to_text(item["csv_path"], title, item["page"])
                elif key == "figures" and item.get("caption"):
                    content = f"Figure on page {item['page']}: {item['caption']}"
                elif key == "equations" and item.get("normalized_text"):
                    content = f"Equation on page {item['page']}: {item['normalized_text']}"
                
                # Normalize the final content string
                normalized_content = normalize_text(content)
                if normalized_content:
                    metadata = item.get("metadata", item) # Chunks have nested metadata
                    metadata["doc_id"] = doc_id
                    metadata["type"] = key
                    metadata["weight"] = weight_map.get(key, 1.0)
                    all_docs_for_indexing.append({
                        "content": normalized_content,
                        "metadata": metadata
                    })
    
    print(f"Created a total of {len(all_docs_for_indexing)} specialized entries for indexing.")
    return all_docs_for_indexing

def build_and_save_indexes(documents: List[Dict], index_dir: Path):
    if not documents:
        print("No documents to index. Aborting.")
        return

    corpus = [doc["content"] for doc in documents]
    
    # --- Build FAISS index ---
    print("\nBuilding FAISS index...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    # --- UPGRADE: More efficient batch encoding ---
    embeddings = encoder.encode(corpus, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(np.array(embeddings, dtype=np.float32))
    faiss.write_index(faiss_index, str(index_dir / "faiss_index.bin"))
    print("  - FAISS index built and saved.")

    # --- Build BM25 index ---
    print("\nBuilding BM25 index...")
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(index_dir / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print("  - BM25 index built and saved.")

    # --- Save the mapping file ---
    print(f"\nSaving document mapping file to {MAPPING_FILE}...")
    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)
    print("  - Mapping file saved.")

if __name__ == "__main__":
    start = time.time()
    print("--- Loading Smart Index Builder Engine ! ---")
    prepared_documents = load_and_prepare_documents(PROCESSED_DIR)
    build_and_save_indexes(prepared_documents, INDEX_DIR)
    print(f"\n--- Indexing Complete ! ---")
    print(f"Total time taken: {time.time() - start:.2f} seconds")