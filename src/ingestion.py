import os
import json
import hashlib
import re
from typing import List, Dict
import fitz  # PyMuPDF
import pdfplumber
import camelot
import pandas as pd
from pathlib import Path
from typing import List, Dict
import tempfile

def generate_doc_id(pdf_path: str, title: str) -> str:
    """Generate unique ID: hash(title + path) for dedupe."""
    hash_input = f"{title}:{pdf_path}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:8]

def extract_structure_and_text(pdf_path: str) -> Dict:
    """Extract title, abstract, sections, full text with page metadata."""
    doc = fitz.open(pdf_path)
    full_text = ""
    pages_text = []
    title = ""
    abstract = ""
    sections = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text() or ""
            full_text += page_text + "\n"
            pages_text.append({"page": page_num, "text": page_text})

    # Heuristics for structure (fallback; improve with GROBID later)
    lines = full_text.split('\n')
    # Title: First non-empty line or largest font (simplified)
    title = lines[0].strip() if lines else ""
    # Abstract: After "Abstract" keyword
    abstract_match = re.search(r'Abstract\s*(.*?)(?=\n\n[A-Z]|References|\Z)', full_text, re.DOTALL | re.IGNORECASE)
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    # Sections: Numbered headers (e.g., 1. Introduction)
    section_pattern = r'^(\d+(?:\.\d+)*)\s+([A-Z][a-z]+.*?)(?=\n\d|\n[A-Z]{2,}|$)'  # Basic regex
    sections = [{"name": match.group(2).strip(), "start_line": i} for i, line in enumerate(lines) if (match := re.match(section_pattern, line))]
    
    doc.close()
    return {
        "title": title,
        "abstract": abstract,
        "sections": sections,
        "pages_text": pages_text,
        "full_text": full_text.strip()
    }

def extract_tables(pdf_path: str) -> List[Dict]:
    """Extract tables via Camelot, save as CSV."""
    tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')  # Try lattice; fallback 'stream'
    extracted = []
    for i, table in enumerate(tables):
        if not table.df.empty:
            df = table.df
            csv_path = f"data/assets/{generate_doc_id(pdf_path, '')}_tbl_{i}.csv"  # Temp doc_id
            df.to_csv(csv_path, index=False)
            extracted.append({
                "id": i,
                "page": table.page,
                "data": df.to_dict('records'),
                "csv_path": csv_path,
                "shape": df.shape
            })
    return extracted

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def extract_images(pdf_path: str, doc_id: str) -> List[Dict]:
    """Extract figures/equations as images with bbox, handle errors gracefully."""
    doc = fitz.open(pdf_path)
    images = []
    os.makedirs(f"data/assets/{doc_id}", exist_ok=True)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                if not base_image:
                    logger.warning(f"Skipping invalid image on page {page_num+1}, index {img_index}")
                    continue
                image_bytes = base_image["image"]
                img_path = f"data/assets/{doc_id}/fig_page{page_num+1}_{img_index}.png"
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                # Handle bbox
                try:
                    bbox = page.get_image_bbox(img)
                    # Convert Rect to list of floats [x0, y0, x1, y1]
                    bbox_serializable = [float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)] if bbox else None
                except ValueError as e:
                    logger.warning(f"No bbox for image on page {page_num+1}: {e}")
                    bbox_serializable = None
                # Detect caption (text near image, e.g., below bbox)
                caption = ""
                if bbox:
                    caption_area = [bbox.x0, bbox.y1, bbox.x1, bbox.y1 + 50]
                    caption = page.get_text("text", clip=caption_area).strip()
                images.append({
                    "id": img_index,
                    "page": page_num + 1,
                    "bbox": bbox_serializable,
                    "path": img_path,
                    "type": "equation" if re.search(r'[\$\\{∫∑]', base_image.get("ext", "") + page.get_text()) else "figure",
                    "caption": caption
                })
            except Exception as e:
                logger.error(f"Error processing image {img_index} on page {page_num+1}: {e}")
                continue
    doc.close()
    logger.info(f"Extracted {len(images)} images for {pdf_path}")
    return images

def chunk_text(full_text: str, chunk_size: 500, overlap: 100) -> List[Dict]:
    """Chunk with overlap; simple char-based for now (improve with token splitter later)."""
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append({"text": chunk.strip(), "start_pos": start, "end_pos": end})
        start = end - overlap
    return chunks

def ingest(pdf_paths: List[str]) -> Dict[str, Dict]:
    """Main pipeline: Batch ingest, validate, extract, save JSON."""
    results = {}
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"Warning: {pdf_path} not found. Skipping.")
            continue
        
        # Validation: Basic check
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                raise ValueError("Empty PDF")
            doc.close()
        except Exception as e:
            print(f"Error validating {pdf_path}: {e}")
            continue
        
        # Extract
        structure = extract_structure_and_text(pdf_path)
        doc_id = generate_doc_id(pdf_path, structure["title"])
        tables = extract_tables(pdf_path)
        images = extract_images(pdf_path, doc_id)
        chunks = chunk_text(structure["full_text"], chunk_size=500, overlap=100)
        
        # Tag chunks with metadata (e.g., sections/images refs; simplified)
        for chunk in chunks:
            chunk["metadata"] = {"doc_id": doc_id, "sections": structure["sections"][:2]}  # Sample refs
        
        # JSON Output
        output = {
            "doc_id": doc_id,
            "title": structure["title"],
            "abstract": structure["abstract"],
            "sections": structure["sections"],
            "chunks": chunks,
            "tables": tables,
            "images": images,
            "full_text": structure["full_text"][:1000]  # Truncate for storage
        }
        
        # Save
        os.makedirs(f"data/{doc_id}", exist_ok=True)
        with open(f"data/{doc_id}/metadata.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        
        # Assets dir
        os.makedirs(f"data/assets/{doc_id}", exist_ok=True)
        
        results[doc_id] = output
        print(f"Ingested {pdf_path} as {doc_id}: {len(chunks)} chunks, {len(tables)} tables, {len(images)} images.")
    
    # Quality check: Ensure non-empty
    if not results:
        raise ValueError("No valid PDFs ingested.")
    
    return results

if __name__ == "__main__":
    sample_pdfs = ["data/sociology-ai.pdf", "data/Startup-Proposal.pdf"]  # Add your files
    results = ingest(sample_pdfs)
    print(f"Ingestion complete: {list(results.keys())}")