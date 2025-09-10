import os
import json
import hashlib
import re
from typing import List, Dict, Any
import time

# --- Core PDF Processing Libraries ---
import fitz  # PyMuPDF
import pdfplumber
import camelot
import pandas as pd
from pathlib import Path
import tempfile
import logging
import warnings

# You will need to install these:
# pip install "unstructured[pdf]" langchain
# also need to install layoutparser and pytesseract for 'unstructured'
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pypdf")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- Unchanged Helper Classes and Functions ---
class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for PyMuPDF objects and pandas DataFrames."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (fitz.Rect, fitz.Point, fitz.Matrix)):
            return list(map(float, obj))
        if hasattr(obj, 'to_dict'):
            return obj.to_dict('records')
        return super().default(obj)

def generate_doc_id(pdf_path: str, title: str) -> str:
    """Generate unique ID: hash(title + path) for dedupe."""
    hash_input = f"{title}:{pdf_path}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:8]

# --- Using a more robust, layout-aware extraction model ---
def extract_structure_with_unstructured(pdf_path: str) -> Dict:
    """
    Extracts title, abstract, sections, and text using the unstructured.io library, my initail logic was hardcoded, used gpt to help me with regex. 
    But later, upon deep research, I found unstructured.io which is a more good for structured extraction.
    """
    logger.info(f"Extracting structure from {pdf_path} using unstructured.io...")
    try:
        # I ran into mistakes while using "fast" strategy for speed; so pivoted to "hi_res" for more accurate... but slower, np
        elements = partition_pdf(pdf_path, strategy="hi_res")
    except Exception as e:
        logger.error(f"Unstructured failed for {pdf_path}: {e}")
        # Fallback to a very basic text extraction if unstructured fails
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return {
            "title": os.path.basename(pdf_path),
            "abstract": "",
            "sections": [],
            "pages_text": [{"page": i+1, "text": page.extract_text() or ""} for i, page in enumerate(pdf.pages)],
            "full_text": text
        }

    title = next((el.text for el in elements if el.category == "Title"), os.path.basename(pdf_path))
    
    pages_text = []
    page_content = {}
    for el in elements:
        page_num = el.metadata.page_number
        if page_num not in page_content:
            page_content[page_num] = []
        page_content[page_num].append(el.text)
    
    for page_num, texts in sorted(page_content.items()):
        pages_text.append({"page": page_num, "text": "\n".join(texts)})
    
    full_text = "\n\n".join([el.text for el in elements])
    
    # Heuristic to find abstract and sections from the structured elements
    abstract = ""
    sections = []
    try:
        abstract_element_indices = [i for i, el in enumerate(elements) if "abstract" in el.text.lower() and el.category == "Title"]
        if abstract_element_indices:
            start_index = abstract_element_indices[0] + 1
            for i in range(start_index, len(elements)):
                if elements[i].category == "Title": # Stop at the next section title
                    break
                abstract += elements[i].text + "\n"
    except Exception:
        pass # Abstract detection is best-effort

    section_pattern = r'^\d+(\.\d+)*\s+.*'
    sections = [{"name": el.text, "page": el.metadata.page_number} for el in elements if el.category == "Title" and re.match(section_pattern, el.text)]

    return {
        "title": title,
        "abstract": abstract.strip(),
        "sections": sections,
        "pages_text": pages_text,
        "full_text": full_text.strip()
    }

def extract_tables(pdf_path: str, doc_id: str) -> List[Dict]:
    """
    It was challanging to pull tables out of a PDF...
    First attempt with Camelot (lattice → stream), if that fails, 
    fall back to pdfplumber. 
    Everything gets saved as CSV for traceability.
    """
    extracted = []

    # use a temp dir to keep Camelot happy (avoids file locking issues)
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # first shot: lattice mode (works if the PDF has nice grid lines)
            tables = camelot.read_pdf(
                pdf_path,
                pages='all',
                flavor='lattice',
                backend='poppler',
                temp_dir=tmpdir
            )

            if not tables:  # sometimes no luck with lattice
                logger.info(f"No lattice tables found in {pdf_path}, trying stream mode")

                # second shot: stream mode (uses spacing instead of grid lines)
                tables = camelot.read_pdf(
                    pdf_path,
                    pages='all',
                    flavor='stream',
                    backend='poppler',
                    temp_dir=tmpdir
                )

        except Exception as e:
            # if Camelot blows up, don’t die — try pdfplumber instead
            logger.warning(f"Camelot failed for {pdf_path}: {e}. Falling back to pdfplumber.")

            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables_plumber = page.extract_tables()

                    for i, table in enumerate(tables_plumber):
                        if table:  # pdfplumber can sometimes spit out empty ones
                            df = pd.DataFrame(table)

                            # keep assets organized by document id
                            os.makedirs(f"data/assets/{doc_id}", exist_ok=True)
                            csv_path = f"data/assets/{doc_id}/tbl_{page_num}_{i}.csv"

                            # dump to CSV (so we can debug easily later)
                            df.to_csv(csv_path, index=False)

                            extracted.append({
                                "id": f"{page_num}_{i}",
                                "page": page_num,
                                "data": df.to_dict('records'),
                                "csv_path": csv_path,
                                "shape": df.shape
                            })

            return extracted

        # if Camelot did return something usable → process it
        for i, table in enumerate(tables):
            if not table.df.empty:
                df = table.df

                os.makedirs(f"data/assets/{doc_id}", exist_ok=True)
                csv_path = f"data/assets/{doc_id}/tbl_{table.page}_{i}.csv"

                df.to_csv(csv_path, index=False)

                extracted.append({
                    "id": f"{table.page}_{i}",
                    "page": table.page,
                    "data": df.to_dict('records'),
                    "csv_path": csv_path,
                    "shape": df.shape
                })

        logger.info(f"Extracted {len(extracted)} tables for {pdf_path}")

    return extracted


def extract_images(pdf_path: str, doc_id: str) -> List[Dict]:
    """
    Pull out figures/equations as images from the PDF.
    Store them in a doc-specific folder, with bounding boxes and captions if possible. I wanted to design it in a way that the system could easily differientiate between figures and equations by feeding the images to the LLM image model, but for now it just judge based on surrounding text.
    """

    doc = fitz.open(pdf_path)  # PyMuPDF document object
    images = []

    # make sure the doc gets its own asset folder
    os.makedirs(f"data/assets/{doc_id}", exist_ok=True)

    for page_num in range(len(doc)):
        page = doc[page_num]

        # grab all image objects (full=True → no skipping)
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            try:
                # every image in PyMuPDF has an "xref" (like a unique ref id)
                xref = img[0]
                base_image = doc.extract_image(xref)

                if not base_image:  # sometimes xref exists but image data is junk
                    logger.warning(f"Skipping invalid image on page {page_num+1}, index {img_index}")
                    continue

                # save the raw bytes → png on disk
                image_bytes = base_image["image"]
                img_path = f"data/assets/{doc_id}/fig_page{page_num+1}_{img_index}.png"
                with open(img_path, "wb") as f:
                    f.write(image_bytes)

                # try to grab bounding box (where on the page this image lives)
                try:
                    bbox = page.get_image_bbox(img)
                    bbox_serializable = [
                        float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)
                    ] if bbox else None
                except ValueError as e:
                    # PyMuPDF can choke on some objects → don’t crash, just log
                    logger.warning(f"No bbox for image on page {page_num+1}: {e}")
                    bbox_serializable = None

                # try to guess the caption (usually text directly under the image box)
                caption = ""
                if bbox:
                    caption_area = [bbox.x0, bbox.y1, bbox.x1, bbox.y1 + 50]  # 50px below
                    caption = page.get_text("text", clip=caption_area).strip()

                # decide if this is a "figure" or an "equation"
                # (very rough: look at extension + surrounding text for LaTeX-ish symbols)
                img_type = "equation" if re.search(
                    r'[\$\\{∫∑]', 
                    base_image.get("ext", "") + page.get_text()
                ) else "figure"

                # stash everything in a structured dict
                images.append({
                    "id": img_index,
                    "page": page_num + 1,
                    "bbox": bbox_serializable,
                    "path": img_path,
                    "type": img_type,
                    "caption": caption
                })

            except Exception as e:
                # generic catch-all so a single bad image doesn’t ruin the run
                logger.error(f"Error processing image {img_index} on page {page_num+1}: {e}")
                continue

    doc.close()
    logger.info(f"Extracted {len(images)} images for {pdf_path}")

    return images

# Using a semantic-aware chunking strategy
def chunk_text_semantic(
    pages_text: List[Dict],
    sections: List[Dict],  # Add sections to the function signature
    doc_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict]:
    """
    Chunks text using a semantic-aware splitter from LangChain to preserve context.
    Each chunk is tagged with its source page number and the relevant section title.
    """
    logger.info(f"Chunking document {doc_id} semantically with section metadata...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    # We sort the sections by page number to process them in order.
    page_to_section_map = {sec["page"]: sec["name"] for sec in sorted(sections, key=lambda x: x["page"])}
    current_section = "Introduction"  # A sensible default for pages before the first section

    all_chunks = []
    for page_info in sorted(pages_text, key=lambda x: x["page"]): # Ensure pages are in order
        page_num = page_info["page"]
        page_content = page_info["text"]

        # If the current page number has a new section starting on it, update our tracker.
        if page_num in page_to_section_map:
            current_section = page_to_section_map[page_num]

        # Split the content of a single page
        docs = text_splitter.create_documents([page_content])

        for i, doc in enumerate(docs):
            chunk_id = f"{doc_id}_p{page_num}_{i}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": doc.page_content,
                "start_pos": doc.metadata.get("start_index"),
                "metadata": {
                    "doc_id": doc_id,
                    "page": page_num,
                    "section": current_section  
                }
            })
    return all_chunks


#  Helper Functions
def validate_output(output: Dict) -> bool:
    """Validate JSON output structure."""
    return all([
        isinstance(output.get("doc_id"), str),
        isinstance(output.get("chunks"), list),
        isinstance(output.get("tables"), list),
        isinstance(output.get("images"), list)
    ])

# Initially didnt care about this. but upon closer look at the task given, I decided to get this right ...
# This function scans text for sensitive info (emails, SSNs) 
# and replaces them with [REDACTED] so we don’t leak PII (Personal info stuff) in logs/outputs.
def redact_pii(text: str) -> str:
    """Redact PII (emails, SSNs) from text for security."""
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[REDACTED]', text)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED]', text)
    return text


# Main Ingestion Pipeline 
def ingest(pdf_paths: List[str]) -> Dict[str, Dict]:
    """Main pipeline: Batch ingest, validate, extract, save JSON."""
    results = {}
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            logger.warning(f"{pdf_path} not found. Skipping.")
            continue
        
        try:
            # Step 1: robust structure extraction
            structure = extract_structure_with_unstructured(pdf_path)
            doc_id = generate_doc_id(pdf_path, structure["title"])
            
            # Step 2: excellent table and image extractors    ###todo: diff btw image and equation? fix eq latwr
            tables = extract_tables(pdf_path, doc_id)
            images = extract_images(pdf_path, doc_id)
            
            # Step 3: Use the new semantic chunking method
            chunks = chunk_text_semantic(structure["pages_text"], structure["sections"], doc_id)
            
            # Step 4: Redact PII from the full text for storage
            redacted_full_text = redact_pii(structure["full_text"])
            
            # Step 5: Assemble the final output
            output = {
                "doc_id": doc_id,
                "title": structure["title"],
                "abstract": structure["abstract"],
                "sections": structure["sections"],
                "chunks": chunks,
                "tables": tables,
                "images": images,
                "full_text_preview": redacted_full_text[:2000] # Store a preview
            }
            
            if not validate_output(output):
                logger.error(f"Invalid output for {doc_id}")
                continue
            
            # Step 6: Save the output !
            output_dir = Path("data/processed") / doc_id
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=4, cls=CustomJSONEncoder)
            
            results[doc_id] = output
            logger.info(f"Successfully ingested {pdf_path} as {doc_id}: {len(chunks)} chunks, {len(tables)} tables, {len(images)} images.")
        
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}", exc_info=True)
            continue
    
    if not results:
        raise ValueError("No valid PDFs were ingested.")
    
    return results


if __name__ == "__main__":
    # Ensure you have the required libraries installed, pls follow the github Readme.md for installation steps:
    Start = time.time()
    sample_pdfs = ["data/All you need is attention.pdf"] # Add your files
    if not os.path.exists(sample_pdfs[0]):
        print(f"Error: Sample PDF not found at {sample_pdfs[0]}")
        print("Please add a PDF to the 'data/' directory to run this script.")
        end = time.time()
        print(f"Time taken: {end - Start} seconds, or {(end - Start)/60} minutes")
    else:
        results = ingest(sample_pdfs)
        print(f"\nIngestion complete. Processed doc_ids: {list(results.keys())}")
        end = time.time()
        print(f"\nTime taken: {end - Start} seconds, or {(end - Start)/60} minutes")