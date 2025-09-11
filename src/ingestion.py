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

# --- REMOVED HEAVY IMPORTS: easyocr, pix2tex, cv2, numpy are no longer needed ---

# --- IMPORTS for lightweight image and text processing ---
from PIL import Image
import io

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


# --- MODIFIED FUNCTION: Simplified to only extract figures ---
def extract_figures(pdf_path: str, doc_id: str) -> List[Dict]:
    """
    Extracts all images from a PDF and saves them as figures,
    along with their bounding boxes and captions if possible.
    """
    doc = fitz.open(pdf_path)
    figures = []

    # make sure the doc gets its own asset folder
    os.makedirs(f"data/assets/{doc_id}", exist_ok=True)

    # Loop through each page and image
    for page_num, page in enumerate(doc, 1):
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            try:
                # Extract the image bytes
                xref = img[0]
                base_image = doc.extract_image(xref)

                if not base_image:  # sometimes xref exists but image data is junk
                    logger.warning(f"Skipping invalid image on page {page_num}, index {img_index}")
                    continue

                # try to grab bounding box (where on the page this image lives)
                bbox_serializable = None
                try:
                    bbox = page.get_image_bbox(img)
                    bbox_serializable = [float(b) for b in bbox] if bbox else None
                except ValueError as e:
                    # PyMuPDF can choke on some objects → don’t crash, just log
                    logger.warning(f"No bbox for image on page {page_num}: {e}")

                # save the raw bytes → png on disk
                img_path = f"data/assets/{doc_id}/fig_page{page_num}_{img_index}.png"
                with open(img_path, "wb") as f:
                    f.write(base_image["image"])

                # try to guess the caption (usually text directly under the image box)
                caption = ""
                if bbox_serializable:
                    caption_area = [bbox_serializable[0], bbox_serializable[3], bbox_serializable[2], bbox_serializable[3] + 50]  # 50px below
                    caption = page.get_text("text", clip=caption_area).strip()
                
                # Append the figure data to the list
                figures.append({
                    "id": f"fig_{page_num}_{img_index}",
                    "page": page_num,
                    "bbox": bbox_serializable,
                    "path": img_path,
                    "caption": caption
                })

            except Exception as e:
                # generic catch-all so a single bad image doesn’t ruin the run
                logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                continue

    doc.close()
    logger.info(f"Extracted {len(figures)} figures for {pdf_path}")
    return figures


# --- ADDED: Helper functions for Text-Based Equation Detection ---
STRONG_OPS = set("=+×÷*/^_{}[]()∑∫√\\")
TOKEN_DIGIT_RE = re.compile(r'\d')

def token_is_strict_math(token: str) -> bool:
    """A helper function to conservatively check if a single token is mathematical."""
    t = token.strip()
    if not t: return False
    if '\\' in t or any(op in t for op in STRONG_OPS) or re.search(r'\d+/\d+|\d[A-Za-z]|[A-Za-z]\d|[\^_]', t):
        return True
    if re.search(r'(alpha|beta|gamma|delta|lambda|mu|sigma|omega|theta|pi)', t, re.I):
        return True
    if TOKEN_DIGIT_RE.search(t) and len(re.sub(r'[^A-Za-z0-9]', '', t)) <= 3:
        return True
    return False

def text_line_features(line: str) -> Dict:
    """This computes features for a line of text to help classify it."""
    tokens = [tk for tk in re.split(r'\s+', line.strip()) if tk]
    if not tokens:
        return {"tokens_count": 0}
    
    math_flags = [token_is_strict_math(tk) for tk in tokens]
    token_math_ratio = sum(math_flags) / len(tokens)
    op_count = sum(c in STRONG_OPS for c in line)
    digit_count = sum(c.isdigit() for c in line)
    alpha_tokens = sum(1 for tk in tokens if re.fullmatch(r'[a-zA-Z]{2,}', tk))
    alpha_ratio = alpha_tokens / len(tokens)
    avg_tok_len = sum(len(tk) for tk in tokens) / len(tokens)
    
    return {
        "tokens": tokens, "token_math_ratio": token_math_ratio,
        "op_count": op_count, "digit_count": digit_count,
        "alpha_ratio": alpha_ratio, "avg_tok_len": avg_tok_len,
        "tokens_count": len(tokens)
    }

def heuristic_accept(features: Dict) -> bool:
    """This is the main heuristic that decides if a line is an equation."""
    if features.get("tokens_count", 0) == 0:
        return False
    # These rules are tuned to find lines that are mostly math
    if features["token_math_ratio"] < 0.5:
        return False
    if features["op_count"] < 1 and features["digit_count"] < 1:
        return False
    if features["alpha_ratio"] > 0.5 and features["avg_tok_len"] > 4:
        return False
    return True


# --- NEW FUNCTION: The "sexy" lightweight text-based equation extraction, lol.
def extract_text_equations(chunks: List[Dict]) -> List[Dict]:
    """
    Extracts inline text-based equations from text chunks using a sophisticated heuristic.
    This is a fast and lightweight alternative to image-based detection.
    """
    text_equations = []
    
    for chunk in chunks:
        # We look for equations line-by-line within each text chunk
        lines = chunk['text'].split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) > 300: # Skip empty or overly long lines
                continue
            
            # Here we apply the powerful heuristic logic
            features = text_line_features(line)
            if heuristic_accept(features):
                logger.info(f"Detected text-based equation on page {chunk['metadata']['page']}: '{line}'")
                text_equations.append({
                    "id": f"txt_eqn_{chunk['chunk_id']}_{i}",
                    "page": chunk['metadata']['page'],
                    "section": chunk['metadata']['section'],
                    "equation_text": line
                })
                
    logger.info(f"Extracted {len(text_equations)} text-based equations from chunks.")
    return text_equations


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
        isinstance(output.get("figures"), list),
        isinstance(output.get("equations"), list)
    ])

# Initially didnt care about this. but upon closer look at the task given, I decided to get this right ...
# This function scans text for sensitive info (emails, SSNs) 
# and replaces them with [REDACTED] so we don’t leak PII (Personal info stuff) in logs/outputs.
def redact_pii(text: str) -> str:
    """Redact PII (emails, SSNs) from text for security."""
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\w+\b', '[REDACTED]', text)
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
            
            # --- UPDATED LOGIC ---
            # Step 2: Extract tables and figures.
            tables = extract_tables(pdf_path, doc_id)
            figures = extract_figures(pdf_path, doc_id)
            
            # Step 3: Use the new semantic chunking method
            chunks = chunk_text_semantic(structure["pages_text"], structure["sections"], doc_id)
            
            # Step 3.5: Extract equations from the text chunks using the new logic
            equations = extract_text_equations(chunks)
            
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
                "figures": figures,
                "equations": equations, # Now contains sophisticated text-based results
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
            logger.info(f"Successfully ingested {pdf_path} as {doc_id}: {len(chunks)} chunks, {len(tables)} tables, {len(figures)} figures, and {len(equations)} text equations.")
        
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}", exc_info=True)
            continue
    
    if not results:
        raise ValueError("No valid PDFs were ingested.")
    
    return results


if __name__ == "__main__":
    # Ensure you have the required libraries installed, pls follow the github Readme.md for installation steps:
    Start = time.time()
    sample_pdfs = ["data/test.pdf"] # Add your files
    if not sample_pdfs or not os.path.exists(sample_pdfs[0]):
        print(f"Error: Sample PDF not found at '{sample_pdfs[0] if sample_pdfs else 'N/A'}'")
        print("Please add a PDF to the 'data/' directory to run this script.")
    else:
        results = ingest(sample_pdfs)
        if results:
            print(f"\nIngestion complete. Processed doc_ids: {list(results.keys())}")
    
    end = time.time()
    print(f"\nTime taken: {end - Start:.2f} seconds, or {(end - Start)/60:.2f} minutes")
