import json
import hashlib
import re
from typing import List, Dict, Any
import time
import os
import csv
# --- Core PDF Processing Libraries ---
import fitz  # PyMuPDF
import pdfplumber
import camelot
import pandas as pd
from pathlib import Path
import tempfile
import logging
import warnings

# pls install these:
# pip install "unstructured[pdf]" langchain
# also need to install layoutparser and pytesseract for 'unstructured'
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pypdf")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SRC_DIR = Path("src")
DATA_DIR = SRC_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ASSETS_DIR = DATA_DIR / "assets"


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
        elements = partition_pdf(pdf_path, strategy="fast")
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
    
    abstract = ""
    sections = []
    try:
        abstract_element_indices = [i for i, el in enumerate(elements) if "abstract" in el.text.lower() and el.category == "Title"]
        if abstract_element_indices:
            start_index = abstract_element_indices[0] + 1
            for i in range(start_index, len(elements)):
                if elements[i].category == "Title":
                    break
                abstract += elements[i].text + "\n"
    except Exception:
        pass

    section_pattern = r'^\d+(\.\d+)*\s+.*'
    sections = [{"name": el.text, "page": el.metadata.page_number} for el in elements if el.category == "Title" and re.match(section_pattern, el.text)]

    return {
        "title": title,
        "abstract": abstract.strip(),
        "sections": sections,
        "pages_text": pages_text,
        "full_text": full_text.strip()
    }


def _sanitize_cell(cell_data: Any) -> str:
    """A helper function to clean individual table cells."""
    if cell_data is None:
        return ""
    # Convert to string, remove null bytes, and replace newlines with a space
    return str(cell_data).replace('\x00', '').replace('\n', ' ').replace('\r', ' ')

def extract_tables(pdf_path: str, doc_id: str) -> List[Dict]:
    """
    It was challanging to pull tables out of a PDF...
    First attempt with Camelot (lattice → stream), if that fails, 
    fall back to pdfplumber. 
    Everything gets saved as CSV for traceability.
    """
    extracted = []
    tables = None # Initialize to ensure variable exists

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # First shot: lattice mode
            tables = camelot.read_pdf(
                pdf_path,
                pages='all',
                flavor='lattice'
            )
            if not tables:
                logger.info(f"No lattice tables found in {pdf_path}, trying stream mode")
                # Second shot, if first fails: stream mode
                tables = camelot.read_pdf(
                    pdf_path,
                    pages='all',
                    flavor='stream',
                    temp_dir=tmpdir
                )
        except Exception as e:
            # Fallback to pdfplumber
            logger.warning(f"Camelot failed for {pdf_path}: {e}. Falling back to pdfplumber.")
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    raw_tables = page.extract_tables()
                    for i, table in enumerate(raw_tables):
                        if table:
                            cleaned_table = [[_sanitize_cell(cell) for cell in row] for row in table]
                            df = pd.DataFrame(cleaned_table)
                            
                            doc_assets_dir = ASSETS_DIR / doc_id
                            doc_assets_dir.mkdir(parents=True, exist_ok=True)
                            csv_path = doc_assets_dir / f"{doc_id}_tbl_{page_num}_{i}.csv"

                            df.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL)
                            extracted.append({
                                "id": f"{page_num}_{i}", "page": page_num,
                                "csv_path": str(csv_path), "shape": df.shape
                            })
            return extracted # Return here since we used the fallback
    
    # Process the tables returned by Camelot if the try block succeeded
    if tables:
        for i, table in enumerate(tables):
            if not table.df.empty:
                # FIX: Replaced deprecated 'applymap' with modern 'map'
                df = table.df.map(_sanitize_cell)

                doc_assets_dir = ASSETS_DIR / doc_id
                doc_assets_dir.mkdir(parents=True, exist_ok=True)
                csv_path = doc_assets_dir / f"{doc_id}_tbl_{table.page}_{i}.csv"

                df.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL)
                extracted.append({
                    "id": f"{table.page}_{i}", "page": table.page,
                    "csv_path": str(csv_path), "shape": df.shape
                })

    logger.info(f"Extracted {len(extracted)} tables for {pdf_path}")
    return extracted
        
    #     # Process the tables returned by Camelot
    #     for i, table in enumerate(tables):
    #         if not table.df.empty:
    #             # --- SANITIZE DATA HERE ---
    #             df = table.df.applymap(_sanitize_cell)

    #             doc_assets_dir = ASSETS_DIR / doc_id
    #             doc_assets_dir.mkdir(parents=True, exist_ok=True)
    #             csv_path = doc_assets_dir / f"tbl_{table.page}_{i}.csv"

    #             df.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL)
    #             extracted.append({
    #                 "id": f"{table.page}_{i}", "page": table.page,
    #                 "csv_path": str(csv_path), "shape": df.shape # Convert Path to string for JSON
    #             })

    #     logger.info(f"Extracted {len(extracted)} tables for {pdf_path}")
    # return extracted

def extract_figures(pdf_path: str, doc_id: str) -> List[Dict]:
    """
    Extracts all images from a PDF and saves them as figures,
    along with their bounding boxes and captions if possible.
    """
    doc = fitz.open(pdf_path)
    figures = []

    # Use the global ASSETS_DIR to create the correct path
    doc_assets_dir = ASSETS_DIR / doc_id
    doc_assets_dir.mkdir(parents=True, exist_ok=True)

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
                    # PyMuPDF can choke on some objects, so don’t crash, just log
                    logger.warning(f"No bbox for image on page {page_num}: {e}")

                # save the raw bytes to png on disk
                img_path = doc_assets_dir / f"fig_page{page_num}_{img_index}.png"
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
                    "path": str(img_path), # Convert Path to string for JSON
                    "caption": caption
                })

            except Exception as e:
                # generic catch-all so a single bad image doesn’t ruin the run
                logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                continue

    doc.close()
    logger.info(f"Extracted {len(figures)} figures for {pdf_path}")
    return figures


# --- Final, Top-Notch Text Equation Extraction Block ---
def is_scientific_equation(line: str) -> bool:
    """
    A top-notch, production-ready heuristic to verify if a line of text is a
    scientific equation. This version uses aggressive rejection filters for high precision.
    """
    line = line.strip()

    # --- Stage 1: Aggressive Rejection Filters ---
    if len(line.replace(" ", "")) < 3: return False
    if re.fullmatch(r'\s*\([\w\.\-]+\)\s*', line): return False
    if re.search(r'\b(19|20)\d{2}\b', line) or re.search(r'\b(Vol|pp|et al)\b', line, re.IGNORECASE) or re.search(r'\d+\(\d+\):\d+–\d+', line): return False
    data_like_tokens = re.findall(r'\b(\d+\.\d+|\d+|[A-Za-z]+\d+)\b', line)
    if '=' not in line and len(data_like_tokens) > 3: return False
    if re.match(r'^\d+(\.\d+)*\s+[A-Za-z]{4,}', line): return False

    # --- Stage 2: Acceptance Heuristics ---
    has_equals = '=' in line
    math_symbols = re.findall(r'[+\-*/^_{}()\[\]|∑∫√]', line)
    if not has_equals and len(math_symbols) < 2: return False
    if line.count('(') != line.count(')'): return False
    long_words = re.findall(r'[a-zA-Z]{5,}', line)
    if len(long_words) > 2: return False
    
    return True

def deduplicate_equations(equations: List[Dict]) -> List[Dict]:
    """Removes duplicate equations found in overlapping chunks."""
    seen = set()
    unique_equations = []
    for eqn in equations:
        normalized_text = re.sub(r'\s+', '', eqn['equation_text'])
        if normalized_text not in seen:
            seen.add(normalized_text)
            unique_equations.append(eqn)
    return unique_equations

def extract_text_equations(chunks: List[Dict]) -> List[Dict]:
    """
    Extracts text equations using the final high-precision, multi-line assembly strategy.
    """
    text_equations = []
    
    for chunk in chunks:
        lines = chunk['text'].split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if is_scientific_equation(line):
                equation_buffer = [line]
                j = i + 1
                while j < len(lines) and (lines[j].startswith('  ') or len(lines[j].strip()) < 20):
                    equation_buffer.append(lines[j].strip())
                    j += 1
                merged_text = " ".join(equation_buffer)
                if is_scientific_equation(merged_text):
                    text_equations.append({
                        "id": f"txt_eqn_{chunk['chunk_id']}_{i}",
                        "page": chunk['metadata']['page'],
                        "section": chunk['metadata']['section'],
                        "type": "heuristic",
                        "equation_text": merged_text
                    })
                i = j
            else:
                i += 1

    unique_equations = deduplicate_equations(text_equations)
    logger.info(f"Extracted {len(unique_equations)} unique text-based equations from chunks.")
    return unique_equations

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

    page_to_section_map = {sec["page"]: sec["name"] for sec in sorted(sections, key=lambda x: x["page"])}
    current_section = "Introduction"

    all_chunks = []
    for page_info in sorted(pages_text, key=lambda x: x["page"]):
        page_num, page_content = page_info["page"], page_info["text"]
        if page_num in page_to_section_map: current_section = page_to_section_map[page_num]

        docs = text_splitter.create_documents([page_content])
        for i, doc in enumerate(docs):
            all_chunks.append({
                "chunk_id": f"{doc_id}_p{page_num}_{i}",
                "text": doc.page_content,
                "start_pos": doc.metadata.get("start_index"),
                "metadata": {"doc_id": doc_id, "page": page_num, "section": current_section}
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
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[REDACTED]', text)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED]', text)
    return text

# --- ADDED: Normalization function for LLM-friendly output ---
def normalize_equation_text(text: str) -> str:
    """
    Replaces common Unicode math symbols with their LaTeX equivalents.
    Ensures equations are standardized before further processing.
    """
    normalization_dict = {
        '−': '-', '–': '-', '—': '-', '×': '\\times', '·': '\\cdot', '÷': '\\div', '≤': '\\leq',
        '≥': '\\geq', '≠': '\\neq', '≈': '\\approx', '±': '\\pm', '∞': '\\infty', '∑': '\\sum',
        '∏': '\\prod', '∫': '\\int', '√': '\\sqrt', '→': '\\to', '←': '\\leftarrow', '↔': '\\leftrightarrow',
        '⇔': '\\iff', '°': '^{\\circ}', 'π': '\\pi', 'θ': '\\theta', 'λ': '\\lambda', 'μ': '\\mu',
        'σ': '\\sigma', 'Δ': '\\Delta', 'δ': '\\delta', 'α': '\\alpha', 'β': '\\beta', 'γ': '\\gamma',
    }
    for uni, latex in normalization_dict.items():
        text = text.replace(uni, latex)
    return text.strip()


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
            
            # Step 2: Extract tables and figures.
            tables = extract_tables(pdf_path, doc_id)
            figures = extract_figures(pdf_path, doc_id)
            
            # Step 3: Use the new semantic chunking method
            chunks = chunk_text_semantic(structure["pages_text"], structure["sections"], doc_id)
            
            # Step 3.5: Extract equations from the text chunks using the new logic
            equations = extract_text_equations(chunks)
            
            # --- ADDED NORMALIZATION STEP ---
            # This loop adds a clean, LLM-ready version of the equation text.
            for eqn in equations:
                eqn["normalized_text"] = normalize_equation_text(eqn["equation_text"])
            
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
                "equations": equations,
                "full_text_preview": redacted_full_text[:2000]
            }
            
            if not validate_output(output):
                logger.error(f"Invalid output for {doc_id}")
                continue
            
            # Step 6: Save the output !
            output_dir = PROCESSED_DIR / doc_id
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
    
    # This logic automatically finds all PDFs in the data directory
    pdfs_to_process = [str(p) for p in DATA_DIR.glob("*.pdf")]
    
    if not pdfs_to_process:
        print("No PDFs found in the data directory to process.")
    else:
        print(f"Found {len(pdfs_to_process)} PDF(s) to process in '{DATA_DIR}'.")
        results = ingest(pdfs_to_process)
        if results:
            print(f"\nIngestion complete. Processed doc_ids: {list(results.keys())}")
    
    end = time.time()
    print(f"\nTime taken: {end - Start:.2f} seconds, or {(end - Start)/60:.2f} minutes")

