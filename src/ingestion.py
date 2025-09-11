import os
import json
import hashlib
import re
from typing import List, Dict, Any, Tuple
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

# --- NEW IMPORTS for Advanced Equation Detection ---
import numpy as np
import cv2
from easyocr import Reader
from pix2tex.cli import LatexOCR
# ---

# --- NEW IMPORTS for Tesseract OCR ---
import pytesseract
from PIL import Image
import io
# ---

# You will need to install these:
# pip install "unstructured[pdf]" langchain
# also need to install layoutparser and pytesseract for 'unstructured'
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pypdf")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- ADDED: Model Initialization (Initialized once for efficiency) ---
print("Initializing OCR and LaTeX models...")
try:
    easy_reader = Reader(["en"])
    latex_model = LatexOCR()
    print("Models initialized successfully.")
except Exception as e:
    print(f"CRITICAL: Failed to initialize OCR/LaTeX models: {e}")
    print("Please ensure EasyOCR and pix2tex are installed correctly.")
# ---

# --- ADDED: Configuration for Equation Detection ---
CONFIG = {
    "Y_TOL": 18,
    "SCORE_THRESH": 0.9,
    "CONFIRM_WITH_MODEL": True,
    "PAD_FRAC": 0.18,
    "MIN_CROP_W": 8,
    "MIN_CROP_H": 8,
}
# ---


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
        elements = partition_pdf(pdf_path, strategy="fast") # UPDATED TO hi_res
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


# --- ADDED: Helper functions for the new equation pipeline ---
STRONG_OPS = set("=+×÷*/^_{}[]()∑∫√\\")
LATEX_MARKERS = ["\\frac", "\\sum", "\\int", "\\sqrt", "\\left", "\\right", "\\cdot", "\\pi", "\\theta"]

def normalize_bbox(bbox) -> Tuple[int, int, int, int]:
    pts = [(float(p[0]), float(p[1])) for p in bbox]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

def group_linewise(results: List[Tuple], y_tol: float = CONFIG["Y_TOL"]) -> List[List[Tuple]]:
    if not results: return []
    tokens = sorted(
        [(idx, r, (normalize_bbox(r[0])[1] + normalize_bbox(r[0])[3]) / 2.0, normalize_bbox(r[0])[0]) for idx, r in enumerate(results)],
        key=lambda t: (t[2], t[3])
    )
    groups, cur_group, cur_y = [], [tokens[0][1]], tokens[0][2]
    for t in tokens[1:]:
        if abs(t[2] - cur_y) <= y_tol:
            cur_group.append(t[1])
            cur_y = (cur_y * len(cur_group) + t[2]) / (len(cur_group) + 1)
        else:
            groups.append(cur_group)
            cur_group, cur_y = [t[1]], t[2]
    groups.append(cur_group)
    return groups

TOKEN_DIGIT_RE = re.compile(r'\d')
def token_is_strict_math(token: str) -> bool:
    t = token.strip()
    if not t: return False
    if '\\' in t or any(op in t for op in STRONG_OPS) or re.search(r'\d+/\d+|\d[A-Za-z]|[A-Za-z]\d|[\^_]', t):
        return True
    if re.search(r'(alpha|beta|gamma|delta|lambda|mu|sigma|omega|theta|pi)', t, re.I):
        return True
    if TOKEN_DIGIT_RE.search(t) and len(re.sub(r'[^A-Za-z0-9]', '', t)) <= 3:
        return True
    return False

def group_features(group: List[Tuple]) -> Dict:
    texts = [t[1] for t in group]
    joined = " ".join(texts).strip()
    tokens = [tk for tk in re.split(r'\s+', joined) if tk]
    if not tokens: return {}
    math_flags = [token_is_strict_math(tk) for tk in tokens]
    token_math_ratio = sum(math_flags) / len(tokens)
    op_count = sum(c in STRONG_OPS for c in joined)
    digit_count = sum(c.isdigit() for c in joined)
    bboxes = [normalize_bbox(t[0]) for t in group]
    x1, y1 = min(b[0] for b in bboxes), min(b[1] for b in bboxes)
    x2, y2 = max(b[2] for b in bboxes), max(b[3] for b in bboxes)
    return {
        "joined": joined, "token_math_ratio": token_math_ratio,
        "op_count": op_count, "digit_count": digit_count,
        "bbox": (x1, y1, x2, y2)
    }

def confirm_with_pix2tex(crop_pil: Image.Image) -> Tuple[bool, str]:
    try:
        latex = latex_model(crop_pil)
        if not latex or not latex.strip(): return False, ""
        if any(m in latex for m in LATEX_MARKERS) or (re.search(r'[\d]', latex) and re.search(r'[=+\-^_{}\\/]', latex)):
            return True, latex
    except Exception: return False, ""
    return False, latex

def find_equations_on_page(page_image_path: str) -> Tuple[List[Dict], np.ndarray]:
    """Runs the full equation detection pipeline on a single page image."""
    easy_results = easy_reader.readtext(page_image_path, detail=1)
    groups = group_linewise(easy_results)
    cv_img = cv2.imread(page_image_path)
    h, w = cv_img.shape[:2]
    detections = []
    for group in groups:
        feats = group_features(group)
        if not feats: continue
        score = 2.0 * feats["token_math_ratio"] + 0.8 * min(1.0, feats["op_count"] / 3.0)
        if score < CONFIG["SCORE_THRESH"]: continue
        x1, y1, x2, y2 = feats["bbox"]
        pad = int((y2 - y1) * CONFIG["PAD_FRAC"])
        x1c, y1c, x2c, y2c = max(0, x1 - pad), max(0, y1 - pad), min(w - 1, x2 + pad), min(h - 1, y2 + pad)
        if (x2c - x1c) < CONFIG["MIN_CROP_W"] or (y2c - y1c) < CONFIG["MIN_CROP_H"]: continue
        crop_pil = Image.fromarray(cv2.cvtColor(cv_img[y1c:y2c, x1c:x2c], cv2.COLOR_BGR2RGB))
        is_math, latex_str = confirm_with_pix2tex(crop_pil) if CONFIG["CONFIRM_WITH_MODEL"] else (True, "")
        if is_math:
            cv2.rectangle(cv_img, (x1c, y1c), (x2c, y2c), (0, 220, 0), 2)
            detections.append({"bbox": [x1c, y1c, x2c, y2c], "ocr_text": feats["joined"], "latex_text": latex_str})
    return detections, cv_img
# ---


# --- REPLACED FUNCTION ---
def extract_figures_and_equations(pdf_path: str, doc_id: str) -> Tuple[List[Dict], List[Dict], str]:
    """
    This is the new integrated function. It extracts embedded images as figures,
    then renders each page to run a sophisticated OCR pipeline to detect and
    extract equations, returning an annotated overview image.
    """
    doc = fitz.open(pdf_path)
    figures = []
    equations = [] # This will store all detected equations
    annotated_page_paths = []
    
    assets_dir = f"data/assets/{doc_id}"
    os.makedirs(assets_dir, exist_ok=True)

    # This function now processes the PDF page by page
    for page_num, page in enumerate(doc, 1):
        logger.info(f"Processing Page {page_num}/{len(doc)}...")

        # Part 1: Extract embedded images as figures, same as your original code
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                base_image = doc.extract_image(img[0])
                if not base_image: continue
                
                bbox = page.get_image_bbox(img)
                bbox_serializable = [float(b) for b in bbox] if bbox else None
                
                img_path = os.path.join(assets_dir, f"fig_page{page_num}_{img_index}.png")
                with open(img_path, "wb") as f: f.write(base_image["image"])

                caption = ""
                if bbox_serializable:
                    caption_area = [bbox_serializable[0], bbox_serializable[3], bbox_serializable[2], bbox_serializable[3] + 50]
                    caption = page.get_text("text", clip=caption_area).strip()
                
                figures.append({
                    "id": f"fig_{page_num}_{img_index}", "page": page_num,
                    "bbox": bbox_serializable, "path": img_path, "caption": caption
                })
            except Exception as e:
                logger.error(f"Error extracting figure on page {page_num}: {e}")

        # Part 2: Render the entire page and run the advanced equation detector on it
        page_image_path = os.path.join(assets_dir, f"page_{page_num}_render.png")
        pix = page.get_pixmap(dpi=300) # High DPI is crucial for good OCR
        pix.save(page_image_path)
        
        page_detections, annotated_img = find_equations_on_page(page_image_path)
        
        # Add the current page number to each detection and add to the main list
        for det in page_detections:
            det["page"] = page_num
        equations.extend(page_detections)
        
        # We save the annotated version of this page to combine later
        annotated_page_path = os.path.join(assets_dir, f"page_{page_num}_annotated.png")
        cv2.imwrite(annotated_page_path, annotated_img)
        annotated_page_paths.append(annotated_page_path)

    doc.close()

    # Part 3: After processing all pages, stitch the annotated images together
    final_annotated_path = os.path.join(assets_dir, f"{doc_id}_equations_overview.png")
    if annotated_page_paths:
        annotated_images = [cv2.imread(p) for p in annotated_page_paths]
        # To stack images vertically, they must have the same width
        max_width = max(img.shape[1] for img in annotated_images)
        resized_images = [
            cv2.copyMakeBorder(img, 0, 0, 0, max_width - img.shape[1], cv2.BORDER_CONSTANT, value=[255, 255, 255])
            for img in annotated_images
        ]
        combined_image = cv2.vconcat(resized_images)
        cv2.imwrite(final_annotated_path, combined_image)
    else:
        final_annotated_path = None # No equations were found anywhere

    logger.info(f"Extraction complete. Found {len(figures)} figures and {len(equations)} equations.")
    return figures, equations, final_annotated_path


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
# --- UPDATED FUNCTION ---
def validate_output(output: Dict) -> bool:
    """Validate JSON output structure."""
    return all([
        isinstance(output.get("doc_id"), str),
        isinstance(output.get("chunks"), list),
        isinstance(output.get("tables"), list),
        isinstance(output.get("figures"), list), # UPDATED
        isinstance(output.get("equations"), list) # UPDATED
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

            # --- UPDATED LOGIC ---
            # Step 2: Extract tables, figures, and equations separately
            tables = extract_tables(pdf_path, doc_id)
            # This is the new, powerful function call
            figures, equations, annotated_path = extract_figures_and_equations(pdf_path, doc_id)

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
                "figures": figures,
                "equations": equations,
                "annotated_overview_path": annotated_path, # New field for the debug image
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
            # --- UPDATED LOG ---
            logger.info(f"Successfully ingested {pdf_path} as {doc_id}: {len(chunks)} chunks, {len(tables)} tables, {len(figures)} figures, and {len(equations)} equations.")

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
    if not os.path.exists(sample_pdfs[0]):
        print(f"Error: Sample PDF not found at {sample_pdfs[0]}")
        print("Please add a PDF to the 'data/' directory to run this script.")
    else:
        results = ingest(sample_pdfs)
        if results:
            print(f"\nIngestion complete. Processed doc_ids: {list(results.keys())}")

    end = time.time()
    print(f"\nTime taken: {end - Start:.2f} seconds, or {(end - Start)/60:.2f} minutes")
