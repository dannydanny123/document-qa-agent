# README Ingestion Pipeline (`ingestion.py`)

This script is the backbone of the AI Document Q\&A Agent. Its main job is to take raw, unstructured PDF documents and turn them into structured, machine-readable JSON files that can be used to build a high-performance search index.

The pipeline is built to be enterprise-ready, with features like robust error handling, multi-modal content extraction, and intelligent state tracking to avoid unnecessary work.

---

## Core Features

### 1. Multi-Modal Content Extraction

This pipeline goes beyond simple text extraction to create a rich, structured representation of each document:

* **Layout-Aware Text Parsing:** Uses `unstructured.io` with a high-resolution strategy to analyze the visual layout of each page. This helps accurately identify titles, abstracts, section headers, and more.
* **Figure Extraction:** Embedded images are extracted and saved as `.png` files. The script also tries to capture the associated caption by analyzing the text below each image.
* **Robust Table Parsing:** Uses a two-step approach for tables. First, it tries `camelot` for tables with clear grid lines. If that fails, it falls back to `pdfplumber`. All tables are sanitized and saved as `.csv` files for reliability and traceability.
* **High-Precision Equation Detection:** A multi-stage heuristic identifies and extracts mathematical equations from the text. It’s tuned for scientific papers to separate complex formulas from simple numbers or table rows, minimizing false positives.

---

### 2. Intelligent Text Chunking

To make the documents ready for a Retrieval-Augmented Generation (RAG) model, the script uses `langchain`'s `RecursiveCharacterTextSplitter`.

* Breaks documents into coherent chunks that respect sentence and paragraph boundaries.
* Each chunk is enriched with metadata, like page number and section, to provide context for downstream tasks.

---

### 3. Enterprise-Ready Features

* **Stateful Processing:** Tracks processed PDFs and file hashes in `ingestion_state.json`. This means only new or modified files are processed on subsequent runs, saving time and resources.
* **Data Integrity:** If a PDF is deleted from the data directory, the script triggers a full re-processing to ensure the search index never contains outdated data.
* **Security & Privacy:** Includes a PII redaction step to remove sensitive data such as emails or Social Security Numbers before saving.

---

## How to Run

You can run this script standalone or let it be called by the main orchestrator (`agent.py`) which handles files automatically.

1. **Place PDFs:** Add your PDF documents into `src/data/`.
2. **Run from Terminal:** Navigate to your project root (`document-qa-agent`) and execute:

```bash
python "src/tasks/Stage 1/ingestion.py"
```

The script will automatically detect and process all new or updated PDFs in `src/data/`.

---

## Output Structure

After running the script, you’ll see a `processed` and an `assets` folder inside `src/data/`. Each processed PDF gets its own folder, identified by a unique `doc_id`.

* **`src/data/processed/<doc_id>/metadata.json`**
  Contains all structured information extracted from the PDF, including:

  * `doc_id`, `title`, `abstract`, `sections`
  * List of text chunks with metadata
  * List of extracted tables, with paths to `.csv` files
  * List of extracted figures, with paths to `.png` files
  * List of detected text-based equations, both original and normalized

* **`src/data/assets/<doc_id>/`**
  Stores all raw assets extracted from the PDF, such as images (`.png`) and structured table data (`.csv`).

