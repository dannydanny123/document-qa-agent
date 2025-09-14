# Smart Index Builder (Index_builder.py)

This script is the engine behind the AI agent's fast and intelligent search. It takes the structured JSON data produced by `ingestion.py` and builds highly-optimized search indexes that let the agent find relevant info in milliseconds. Think of it like a “card catalog” for all your documents.

## Key Features

### 1. Multi-Representation Indexing
The script doesn’t just store raw text—it organizes content intelligently:

- **Summaries:** Important info from titles and abstracts.  
- **Text Chunks:** Includes section titles for context.  
- **Tables:** Converts CSV tables into searchable text.  
- **Figure Captions:** Indexed as descriptive text.  
- **Equations:** Cleaned and normalized text of formulas is searchable.  

### 2. Hybrid Search Indexes
It builds two types of indexes for maximum flexibility:

- **FAISS Index (`faiss_index.bin`):** Fast semantic search based on meaning and context.  
- **BM25 Index (`bm25_index.pkl`):** Keyword search to handle exact matches like acronyms or codes.  

### 3. Context & Cleaning
Before indexing, the script cleans and enriches the data:

- Removes noisy headers like arXiv footers.  
- Normalizes whitespace and text formatting.  
- Prepends section titles to chunks so the model understands context better.  

### 4. Importance Weighting
Each piece of content gets a weight based on its type (e.g., summaries are more important than normal chunks). This helps the agent prioritize the most relevant info during searches.  

## How to Run
This is an offline script, run after `ingestion.py`. You can run it through the main `agent.py` orchestrator or standalone.

**Steps:**
1. Make sure `ingestion.py` has run and `src/data/processed` contains metadata files.  
2. From the project root, run:

```bash
python "src/tasks/Stage 2/Index_builder.py"

### Output Files

faiss_index.bin – Vector embeddings for semantic search.

bm25_index.pkl – Pickled BM25 keyword index.

indexed_documents.json – Maps all indexed text to metadata like doc ID, page, section, type, and weight. This helps the agent retrieve the original content.