# AI Document Q&A Agent (app.py)

This script is the user-facing frontend for the Document Q&A pipeline. It launches an interactive web app using **Streamlit**, letting users chat with an AI agent that has deep knowledge of all ingested documents.

The app is built to be **fast, accurate, secure, and enterprise-ready**.

## Key Features

### 1. Grounded Answering from Local Documents
The agent answers questions using a **Retrieval-Augmented Generation (RAG) pipeline**. It finds the most relevant info from processed documents to provide precise answers.

### 2. Intelligent Arxiv Search
After answering from local docs, the agent performs a **context-aware search on Arxiv** to find related research papers, giving users broader insights.

### 3. Advanced RAG Pipeline
- **Hybrid Search:** Combines FAISS (semantic) and BM25 (keyword) search for both conceptual and exact matches.  
- **Cross-Encoder Reranking:** Filters and reranks results for maximum relevance.  
- **Importance Weighting:** Gives priority to key sections like titles and abstracts.

### 4. Smart Arxiv Integration
- **Context-Aware Queries:** Uses the user’s question and local answer to generate precise Arxiv queries.  
- **Filtering & Ranking:** Filters by category and recency; ranks by title, abstract, and journal prestige.  
- **Automatic Metric Extraction:** Highlights relevant quantitative metrics if mentioned (e.g., accuracy, F1-score).

## Technical Architecture
Query Flow:
User Query → Input Sanitization → [Local RAG] → Display Local Answer & Sources → [Smart Arxiv Search] → Display Arxiv Results

This ensures users get a **fast, grounded answer first**, then broader, public research immediately.

## How to Run

**Prerequisites:**  
- Install all libraries from `requirements.txt`.  
- Run data prep first:

```bash
python "src/tasks/Stage 1/ingestion.py"
python "src/tasks/Stage 2/Index_builder.py"

Create a .env file in src/ and add your Google Gemini API key:
Launch the app: 
streamlit run "src/tasks/Stage 3/app.py"


Enterprise-Grade Features

Security:

Prompt Injection Defense: Cleans user input to remove malicious instructions.

Prompt Guardrails: Ensures the agent answers only from the provided context.

Performance:

Model Caching: Heavy models and indexes are cached for fast responses.

Robustness:

Index Check: Verifies required files exist, preventing crashes.

Error Handling: Safe, user-friendly messages for any API or runtime errors.

Verifiability:

Citable Sources: Every answer includes a "Sources Used" expander showing exact text chunks from original PDFs for transparency and fact-checking.