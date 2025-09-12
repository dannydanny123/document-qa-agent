import os
import json
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import pickle
import google.generativeai as genai
from typing import List, Dict
import re

# --- Enhanced Arxiv Import with Error Handling ---
try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    # This will display an error at the top of the app if the library is missing.
    st.error("Arxiv library not available. Please install it with: pip install arxiv")
    ARXIV_AVAILABLE = False

# --- Configuration ---
load_dotenv()
INDEX_DIR = Path("data/index")
FAISS_PATH = INDEX_DIR / "faiss_index.bin"
BM25_PATH = INDEX_DIR / "bm25_index.pkl"
MAPPING_PATH = INDEX_DIR / "indexed_documents.json"

# --- Self-Healing Index Check ---
def check_for_index_files():
    if not (FAISS_PATH.exists() and BM25_PATH.exists() and MAPPING_PATH.exists()):
        st.error(
            "Index files not found! 🚨\n"
            "Please run the `build_index.py` script first to process your documents."
        )
        st.code("python src/build_index.py")
        st.stop()

# --- Cached Model Loading ---
@st.cache_resource
def load_models_and_data():
    ## Load all models and data (yeah, takes a few seconds,meanwhile chai peelo, streach out a btit lol.
    print("Loading models and data...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    faiss_index = faiss.read_index(str(FAISS_PATH))
    with open(BM25_PATH, "rb") as f: bm25_index = pickle.load(f)
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f: indexed_docs = json.load(f)
    try:
        api_key = os.getenv("GEMINI_API_KEY") # Corrected to GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found. Please check your .env file.")
        genai.configure(api_key=api_key)
        llm_model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}")
        llm_model = None
    print("Models and data loaded.")
    return encoder, cross_encoder, faiss_index, bm25_index, indexed_docs, llm_model

# --- Enhanced Arxiv Search Algorithm ---
def search_arxiv(query: str, max_results: int = 5) -> Dict:
    """
    Enhanced Arxiv search with better query parsing and error handling.
    Returns a dictionary with results and status.
    """
    if not ARXIV_AVAILABLE:
        return {
            "status": "error",
            "message": "Arxiv library not available. Please install with: pip install arxiv",
            "results": []
        }
    
    try:
        # Clean and prepare query
        query = re.sub(r'[^\w\s\-\.]', '', query)  # Remove special characters
        query = ' '.join(query.split()[:10])  # Limit query length
        
        # Search parameters
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        
        results = []
        for result in search.results():
            # Extract publication year
            year = result.published.year if result.published else "Unknown"
            
            # Clean summary
            summary = result.summary.replace('\n', ' ').strip()
            if len(summary) > 300:
                summary = summary[:300] + "..."
            
            results.append({
                "title": result.title,
                "authors": [str(author) for author in result.authors],
                "published": result.published.strftime('%Y-%m-%d') if result.published else "Unknown",
                "year": year,
                "pdf_url": result.pdf_url,
                "arxiv_url": result.entry_id,
                "summary": summary,
                "primary_category": result.primary_category if result.primary_category else "Unknown"
            })
        
        return {
            "status": "success",
            "message": f"Found {len(results)} results",
            "results": results
        }
        
    except arxiv.ArxivError as e:
        return {
            "status": "error",
            "message": f"Arxiv API error: {str(e)}",
            "results": []
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
            "results": []
        }

# --- Format Arxiv Results in Markdown ---
def format_arxiv_results(arxiv_data: Dict) -> str:
    """Formats Arxiv search results in markdown with proper styling."""
    if arxiv_data["status"] != "success" or not arxiv_data["results"]:
        return f"**Arxiv Search Results**: {arxiv_data['message']}"
    
    markdown_output = "### 📚 Related Papers from Arxiv\n\n"
    
    for i, paper in enumerate(arxiv_data["results"], 1):
        markdown_output += f"#### {i}. {paper['title']}\n\n"
        markdown_output += f"**Authors:** {', '.join(paper['authors'][:3])}"
        if len(paper['authors']) > 3:
            markdown_output += f" *+{len(paper['authors']) - 3} more*\n"
        else:
            markdown_output += "\n"
            
        markdown_output += f"**Published:** {paper['published']} | "
        markdown_output += f"**Category:** {paper['primary_category']}\n\n"
        markdown_output += f"**Summary:** {paper['summary']}\n\n"
        markdown_output += f"**Links:** [PDF]({paper['pdf_url']}) | [Arxiv Page]({paper['arxiv_url']})\n\n"
        markdown_output += "---\n\n"
    
    return markdown_output

# --- Tool 1: LocalSearchTool (unchanged) ---
class LocalSearchTool:
    """A tool for searching the local, pre-indexed collection of documents."""
    def __init__(self, encoder, cross_encoder, faiss_index, bm25_index, indexed_docs, llm_model):
        self.encoder = encoder
        self.cross_encoder = cross_encoder
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.indexed_docs = indexed_docs
        self.llm_model = llm_model

    def _search(self, query: str, k: int = 30) -> List[Dict]:
        """Performs hybrid search."""
        query_embedding = self.encoder.encode(query)
        _, faiss_indices = self.faiss_index.search(np.array([query_embedding], dtype=np.float32), k)
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:k]
        combined_indices = list(set(faiss_indices[0]) | set(bm25_indices))
        return [self.indexed_docs[i] for i in combined_indices]

    def _rerank(self, query: str, docs: List[Dict], top_n: int = 5) -> List[Dict]:
        """Reranks documents and applies importance weighting."""
        pairs = [[query, doc["content"]] for doc in docs]
        scores = self.cross_encoder.predict(pairs)
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = score
            doc["final_score"] = score * doc["metadata"].get("weight", 1.0)
        docs.sort(key=lambda x: x["final_score"], reverse=True)
        return docs[:top_n]

    def run(self, query: str) -> Dict:
        """
        The main execution method for the tool. This finds the best context,
        generates an answer, and returns both the answer and the sources.
        """
        if not self.llm_model:
            return {"answer": "Error: LLM model is not available. Check API key.", "sources": []}

        retrieved_docs = self._search(query)
        if not retrieved_docs:
            return {"answer": "No relevant information found in the local documents to answer this query.", "sources": []}

        reranked_docs = self._rerank(query, retrieved_docs)
        
        context = "\n\n---\n\n".join([doc["content"] for doc in reranked_docs])
        prompt = f"""
        You are a research assistant tasked with answering questions based on the provided document context. 

        Instructions: 
        1. Provide a clear, concise answer to the user's question, formatted neatly in Markdown. 
        2. Structure it by Using headings (e.g., `### Summary`), bullet points (`*`), and bold text (`**text**`) for clarity. For all mathematical formulas, variables, or code snippets, you MUST enclose them in single backticks (` `).
        3. Support your answer with citations, explicitly mentioning the document title and page number. 
        4. Do not speculate or add external knowledge. Stay strictly within the given context. 

        CONTEXT: 
        {context} 

        QUESTION: 
        {query} 
        """
        try:
            response = self.llm_model.generate_content(prompt)
            # The tool now returns a dictionary, separating the answer from the sources
            return {"answer": response.text, "sources": reranked_docs}
        except Exception as e:
            return {"answer": f"Error during generation: {e}", "sources": reranked_docs}

# --- Streamlit User Interface ---
st.set_page_config(layout="wide")
st.title("Enterprise Document Q&A Agent 📄")
st.markdown("This agent first answers from indexed documents, then searches Arxiv for related papers.")

check_for_index_files()
encoder, cross_encoder, faiss_index, bm25_index, indexed_docs, llm_model = load_models_and_data()

# --- Initialize the Local Search Tool ---
local_search_tool = LocalSearchTool(encoder, cross_encoder, faiss_index, bm25_index, indexed_docs, llm_model)

st.sidebar.title("Indexed Documents")
indexed_titles = sorted(list(set(doc["metadata"].get("title", "Unknown") for doc in indexed_docs if doc["metadata"]["type"] == "summary")))
for title in indexed_titles:
    st.sidebar.info(f"- {title}")

# --- Arxiv Search Toggle ---
arxiv_enabled = st.sidebar.checkbox("Enable Arxiv Search", value=True, help="Toggle to enable/disable automatic Arxiv searches")

# --- Main Chat Logic with Enhanced Arxiv Search ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Sources Used"):
                for source in message["sources"]:
                     meta = source["metadata"]
                     st.info(f"**{meta.get('type', 'chunk').capitalize()}** from '{meta.get('title', 'N/A')}' (Page {meta.get('page', 'N/A')})")
                     st.caption(source["content"])

if query := st.chat_input("Ask a question about the documents..."):
    # Add user's message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # This is the main response block for the assistant
    with st.chat_message("assistant"):
        full_response = ""
        sources_for_display = []
        
        # --- STAGE 1: Answer from Local Documents ---
        with st.spinner("Thinking... Searching indexed documents..."):
            local_result = local_search_tool.run(query)
            local_answer = local_result["answer"]
            sources_for_display = local_result["sources"]
            
            st.markdown(local_answer)
            full_response += local_answer
            
            # Display the sources expander immediately after the first answer
            if sources_for_display:
                with st.expander("Sources Used for this Answer"):
                    for source in sources_for_display:
                        meta = source["metadata"]
                        st.info(f"**{meta.get('type', 'chunk').capitalize()}** from '{meta.get('title', 'N/A')}' (Page {meta.get('page', 'N/A')})")
                        st.caption(source["content"])

        # --- STAGE 2: Enhanced Arxiv Search ---
        if arxiv_enabled and ARXIV_AVAILABLE:
            with st.spinner("🔍 Searching Arxiv for related research papers..."):
                # Improved query formulation for Arxiv
                arxiv_query = query
                if local_answer and "No relevant information found" not in local_answer and len(local_answer) > 50:
                    # Extract key terms from the local answer to improve Arxiv search
                    important_terms = ' '.join(local_answer.split()[:20])
                    arxiv_query = f"{query} {important_terms}"
                
                # Perform Arxiv search
                arxiv_results = search_arxiv(arxiv_query, max_results=5)
                
                # Format and display results
                arxiv_markdown = format_arxiv_results(arxiv_results)
                st.markdown(arxiv_markdown)
                full_response += "\n\n" + arxiv_markdown
        elif arxiv_enabled and not ARXIV_AVAILABLE:
            st.warning("Arxiv search is enabled, but the library is not available. Please install it with: pip install arxiv")

    # Save the complete, multi-part response to the session state
    st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": sources_for_display})

