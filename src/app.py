# Standard imports of stuff
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
# --- Configuration ---
load_dotenv()
INDEX_DIR = Path("data/index")
FAISS_PATH = INDEX_DIR / "faiss_index.bin"
BM25_PATH = INDEX_DIR / "bm25_index.pkl"
MAPPING_PATH = INDEX_DIR / "indexed_documents.json"

# --- UPGRADE: Self-Healing Index Check ---
# This check runs before the app starts to ensure the necessary index files exist.
def check_for_index_files():
    if not (FAISS_PATH.exists() and BM25_PATH.exists() and MAPPING_PATH.exists()):
        st.error(
            "Index files not found! 🚨\n"
            "Please run the `build_index.py` script first to process your documents."
        )
        st.code("python src/build_index.py")
        st.stop() # Stop the app from running further.

@st.cache_resource
def load_models_and_data():
    # Loads all necessary models and data from disk, cached for performance.
    print("Loading models and data...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    faiss_index = faiss.read_index(str(FAISS_PATH))
    with open(BM25_PATH, "rb") as f: bm25_index = pickle.load(f)
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f: indexed_docs = json.load(f)
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        llm_model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}")
        llm_model = None
    print("Models and data loaded.")
    return encoder, cross_encoder, faiss_index, bm25_index, indexed_docs, llm_model

# --- Core 'RAG' Agent Logic ---
class QAAgent:
    def __init__(self, encoder, cross_encoder, faiss_index, bm25_index, indexed_docs, llm_model):
        self.encoder = encoder
        self.cross_encoder = cross_encoder
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.indexed_docs = indexed_docs
        self.llm_model = llm_model

    def search(self, query: str, k: int = 30) -> List[Dict]:
        """Performs hybrid search."""
        query_embedding = self.encoder.encode(query)
        _, faiss_indices = self.faiss_index.search(np.array([query_embedding], dtype=np.float32), k)
        
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:k]
        
        combined_indices = list(set(faiss_indices[0]) | set(bm25_indices))
        return [self.indexed_docs[i] for i in combined_indices]

    def rerank(self, query: str, docs: List[Dict], top_n: int = 5) -> List[Dict]:
        """Reranks documents and applies importance weighting."""
        pairs = [[query, doc["content"]] for doc in docs]
        scores = self.cross_encoder.predict(pairs)
        
        # The final score is a combination of the relevance score and the pre-assigned content weight!
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = score
            doc["final_score"] = score * doc["metadata"].get("weight", 1.0)
            
        docs.sort(key=lambda x: x["final_score"], reverse=True)
        return docs[:top_n]

    def answer_question(self, query: str):
        if not self.llm_model:
            return {"answer": "LLM model is not available. Check API key.", "sources": []}

        # 1. Retrieve
        retrieved_docs = self.search(query)
        if not retrieved_docs:
            return {"answer": "No relevant information found.", "sources": []}

        # 2. Rerank
        reranked_docs = self.rerank(query, retrieved_docs)
        
        # 3. Generate itt babyy !
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
            return {"answer": response.text, "sources": reranked_docs}
        except Exception as e:
            st.error(f"Error during generation: {e}")
            return {"answer": "Error generating answer.", "sources": reranked_docs}

# --- Streamlit User Interface ---
st.set_page_config(layout="wide")
st.title("Enterprise Document Q&A Agent 📄")
st.markdown("This agent uses a smart-indexed, Retrieve-Rerank-Generate pipeline to answer questions.")

# Run the self-healing check first
check_for_index_files()

# Load all models and data
encoder, cross_encoder, faiss_index, bm25_index, indexed_docs, llm_model = load_models_and_data()

# Initialize the agent
agent = QAAgent(encoder, cross_encoder, faiss_index, bm25_index, indexed_docs, llm_model)

st.sidebar.title("Indexed Documents")
indexed_titles = sorted(list(set(doc["metadata"].get("title", "Unknown") for doc in indexed_docs if doc["metadata"]["type"] == "summary")))
for title in indexed_titles:
    st.sidebar.info(f"- {title}")

# User input querey
query = st.text_input("Ask a question about the indexed documents:", placeholder="e.g., What is the Hubble tension?")

if query:
    with st.spinner("Thinking..."):
        result = agent.answer_question(query)
        st.success("Answer generated.")
        
        st.markdown("Ai Agent's Answer")
        st.write(result["answer"])
        
        st.markdown("---")
        st.markdown("### Sources Used from the given documents:")
        if result["sources"]:
            for source in result["sources"]:
                meta = source["metadata"]
                with st.expander(f"**{meta.get('type', 'chunk').capitalize()}** from '{meta.get('title', 'N/A')}' (Page {meta.get('page', 'N/A')}) - Score: {source.get('final_score', 0):.2f}"):
                    st.write(source["content"])