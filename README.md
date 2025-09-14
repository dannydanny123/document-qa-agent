Introducing: Document Q&A AI Agent 📄!

An enterprise-ready AI agent prototype that ingests PDF documents, extracts structured content, and answers user queries using LLM APIs (Google Gemini).

🚀 Features:
    📂 Multi-PDF ingestion pipeline (titles, abstracts, sections, tables, Equations, figures/pics).
    🤖 User can Ask questions over extracted content.
    📝 Summarization & evaluation metric extraction.
    ⚡ Enterprise-ready optimizations.
    🔌 Give refrences from Arxiv Database (Extensible to API calls).

⚙️ Setup Instructions
    1️⃣ Clone Repo 
    git clone https://github.com/dannydanny123/document-qa-agent.git
    cd document-qa-agent

    2️⃣ Create Conda Environment (Conda is the best)
    conda env create -f environment.yml
    conda activate docqa

    3️⃣ Install Dependencies
    pip install -r requirements.txt

    ## System-Level Dependencies (Crucial!)
    Some of the Python libraries require external tools to be installed on the operating system. These must be installed before running pip install. like:
    ### 1. Tesseract-OCR
    Required by unstructured for Optical Character Recognition.
    macOS: brew install tesseract
    Windows: Download and run the installer from the Tesseract at UB-Mannheim page.

    ### unstructured
    pip install "unstructured[pdf]" langchain
    also need to install layoutparser and pytesseract for 'unstructured'

    ### Poppler
    Required by unstructured and camelot for PDF rendering.
    macOS: brew install poppler
    Windows: Download the binaries, extract them, and add the bin/ folder to your system's PATH.

    4️⃣ Add API Keys
    Create a .env file in project root:
    GOOGLE_API_KEY=your_key_here

    5️⃣ Run 'agent.py' with command line args mentioning Multiple pdfs path seperated with a space in btw in the terminal of the root dir 'document-qa-agent'
    python agent.py "pdf_path1" "pdf_path2" "pdf_path3" "pdf_path4" "pdf_path_N"
    or if you want to run the code again with the same previously processed pdf, use this command line simply: python agent.py... You dont need to go from stage 1 to 3 for already processed pdfs, only for new ones.

    🖥️ Operations
    I made 'agent.py' as the entry to my app that calls of 3 python scripts Automatically: 'src\tasks\Stage 1\ingestion.py', 'src\tasks\Stage 2\Index_builder.py' and 'src\tasks\Stage 3\app.py'
    🚀(Phase 1 → Examine the given Docs, Extracting features of the pdf and store in Data dir)
    🚀(Phase 2 → A hybrid pipeline: Build FAISS index, Build BM25 index → Indexing Complete Store in data/index)
    🚀 (Phase 3 → Streamlit app for user-friendly interface, Renders an Intutive platform for Agent-User interraction)
    
    🔌Query the Agent in the Streamlit Web page
    "What is the Summary of Paper X?"
    -> Toggle 'Enable Arxiv Search' in the sidebar of the UI for activating Arxiv Search.
    "Summarise the Document, also find more papers related to the given documents !"

SCREENSHOTS: TO DO

🔮 Details on developing this Project
    Project Research - Getting Familier with the buiding of Ai Agents. (Time taken: 2 Days)
    Project setup + CODE ingestion pipeline (Time taken: 2 Days with Crazy Mode on)
    CODE RAG pipeline + Arxiv API call pipeline + CODE Streamlit User Interface (Time Taken: 1 Days)

📹 Demo
    (A demo video .mp4 will be added after final implementation.)

👤 Author
    Daniel Danny Kennedy