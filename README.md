📄 Document Q&A AI Agent

An enterprise-ready AI agent prototype that ingests PDF documents, extracts structured content, and answers user queries using LLM APIs (Google Gemini / OpenAI).

🚀 Features:
    📂 Multi-PDF ingestion pipeline (titles, abstracts, sections, tables).
    🤖 Query answering over extracted content.
    📝 Summarization & evaluation metric extraction.
    ⚡ Enterprise-ready optimizations (Modular design, version control).
    🔌 Extensible to API calls (e.g., Arxiv search).

⚙️ Setup Instructions
    1️⃣ Clone Repo
    git clone https://github.com/<your-username>/document-qa-agent.git
    cd document-qa-agent

    2️⃣ Create Conda Environment
    conda env create -f environment.yml
    conda activate doc-qa-env

    3️⃣ Install Dependencies
    pip install -r requirements.txt

    ## System-Level Dependencies (Crucial!)
    Some of the Python libraries require external tools to be installed on the operating system. These must be installed before running pip install.

    ### Tesseract-OCR
    Required by unstructured for Optical Character Recognition.

    macOS: brew install tesseract

    Debian/Ubuntu: sudo apt-get install tesseract-ocr

    Windows: Download and run the installer from the Tesseract at UB-Mannheim page.

    ### Poppler
    Required by unstructured and camelot for PDF rendering.

    macOS: brew install poppler

    Debian/Ubuntu: sudo apt-get install poppler-utils

    Windows: Download the binaries, extract them, and add the bin/ folder to your system's PATH.

    This complete list will ensure that anyone, including the evaluator, can set up and run your project smoothly.

    4️⃣ Add API Keys
    Create a .env file in project root:

    GEMINI_API_KEY=your_key_here
    OPENAI_API_KEY=your_key_here   # optional

    5️⃣ Run Basic Test
    pytest

    🖥️ Usage
    Run ingestion pipeline
    python src/ingest.py data/sample.pdf

    Query the Agent
    python src/agent.py "What is the conclusion of Paper X?"


    (Phase 2 → Streamlit app for user-friendly interface)

✅ CI/CD
    Every push triggers:
    Dependency install
    Pytest unit tests
    Build verification
    Configured via GitHub Actions in .github/workflows/ci.yml.

🔮 Roadmap
    Project setup (Phase 1)
    Document ingestion pipeline (Phase 2)
    Q&A agent integration (Phase 3)
    Summarization + evaluation metric extraction (Phase 4)
    also, Bonus: Arxiv API functional calling

📹 Demo
    (A demo video .mp4 will be added after final implementation.)

👤 Author
    Danny Kennedy