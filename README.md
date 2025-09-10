📄 Document Q&A AI Agent

An enterprise-ready AI agent prototype that ingests PDF documents, extracts structured content, and answers user queries using LLM APIs (Google Gemini / OpenAI).

🚀 Features:
    📂 Multi-PDF ingestion pipeline (titles, abstracts, sections, tables).
    🤖 Query answering over extracted content.
    📝 Summarization & evaluation metric extraction.
    ⚡ Enterprise-ready optimizations (CI/CD, modular design, version control).
    🔌 Extensible to API calls (e.g., Arxiv search).

📦 Project Structure:
    document-qa-agent/
    │── data/                 # Sample PDFs
    │── src/                  # Source code
    │   ├── ingest.py         # PDF ingestion & parsing
    │   ├── agent.py          # Q&A agent logic
    │   └── interface.py      # CLI / Streamlit interface
    │── tests/                # Unit tests
    │   └── test_sanity.py
    │── environment.yml       # Conda environment setup
    │── requirements.txt      # Pip dependencies
    │── .gitignore
    │── .github/
    │   └── workflows/
    │       └── ci.yml        # GitHub Actions (CI pipeline)
    │── README.md             # Project documentation

⚙️ Setup Instructions
    1️⃣ Clone Repo
    git clone https://github.com/<your-username>/document-qa-agent.git
    cd document-qa-agent

    2️⃣ Create Conda Environment
    conda env create -f environment.yml
    conda activate doc-qa-env

    3️⃣ Install Dependencies
    pip install -r requirements.txt

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