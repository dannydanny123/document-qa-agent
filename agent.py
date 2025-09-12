import sys
import subprocess
from pathlib import Path
import shutil
import time

# --- Configuration ---
# This script is designed to be run from the project root directory: 'document-qa-agent'
# All paths are relative to this root, matching your specified structure.
SRC_DIR = Path("src")
TASKS_DIR = SRC_DIR / "tasks"
DATA_DIR = SRC_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
# --- NEW: Define the assets directory path for cleaning ---
ASSETS_DIR = DATA_DIR / "assets"

INGESTION_SCRIPT = TASKS_DIR / "Stage 1" / "ingestion.py"
INDEX_BUILDER_SCRIPT = TASKS_DIR / "Stage 2" / "Index_builder.py"
APP_SCRIPT = TASKS_DIR / "Stage 3" / "app.py"

def run_command(command: list, step_name: str):
    """A helper function to run a command and print its status."""
    print(f"\n--- 🚀 Starting Step: {step_name} ---")
    start_time = time.time()
    try:
        # Executes a python script as a separate process
        subprocess.run([sys.executable, *command], check=True, text=True)
        end_time = time.time()
        print(f"--- ✅ Finished Step: {step_name} (Took {end_time - start_time:.2f} seconds) ---")
    except Exception as e:
        print(f"--- ❌ ERROR during Step: {step_name}: {e} ---")
        sys.exit(1)

def main():
    """The main function to orchestrate the entire pipeline."""
    print("--- Document Q&A Agent Pipeline Initializing ---")
    
    # --- New Logic: Check if the user provided PDF paths ---
    pdf_source_paths = [Path(p) for p in sys.argv[1:]]
    
    # --- MODE 1: Process Flow (if PDF paths are provided) ---
    if pdf_source_paths:
        print(f"📁 Process Mode Activated for {len(pdf_source_paths)} PDF(s).")
        
        # Validate all provided paths
        for path in pdf_source_paths:
            if not path.is_file() or path.suffix.lower() != '.pdf':
                print(f"\n❌ ERROR: The provided path '{path}' is not a valid PDF file.")
                sys.exit(1)

        # 1. Clean up ALL previous data for a fresh start. This is the core of the new logic.
        print("\n--- 🧹 Cleaning old data and indexes for a fresh build ---")
        if PROCESSED_DIR.exists():
            shutil.rmtree(PROCESSED_DIR)
        if INDEX_DIR.exists():
            shutil.rmtree(INDEX_DIR)
        # --- CRITICAL FIX: The assets directory is now also cleaned ---
        if ASSETS_DIR.exists():
            shutil.rmtree(ASSETS_DIR)
        
        # Clean any old PDFs from the data directory as well
        for old_pdf in DATA_DIR.glob("*.pdf"):
            old_pdf.unlink()

        # 2. Copy the new PDFs into the project's data directory
        DATA_DIR.mkdir(exist_ok=True)
        print(f"  - Copying {len(pdf_source_paths)} new PDF(s) to '{DATA_DIR}'...")
        for pdf_file in pdf_source_paths:
            shutil.copy(pdf_file, DATA_DIR)
        
        # 3. Run the full pipeline
        run_command([str(INGESTION_SCRIPT)], "PDF Ingestion")
        run_command([str(INDEX_BUILDER_SCRIPT)], "Index Building")
        
    # --- MODE 2: Launch-Only Flow (if no PDF paths are provided) ---
    else:
        print("🚀 Launch-Only Mode Activated. Attempting to start the application...")
        # A crucial check to ensure the app doesn't start without data
        if not INDEX_DIR.exists() or not list(INDEX_DIR.glob("*")):
            print("\n❌ ERROR: No index found. The application cannot start.")
            print("   Please provide paths to some PDFs to process them first.")
            print("   Usage: python agent.py <path_to_your_pdf>")
            sys.exit(1)

    # --- Final Step: Launch the Streamlit Application ---
    print(f"\n--- 🎉 Pipeline Complete! Launching the Q&A Application ---")
    
    try:
        subprocess.run(["streamlit", "run", str(APP_SCRIPT)])
    except Exception as e:
        print(f"\n--- ❌ ERROR launching Streamlit: {e} ---")
        sys.exit(1)

if __name__ == "__main__":
    main()

