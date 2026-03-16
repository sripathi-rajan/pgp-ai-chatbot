"""
scripts/ingest_pdfs.py
----------------------
One-shot runner: extract PDFs → chunk → index into FAISS.

Three-step pipeline
-------------------
1. extract_pdfs_to_raw()   — reads every PDF in mastersunion_files/ with
                              PyMuPDF, cleans text, and saves one JSON file
                              per PDF into data/raw/<category>/.
2. ingest_pdf_data()       — loads those JSON files, splits each page into
                              ~400-word chunks (50-word overlap), and returns
                              LangChain Document objects with rich metadata.
3. index_pdf_chunks_to_faiss() — merges new chunks into the FAISS index,
                                  skipping IDs already recorded in the manifest.

Usage
-----
    # Incremental (skips already-extracted PDFs and already-indexed chunks):
    python scripts/ingest_pdfs.py

    # Force full rebuild (deletes existing index + all JSON intermediates):
    python scripts/ingest_pdfs.py --force

After running, start the app normally:
    streamlit run app.py
"""

import argparse
import shutil
import sys
from pathlib import Path

# ── Make project-root packages importable regardless of CWD ──────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.mastersunion_scraper import extract_pdfs_to_raw   # noqa: E402
from core.pipeline import ingest_pdf_data                  # noqa: E402
from core.retriever import index_pdf_chunks_to_faiss       # noqa: E402

FAISS_INDEX_PATH = str(ROOT / "faiss_index")
FAISS_MANIFEST   = str(ROOT / "faiss_index" / "pdf_manifest.json")
FAISS_SENTINEL   = str(ROOT / "faiss_index" / ".built_by_app")
RAW_DIR          = str(ROOT / "data" / "raw")
PDF_SOURCE_DIR   = str(ROOT / "mastersunion_files")


def _wipe_state():
    """Remove the FAISS index directory and all intermediate JSON files."""
    idx = Path(FAISS_INDEX_PATH)
    if idx.exists():
        shutil.rmtree(idx)
        print(f"[FORCE] Removed FAISS index at {idx}")

    removed_json = 0
    for jf in Path(RAW_DIR).glob("**/*.json"):
        jf.unlink()
        removed_json += 1
    if removed_json:
        print(f"[FORCE] Removed {removed_json} intermediate JSON file(s) from {RAW_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract PDFs and index them into the RAG pipeline."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Wipe the existing FAISS index and all intermediate JSON files, "
            "then rebuild everything from scratch."
        ),
    )
    args = parser.parse_args()

    # ── --force: clear previously indexed data ────────────────────────────────
    if args.force:
        print("=" * 60)
        print("--force: wiping existing state")
        print("=" * 60)
        _wipe_state()
        print()

    # ── Step 1: Extract PDFs → data/raw/<category>/<stem>.json ───────────────
    print("=" * 60)
    print("STEP 1 — Extract PDFs → data/raw/<category>/*.json")
    print("=" * 60)
    extract_stats = extract_pdfs_to_raw(
        source_dir=PDF_SOURCE_DIR,
        output_dir=RAW_DIR,
    )

    # ── Step 2: Load JSON files and create LangChain Document chunks ─────────
    print()
    print("=" * 60)
    print("STEP 2 — Load JSON files and create text chunks")
    print("=" * 60)
    chunks = ingest_pdf_data(RAW_DIR)

    if not chunks:
        print(
            "[INGEST] No chunks produced. "
            "Check that mastersunion_files/ contains PDF files and re-run."
        )
        sys.exit(0)

    # ── Step 3: Index new chunks into FAISS (skips already-indexed) ──────────
    print()
    print("=" * 60)
    print("STEP 3 — Index chunks into FAISS")
    print("=" * 60)

    # Lazy-import here so the rest of the script is usable even without
    # sentence-transformers installed (e.g. for dry-run testing)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        print("[ERROR] langchain-huggingface not installed. Run: pip install langchain-huggingface")
        sys.exit(1)

    print("[EMBED] Loading sentence-transformers/all-MiniLM-L6-v2 ...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    added, skipped_idx = index_pdf_chunks_to_faiss(
        chunks,
        embeddings,
        faiss_index_path=FAISS_INDEX_PATH,
        manifest_path=FAISS_MANIFEST,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  PDFs processed (new)   : {extract_stats.get('processed', 0)}")
    print(f"  PDFs skipped (existing): {extract_stats.get('skipped',   0)}")
    print(f"  PDFs failed            : {extract_stats.get('failed',    0)}")
    print(f"  Chunks created         : {len(chunks)}")
    print(f"  Chunks indexed (new)   : {added}")
    print(f"  Chunks skipped (dup)   : {skipped_idx}")
    print(f"  FAISS index location   : {FAISS_INDEX_PATH}/")
    print(f"  Manifest location      : {FAISS_MANIFEST}")
    print("=" * 60)
    print()
    print("All done! Start the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
