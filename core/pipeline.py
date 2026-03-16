import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import pdfplumber
import streamlit as st

from utils.ocr_cleaner import clean_ocr


@dataclass
class Pipeline:
    db: Any
    bm25: Any
    texts: list
    chunks: list
    embeddings: Any
    llm: Any


def _get_content_type(filename: str) -> str:
    """Infer a content_type tag from the filename for better retrieval filtering."""
    f = filename.lower()
    if "placement" in f:
        return "placement"
    if "brochure" in f:
        return "brochure"
    if "curriculum" in f:
        return "curriculum"
    if "fee" in f or "admission" in f:
        return "admissions"
    if "immersion" in f or "gip" in f:
        return "immersion"
    return "general"


def clean_chunk_text(text: str) -> str:
    """Fix common PDF extraction issues: missing spaces, garbled runs, broken hyphens."""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)   # camelCase → camel Case
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)       # broken hy- phens
    text = re.sub(r'\s+', ' ', text)                    # collapse whitespace
    return text.strip()

def extract_pdf(pdf_path):
    documents = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                table_text = ""
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        clean_row = [str(cell).strip() for cell in row if cell and str(cell).strip()]
                        if clean_row:
                            table_text += " | ".join(clean_row) + "\n"
                full_text = clean_ocr(text + "\n" + table_text)
                if len(full_text.strip()) > 50:
                    documents.append(Document(
                        page_content=full_text,
                        metadata={"source": "brochure.pdf", "page": page_num + 1}
                    ))
        print(f"[PDF] Extracted {len(documents)} pages")
    except Exception as e:
        print(f"[PDF] Error: {e}")
    return documents

def load_scraped_data(raw_dir="data/raw"):
    documents = []
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        print("[SCRAPER] data/raw/ not found — run: python src/mastersunion_scraper.py")
        return documents

    combined_files = list(raw_path.glob("*_full.txt"))
    if not combined_files:
        combined_files = list(raw_path.glob("**/*.txt"))

    for file in combined_files:
        try:
            text = file.read_text(encoding="utf-8")
            if len(text.strip()) < 100:
                continue
            first_line = text.split("\n")[0]
            course_name = first_line.replace("COURSE:", "").strip() if "COURSE:" in first_line else file.stem
            documents.append(Document(
                page_content=text,
                metadata={
                    "source":       str(file),
                    "course":       course_name,
                    "page":         file.stem,
                    "content_type": _get_content_type(file.name),
                }
            ))
            print(f"[SCRAPER] Loaded {file.name} ({len(text):,} chars)")
        except Exception as e:
            print(f"[SCRAPER] Error loading {file}: {e}")

    print(f"[SCRAPER] Total: {len(documents)} documents loaded")
    return documents


def ingest_pdf_data(raw_dir: str = "data/raw") -> list:
    """
    Load all JSON files produced by extract_pdfs_to_raw() from data/raw/,
    split each page's text into ~400-word chunks (50-word overlap), and
    return a list of LangChain Document objects with rich metadata.

    Metadata keys on every chunk:
        source   — original PDF filename  (e.g. "pgp_brochure_2024.pdf")
        category — pgp / ug / executive / general
        page     — 1-based page number the chunk came from
        type     — "pdf"  (allows callers to distinguish PDF chunks from scraped text)
        chunk_id — unique stable ID: pdf_<stem>_pg<page>_<idx>
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        print("[PDF-INGEST] data/raw/ not found — run scripts/ingest_pdfs.py first")
        return []

    # ~600 words ≈ 3 600 chars; 100-word overlap ≈ 600 chars
    pdf_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3600,
        chunk_overlap=600,
        separators=["\n\n", "\n", ". ", " "],
    )

    documents = []
    json_files = list(raw_path.glob("**/*.json"))
    print(f"[PDF-INGEST] Found {len(json_files)} JSON file(s) in '{raw_path}'")

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[PDF-INGEST] Failed to load {jf}: {exc}")
            continue

        filename = data.get("filename", jf.name)
        stem     = data.get("stem", jf.stem)
        category = data.get("category", "general")

        for page in data.get("pages", []):
            page_num  = page.get("page_num", 0)
            page_text = page.get("text", "")
            if not page_text.strip():
                continue

            # Split this page into overlapping word-bounded chunks
            page_chunks = pdf_splitter.split_text(page_text)
            for chunk_idx, chunk_text in enumerate(page_chunks):
                cleaned = clean_chunk_text(chunk_text)
                if len(cleaned.strip()) < 50:   # skip near-empty fragments
                    continue

                # Stable, unique ID used for dedup during incremental indexing
                chunk_id = f"pdf_{stem}_pg{page_num}_{chunk_idx}"

                documents.append(Document(
                    page_content=cleaned,
                    metadata={
                        "source":       filename,
                        "category":     category,
                        "page":         page_num,
                        "type":         "pdf",
                        "chunk_id":     chunk_id,
                        "content_type": _get_content_type(filename),
                    },
                ))

    print(f"[PDF-INGEST] Created {len(documents)} chunk(s) from PDF JSON files")
    return documents


@st.cache_resource
def load_pipeline():
    # Step 1: Load PDF brochure
    pdf_docs = extract_pdf("data/brochure.pdf")

    # Step 2: Load TXT documentation
    txt_docs = []
    try:
        with open("data/program_data.txt", "r", encoding="utf-8") as f:
            txt_content = f.read()
        txt_docs.append(Document(
            page_content=txt_content,
            metadata={"source": "program_data.txt", "page": "documentation"}
        ))
        print(f"[TXT] Loaded program_data.txt ({len(txt_content)} chars)")
    except Exception as e:
        print(f"[TXT] Error: {e}")

    # Step 3: Scraped data (pre-generated by src/mastersunion_scraper.py)
    scraped_docs = load_scraped_data("data/raw")

    # Step 4: PDF chunks extracted by scripts/ingest_pdfs.py → data/raw/**/*.json
    pdf_json_docs = ingest_pdf_data("data/raw")

    all_docs = pdf_docs + txt_docs + scraped_docs + pdf_json_docs
    print(
        f"[PIPELINE] PDF:{len(pdf_docs)} TXT:{len(txt_docs)} "
        f"Scraped:{len(scraped_docs)} PDFJson:{len(pdf_json_docs)} Total:{len(all_docs)}"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3600,
        chunk_overlap=600,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(all_docs)
    for c in chunks:
        c.page_content = clean_chunk_text(c.page_content)
    texts = [c.page_content for c in chunks]
    print(f"[PIPELINE] Total chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    FAISS_INDEX_PATH = "./faiss_index"

    # Rebuild index if any source data is newer than the saved index
    if os.path.exists(FAISS_INDEX_PATH):
        index_mtime = os.path.getmtime(FAISS_INDEX_PATH)
        data_files  = ["data/program_data.txt", "data/brochure.pdf"]
        latest_data = max(
            (os.path.getmtime(f) for f in data_files if os.path.exists(f)),
            default=0,
        )
        if latest_data > index_mtime:
            shutil.rmtree(FAISS_INDEX_PATH)
            print("[FAISS] Source data newer than index — rebuilding")

    FAISS_SENTINEL = os.path.join(FAISS_INDEX_PATH, ".built_by_app")

    if os.path.exists(FAISS_INDEX_PATH):
        if not os.path.exists(FAISS_SENTINEL):
            # Index directory exists but was not created by this app — rebuild to be safe
            shutil.rmtree(FAISS_INDEX_PATH)
            print("[FAISS] Sentinel missing — untrusted index removed, rebuilding")

    if os.path.exists(FAISS_INDEX_PATH):
        # allow_dangerous_deserialization is required by FAISS (uses pickle).
        # The sentinel file above ensures this index was written by this application
        # and has not been replaced by an external actor.
        db = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("[FAISS] Loaded existing index")
    else:
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(FAISS_INDEX_PATH)
        # Write sentinel to mark this index as trusted
        Path(FAISS_SENTINEL).touch()
        print("[FAISS] Built and saved new index")

    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    print("[PIPELINE] Using fast keyword intent classifier")

    # ── LLM toggle: read from st.secrets first, fall back to env ─────────────
    USE_LOCAL_LLM = (
        st.secrets.get("USE_LOCAL_LLM", os.environ.get("USE_LOCAL_LLM", "false"))
    ).lower() == "true"

    if USE_LOCAL_LLM:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model="qwen3:1.7b",
            temperature=0,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            num_ctx=4096,
            num_predict=2048,
            num_thread=8,
            num_batch=512,
            repeat_penalty=1.1,
            stream=True,
            think=False,  # Disable chain-of-thought — avoids empty answers when token budget runs out
        )
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        )

    return Pipeline(db=db, bm25=bm25, texts=texts, chunks=chunks, embeddings=embeddings, llm=llm)


def load_pipeline_flask():
    """
    Streamlit-free version of load_pipeline() for use with Flask (python app.py).
    Reads all config from environment variables — no st.secrets, no st.cache_resource.
    """
    # Step 1: Load PDF brochure
    pdf_docs = extract_pdf("data/brochure.pdf")

    # Step 2: Load TXT documentation
    txt_docs = []
    try:
        with open("data/program_data.txt", "r", encoding="utf-8") as f:
            txt_content = f.read()
        txt_docs.append(Document(
            page_content=txt_content,
            metadata={"source": "program_data.txt", "page": "documentation"}
        ))
        print(f"[TXT] Loaded program_data.txt ({len(txt_content)} chars)")
    except Exception as e:
        print(f"[TXT] Error: {e}")

    # Step 3: Scraped data (pre-generated by src/mastersunion_scraper.py)
    scraped_docs = load_scraped_data("data/raw")

    # Step 4: PDF chunks extracted by scripts/ingest_pdfs.py → data/raw/**/*.json
    pdf_json_docs = ingest_pdf_data("data/raw")

    all_docs = pdf_docs + txt_docs + scraped_docs + pdf_json_docs
    print(
        f"[PIPELINE] PDF:{len(pdf_docs)} TXT:{len(txt_docs)} "
        f"Scraped:{len(scraped_docs)} PDFJson:{len(pdf_json_docs)} Total:{len(all_docs)}"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3600,
        chunk_overlap=600,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(all_docs)
    for c in chunks:
        c.page_content = clean_chunk_text(c.page_content)
    texts = [c.page_content for c in chunks]
    print(f"[PIPELINE] Total chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    FAISS_INDEX_PATH = "./faiss_index"

    # Rebuild index if any source data is newer than the saved index
    if os.path.exists(FAISS_INDEX_PATH):
        index_mtime = os.path.getmtime(FAISS_INDEX_PATH)
        data_files  = ["data/program_data.txt", "data/brochure.pdf"]
        latest_data = max(
            (os.path.getmtime(f) for f in data_files if os.path.exists(f)),
            default=0,
        )
        if latest_data > index_mtime:
            shutil.rmtree(FAISS_INDEX_PATH)
            print("[FAISS] Source data newer than index — rebuilding")

    FAISS_SENTINEL = os.path.join(FAISS_INDEX_PATH, ".built_by_app")

    if os.path.exists(FAISS_INDEX_PATH):
        if not os.path.exists(FAISS_SENTINEL):
            shutil.rmtree(FAISS_INDEX_PATH)
            print("[FAISS] Sentinel missing — untrusted index removed, rebuilding")

    if os.path.exists(FAISS_INDEX_PATH):
        db = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("[FAISS] Loaded existing index")
    else:
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(FAISS_INDEX_PATH)
        Path(FAISS_SENTINEL).touch()
        print("[FAISS] Built and saved new index")

    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    print("[PIPELINE] Using fast keyword intent classifier")

    # ── LLM selection — priority: Groq > OpenAI > Local Ollama ──────────────
    USE_LOCAL_LLM = os.environ.get("USE_LOCAL_LLM", "false").lower() == "true"
    groq_key      = os.environ.get("GROQ_API_KEY", "")
    openai_key    = os.environ.get("OPENAI_API_KEY", "")

    if groq_key:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=groq_key,
            temperature=0,
        )
        print("[LLM] Using Groq llama-3.1-8b-instant")
    elif USE_LOCAL_LLM:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model="qwen3:1.7b",
            temperature=0,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            num_ctx=4096,
            num_predict=2048,
            num_thread=8,
            num_batch=512,
            repeat_penalty=1.1,
            stream=True,
            think=False,  # Disable chain-of-thought — avoids empty answers when token budget runs out
        )
        print("[LLM] Using local Qwen3 1.7B via Ollama")
    else:
        from langchain_openai import ChatOpenAI
        if not openai_key:
            print("[WARN] OPENAI_API_KEY not set — LLM calls will fail")
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_key,
        )
        print("[LLM] Using OpenAI gpt-4o-mini")

    return Pipeline(db=db, bm25=bm25, texts=texts, chunks=chunks, embeddings=embeddings, llm=llm)