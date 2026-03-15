import os
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import pdfplumber
import streamlit as st

from utils.ocr_cleaner import clean_ocr

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

def scrape_website(urls):
    documents = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                print(f"[WEB] Failed {url} — status {resp.status_code}")
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer",
                             "header", "aside", "meta", "noscript"]):
                tag.decompose()
            sections = []
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4']):
                title = heading.get_text(strip=True)
                content_parts = []
                for sibling in heading.find_next_siblings():
                    if sibling.name in ['h1', 'h2', 'h3', 'h4']:
                        break
                    text = sibling.get_text(separator=" ", strip=True)
                    if len(text) > 20:
                        content_parts.append(text)
                if content_parts:
                    sections.append(clean_ocr(f"{title}\n" + "\n".join(content_parts)))
            if not sections:
                paras = [p.get_text(separator=" ", strip=True)
                         for p in soup.find_all('p')
                         if len(p.get_text(strip=True)) > 40]
                sections = [clean_ocr(p) for p in paras]
            combined = "\n\n".join(sections)
            if len(combined.strip()) > 100:
                documents.append(Document(
                    page_content=combined,
                    metadata={"source": url, "page": "web"}
                ))
        except Exception as e:
            print(f"[WEB] Error scraping {url}: {e}")
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

    urls = [
        # ✅ Main program page (confirmed working)
        "https://mastersunion.org/pgp-in-applied-ai-and-agentic-systems",

        # ✅ Apply now page (has admissions + fees info)
        "https://mastersunion.org/pgp-in-applied-ai-and-agentic-systems-applynow",

        # ✅ Main site (awards, rankings, accreditations)
        "https://mastersunion.org/",

        # ✅ Contact page
        "https://mastersunion.org/contact-us",
    ]
    web_docs = scrape_website(urls)

    all_docs = pdf_docs + txt_docs + web_docs
    print(f"[PIPELINE] PDF: {len(pdf_docs)} | TXT: {len(txt_docs)} | Web: {len(web_docs)} | Total: {len(all_docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "]
    )
    chunks = splitter.split_documents(all_docs)
    texts = [c.page_content for c in chunks]
    print(f"[PIPELINE] Total chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db"
    )

    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    # Step 7: Lightweight intent classifier (no model download needed)
    classifier = None
    print("[PIPELINE] Using fast keyword intent classifier")

    # ── LLM toggle: set USE_LOCAL_LLM = True to use Ollama, False for Groq ──
    USE_LOCAL_LLM = True

    if USE_LOCAL_LLM:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model="qwen3:1.7b",
            temperature=0,
            base_url="http://10.221.176.97:11434",
            num_ctx=4096,
            num_predict=1500,  # Covers all answer types; Ollama stops at EOS naturally
            num_thread=8,
            num_batch=512,
            repeat_penalty=1.1,
            stream=True,
        )
    else:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
        )

    return db, bm25, texts, chunks, embeddings, classifier, llm