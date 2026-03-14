import os
import streamlit as st

# Read API key
groq_key = st.secrets.get("GROQ_API_KEY", "") or os.environ.get("GROQ_API_KEY", "")
os.environ["GROQ_API_KEY"] = groq_key

import re
import numpy as np
import requests
import pdfplumber
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
from transformers import pipeline as hf_pipeline

st.set_page_config(page_title="PGP AI Assistant", page_icon="🎓")
st.title("🎓 PGP AI Program Assistant")
st.caption("Hybrid RAG · Intent Routing · Hallucination Guardrails")

# ── OCR Cleaner ───────────────────────────────────────────────────────────────
def clean_ocr(text):
    text = text.replace('\xa0', ' ')
    text = re.sub(r'(?<=[A-Za-z])\s(?=[A-Za-z])', '', text)
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    text = re.sub(r'INR\s*([\d,]+)\s*/\s*-', r'₹\1', text)
    text = re.sub(r'INR\s*([\d,]+)', r'₹\1', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ── PDF Extractor ─────────────────────────────────────────────────────────────
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

# ── Website Scraper ───────────────────────────────────────────────────────────
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
                    metadata={"source": url, "page": url.split("/")[-1]}
                ))
                print(f"[WEB] Scraped {len(combined)} chars from {url}")
        except Exception as e:
            print(f"[WEB] Error scraping {url}: {e}")
    return documents

# ── Assign Title ──────────────────────────────────────────────────────────────
def assign_title(text, source=""):
    t = text.lower()
    src = "🌐 Web" if "http" in str(source) else "📄 PDF"
    if any(w in t for w in ['fee', '₹', 'scholarship', 'emi', 'payment', 'cost', 'installment']):
        return f"{src} · 💰 Fee Structure"
    elif any(w in t for w in ['curriculum', 'term', 'module', 'course', 'syllabus', 'subject']):
        return f"{src} · 📚 Curriculum"
    elif any(w in t for w in ['admission', 'eligibility', 'apply', 'selection', 'interview']):
        return f"{src} · 📋 Admissions"
    elif any(w in t for w in ['career', 'placement', 'hire', 'salary', 'company', 'job']):
        return f"{src} · 🚀 Career Outcomes"
    elif any(w in t for w in ['faculty', 'professor', 'mentor', 'instructor']):
        return f"{src} · 👨‍🏫 Faculty"
    elif any(w in t for w in ['duration', 'month', 'schedule', 'format', 'online', 'offline']):
        return f"{src} · 🗓️ Program Structure"
    else:
        return f"{src} · 📄 Program Information"

# ── Load Pipeline ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PDF_PATH = os.path.join(BASE_DIR, "brochure.pdf")
    pdf_docs = extract_pdf(PDF_PATH)

    urls = [
        # Program pages
        "https://mastersunion.org/programs/pgp-applied-ai-agentic-systems",
        "https://mastersunion.org/programs/pgp-applied-ai-agentic-systems/curriculum",
        "https://mastersunion.org/programs/pgp-applied-ai-agentic-systems/admissions",
        "https://mastersunion.org/programs/pgp-applied-ai-agentic-systems/fees",
        "https://mastersunion.org/programs/pgp-applied-ai-agentic-systems/careers",
        "https://mastersunion.org/programs/pgp-applied-ai-agentic-systems/faculty",
        "https://mastersunion.org/programs/pgp-applied-ai-agentic-systems/placements",
        # Main site pages
        "https://mastersunion.org/faculty",
        "https://mastersunion.org/placements",
    ]
    web_docs = scrape_website(urls)

    all_docs = pdf_docs + web_docs
    print(f"[PIPELINE] PDF: {len(pdf_docs)} | Web: {len(web_docs)} | Total: {len(all_docs)}")

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
    db = Chroma.from_documents(chunks, embeddings)

    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    classifier = hf_pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    return db, bm25, texts, chunks, embeddings, classifier, llm

db, bm25, texts, chunks, embeddings, classifier, llm = load_pipeline()

# ── Session State — initialize BEFORE any usage ───────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Intent Detection ──────────────────────────────────────────────────────────
def detect_intent(query):
    labels = [
        "fees, cost, scholarships and payment",
        "curriculum, syllabus, modules and course topics",
        "admissions, eligibility, how to apply and application process",
        "career outcomes, placements, hiring companies and salary",
        "program overview, duration, format and structure"
    ]
    result = classifier(query, candidate_labels=labels)
    top = result["labels"][0]
    score = result["scores"][0]
    intent_map = {
        "fees, cost, scholarships and payment": "💰 Fees",
        "curriculum, syllabus, modules and course topics": "📚 Curriculum",
        "admissions, eligibility, how to apply and application process": "📋 Admissions",
        "career outcomes, placements, hiring companies and salary": "🚀 Career",
        "program overview, duration, format and structure": "🎓 Overview"
    }
    return intent_map[top], score

# ── Hybrid Retrieval ──────────────────────────────────────────────────────────
def hybrid_retrieve(query, k=5):
    semantic_results = db.similarity_search_with_relevance_scores(query, k=k*2)
    semantic_docs = [r[0].page_content for r in semantic_results]
    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:k*2]
    bm25_docs = [texts[i] for i in bm25_top_idx]
    rrf_scores = {}
    for rank, doc in enumerate(semantic_docs):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (60 + rank)
    for rank, doc in enumerate(bm25_docs):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (60 + rank)
    sorted_docs = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
    return sorted_docs[:k]

# ── Best Sentence ─────────────────────────────────────────────────────────────
def get_best_sentence(query, chunk_text):
    sentences = [
        s.strip() for s in re.split(r'[.\n]', chunk_text)
        if len(s.strip()) > 40
    ]
    if not sentences:
        return chunk_text[:200], 0.0
    query_vec = embeddings.embed_query(query)
    sentence_vecs = embeddings.embed_documents(sentences)
    sims = cosine_similarity([query_vec], sentence_vecs)[0]
    best_idx = int(np.argmax(sims))
    return sentences[best_idx], float(sims[best_idx])

# ── Web Search Fallback ───────────────────────────────────────────────────────
def web_search_fallback(query):
    """
    Search the web in real time when local knowledge base
    doesn't have enough information.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(
                f"Masters Union PGP AI program {query}",
                max_results=4
            ))
        if not results:
            return ""
        combined = ""
        for r in results:
            combined += f"{r.get('title', '')}\n{r.get('body', '')}\n\n"
        return combined.strip()
    except Exception as e:
        print(f"[SEARCH] Web search failed: {e}")
        return ""

# ── Prompt with simple history ────────────────────────────────────────────────
def build_prompt(query, context):
    history_text = ""
    if st.session_state.chat_history:
        history_text = "\n\nCONVERSATION SO FAR:\n"
        for role, content in st.session_state.chat_history[-6:]:
            history_text += f"{role}: {content}\n"

    return f"""You are a helpful assistant for the PGP AI program at Masters' Union.

STRICT RULES:
1. Answer using the context below — which includes brochure, website, and live web search results.
2. Prefer brochure/website info. Use web search results only to fill gaps.
3. Use conversation history to resolve "that", "it", "those" etc.
4. If someone asks about faculty, list names and roles if found in context.
5. If someone asks about companies, list company names if found in context.
6. If truly not found anywhere, say: "I don't have that detail. Please contact admissions at pgadmissions@mastersunion.org"
7. Never invent numbers, dates, or names.
8. Be concise, use bullet points, format fees with ₹ symbol.
{history_text}
CONTEXT (Brochure + Website + Web Search):
{context}

Question: {query}
Answer:"""

# ── Process Query ─────────────────────────────────────────────────────────────
def process_query(query):
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            intent, confidence = detect_intent(query)

            if confidence < 0.45:
                clarification_map = {
                    "💰 Fees":       "fees or scholarships",
                    "📚 Curriculum": "course topics or syllabus",
                    "📋 Admissions": "eligibility or application process",
                    "🚀 Career":     "placements or career outcomes",
                    "🎓 Overview":   "program structure or duration",
                }
                hint = clarification_map.get(intent, "the program")
                st.info(f"Could you clarify — are you asking about **{hint}**? Feel free to rephrase.")

            top_docs = hybrid_retrieve(query, k=5)
            context = "\n\n".join(top_docs)

            # Check if local context is thin — trigger web search
            low_context = len(context.strip()) < 300
            faculty_query = any(w in query.lower() for w in
                ["faculty", "professor", "mentor", "instructor", "teacher"])
            company_query = any(w in query.lower() for w in
                ["compan", "hire", "recruit", "employer", "who hires"])

            web_context = ""
            if low_context or faculty_query or company_query:
                with st.spinner("Searching web for latest info..."):
                    web_context = web_search_fallback(query)
                if web_context:
                    context = context + "\n\nWEB SEARCH RESULTS:\n" + web_context

            prompt = build_prompt(query, context)
            response = llm.invoke(prompt)
            answer = response.content

        st.write(answer)

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"Intent: {intent}")
        with col2:
            st.info(f"Confidence: {confidence:.0%}")

        with st.expander("📄 Sources used to answer"):
            intent_filter = {
                "💰 Fees":       ["fee", "₹", "scholarship", "payment", "cost", "installment"],
                "📚 Curriculum": ["curriculum", "term", "module", "course", "syllabus"],
                "📋 Admissions": ["admission", "eligibility", "apply", "selection"],
                "🚀 Career":     ["career", "placement", "hire", "salary", "company", "job"],
                "🎓 Overview":   ["duration", "month", "program", "format", "online"],
            }
            current_keywords = next(
                (v for k, v in intent_filter.items() if k in intent), []
            )

            high_sources = []
            low_sources = []

            for i, doc in enumerate(top_docs):
                page_num = "?"
                source_url = ""
                for chunk in chunks:
                    if chunk.page_content == doc:
                        page_num = chunk.metadata.get("page", "?")
                        source_url = chunk.metadata.get("source", "")
                        break

                best_sentence, relevance = get_best_sentence(query, doc)
                title = assign_title(doc, source_url)

                if any(kw in doc.lower() for kw in current_keywords):
                    relevance = min(relevance + 0.2, 1.0)

                if relevance >= 0.30:
                    high_sources.append((page_num, title, best_sentence, relevance, source_url))
                else:
                    low_sources.append((page_num, title, source_url))

            high_sources.sort(key=lambda x: x[3], reverse=True)

            for page_num, title, sentence, relevance, source_url in high_sources:
                color = "#1D9E75" if relevance > 0.5 else "#3B8BD4"
                badge = "✅ High" if relevance > 0.5 else "🔵 Medium"
                source_label = f"🌐 {source_url}" if "http" in str(source_url) else f"📄 Page {page_num}"
                st.markdown(f"""
                <div style="border:0.5px solid #e0e0e0;border-left:4px solid {color};
                            border-radius:8px;padding:14px 16px;margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;
                                align-items:center;margin-bottom:8px;">
                        <span style="font-size:13px;font-weight:600;color:{color};">{title}</span>
                        <span style="font-size:11px;color:{color};background:{color}18;
                                     padding:2px 8px;border-radius:10px;">
                            {badge} · {relevance:.0%}
                        </span>
                    </div>
                    <div style="font-size:11px;color:#999;margin-bottom:6px;">{source_label}</div>
                    <div style="font-size:13px;line-height:1.7;color:#333;">{sentence}</div>
                </div>
                """, unsafe_allow_html=True)

            if low_sources:
                st.markdown(
                    "<div style='font-size:12px;color:#999;margin-top:8px;"
                    "margin-bottom:4px;'>📎 Additional References</div>",
                    unsafe_allow_html=True
                )
                for page_num, title, source_url in low_sources:
                    ref = source_url if "http" in str(source_url) else f"Page {page_num}"
                    st.markdown(
                        f"<div style='font-size:12px;color:#888;padding:4px 0;'>"
                        f"· {title} — {ref}</div>",
                        unsafe_allow_html=True
                    )

    # Save to simple chat history — no external dependency
    st.session_state.chat_history.append(("Student", query))
    st.session_state.chat_history.append(("Assistant", answer))
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "intent": f"{intent} ({confidence:.0%})"
    })

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("💡 Try asking...")
    sample_questions = [
        "What is the program fee?",
        "What are the eligibility criteria?",
        "What topics are covered?",
        "What is the program duration?",
        "What companies hire graduates?",
        "Who are the faculty members?",
        "What is the selection process?",
        "What is the average salary after graduation?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.session_state["pending_query"] = q
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("🧠 Powered by")
    st.caption("• Llama 3.3 70B (LLM)")
    st.caption("• MiniLM-L6 (Embeddings)")
    st.caption("• BART-MNLI (Intent ML)")
    st.caption("• BM25 + RRF (Hybrid)")
    st.caption("• pdfplumber (PDF)")
    st.caption("• BeautifulSoup (Web)")

# ── Chat History Display ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "intent" in msg:
            st.caption(f"Intent: {msg['intent']}")

# ── Pending sidebar query ─────────────────────────────────────────────────────
if "pending_query" in st.session_state:
    q = st.session_state.pop("pending_query")
    process_query(q)

# ── Chat Input ────────────────────────────────────────────────────────────────
if query := st.chat_input("Ask about the PGP AI program..."):
    st.session_state.messages.append({"role": "user", "content": query})
    process_query(query)