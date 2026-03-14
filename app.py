import os

import re
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferWindowMemory
from rank_bm25 import BM25Okapi
from transformers import pipeline as hf_pipeline

st.set_page_config(page_title="PGP AI Assistant", page_icon="🎓")
st.title("🎓 PGP AI Program Assistant")
st.caption("Hybrid RAG · Intent Routing · Hallucination Guardrails")

# ── OCR Text Cleaner ──────────────────────────────────────────────────────────
def clean_ocr(text):
    text = text.replace('\xa0', ' ')
    text = re.sub(r'(?<=[A-Za-z])\s(?=[A-Za-z])', '', text)
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    text = re.sub(r'\s*/\s*-', '/-', text)
    # Convert INR amounts to ₹ symbol
    text = re.sub(r'INR\s*([\d,]+)\s*/\s*-', r'₹\1', text)
    text = re.sub(r'INR\s*([\d,]+)', r'₹\1', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ── Assign Contextual Title to a chunk ───────────────────────────────────────
def assign_title(text):
    t = text.lower()
    if any(w in t for w in ['fee', 'inr', '₹', 'scholarship', 'emi', 'payment', 'cost']):
        return "💰 Fee Structure"
    elif any(w in t for w in ['curriculum', 'term', 'module', 'course', 'syllabus', 'subject']):
        return "📚 Curriculum"
    elif any(w in t for w in ['admission', 'eligibility', 'apply', 'selection', 'step', 'interview']):
        return "📋 Admissions & Eligibility"
    elif any(w in t for w in ['career', 'placement', 'hire', 'salary', 'company', 'job', 'outcome']):
        return "🚀 Career Outcomes"
    elif any(w in t for w in ['faculty', 'professor', 'mentor', 'instructor']):
        return "👨‍🏫 Faculty & Mentors"
    elif any(w in t for w in ['duration', 'month', 'term', 'schedule', 'format', 'online', 'offline']):
        return "🗓️ Program Structure"
    elif any(w in t for w in ['ai', 'machine learning', 'deep learning', 'llm', 'agent']):
        return "🤖 AI & Technology"
    else:
        return "📄 Program Information"

# ── Load Pipeline ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    loader = PyPDFLoader("brochure.pdf")
    documents = loader.load()
    for doc in documents:
        doc.page_content = clean_ocr(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    texts = [c.page_content for c in chunks]

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

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=st.secrets["GROQ_API_KEY"])

    return db, bm25, texts, chunks, embeddings, classifier, llm

db, bm25, texts, chunks, embeddings, classifier, llm = load_pipeline()

# ── Intent Detection ──────────────────────────────────────────────────────────
def detect_intent(query):
    labels = [
        "fees and cost",
        "curriculum and syllabus",
        "admissions and eligibility",
        "career outcomes and placements",
        "program overview and duration"
    ]
    result = classifier(query, candidate_labels=labels)
    top = result["labels"][0]
    score = result["scores"][0]
    intent_map = {
        "fees and cost": "💰 Fees",
        "curriculum and syllabus": "📚 Curriculum",
        "admissions and eligibility": "📋 Admissions",
        "career outcomes and placements": "🚀 Career",
        "program overview and duration": "🎓 Overview"
    }
    return intent_map[top], score

# ── Hybrid Retrieval ──────────────────────────────────────────────────────────
def hybrid_retrieve(query, k=4):
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

# ── Best Sentence via Cosine Similarity ───────────────────────────────────────
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

# ── Prompt ────────────────────────────────────────────────────────────────────
def build_prompt(query, context):
    history = st.session_state.memory.load_memory_variables({})
    history_text = ""

    if history.get("chat_history"):
        history_text = "\n\nCONVERSATION SO FAR:\n"
        for msg in history["chat_history"]:
            if hasattr(msg, "type"):
                role = "Student" if msg.type == "human" else "Assistant"
                history_text += f"{role}: {msg.content}\n"

    return f"""You are a helpful assistant for the PGP AI program at Masters' Union.

STRICT RULES:
1. Answer ONLY using the context and conversation history below.
2. Use conversation history to resolve "that", "it", "those", "the fee" etc.
3. If the context contains payment installment info, treat that as fee/payment info.
4. If someone asks about EMI or installments, look for payment schedule info in context.
5. If truly not in context, say: "I don't have that specific detail. Please contact the admissions team at Masters' Union."
6. Never invent numbers, dates, or names.
7. Be concise and use bullet points.
{history_text}
CONTEXT FROM BROCHURE:
{context}

Current Question: {query}
Answer:"""

# ── Process a query and display response ─────────────────────────────────────
# FIX 1: Extracted into a function so both chat input
#         AND sidebar buttons call the exact same logic
def process_query(query):
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            intent, confidence = detect_intent(query)
            # If intent confidence is low, ask for clarification
            if confidence < 0.50:
                clarification_map = {
                    "💰 Fees":       "fees or scholarships",
                    "📚 Curriculum": "course topics or syllabus",
                    "📋 Admissions": "eligibility or application process",
                    "🚀 Career":     "placements or career outcomes",
                    "🎓 Overview":   "program structure or duration",
                }
                hint = clarification_map.get(intent, "the program")
                st.info(
                    f"I want to make sure I answer correctly — "
                    f"are you asking about **{hint}**? "
                    f"Feel free to rephrase and I'll give you a more precise answer."
                )
                return
            top_docs = hybrid_retrieve(query, k=4)
            context = "\n\n".join(top_docs)
            prompt = build_prompt(query, context)
            response = llm.invoke(prompt)
            answer = response.content

        st.write(answer)

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"Intent: {intent}")
        with col2:
            st.info(f"Confidence: {confidence:.0%}")

        # ── FIX 2: Smart Source Cards ─────────────────────────────────────────
        with st.expander("📄 Sources used to answer"):

            high_sources = []
            low_sources = []

            # Intent → expected title keywords for filtering
            intent_filter = {
                "💰 Fees":        ["fee", "₹", "inr", "scholarship", "payment", "cost", "installment"],
                "📚 Curriculum":  ["curriculum", "term", "module", "course", "syllabus"],
                "📋 Admissions":  ["admission", "eligibility", "apply", "selection", "step"],
                "🚀 Career":      ["career", "placement", "hire", "salary", "company", "job"],
                "🎓 Overview":    ["duration", "month", "program", "format", "online", "offline"],
            }

            # Get keywords for current intent
            current_keywords = []
            for key, keywords in intent_filter.items():
                if key in intent:
                    current_keywords = keywords
                    break

            for i, doc in enumerate(top_docs):
                # Page number
                page_num = "?"
                for chunk in chunks:
                    if chunk.page_content == doc:
                        page_num = chunk.metadata.get("page", 0) + 1
                        break

                best_sentence, relevance = get_best_sentence(query, doc)
                title = assign_title(doc)

                # Boost relevance if doc matches intent keywords
                doc_lower = doc.lower()
                intent_match = any(kw in doc_lower for kw in current_keywords)
                if intent_match:
                    relevance = min(relevance + 0.2, 1.0)  # boost matched sources

                if relevance >= 0.30:
                    high_sources.append((page_num, title, best_sentence, relevance, intent_match))
                else:
                    low_sources.append((page_num, title))

            # Sort — intent-matched sources first
            high_sources.sort(key=lambda x: (x[4], x[3]), reverse=True)

            # Primary source cards
            for page_num, title, sentence, relevance, intent_match in high_sources:
                if relevance > 0.5:
                    color = "#1D9E75"
                    badge = "✅ High"
                else:
                    color = "#3B8BD4"
                    badge = "🔵 Medium"

                # Extra highlight if intent matched
                border_style = "4px" if intent_match else "2px"

                st.markdown(f"""
                <div style="
                    border: 0.5px solid #e0e0e0;
                    border-left: {border_style} solid {color};
                    border-radius: 8px;
                    padding: 14px 16px;
                    margin-bottom: 10px;
                ">
                    <div style="display:flex;justify-content:space-between;
                                align-items:center;margin-bottom:8px;">
                        <span style="font-size:13px;font-weight:600;color:{color};">
                            {title}
                        </span>
                        <span style="font-size:11px;color:{color};
                                     background:{color}18;padding:2px 8px;
                                     border-radius:10px;">
                            {badge} · {relevance:.0%}
                        </span>
                    </div>
                    <div style="font-size:11px;color:#999;margin-bottom:6px;">
                        Page {page_num}
                    </div>
                    <div style="font-size:13px;line-height:1.7;color:#333;">
                        {sentence}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Additional references
            if low_sources:
                st.markdown(
                    "<div style='font-size:12px;color:#999;"
                    "margin-top:8px;margin-bottom:4px;'>"
                    "📎 Additional References</div>",
                    unsafe_allow_html=True
                )
                for page_num, title in low_sources:
                    st.markdown(
                        f"<div style='font-size:12px;color:#888;padding:4px 0;'>"
                        f"· {title} — Page {page_num}</div>",
                        unsafe_allow_html=True
                    )

    # Save this exchange to memory
    st.session_state.memory.save_context(
        {"input": query},
        {"output": answer}
    )

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
        "Is there an EMI option?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            # FIX 1: append to messages AND set a flag to process immediately
            st.session_state.messages.append({"role": "user", "content": q})
            st.session_state["pending_query"] = q
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()
    st.caption("🧠 Powered by")
    st.caption("• Llama 3.3 70B (LLM)")
    st.caption("• MiniLM-L6 (Embeddings)")
    st.caption("• BART-MNLI (Intent ML)")
    st.caption("• BM25 + RRF (Hybrid Retrieval)")

# ── Chat History ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=5,  # remembers last 5 exchanges
        return_messages=True,
        memory_key="chat_history"
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "intent" in msg:
            st.caption(f"Intent: {msg['intent']}")

# ── FIX 1: Process pending query from sidebar ─────────────────────────────────
if "pending_query" in st.session_state:
    q = st.session_state.pop("pending_query")
    process_query(q)

# ── Chat Input ────────────────────────────────────────────────────────────────
if query := st.chat_input("Ask about the PGP AI program..."):
    st.session_state.messages.append({"role": "user", "content": query})
    process_query(query)