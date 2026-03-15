import os
import streamlit as st

# Read API key
groq_key = st.secrets.get("GROQ_API_KEY", "") or os.environ.get("GROQ_API_KEY", "")
os.environ["GROQ_API_KEY"] = groq_key

serper_key = st.secrets.get("SERPER_API_KEY") or os.environ.get("SERPER_API_KEY")
os.environ["SERPER_API_KEY"] = serper_key or ""


import re
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

# Import our modules
from core.pipeline import load_pipeline
from core.retriever import hybrid_retrieve, get_best_sentence
from core.intent import detect_intent
from core.web_search import web_search_fallback
from core.prompt import build_prompt
from utils.ocr_cleaner import clean_ocr
from utils.notifier import notify_admin

st.set_page_config(page_title="PGP AI Assistant", page_icon="🎓")
st.title("🎓 PGP AI Program Assistant")
st.caption("Hybrid RAG · Intent Routing · Hallucination Guardrails")




db, bm25, texts, chunks, embeddings, classifier, llm = load_pipeline()

# ── Session State — initialize BEFORE any usage ───────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
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

            top_docs = hybrid_retrieve(query, db, bm25, texts, embeddings, k=5)
            context = "\n\n".join(top_docs)

            # Check if local context is thin — trigger web search
            low_context = len(context.strip()) < 500
            faculty_query = any(w in query.lower() for w in
                ["faculty", "professor", "mentor", "instructor", "teacher"])
            company_query = any(w in query.lower() for w in
                ["compan", "hire", "recruit", "employer", "who hires"])
            news_query = any(w in query.lower() for w in
                ["award", "news", "recent", "latest", "2024", "2025",
                 "reddit", "quora", "review", "ranking", "rank",
                 "batch", "intake", "size", "accreditat", "recogni"])
            low_confidence = confidence < 0.60

            # Always search web if any trigger fires
            should_search = low_context or faculty_query or company_query or news_query or low_confidence

            print(f"[DEBUG] Query: {query}")
            print(f"[DEBUG] Triggers: low_context={low_context} faculty={faculty_query} "
                  f"company={company_query} news={news_query} low_conf={low_confidence}")
            print(f"[DEBUG] Will search web: {should_search}")

            web_context = ""
            if should_search:
                with st.spinner("🌐 Searching web for latest info..."):
                    web_context = web_search_fallback(query)
                print(f"[DEBUG] Web returned {len(web_context)} chars")
                if web_context:
                    context = context + "\n\nWEB SEARCH RESULTS:\n" + web_context
                    print(f"[DEBUG] Preview: {web_context[:300]}")
                else:
                    print("[DEBUG] Web search returned empty — no web context added")
            else:
                print("[DEBUG] Web search NOT triggered")

            prompt = build_prompt(query, context, st.session_state.chat_history)

            # Streaming response
            placeholder = st.empty()
            full_response = ""
            for chunk in llm.stream(prompt):
                full_response += chunk.content
                placeholder.markdown(full_response + "▌")

        answer = full_response

        # Hallucination Guardrail
        guard_prompt = f"""Does the answer below contain any information not present in the CONTEXT?
Answer only YES or NO.

CONTEXT: {context[:8000]}
ANSWER: {answer}"""

        guard_response = llm.invoke(guard_prompt)
        if guard_response.content.strip().upper() == "YES":
            answer = "⚠️ I couldn't verify this part. " + answer

        # ── Source relevance computation (needed for detection + display) ──────
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

        for doc in top_docs:
            page_num = "?"
            source_url = ""
            for chunk in chunks:
                if chunk.page_content == doc:
                    page_num = chunk.metadata.get("page", "?")
                    source_url = chunk.metadata.get("source", "")
                    break

            best_sentence, relevance = get_best_sentence(query, doc, embeddings)
            title = assign_title(doc, source_url)

            if any(kw in doc.lower() for kw in current_keywords):
                relevance = min(relevance + 0.2, 1.0)

            if relevance >= 0.30:
                high_sources.append((page_num, title, best_sentence, relevance, source_url))
            else:
                low_sources.append((page_num, title, source_url))

        high_sources.sort(key=lambda x: x[3], reverse=True)

        # ── Low-confidence detection & admin notification ─────────────────────
        is_uncertain = any(phrase in answer.lower() for phrase in [
            "i don't have", "not found", "no information", "contact admissions",
            "please contact", "i couldn't find", "not sure", "don't know",
        ])
        context_too_short = len(context.strip()) < 400
        low_relevance = not high_sources or all(r < 0.28 for _, _, _, r, _ in high_sources)

        if is_uncertain or context_too_short or low_relevance:
            notify_admin(query, st.session_state.chat_history, answer)
            answer += (
                "\n\n⚠️ This question isn't fully covered yet — "
                "our team has been notified and will improve the knowledge base soon. Thank you!"
            )

        placeholder.markdown(answer)

        # Feedback buttons
        col_feedback = st.columns(3)
        with col_feedback[0]:
            if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.messages)}"):
                st.success("Thanks for the feedback!")
        with col_feedback[1]:
            if st.button("👎 Not helpful", key=f"not_helpful_{len(st.session_state.messages)}"):
                st.error("Sorry about that. We'll improve!")
        with col_feedback[2]:
            if st.button("🎤 Voice Input", key=f"voice_{len(st.session_state.messages)}"):
                st.info("Voice input feature coming soon!")

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"Intent: {intent}")
        with col2:
            st.info(f"Confidence: {confidence:.0%}")

        with st.expander("📄 Sources used to answer"):
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
    st.caption("• BART-MNLI (Intent)")
    st.caption("• BGE Reranker (Relevance)")
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