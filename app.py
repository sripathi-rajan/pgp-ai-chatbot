import os
import re
import streamlit as st

# Read API keys
groq_key = st.secrets.get("GROQ_API_KEY", "") or os.environ.get("GROQ_API_KEY", "")
os.environ["GROQ_API_KEY"] = groq_key

serper_key = st.secrets.get("SERPER_API_KEY") or os.environ.get("SERPER_API_KEY")
os.environ["SERPER_API_KEY"] = serper_key or ""

# Import our modules
from core.pipeline import load_pipeline
from core.retriever import hybrid_retrieve, get_best_sentence
from core.intent import detect_intent
from core.web_search import web_search_fallback
from core.prompt import build_prompt
from utils.notifier import notify_admin

import streamlit.components.v1 as components

st.set_page_config(page_title="PGP AI Assistant", page_icon="🎓")
st.title("🎓 PGP AI Program Assistant")
st.caption("Hybrid RAG · Intent Routing · Hallucination Guardrails")




@st.cache_resource
def load_cached_pipeline():
    return load_pipeline()

db, bm25, texts, chunks, embeddings, classifier, llm = load_cached_pipeline()

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
    st.session_state.messages.append({"role": "user", "content": query})
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

            long_answer_triggers = [
                "topics", "covered", "curriculum", "modules", "subjects", "terms", "syllabus",
                "fee", "cost", "payment", "schedule", "installment", "scholarship",
                "eligibility", "criteria", "process", "selection",
                "companies", "career", "placement", "faculty", "who are", "what all",
            ]
            long_query = any(w in query.lower() for w in long_answer_triggers)
            top_docs = hybrid_retrieve(query, db, bm25, texts, embeddings, k=3)
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
            # Always search web if any trigger fires (low_confidence removed — pollutes local answers)
            should_search = low_context or faculty_query or company_query or news_query

            print(f"[DEBUG] Query: {query}")
            print(f"[DEBUG] Triggers: low_context={low_context} faculty={faculty_query} "
                  f"company={company_query} news={news_query}")
            print(f"[DEBUG] Will search web: {should_search}")

            web_context = ""
            if should_search:
                with st.spinner("🌐 Searching web for latest info..."):
                    web_context = web_search_fallback(query)
                print(f"[DEBUG] Web returned {len(web_context)} chars")
                if web_context:
                    context = context + "\n\nWEB SEARCH RESULTS:\n" + web_context[:1500]
                    print(f"[DEBUG] Preview: {web_context[:300]}")
                else:
                    print("[DEBUG] Web search returned empty — no web context added")
            else:
                print("[DEBUG] Web search NOT triggered")

            prompt = build_prompt(query, context, st.session_state.chat_history)

            if long_query:
                prompt += "\n\nIMPORTANT: Provide the COMPLETE answer. Do not truncate, abbreviate, or add '...' — write out every item in full."

            # Streaming response
            placeholder = st.empty()
            full_response = ""
            try:
                for chunk in llm.stream(prompt):
                    # chunk.content is a str — avoid leaking repr of AIMessageChunk
                    token = chunk.content if isinstance(chunk.content, str) else ""
                    if token:
                        full_response += token
                        display = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
                        placeholder.markdown(display + "▌")
            except Exception as e:
                st.error(f"LLM stream error: {e}")
                return

        # Final answer with think tags removed
        answer = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()

        if not answer:
            st.error("No response received from the model.")
            return

        # Hallucination Guardrail — phrase-based (no second LLM call)
        HALLUCINATION_PHRASES = [
            "as of my knowledge cutoff",
            "i don't have access",
            "i cannot confirm",
            "not in my training",
            "i made up",
            "i fabricated",
        ]
        if any(p in answer.lower() for p in HALLUCINATION_PHRASES):
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
        col_feedback = st.columns(2)
        with col_feedback[0]:
            if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.messages)}"):
                st.success("Thanks for the feedback!")
        with col_feedback[1]:
            if st.button("👎 Not helpful", key=f"not_helpful_{len(st.session_state.messages)}"):
                st.error("Sorry about that. We'll improve!")

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
            st.session_state["pending_query"] = q
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption("🧠 Powered by")
    st.caption("• Qwen3 1.7B / Llama 3.3 70B (LLM)")
    st.caption("• Whisper (Voice)")
    st.caption("• MiniLM-L6 (Embeddings)")
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

# ── Chat bar styling + inline voice button injection ──────────────────────────
components.html("""
<script>
(function() {
  const pdoc = window.parent.document;
  if (pdoc.getElementById('pgp-voice-btn')) return;

  // ── Chat bar CSS: single flex row, no extra wrappers breaking alignment ─────
  const style = pdoc.createElement('style');
  style.textContent = `
    /* Outer container → flex row */
    [data-testid="stChatInput"] > div {
      display: flex !important;
      align-items: center !important;
      width: 100% !important;
      padding: 8px 12px !important;
      border-radius: 12px !important;
      gap: 8px !important;
      box-sizing: border-box !important;
    }

    /* Textarea: flex: 1 to fill available width */
    [data-testid="stChatInputTextArea"] {
      flex: 1 !important;
      border: none !important;
      outline: none !important;
      background: transparent !important;
      font-size: 14px !important;
      resize: none !important;
      min-height: 36px !important;
      max-height: 36px !important;
      padding: 0 !important;
      line-height: 36px !important;
      overflow: hidden !important;
      align-self: center !important;
    }

    /* Actions wrapper: flex row, zero extra spacing */
    [data-testid="stChatInput"] > div > div:last-child {
      display: flex !important;
      align-items: center !important;
      gap: 6px !important;
      margin: 0 !important;
      padding: 0 !important;
      flex-shrink: 0 !important;
    }

    /* Send button */
    [data-testid="stChatInputSubmitButton"] button {
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      width: 36px !important;
      height: 36px !important;
      border-radius: 8px !important;
      margin: 0 !important;
      padding: 0 !important;
      flex-shrink: 0 !important;
    }
    [data-testid="stChatInputSubmitButton"] button:hover {
      background: rgba(255,255,255,0.08) !important;
    }

    /* Mic button: same size/style as send button */
    #pgp-voice-btn {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 36px;
      height: 36px;
      border-radius: 8px;
      background: transparent;
      border: none;
      cursor: pointer;
      color: #888;
      flex-shrink: 0;
      transition: background .2s, color .2s;
    }
    #pgp-voice-btn:hover  { background: rgba(255,255,255,0.08); color: #fff; }
    #pgp-voice-btn.active { color: #ff4b4b; background: #ff4b4b22; animation: pgp-pulse 1s infinite; }
    @keyframes pgp-pulse  { 0%,100%{opacity:1} 50%{opacity:.4} }
  `;
  pdoc.head.appendChild(style);

  // ── Mic button ──────────────────────────────────────────────────────────────
  const MIC_SVG = `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4zm0 2a2 2 0 0 0-2 2v6a2 2 0 0 0 4 0V5a2 2 0 0 0-2-2zM4.5 12h2A5.5 5.5 0 0 0 12 17.5 5.5 5.5 0 0 0 17.5 12h2A7.5 7.5 0 0 1 13 19.43V22h-2v-2.57A7.5 7.5 0 0 1 4.5 12z"/>
  </svg>`;
  const STOP_SVG = `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
    <rect x="5" y="5" width="14" height="14" rx="2"/>
  </svg>`;

  const btn = pdoc.createElement('button');
  btn.id = 'pgp-voice-btn';
  btn.title = 'Voice input (Chrome/Edge)';
  btn.innerHTML = MIC_SVG;

  // ── Insert mic inline, before the send button ───────────────────────────────
  function insertMicBtn() {
    const submitWrapper = pdoc.querySelector('[data-testid="stChatInputSubmitButton"]');
    if (submitWrapper && !pdoc.getElementById('pgp-voice-btn')) {
      submitWrapper.parentNode.insertBefore(btn, submitWrapper);
      return true;
    }
    return false;
  }

  if (!insertMicBtn()) {
    const observer = new MutationObserver(() => {
      if (insertMicBtn()) observer.disconnect();
    });
    observer.observe(pdoc.body, { childList: true, subtree: true });
  }

  // ── Speech recognition ──────────────────────────────────────────────────────
  const SR = window.parent.SpeechRecognition || window.parent.webkitSpeechRecognition;
  let rec = null, active = false;

  btn.addEventListener('click', () => {
    if (!SR) { alert('Voice input requires Chrome or Edge.'); return; }
    if (active) { rec.stop(); return; }

    rec = new SR();
    rec.lang = 'en-US';
    rec.interimResults = false;

    rec.onstart = () => {
      active = true;
      btn.classList.add('active');
      btn.innerHTML = STOP_SVG;
    };

    rec.onresult = (e) => {
      const text = e.results[0][0].transcript;
      const ta = pdoc.querySelector('[data-testid="stChatInputTextArea"]');
      if (ta) {
        const setter = Object.getOwnPropertyDescriptor(
          window.parent.HTMLTextAreaElement.prototype, 'value'
        ).set;
        setter.call(ta, text);
        ta.dispatchEvent(new Event('input', { bubbles: true }));
        ta.focus();
      }
    };

    rec.onend = () => {
      active = false;
      btn.classList.remove('active');
      btn.innerHTML = MIC_SVG;
    };

    rec.onerror = (e) => { console.error('Voice error:', e.error); rec.stop(); };
    rec.start();
  });
})();
</script>
""", height=0)

# ── Chat Input ────────────────────────────────────────────────────────────────
if query := st.chat_input("Ask about the PGP AI program..."):
    process_query(query)