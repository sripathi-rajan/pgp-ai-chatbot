import html as _html
import os
import re
import threading
import streamlit as st

# Read API keys
openai_key = st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = openai_key

serper_key = st.secrets.get("SERPER_API_KEY") or os.environ.get("SERPER_API_KEY")
if not serper_key:
    print("[WARN] SERPER_API_KEY not set — web search fallback disabled")
os.environ["SERPER_API_KEY"] = serper_key or ""

# Import our modules
from core.pipeline import load_pipeline
from core.retriever import hybrid_retrieve, broad_retrieve, get_best_sentence
from core.intent import detect_intent, NO_WARNING_INTENTS
from core.prompt import build_prompt, format_context
from utils.notifier import notify_admin

import streamlit.components.v1 as components

st.set_page_config(page_title="PGP AI Assistant", page_icon="🎓")
st.title("🎓 PGP AI Program Assistant")
st.caption("Hybrid RAG · Intent Routing · Hallucination Guardrails")




@st.cache_resource
def load_cached_pipeline():
    return load_pipeline()

_pipeline = load_cached_pipeline()
db        = _pipeline.db
bm25      = _pipeline.bm25
texts     = _pipeline.texts
chunks    = _pipeline.chunks
embeddings = _pipeline.embeddings
llm       = _pipeline.llm

# Build metadata lookup once — avoids O(n²) scan per query
chunk_metadata = {c.page_content: c.metadata for c in chunks}

# ── Session State — initialize BEFORE any usage ───────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# ── Assign Title — extract first meaningful line from content ──────────────────
def assign_title(text, source=""):
    src = "🌐 Web" if "http" in str(source) else "📄 PDF"
    for line in text.split("\n"):
        line = line.strip()
        if 10 < len(line) < 80:
            return f"{src} · {line[:60]}"
    return f"{src} · Program Information"

# ── Small talk detector ────────────────────────────────────────────────────────
SMALL_TALK = [
    "hi", "hello", "hey", "howdy", "hiya",
    "good morning", "good evening", "good afternoon", "good night",
    "thanks", "thank you", "thx", "ty",
    "bye", "goodbye", "see you", "take care",
    "ok", "okay", "cool", "got it", "sure", "great",
    "who are you", "what are you", "what can you do",
]

def is_small_talk(query: str) -> str | None:
    """Returns a canned reply if small talk, else None."""
    q = query.strip().lower().rstrip("!?.")
    if q not in SMALL_TALK:
        return None
    if any(w in q for w in ["hi", "hello", "hey", "howdy", "hiya", "morning", "evening", "afternoon", "night"]):
        return "Hello! How can I help you with the PGP AI program today?"
    if any(w in q for w in ["thanks", "thank", "thx", "ty"]):
        return "You're welcome! Feel free to ask anything else."
    if any(w in q for w in ["bye", "goodbye", "see you", "take care"]):
        return "Goodbye! Come back anytime if you have more questions."
    if any(w in q for w in ["who are you", "what are you", "what can you do"]):
        return "I'm the PGP AI Program Assistant. Ask me about fees, curriculum, admissions, career outcomes, or anything else about the program!"
    return "Got it! Feel free to ask me anything about the PGP AI program."

# ── Source display ────────────────────────────────────────────────────────────
def display_sources(high_sources, low_sources):
    """Render high-relevance and low-relevance sources in clean, clickable expanders."""
    if not high_sources and not low_sources:
        return

    with st.expander("📄 Sources used to answer"):
        for page_num, title, sentence, full_chunk, relevance, source_url in high_sources:
            raw_url = str(source_url) if source_url else ""
            # Only allow http/https URLs — strip anything else to prevent javascript: URIs
            safe_url = raw_url if raw_url.startswith(("http://", "https://")) else ""
            has_url = bool(safe_url)
            filename = raw_url.split("/")[-1].split("\\")[-1] if raw_url else "Document"
            score_pct = f"{relevance:.0%}"
            badge = f"🟢 {score_pct}" if relevance > 0.5 else f"🟡 {score_pct}"
            page_display = f"Page {page_num}" if str(page_num) != "?" else "PDF"

            if has_url:
                # Web/scraped source — full clickable card
                color = "#1D9E75" if relevance > 0.5 else "#3B8BD4"
                preview = _html.escape(" ".join(sentence.split())[:120])
                safe_title = _html.escape(title)
                safe_url_attr = _html.escape(safe_url, quote=True)
                ellipsis = "..." if len(sentence) > 120 else ""
                st.markdown(f"""
                <a href="{safe_url_attr}" target="_blank" rel="noopener noreferrer" style="text-decoration:none;">
                <div style="border:0.5px solid #e0e0e0;border-left:4px solid {color};
                            border-radius:8px;padding:14px 16px;margin-bottom:10px;cursor:pointer;"
                     onmouseover="this.style.boxShadow='0 2px 8px rgba(0,0,0,0.12)'"
                     onmouseout="this.style.boxShadow='none'">
                    <div style="display:flex;justify-content:space-between;
                                align-items:center;margin-bottom:8px;">
                        <span style="font-size:13px;font-weight:600;color:{color};">{safe_title}</span>
                        <span style="font-size:11px;color:{color};background:{color}18;
                                     padding:2px 8px;border-radius:10px;">
                            {badge} · 🔗 View source
                        </span>
                    </div>
                    <div style="font-size:11px;color:#999;margin-bottom:6px;">🌐 {_html.escape(safe_url)}</div>
                    <div style="font-size:13px;line-height:1.7;color:#333;">{preview}{ellipsis}</div>
                </div>
                </a>
                """, unsafe_allow_html=True)
            else:
                # PDF/doc source — expandable with full chunk as proof
                label = f"📎 {filename}  |  {page_display}  {badge}"
                with st.expander(label, expanded=False):
                    st.markdown(f"**📄 File:** `{filename}`")
                    st.markdown(f"**📖 Page:** `{page_display}`")
                    st.markdown(f"**🎯 Relevance:** `{score_pct}`")
                    st.divider()
                    st.markdown("**📝 Excerpt:**")
                    preview = " ".join(full_chunk.split())[:120]
                    st.info(preview + ("..." if len(full_chunk) > 120 else ""))

        if low_sources:
            st.markdown(
                "<div style='font-size:12px;color:#999;margin-top:8px;"
                "margin-bottom:4px;'>📎 Additional References</div>",
                unsafe_allow_html=True
            )
            for page_num, title, source_url in low_sources:
                raw_url = str(source_url) if source_url else ""
                safe_url = raw_url if raw_url.startswith(("http://", "https://")) else ""
                has_url = bool(safe_url)
                if has_url:
                    st.markdown(
                        f"<div style='font-size:12px;color:#888;padding:4px 0;'>"
                        f"· <a href='{_html.escape(safe_url, quote=True)}' target='_blank'"
                        f" rel='noopener noreferrer' style='color:#3B8BD4;'>"
                        f"{_html.escape(title)}</a> — {_html.escape(safe_url)}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    page_display = f"Page {page_num}" if str(page_num) != "?" else "PDF"
                    st.markdown(
                        f"<div style='font-size:12px;color:#888;padding:4px 0;'>"
                        f"· {_html.escape(title)} — {page_display}</div>",
                        unsafe_allow_html=True
                    )


# ── Input sanitization ────────────────────────────────────────────────────────
MAX_QUERY_LENGTH = 500
_INJECTION_PATTERNS = re.compile(
    r"ignore (all |previous |prior )?instructions|you are now|system prompt|forget everything",
    re.IGNORECASE,
)

def sanitize_query(query: str) -> str | None:
    """Returns cleaned query, or None if the query should be rejected."""
    if not isinstance(query, str):
        return None
    query = query.strip()[:MAX_QUERY_LENGTH]
    if _INJECTION_PATTERNS.search(query):
        return None
    return query

# ── Out-of-scope keywords ──────────────────────────────────────────────────────
OUT_OF_SCOPE = [
    "cricket", "football", "movie", "stock price", "weather",
    "recipe", "joke", "poem", "write an essay",
]

# ── Module-level constants (avoids re-allocation on every query) ───────────────
INFERENCE_TRIGGERS = [
    "suitable for", "good for", "right for me", "fit for me",
    "can someone", "can i join", "am i eligible", "will i qualify",
    "should i join", "is it worth", "is this for me",
    "non-tech", "non technical", "arts background", "commerce background",
    "manager", "fresher", "without experience", "without degree", "without technical",
]

HALLUCINATION_PHRASES = [
    "as of my knowledge cutoff",
    "i don't have access",
    "i cannot confirm",
    "not in my training",
    "i made up",
    "i fabricated",
]

GENUINE_GAPS = [
    "i don't have that specific detail",
    "i don't have that information",
    "not mentioned anywhere",
    "no data available",
    "cannot find any reference",
]

INFERABLE_PATTERNS = [
    "suitable for", "good for", "can i join", "am i eligible",
    "background", "experience", "non-tech", "manager", "fresher",
    "should i", "is it worth", "will i", "can someone",
]

BROAD_QUERY_TRIGGERS = [
    "tell me everything", "tell me about", "explain the course",
    "explain the program", "overview", "summarize", "all about",
    "what is this program", "describe the program", "give me details",
    "what does this program offer", "full details", "complete information",
]

# ── Inference handler — eligibility/suitability questions ────────────────────
def _handle_inference(query, intent, confidence):
    # Fetch eligibility context live from the knowledge base so answers
    # always reflect the latest brochure / program_data.txt content.
    eligibility_docs = hybrid_retrieve(
        "eligibility criteria background experience requirements STEM apply",
        db, bm25, texts, embeddings, k=4
    )
    eligibility_context = "\n\n".join(eligibility_docs) if eligibility_docs else (
        "No eligibility context found — please contact pgadmissions@mastersunion.org"
    )

    inference_prompt = f"""You are a helpful admissions counselor for Masters' Union.

Using the eligibility context retrieved from the program knowledge base below, \
answer the user's question with clear reasoning.
- Be direct: say YES, NO, or POSSIBLY with a brief explanation.
- Mention what specifically makes them eligible or ineligible.
- If borderline, explain what would strengthen their application.
- Keep it under 5 sentences. No sources needed — this is your expert assessment.
- If the context does not contain enough information, say so honestly.

ELIGIBILITY CONTEXT (from knowledge base):
{eligibility_context}

USER QUESTION: {query}

ANSWER:"""

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            placeholder = st.empty()
            full_response = ""
            try:
                for chunk in llm.stream(inference_prompt):
                    token = chunk.content if isinstance(chunk.content, str) else ""
                    if token:
                        full_response += token
                        display = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
                        placeholder.markdown(display + "▌")
            except Exception as e:
                st.error(f"Error: {e}")
                return

        answer = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
        placeholder.markdown(answer)

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"Intent: {intent}")
        with col2:
            st.info(f"Confidence: {confidence:.0%}")

    st.session_state.chat_history.append(("Student", query))
    st.session_state.chat_history.append(("Assistant", answer))
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "intent": f"{intent} ({confidence:.0%})"
    })


# ── Process Query ─────────────────────────────────────────────────────────────
def process_query(query):
    query = sanitize_query(query)
    if not query:
        with st.chat_message("assistant"):
            st.warning("Your message couldn't be processed. Please rephrase and try again.")
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # ── Small talk shortcut — bypass RAG entirely ─────────────────────────────
    small_talk_reply = is_small_talk(query)
    if small_talk_reply:
        with st.chat_message("assistant"):
            st.markdown(small_talk_reply)
        st.session_state.chat_history.append(("Student", query))
        st.session_state.chat_history.append(("Assistant", small_talk_reply))
        st.session_state.messages.append({"role": "assistant", "content": small_talk_reply})
        return

    # ── Out-of-scope redirect ─────────────────────────────────────────────────
    if any(w in query.lower() for w in OUT_OF_SCOPE):
        reply = "I'm specialised in answering questions about the PGP AI program. Could you ask me something related to the program?"
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        return

    # ── Inference routing — suitability/eligibility questions ────────────────
    if any(t in query.lower() for t in INFERENCE_TRIGGERS):
        intent, confidence = detect_intent(query, llm=llm)
        _handle_inference(query, intent, confidence)
        return

    # ── RAG pipeline ──────────────────────────────────────────────────────────
    intent, confidence = detect_intent(query, llm=llm)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

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

            is_broad = any(w in query.lower() for w in BROAD_QUERY_TRIGGERS)

            if is_broad:
                top_docs = broad_retrieve(query, db, bm25, texts, embeddings, chunks_per_topic=2)
            else:
                top_docs = hybrid_retrieve(query, db, bm25, texts, embeddings, k=3)
            # Build context: PDF chunks get [Source|Category|Page] headers;
            # scraped text is passed through as plain text.
            context_parts = []
            for doc_text in top_docs:
                meta = chunk_metadata.get(doc_text, {})
                if meta.get("type") == "pdf":
                    source   = meta.get("source", "unknown")
                    category = meta.get("category", "general").upper()
                    page     = meta.get("page", "")
                    pg_tag   = f" | Page: {page}" if page else ""
                    header   = f"[Source: {source} | Category: {category}{pg_tag}]"
                    context_parts.append(f"{header}\n{doc_text}")
                else:
                    context_parts.append(doc_text)
            context = "\n\n".join(context_parts)

            prompt = build_prompt(query, context, st.session_state.chat_history, intent=intent)

            if is_broad:
                prompt += """

IMPORTANT: The user wants a complete overview. Structure your answer with these sections \
(only include sections that have information in the context):

## Program Overview
## Curriculum & Topics
## Fees & Scholarships
## Eligibility & Admissions
## Career Outcomes
## Learning Format

Keep each section concise — 3 to 5 bullet points. \
If a section has no data in context, skip it entirely. Do not invent anything."""

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

        # Embed query once; reuse across all source docs
        query_vec = embeddings.embed_query(query)

        for doc in top_docs:
            metadata   = chunk_metadata.get(doc, {})
            page_num   = metadata.get("page", "?")
            source_url = metadata.get("source", "")

            best_sentence, relevance = get_best_sentence(doc, embeddings, query_vec=query_vec)
            title = assign_title(doc, source_url)

            if any(kw in doc.lower() for kw in current_keywords):
                relevance = min(relevance + 0.2, 1.0)

            if relevance >= 0.30:
                high_sources.append((page_num, title, best_sentence, doc, relevance, source_url))
            else:
                low_sources.append((page_num, title, source_url))

        high_sources.sort(key=lambda x: x[4], reverse=True)

        # ── Smarter uncertainty detection & admin notification ───────────────
        is_genuinely_unknown = any(p in answer.lower() for p in GENUINE_GAPS)
        is_inferable = any(p in query.lower() for p in INFERABLE_PATTERNS)
        context_has_relevant = len(context.strip()) > 300
        context_too_short = len(context.strip()) < 100
        low_relevance = not high_sources or all(r < 0.28 for _, _, _, _, r, _ in high_sources)

        should_notify = (
            (is_genuinely_unknown and not is_inferable and not context_has_relevant)
            or (context_too_short and not is_inferable)
            or (low_relevance and not is_inferable and not context_has_relevant)
        )

        if should_notify and intent not in NO_WARNING_INTENTS:
            threading.Thread(
                target=notify_admin,
                args=(query, list(st.session_state.chat_history), answer),
                daemon=True
            ).start()
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

        display_sources(high_sources, low_sources)

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
    st.caption("• GPT-4o-mini / Qwen3 1.7B (LLM)")
    st.caption("• Whisper (Voice)")
    st.caption("• MiniLM-L6 (Embeddings)")
    st.caption("• BM25 + Hybrid Retrieval")
    st.caption("• pdfplumber (PDF)")

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