"""
Flask server for Masters' Union AI Chatbot
Serves the HTML frontend and exposes the /ask POST endpoint.

Run:  python app.py
Open: http://localhost:5000

NOTE: The original Streamlit interface is preserved in streamlit_app.py
      Run it with:  streamlit run streamlit_app.py
"""

import os
import re
import sys

# Fix Windows stdout encoding so emoji in log messages don't crash the server
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ── API Keys (read from environment variables) ───────────────────────────────
groq_key = os.environ.get("GROQ_API_KEY", "")
os.environ["GROQ_API_KEY"] = groq_key

openai_key = os.environ.get("OPENAI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = openai_key

serper_key = os.environ.get("SERPER_API_KEY", "")
if not serper_key:
    print("[WARN] SERPER_API_KEY not set — web search fallback disabled")
os.environ["SERPER_API_KEY"] = serper_key

# ── Core RAG modules ──────────────────────────────────────────────────────────
from core.pipeline import load_pipeline_flask
from core.retriever import hybrid_retrieve, broad_retrieve
from core.intent import detect_intent
from core.prompt import build_prompt, format_context

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the HTML frontend

# ── Load pipeline ONCE at startup (global — avoids reloading per request) ─────
print("[APP] Loading RAG pipeline — this may take a minute on first run...")
_pipeline = load_pipeline_flask()
print("[APP] Pipeline ready. Starting server on http://localhost:5000")

# ── Intent → Category mapping for the frontend badge ─────────────────────────
INTENT_TO_CATEGORY = {
    "💰 Fees":           "Fees",
    "📚 Curriculum":     "Curriculum",
    "📋 Admissions":     "Admissions",
    "🚀 Career":         "Career",
    "📊 Placement":      "Career",
    "✈️ Immersion":      "Overview",
    "📄 Brochure":       "Overview",
    "❓ General":        "Overview",
    "🎓 Overview":       "Overview",
    "👨\u200d🏫 Faculty": "Overview",
    "🎯 Recommendation": "Overview",
    "👋 Greeting":       "Overview",
    "🙏 Thanks":         "Overview",
    "👋 Farewell":       "Overview",
}

# ── Small-talk shortcut (bypass RAG for greetings / thanks / farewells) ───────
_SMALL_TALK = {
    "hi":           "Hello! How can I help you with Masters' Union programmes today?",
    "hello":        "Hello! How can I help you with Masters' Union programmes today?",
    "hey":          "Hello! How can I help you with Masters' Union programmes today?",
    "howdy":        "Hello! How can I help you with Masters' Union programmes today?",
    "hiya":         "Hello! How can I help you with Masters' Union programmes today?",
    "good morning": "Good morning! How can I help you today?",
    "good evening": "Good evening! How can I help you today?",
    "good afternoon":"Good afternoon! How can I help you today?",
    "good night":   "Good night! Feel free to come back with any questions.",
    "thanks":       "You're welcome! Feel free to ask anything else.",
    "thank you":    "You're welcome! Feel free to ask anything else.",
    "thx":          "You're welcome!",
    "ty":           "You're welcome!",
    "bye":          "Goodbye! Come back anytime if you have more questions.",
    "goodbye":      "Goodbye! Come back anytime if you have more questions.",
    "see you":      "See you! Feel free to return with more questions.",
    "take care":    "Take care! All the best.",
}

_WHO_AM_I = {"who are you", "what are you", "what can you do"}

def _small_talk_reply(query: str):
    """Return a canned reply for small-talk, or None."""
    q = query.strip().lower().rstrip("!?.")
    if q in _SMALL_TALK:
        return _SMALL_TALK[q]
    if q in _WHO_AM_I or any(p in q for p in _WHO_AM_I):
        return (
            "I'm the Masters' Union AI Assistant. "
            "Ask me about fees, curriculum, admissions, placements or campus life!"
        )
    return None

# ── Out-of-scope topics ────────────────────────────────────────────────────────
_OUT_OF_SCOPE = [
    "cricket", "football", "movie", "stock price", "weather",
    "recipe", "joke", "poem", "write an essay",
]

# ── Broad query triggers (pull chunks across all topic areas) ─────────────────
_BROAD_TRIGGERS = [
    "tell me everything", "tell me about", "explain the course",
    "explain the program", "overview", "summarize", "all about",
    "what is this program", "describe the program", "give me details",
    "what does this program offer", "full details", "complete information",
]

# ── Course-list query triggers (always inject the programme list first) ────────
_COURSE_LIST_TRIGGERS = [
    "list all", "all courses", "all programmes", "all programs",
    "what courses", "what programmes", "what programs",
    "available courses", "available programmes", "available programs",
    "course list", "programme list", "program list",
    "courses offered", "programmes offered", "programs offered",
    "list of courses", "list of programmes", "list of programs",
    "what can i study", "what do you offer",
    "undergraduate programme", "undergraduate program",
    "postgraduate programme", "postgraduate program",
    "ug programme", "pg programme", "pgp programme",
    "which programmes", "which programs", "which courses",
]

# ── Programme catalogue injected as guaranteed first chunk ────────────────────
_PROGRAMME_CATALOGUE = """MASTERS' UNION — COMPLETE LIST OF ALL PROGRAMMES OFFERED

UNDERGRADUATE PROGRAMMES:
- UG in Technology & Business Management
- UG in Psychology & Marketing
- UG in Data Science & AI
- UG in Finance & Economics
- UG Programme in Design (MUDS)
- UG in Technology & Business Management (Illinois Tech, US)
- UG in Psychology & Marketing (Illinois Tech, US)
- UG in Data Science & AI (Illinois Tech, US)
- UG in Technology & Business Management (Griffith University, Australia)

POSTGRADUATE PROGRAMMES (PGP):
- PGP in Technology & Business Management
- PGP in Technology & Business Management (Young Leaders Cohort)
- PGP in Human Resources & Organisation Strategy
- PGP in Sports Management & Gaming
- PGP in Applied AI & Agentic Systems
- PGP in UI/UX & AI Product Design
- PGP in Sustainability & Business Management

EXECUTIVE PROGRAMMES:
- PGP Rise: General Management
- PGP in Capital Markets and Trading
- PGP in Entrepreneurship and Business Acceleration
- PGP Rise: General Management (Global)
- Bloomberg Equity Research Programme

FAMILY BUSINESS PROGRAMMES:
- PGP Rise: Owners and Promoters Management
- PGP in Entrepreneurship and Business Acceleration

IMMERSION PROGRAMMES:
- PGP Bharat
- Bharat Fellowship"""

# ── Input sanitisation ─────────────────────────────────────────────────────────
_MAX_QUERY_LEN = 500
_INJECTION_RE = re.compile(
    r"ignore (all |previous |prior )?instructions|you are now|system prompt|forget everything",
    re.IGNORECASE,
)

def _sanitize(query: str):
    """Return cleaned query string, or None if it should be rejected."""
    if not isinstance(query, str):
        return None
    query = query.strip()[:_MAX_QUERY_LEN]
    if _INJECTION_RE.search(query):
        return None
    return query or None


# ── Post-process LLM answer for clean, consistent formatting ──────────────────
_TABLE_ROW_RE   = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_SEP_RE   = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")
_RECOMMEND_RE   = re.compile(
    r"((?:💡\s*)?(?:Recommendation|I recommend|Best suited|I suggest)[^\n]*)",
    re.IGNORECASE,
)


def _table_block_to_tree(text: str) -> str:
    """Convert any markdown pipe-table block in text to tree format."""
    lines   = text.splitlines()
    out     = []
    i       = 0
    while i < len(lines):
        line = lines[i]
        # Detect start of a table block
        if _TABLE_ROW_RE.match(line) and not _TABLE_SEP_RE.match(line):
            # Collect the full table block
            block = []
            while i < len(lines) and (
                _TABLE_ROW_RE.match(lines[i]) or _TABLE_SEP_RE.match(lines[i])
            ):
                block.append(lines[i])
                i += 1

            # Parse header + data rows
            header = None
            rows   = []
            for bline in block:
                if _TABLE_SEP_RE.match(bline):
                    continue
                cells = [c.strip() for c in bline.split("|") if c.strip()]
                if not cells:
                    continue
                if header is None:
                    header = cells
                else:
                    rows.append(cells)

            # Emit tree format
            for row in rows:
                if not row:
                    continue
                out.append(row[0].upper())
                for j, cell in enumerate(row[1:], start=1):
                    label     = header[j] if header and j < len(header) else f"Col {j}"
                    connector = "└─" if j == len(row) - 1 else "├─"
                    out.append(f"{connector} {label}: {cell}")
                out.append("---")
        else:
            out.append(line)
            i += 1
    return "\n".join(out)


def post_process_answer(answer: str) -> str:
    """
    Post-process LLM output for consistent display:
    1. Convert markdown pipe-tables → tree format
    2. Wrap recommendation sentences → [RECOMMEND]...[/RECOMMEND]
    3. Collapse excessive blank lines
    4. Strip trailing whitespace per line
    """
    # 1. Convert markdown tables
    answer = _table_block_to_tree(answer)

    # 2. Wrap recommendation sentences
    def _wrap(m: re.Match) -> str:
        return f"[RECOMMEND]{m.group(1).strip()}[/RECOMMEND]"
    answer = _RECOMMEND_RE.sub(_wrap, answer)

    # 3. Collapse 3+ blank lines → 2
    answer = re.sub(r"\n{3,}", "\n\n", answer)

    # 4. Strip trailing spaces per line
    answer = "\n".join(l.rstrip() for l in answer.splitlines())

    return answer.strip()


# ── GET / — serve the HTML chatbot frontend ────────────────────────────────────
@app.route("/")
def index():
    """Serve index.html from the project root."""
    return send_file("index.html")


# ── POST /ask — full RAG pipeline ─────────────────────────────────────────────
@app.route("/ask", methods=["POST"])
def ask():
    """
    Accepts: { "query": "user question" }
    Returns: { "answer": "...", "category": "Fees|Curriculum|Admissions|Career|Overview",
               "model": "model name string" }
    """
    try:
        data    = request.get_json(force=True) or {}
        query   = _sanitize(data.get("query", ""))
        history = data.get("history", [])
        # Validate history: must be list of [role, message] pairs
        if not isinstance(history, list):
            history = []

        if not query:
            return jsonify({
                "answer":   "Please enter a valid question.",
                "category": "Overview",
                "model":    "",
            })

        # ── Out-of-scope redirect ──────────────────────────────────────────
        if any(w in query.lower() for w in _OUT_OF_SCOPE):
            return jsonify({
                "answer": (
                    "I'm specialised in answering questions about Masters' Union "
                    "programmes. Could you ask me something related to the programme?"
                ),
                "category": "Overview",
                "model":    "",
            })

        # ── Small-talk shortcut — bypass RAG entirely ─────────────────────
        reply = _small_talk_reply(query)
        if reply:
            return jsonify({"answer": reply, "category": "Overview", "model": ""})

        # ── 1. Classify intent (keyword-first, LLM fallback on ambiguity) ─
        intent, _ = detect_intent(query, llm=_pipeline.llm)

        # ── 2. Detect course-list queries ──────────────────────────────────
        q_lower = query.lower()
        is_course_list = any(t in q_lower for t in _COURSE_LIST_TRIGGERS)

        # ── 3. Retrieve relevant chunks ────────────────────────────────────
        is_broad = any(w in q_lower for w in _BROAD_TRIGGERS)
        if is_broad or is_course_list:
            top_docs = broad_retrieve(
                query, _pipeline.db, _pipeline.bm25,
                _pipeline.texts, _pipeline.embeddings, chunks_per_topic=2
            )
        else:
            top_docs = hybrid_retrieve(
                query, _pipeline.db, _pipeline.bm25,
                _pipeline.texts, _pipeline.embeddings, k=8,
                chunks=_pipeline.chunks
            )

        # ── 4. For course-list queries inject the catalogue as first chunk ─
        if is_course_list:
            from langchain_core.documents import Document as _Doc
            catalogue_doc = _Doc(
                page_content=_PROGRAMME_CATALOGUE,
                metadata={"source": "programme_catalogue", "content_type": "catalogue"}
            )
            top_docs = [catalogue_doc] + list(top_docs)

        # ── 5. Build context string ────────────────────────────────────────
        context = format_context(top_docs)

        # ── 6. Build prompt ────────────────────────────────────────────────
        prompt = build_prompt(query, context, history, intent=intent)

        if is_broad:
            prompt += """

IMPORTANT: The user wants a complete overview. Use ALL CAPS section names (not ## headers). \
Only include sections that have data in the context. Format:

PROGRAMME OVERVIEW:
[3-5 bullet points]
---
CURRICULUM & TOPICS:
[3-5 bullet points]
---
FEES & SCHOLARSHIPS:
[key: value pairs]
---
ELIGIBILITY & ADMISSIONS:
[3-5 bullet points]
---
CAREER OUTCOMES:
[key: value pairs]
---
LEARNING FORMAT:
[3-5 bullet points]

Skip any section with no data in context. Do not invent anything."""

        # ── 5. Call LLM ────────────────────────────────────────────────────
        response = _pipeline.llm.invoke(prompt)
        answer   = getattr(response, "content", str(response))
        # Strip chain-of-thought tags (Qwen3 local model adds these)
        answer   = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        # Post-process for clean, consistently formatted output
        answer   = post_process_answer(answer)

        if not answer:
            answer = "I couldn't generate a response. Please try rephrasing your question."

        # ── 6. Map intent → frontend category badge ────────────────────────
        category = INTENT_TO_CATEGORY.get(intent, "Overview")

        # ── 7. Resolve model name for display ──────────────────────────────
        model_name = (
            getattr(_pipeline.llm, "model_name", None)
            or getattr(_pipeline.llm, "model", "AI")
        )

        return jsonify({
            "answer":   answer,
            "category": category,
            "model":    model_name,
        })

    except Exception as exc:
        # Safe print on Windows (avoid UnicodeEncodeError from emoji in error msgs)
        try:
            print(f"[ASK] Error: {exc}")
        except UnicodeEncodeError:
            print(f"[ASK] Error: {type(exc).__name__}")

        # Return a user-friendly message based on error type
        exc_str = str(exc).lower()
        if "insufficient_quota" in exc_str or "rate_limit" in exc_str or "429" in exc_str:
            answer = (
                "⚠️ The OpenAI API quota is exhausted. "
                "Please add credits at platform.openai.com, or set USE_LOCAL_LLM=true "
                "to use a local Ollama model."
            )
        elif "auth" in exc_str or "api_key" in exc_str or "401" in exc_str:
            answer = (
                "⚠️ OpenAI API key is invalid or missing. "
                "Set the OPENAI_API_KEY environment variable and restart the server."
            )
        else:
            answer = "Sorry, I could not process your request. Please try again."

        return jsonify({"answer": answer, "category": "Overview", "model": ""}), 200


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
