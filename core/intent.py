NO_WARNING_INTENTS = {"👋 Greeting", "🙏 Thanks", "👋 Farewell", "❓ General", "📄 Brochure"}

EMOJI_MAP = {
    "Fees":           "💰 Fees",
    "Curriculum":     "📚 Curriculum",
    "Admissions":     "📋 Admissions",
    "Career":         "🚀 Career",
    "Faculty":        "👨‍🏫 Faculty",
    "Overview":       "🎓 Overview",
    "Recommendation": "🎯 Recommendation",
    "Greeting":       "👋 Greeting",
    "Farewell":       "👋 Farewell",
    "Thanks":         "🙏 Thanks",
    "General":        "❓ General",
    # ── New PDF-content intents ──────────────────────────────────────────────
    "Placement":      "📊 Placement",    # salary data, placement reports, hiring stats
    "Immersion":      "✈️ Immersion",    # global immersion trips, international exposure
    "Brochure":       "📄 Brochure",     # user wants to see / download a brochure
}

KEYWORD_RULES = [
    # ── Core intents (existing) ──────────────────────────────────────────────
    ("💰 Fees",       [
        "fee", "fees", "₹", "scholarship", "emi", "payment", "cost",
        "installment", "price", "tuition", "discount", "waiver", "refund",
        # PDF-sourced additions: financial aid language from brochures
        "financial aid", "financial support", "stipend", "loan", "bursary",
    ]),
    ("📚 Curriculum", [
        "curriculum", "term", "module", "syllabus", "subject", "topic",
        "learn", "teach", "content", "semester",
        # PDF-sourced additions: programme document language
        "course structure", "learning outcomes", "elective", "specialisation",
        "specialization", "capstone", "project", "workshop", "bootcamp",
    ]),
    ("📋 Admissions", [
        "admission", "eligibility", "apply", "application", "selection",
        "interview", "criteria", "requirement", "qualify", "enroll", "join",
        "deadline",
        # PDF-sourced additions: brochure / form language
        "how to apply", "application form", "last date", "entrance", "gmat",
        "cat score", "work experience", "minimum experience",
    ]),
    # ── Placement before Career so salary/placement queries route correctly ──
    # placement_query: placement reports, median salary, recruiter lists
    ("📊 Placement",  [
        "placement", "placement report", "placement stats", "placement record",
        "average salary", "median salary", "highest salary", "salary",
        "top recruiters", "companies visited", "hiring companies",
        "placed students", "campus placement", "placement season", "lpa", "ctc",
    ]),
    ("🚀 Career",     [
        "career", "hire", "hiring", "company", "job", "recruit",
        "employer", "package", "outcome",
        # "salary" and "placement" are handled by 📊 Placement above
    ]),
    ("👨‍🏫 Faculty",   [
        "faculty", "professor", "mentor", "teacher", "instructor", "who teaches",
    ]),
    ("🎓 Overview",   [
        "duration", "months", "how long", "online", "offline", "format",
        "schedule", "part-time", "full-time", "weekend", "weekday", "mode of",
        "delivery", "when does", "start date", "batch", "how many months",
        "program structure", "class timing", "timing", "location", "campus",
        "hybrid",
        # Added: common overview / catalogue queries
        "programme", "programs", "programmes", "what program", "courses offered",
        "all courses", "list of programs",
    ]),
    ("🎯 Recommendation", [
        "best course", "which course", "recommend", "suitable", "should i",
        "compare", "which program", "right for me",
    ]),
    # immersion_query: global immersion content from immersion-report PDFs
    ("✈️ Immersion",  [
        "immersion", "global immersion", "international trip", "overseas",
        "foreign university", "international exposure", "global experience",
        "study abroad", "exchange program", "international module",
        "bharat immersion", "immersion report",
    ]),
    # brochure_request: user wants to view or download a brochure / cohort report
    ("📄 Brochure",   [
        "brochure", "pdf", "download brochure", "send brochure",
        "cohort report", "cohort profile", "class profile",
        "programme document", "program document", "prospectus",
        "information pack",
    ]),
    # ── Conversational intents (existing) ───────────────────────────────────
    ("👋 Greeting",   ["hello", "hi", "hey", "good morning", "good evening", "good afternoon", "good night", "howdy"]),
    ("🙏 Thanks",     ["thanks", "thank you", "thank", "appreciate", "grateful", "thx"]),
    ("👋 Farewell",   ["bye", "goodbye", "see you", "take care", "cya"]),
]


def _keyword_classify(query: str) -> tuple[str, float]:
    q = query.lower()
    for intent, keywords in KEYWORD_RULES:
        if any(k in q for k in keywords):
            return intent, 0.75
    return "❓ General", 0.5


def _llm_classify(query: str, llm) -> tuple[str, float]:
    prompt = f"""Classify this user query into ONE intent category.
Choose from: Fees, Curriculum, Admissions, Career, Faculty, Overview, Recommendation, Placement, Immersion, Brochure, Greeting, Farewell, Thanks, General.
Also give a confidence score between 0.0 and 1.0.

Query: {query}

Respond in this exact format only:
INTENT: <intent>
CONFIDENCE: <score>"""

    response = llm.invoke(prompt)
    text = getattr(response, "content", "").strip()
    intent_line = next((l for l in text.splitlines() if "INTENT:" in l), "")
    conf_line   = next((l for l in text.splitlines() if "CONFIDENCE:" in l), "")
    intent_raw  = intent_line.split("INTENT:")[-1].strip()
    conf_raw    = conf_line.split("CONFIDENCE:")[-1].strip()
    if not intent_raw or not conf_raw:
        raise ValueError(f"LLM returned unexpected format: {text!r}")
    intent     = EMOJI_MAP.get(intent_raw, f"❓ {intent_raw}")
    confidence = float(conf_raw)
    return intent, min(max(confidence, 0.0), 1.0)


def detect_intent(query: str, llm=None) -> tuple[str, float]:
    """
    Keyword-first intent classifier. Falls back to LLM only when
    keyword matching returns General (ambiguous query).
    """
    intent, confidence = _keyword_classify(query)

    # Only pay for an LLM call when keyword matching is uncertain
    if intent != "❓ General" or llm is None:
        return intent, confidence

    try:
        return _llm_classify(query, llm)
    except Exception as e:
        print(f"[INTENT] LLM classify failed: {e}, using keyword result")
        return intent, confidence
