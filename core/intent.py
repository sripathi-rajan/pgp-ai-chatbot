NO_WARNING_INTENTS = {"👋 Greeting", "🙏 Thanks", "👋 Farewell", "❓ General"}

EMOJI_MAP = {
    "Fees":           "💰 Fees",
    "Curriculum":     "📚 Curriculum",
    "Admissions":     "📋 Admissions",
    "Career":         "🚀 Career",
    "Faculty":        "👨‍🏫 Faculty",
    "Recommendation": "🎯 Recommendation",
    "Greeting":       "👋 Greeting",
    "Farewell":       "👋 Farewell",
    "Thanks":         "🙏 Thanks",
    "General":        "❓ General",
}

KEYWORD_RULES = [
    ("💰 Fees",           ["fee", "fees", "₹", "scholarship", "emi", "payment", "cost", "installment", "price", "tuition", "discount", "waiver", "refund"]),
    ("📚 Curriculum",     ["curriculum", "term", "module", "syllabus", "subject", "topic", "learn", "teach", "content", "semester"]),
    ("📋 Admissions",     ["admission", "eligibility", "apply", "application", "selection", "interview", "criteria", "requirement", "qualify", "enroll", "join", "deadline"]),
    ("🚀 Career",         ["career", "placement", "hire", "hiring", "salary", "company", "job", "recruit", "employer", "package", "outcome", "ctc"]),
    ("👨‍🏫 Faculty",       ["faculty", "professor", "mentor", "teacher", "instructor", "who teaches"]),
    ("🎯 Recommendation", ["best course", "which course", "recommend", "suitable", "should i", "compare", "which program", "right for me"]),
    ("👋 Greeting",       ["hello", "hi", "hey", "good morning", "good evening", "good afternoon", "good night", "howdy"]),
    ("🙏 Thanks",         ["thanks", "thank you", "thank", "appreciate", "grateful", "thx"]),
    ("👋 Farewell",       ["bye", "goodbye", "see you", "take care", "cya"]),
]


def _keyword_classify(query: str) -> tuple[str, float]:
    q = query.lower()
    for intent, keywords in KEYWORD_RULES:
        if any(k in q for k in keywords):
            return intent, 0.75
    return "❓ General", 0.5


def _llm_classify(query: str, llm) -> tuple[str, float]:
    prompt = f"""Classify this user query into ONE intent category.
Choose from: Fees, Curriculum, Admissions, Career, Faculty, Recommendation, Greeting, Farewell, Thanks, General.
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
