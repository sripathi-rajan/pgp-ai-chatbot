NO_WARNING_INTENTS = {"👋 Greeting", "🙏 Thanks", "👋 Farewell", "❓ General"}

def detect_intent(query: str, llm=None) -> tuple[str, float]:
    """
    LLM-based intent classifier with keyword fallback.
    Pass llm= for dynamic classification; omit for instant keyword matching.
    """
    if llm:
        try:
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

            intent     = intent_line.split("INTENT:")[-1].strip()
            confidence = float(conf_line.split("CONFIDENCE:")[-1].strip())

            emoji_map = {
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
            intent = emoji_map.get(intent, f"❓ {intent}")
            return intent, min(max(confidence, 0.0), 1.0)

        except Exception as e:
            print(f"[INTENT] LLM classify failed: {e}, falling back to keywords")

    # ── Keyword fallback ──────────────────────────────────────────────────────
    q = query.lower()
    rules = [
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
    for intent, keywords in rules:
        if any(k in q for k in keywords):
            return intent, 0.75
    return "❓ General", 0.5
