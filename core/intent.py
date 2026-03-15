INTENT_KEYWORDS = {
    "💰 Fees": [
        "fee", "fees", "₹", "scholarship", "emi", "payment", "cost",
        "installment", "price", "tuition", "discount", "waiver", "refund",
    ],
    "📚 Curriculum": [
        "curriculum", "term", "module", "course", "syllabus", "subject",
        "topic", "learn", "teach", "content", "semester", "week",
    ],
    "📋 Admissions": [
        "admission", "eligibility", "apply", "application", "selection",
        "interview", "criteria", "requirement", "qualify", "enroll", "join",
        "deadline", "process",
    ],
    "🚀 Career": [
        "career", "placement", "hire", "hires", "hiring", "salary", "company",
        "job", "recruit", "employer", "package", "outcome", "opportunity",
        "compan", "ctc",
    ],
    "🎓 Overview": [
        "duration", "month", "schedule", "format", "online", "offline",
        "program", "overview", "about", "structure", "hybrid", "pgp",
    ],
}

def detect_intent(text: str):
    """Keyword-based intent classifier — zero RAM, instant."""
    text_lower = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return intent, 0.85
    return "🎓 Overview", 0.50
