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

    # Conversational intents
    "👋 Greeting": [
        "hello", "hi", "hey", "good morning", "good evening", "good afternoon",
        "good night", "howdy", "greetings", "what's up", "sup",
    ],
    "🙏 Thanks": [
        "thank", "thanks", "thank you", "appreciate", "grateful",
        "awesome", "great", "good answer", "good answers", "perfect",
        "helpful", "nice", "well done", "excellent", "wonderful",
    ],
    "👋 Farewell": [
        "bye", "goodbye", "see you", "take care", "later", "cya",
        "good bye", "have a good day",
    ],
    "❓ General": [
        "what", "how", "why", "when", "where", "who", "which",
        "tell me", "explain", "describe", "give me", "can you",
        "do you", "is there", "are there", "help",
    ],
}

# Intents that should NEVER show the ⚠️ warning
NO_WARNING_INTENTS = {"👋 Greeting", "🙏 Thanks", "👋 Farewell", "❓ General"}

def detect_intent(text: str):
    """Keyword-based intent classifier — zero RAM, instant."""
    text_lower = text.lower()

    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return intent, 0.85

    return "❓ General", 0.85
