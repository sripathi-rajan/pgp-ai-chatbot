import streamlit as st
from transformers import pipeline

# Initialize classifier (cached)
@st.cache_resource
def get_classifier():
    return pipeline("zero-shot-classification", 
                    model="facebook/bart-large-mnli", 
                    device="cpu")  # Use "cuda" if GPU available

classifier = get_classifier()
candidate_labels = ["Fees", "Curriculum", "Admissions", "Career Outcomes", "Program Overview"]

def detect_intent(query):
    """
    Zero-shot intent classification using BART-MNLI.
    More accurate than keyword matching.
    """
    result = classifier(query, candidate_labels, multi_label=False)
    intent_label = result["labels"][0]
    confidence = result["scores"][0]
    
    # Map to emoji format
    intent_map = {
        "Fees": "💰 Fees",
        "Curriculum": "📚 Curriculum", 
        "Admissions": "📋 Admissions",
        "Career Outcomes": "🚀 Career",
        "Program Overview": "🎓 Overview"
    }
    
    intent = intent_map.get(intent_label, "🎓 Overview")
    return intent, confidence