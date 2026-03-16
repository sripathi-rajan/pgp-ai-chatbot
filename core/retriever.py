import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

QUERY_EXPANSIONS = {
    "manager":           "working professional experience eligibility",
    "non-tech":          "non-technical background STEM eligibility criteria",
    "non technical":     "non-technical background STEM eligibility criteria",
    "arts background":   "non-STEM background eligibility analytical aptitude",
    "commerce":          "non-technical background eligibility",
    "mba":               "working professional management eligibility",
    "fresher":           "recent graduate entry level eligibility",
    "worth it":          "career outcomes placement salary benefits",
    "good program":      "program overview benefits outcomes rankings",
    "suitable for me":   "eligibility criteria target participants",
}

def expand_query(query: str) -> str:
    q = query.lower()
    expansions = []
    for trigger, expansion in QUERY_EXPANSIONS.items():
        if trigger in q:
            expansions.append(expansion)
    if expansions:
        return query + " " + " ".join(expansions)
    return query

def hybrid_retrieve(query, db, bm25, texts, embeddings, k=5):
    query = expand_query(query)
    semantic_results = db.similarity_search_with_relevance_scores(query, k=k*2)
    semantic_docs = [r[0].page_content for r in semantic_results]

    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:k*2]
    bm25_docs = [texts[i] for i in bm25_top_idx]

    # Ordered dedup merge: semantic results take priority, BM25 fills remaining slots
    seen = set()
    merged = []
    for doc in semantic_docs + bm25_docs:
        if doc not in seen:
            seen.add(doc)
            merged.append(doc)
        if len(merged) >= k:
            break
    return merged

def broad_retrieve(query, db, bm25, texts, embeddings, chunks_per_topic=2):
    """
    For broad/overview queries — fetch chunks across all topic areas
    instead of just the top-k most similar.
    """
    topic_queries = [
        "program fee cost scholarship installment",
        "curriculum topics modules terms syllabus",
        "eligibility admission criteria apply selection",
        "career placement salary companies hiring",
        "faculty mentor professor instructor",
        "program duration format online offline schedule",
    ]
    seen = set()
    all_docs = []
    for topic_query in topic_queries:
        try:
            results = db.similarity_search(topic_query, k=chunks_per_topic)
            for doc in results:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    all_docs.append(doc.page_content)
        except Exception as e:
            print(f"[BROAD] Topic query failed: {e}")
    return all_docs


def get_best_sentence(chunk_text, embeddings, query_vec):
    """Find the most relevant sentence in a chunk using a pre-computed query vector."""
    sentences = [
        s.strip() for s in re.split(r'[.\n]', chunk_text)
        if len(s.strip()) > 40
    ]
    if not sentences:
        return chunk_text[:200], 0.0
    sentence_vecs = embeddings.embed_documents(sentences)
    sims = cosine_similarity([query_vec], sentence_vecs)[0]
    best_idx = int(np.argmax(sims))
    return sentences[best_idx], float(sims[best_idx])