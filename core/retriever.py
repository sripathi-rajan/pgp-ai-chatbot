import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def hybrid_retrieve(query, db, bm25, texts, embeddings, k=5):
    semantic_results = db.similarity_search_with_relevance_scores(query, k=k*2)
    semantic_docs = [r[0].page_content for r in semantic_results]
    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:k*2]
    bm25_docs = [texts[i] for i in bm25_top_idx]
    rrf_scores = {}
    for rank, doc in enumerate(semantic_docs):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (60 + rank)
    for rank, doc in enumerate(bm25_docs):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (60 + rank)
    sorted_docs = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
    return sorted_docs[:k]

def get_best_sentence(query, chunk_text, embeddings):
    sentences = [
        s.strip() for s in re.split(r'[.\n]', chunk_text)
        if len(s.strip()) > 40
    ]
    if not sentences:
        return chunk_text[:200], 0.0
    query_vec = embeddings.embed_query(query)
    sentence_vecs = embeddings.embed_documents(sentences)
    sims = cosine_similarity([query_vec], sentence_vecs)[0]
    best_idx = int(np.argmax(sims))
    return sentences[best_idx], float(sims[best_idx])