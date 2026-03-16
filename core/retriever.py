import json
import re
import numpy as np
from pathlib import Path
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


# ─── PDF CHUNK INDEXING HELPERS ───────────────────────────────────────────────
# Default path for the manifest that tracks which PDF chunk IDs are already
# in the FAISS index. Keeping it inside faiss_index/ means it travels with the
# index when the directory is copied or archived.
_DEFAULT_MANIFEST = "faiss_index/pdf_manifest.json"


def load_pdf_manifest(manifest_path: str = _DEFAULT_MANIFEST) -> set:
    """
    Return the set of chunk IDs that have already been indexed into FAISS.
    Returns an empty set if the manifest file doesn't exist yet.
    """
    p = Path(manifest_path)
    if p.exists():
        try:
            return set(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            return set()
    return set()


def save_pdf_manifest(chunk_ids: set, manifest_path: str = _DEFAULT_MANIFEST):
    """Persist the full set of indexed chunk IDs to the manifest JSON file."""
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(sorted(chunk_ids), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def index_pdf_chunks_to_faiss(
    chunks,                              # list[Document] from ingest_pdf_data()
    embeddings,                          # HuggingFaceEmbeddings (or compatible)
    faiss_index_path: str = "./faiss_index",
    manifest_path: str = _DEFAULT_MANIFEST,
) -> tuple:
    """
    Add PDF chunks to the FAISS index, skipping any chunk whose chunk_id is
    already recorded in the manifest file (incremental / idempotent).

    If the index does not yet exist, a fresh one is built from the supplied
    chunks and the sentinel file (.built_by_app) is written so that
    pipeline.py treats it as trusted.

    Returns (added: int, skipped: int).
    """
    # Lazy import — keeps retriever.py importable without langchain installed
    from langchain_community.vectorstores import FAISS

    # Load the set of already-indexed chunk IDs to skip duplicates
    indexed_ids = load_pdf_manifest(manifest_path)

    # Filter out chunks that have already been indexed
    new_chunks = [
        c for c in chunks
        if c.metadata.get("chunk_id", "") not in indexed_ids
    ]
    skipped = len(chunks) - len(new_chunks)

    if not new_chunks:
        print(f"[FAISS] All {len(chunks)} PDF chunk(s) already indexed — nothing to do")
        return 0, skipped

    index_path   = Path(faiss_index_path)
    sentinel     = index_path / ".built_by_app"

    if index_path.exists() and sentinel.exists():
        # Merge new chunks into the existing trusted index
        existing_db = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,   # safe: sentinel confirms our origin
        )
        new_db = FAISS.from_documents(new_chunks, embeddings)
        existing_db.merge_from(new_db)              # in-place merge
        existing_db.save_local(str(index_path))
        print(f"[FAISS] Merged {len(new_chunks)} new PDF chunk(s) into existing index")
    else:
        # No existing index — build a fresh one from these chunks only
        db = FAISS.from_documents(new_chunks, embeddings)
        db.save_local(str(index_path))
        sentinel.touch()    # mark the index as trusted for pipeline.py
        print(f"[FAISS] Built new FAISS index with {len(new_chunks)} PDF chunk(s)")

    # Persist updated manifest so next run knows what's already indexed
    new_ids = {c.metadata["chunk_id"] for c in new_chunks if "chunk_id" in c.metadata}
    indexed_ids.update(new_ids)
    save_pdf_manifest(indexed_ids, manifest_path)

    return len(new_chunks), skipped


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