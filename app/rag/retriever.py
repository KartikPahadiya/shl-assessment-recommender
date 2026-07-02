import numpy as np
from app.rag.embeddings import embed_text
from app.rag.vectorstore import metadata, index
from app.rag.filter import metadata_filter
from app.rag.keyword_search import bm25_search

# Common stopwords to ignore when extracting query keywords
_STOPWORDS = {
    "need", "for", "the", "and", "with", "but", "can", "you", "please", "i", "a", "an", "that", "this", "those", "have", "has", "had", "would", "could", "should", "be", "was", "were", "is", "are", "been", "any", "one", "two", "three", "new"
}


def retrieve(query, constraints=None, k=5):
    if constraints is None:
        constraints = {}

    # Extract meaningful keywords from the query (length > 3, not stopwords)
    query_words = set(w.lower() for w in query.split() if len(w) > 3)
    keywords = [w for w in query_words if w not in _STOPWORDS]
    # Also keep single-word programming language terms like "C++", "R", "Go" (but Go is ambiguous)
    for w in query.split():
        if w.lower() in ("c++", "c#", "r"):
            keywords.append(w.lower())
    keywords = list(set(keywords))

    # --- Stage 1: FAISS semantic search (define the neighborhood) ---
    query_embedding = embed_text([query])
    query_embedding = np.array(query_embedding).astype("float32")
    search_k = min(50, len(metadata))
    scores, indices = index.search(query_embedding, search_k)

    faiss_candidates = []
    for score, idx in zip(scores[0], indices[0]):
        item = metadata[idx].copy()
        item["vector_score"] = float(score)
        faiss_candidates.append(item)

    # --- Stage 2: Post-filter — keep only candidates matching at least one keyword ---
    if keywords:
        filtered = []
        for item in faiss_candidates:
            name = item.get("name", "").lower()
            desc = item.get("description", "").lower()
            if any(kw in name or kw in desc for kw in keywords):
                filtered.append(item)
        # If filter is too aggressive, fall back to top N unfiltered
        if len(filtered) >= k:
            candidates = filtered
        else:
            # Keep filtered + fill from unfiltered up to 50
            seen = {id(item) for item in filtered}
            for item in faiss_candidates:
                if id(item) not in seen:
                    filtered.append(item)
            candidates = filtered
    else:
        candidates = faiss_candidates

    # --- Stage 3: Apply constraints ---
    candidates = metadata_filter(candidates, constraints)
    if len(candidates) == 0:
        candidates = faiss_candidates

    # --- Stage 4: BM25 keyword scoring on candidates ---
    keyword_scores = np.array(bm25_search(query, candidates))
    vector_scores = np.array([item["vector_score"] for item in candidates])

    # --- Stage 5: Normalize ---
    vector_scores = (vector_scores - vector_scores.min()) / (
        vector_scores.max() - vector_scores.min() + 1e-8
    )
    keyword_scores = (keyword_scores - keyword_scores.min()) / (
        keyword_scores.max() - keyword_scores.min() + 1e-8
    )

    # --- Stage 6: Hybrid — 90% semantic, 10% keyword ---
    final_scores = 0.9 * vector_scores + 0.1 * keyword_scores

    for i, item in enumerate(candidates):
        item["score"] = float(final_scores[i])

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:k]
