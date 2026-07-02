import numpy as np
from app.rag.embeddings import embed_text
from app.rag.vectorstore import metadata, index
from app.rag.filter import metadata_filter
from app.rag.keyword_search import bm25_search


def retrieve(query, constraints=None, k=5):
    if constraints is None:
        constraints = {}

    # --- Stage 1: Keyword pre-filter (BM25 on ALL metadata) ---
    # This prevents FAISS from pulling in semantically-similar-but-wrong items
    all_keyword_scores = bm25_search(query, metadata)

    # Keep top N by BM25 for further semantic re-ranking
    # N = 3*k ensures enough candidates while filtering out noise
    bm25_threshold_idx = max(3 * k, 20)
    bm25_threshold = sorted(all_keyword_scores, reverse=True)[min(bm25_threshold_idx, len(all_keyword_scores) - 1)]

    keyword_candidates = []
    for item, score in zip(metadata, all_keyword_scores):
        if score >= bm25_threshold or score > 0:
            item_copy = item.copy()
            item_copy["keyword_score"] = float(score)
            keyword_candidates.append(item_copy)

    # If keyword filter is too aggressive, fall back to all metadata
    if len(keyword_candidates) < k:
        keyword_candidates = [item.copy() for item in metadata]
        for item, score in zip(keyword_candidates, all_keyword_scores):
            item["keyword_score"] = float(score)

    # --- Stage 2: Semantic re-rank (FAISS on keyword-filtered subset) ---
    query_embedding = embed_text([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # FAISS search on the full index to get vector scores for candidates
    search_k = min(50, len(metadata))
    scores, indices = index.search(query_embedding, search_k)

    # Build a lookup of vector scores by index
    vector_score_map = {}
    for score, idx in zip(scores[0], indices[0]):
        vector_score_map[int(idx)] = float(score)

    # Assign vector scores to keyword candidates
    for item in keyword_candidates:
        idx = item.get("_idx")
        if idx is not None and idx in vector_score_map:
            item["vector_score"] = vector_score_map[idx]
        else:
            item["vector_score"] = 0.0

    # --- Stage 3: Apply constraints ---
    candidates = metadata_filter(keyword_candidates, constraints)
    if len(candidates) == 0:
        candidates = keyword_candidates

    # --- Stage 4: Normalize and combine ---
    vector_scores = np.array([item["vector_score"] for item in candidates])
    keyword_scores = np.array([item["keyword_score"] for item in candidates])

    # Normalize
    vector_scores = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min() + 1e-8)
    keyword_scores = (keyword_scores - keyword_scores.min()) / (keyword_scores.max() - keyword_scores.min() + 1e-8)

    # Hybrid: BM25 dominates (0.75), FAISS refines (0.25)
    final_scores = 0.25 * vector_scores + 0.75 * keyword_scores

    for i, item in enumerate(candidates):
        item["score"] = float(final_scores[i])

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:k]
