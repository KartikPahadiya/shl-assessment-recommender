import numpy as np
from app.rag.embeddings import embed_text
from app.rag.vectorstore import metadata, index
from app.rag.filter import metadata_filter
from app.rag.keyword_search import bm25_search


def retrieve(query, constraints=None, k=5):
    if constraints is None:
        constraints = {}

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

    # --- Stage 2: Apply constraints ---
    candidates = metadata_filter(faiss_candidates, constraints)
    if len(candidates) == 0:
        candidates = faiss_candidates

    # --- Stage 3: BM25 keyword scoring on FAISS candidates ---
    keyword_scores = np.array(bm25_search(query, candidates))
    vector_scores = np.array([item["vector_score"] for item in candidates])

    # --- Stage 4: Normalize ---
    vector_scores = (vector_scores - vector_scores.min()) / (
        vector_scores.max() - vector_scores.min() + 1e-8
    )
    keyword_scores = (keyword_scores - keyword_scores.min()) / (
        keyword_scores.max() - keyword_scores.min() + 1e-8
    )

    # --- Stage 5: Hybrid — 90% semantic, 10% keyword ---
    final_scores = 0.9 * vector_scores + 0.1 * keyword_scores

    for i, item in enumerate(candidates):
        item["score"] = float(final_scores[i])

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:k]
