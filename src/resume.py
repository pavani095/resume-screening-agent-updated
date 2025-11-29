# src/ranker.py
from typing import List, Any
import numpy as np
import re

def _cosine_sim(a: Any, b: Any) -> float:
    """
    Compute cosine similarity between two vectors.
    Ensure both are 1-D numpy arrays (flattened).
    Returns 0.0 if shapes are incompatible.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()

    if a.size == 0 or b.size == 0:
        return 0.0
    if a.shape[0] != b.shape[0]:
        # incompatible sizes
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _has_embedding(emb):
    """
    Return (bool, np.array or None) — whether embedding is present and the numpy array.
    Handles lists, tuples, 2D arrays (converts to flat 1D), or None.
    """
    if emb is None:
        return False, None
    try:
        arr = np.asarray(emb, dtype=float).ravel()
        if arr.size == 0:
            return False, None
        return True, arr
    except Exception:
        return False, None

def score_and_rank(jd_embedding: List[float], vector_store, top_k: int = 5):
    """
    Query vector store and re-rank using cosine similarity when embeddings are available.
    Adds a small boost for years of experience found in the resume text.
    """
    # Overfetch to have more candidates to re-rank
    results = vector_store.query(query_embedding=jd_embedding, top_k=top_k * 5)
    scored = []

    # prepare jd embedding as numpy array once
    jd_has, jd_arr = _has_embedding(jd_embedding)

    for r in results:
        text = (r.get('metadata') or {}).get('text', '') or r.get('document', '') or ''
        years_boost = 0.0
        m = re.search(r"(\d{1,2})\+?\s+years", text.lower())
        if m:
            try:
                years = int(m.group(1))
                years_boost = min(0.2, years * 0.01)
            except:
                years_boost = 0.0

        emb = r.get('embedding', None)
        has_emb, emb_arr = _has_embedding(emb)

        if has_emb and jd_has:
            sim = _cosine_sim(jd_arr, emb_arr)
            combined = sim + years_boost
        else:
            # fallback to use any score returned by the vector store (may be None)
            base_score = r.get('score', 0.0) or 0.0
            combined = base_score + years_boost

        scored.append({
            'id': r.get('id'),
            'score': combined,
            'metadata': r.get('metadata', {}),
            'document': r.get('document', '')
        })

    scored = sorted(scored, key=lambda x: x['score'], reverse=True)
    return scored[:top_k]
