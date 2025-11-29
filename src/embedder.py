# src/embedder.py
import os, time, traceback
from typing import List
# Try to import new OpenAI client; if absent it'll fail and we fallback to local embeddings.
USE_OPENAI = bool(os.environ.get('OPENAI_API_KEY'))
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY')) if USE_OPENAI else None
except Exception:
    client = None
    USE_OPENAI = False

# caching helper
try:
    from src.cache_embed import get_cached_embedding, set_cached_embedding
except Exception:
    # fallback no-op cache
    def get_cached_embedding(text): return None
    def set_cached_embedding(text, emb): pass

import hashlib, numpy as np

EMBED_DIM = 384
EMBED_MODEL = "text-embedding-3-small"

def _text_to_seed_vector(text: str, dim: int = EMBED_DIM):
    if not text:
        return [0.0] * dim
    h = hashlib.sha256(text.encode('utf-8')).digest()
    out = []
    i = 0
    while len(out) < dim:
        i_bytes = (i).to_bytes(2, 'big')
        chunk = hashlib.sha256(h + i_bytes).digest()
        for b in chunk:
            out.append(b)
            if len(out) >= dim:
                break
        i += 1
    arr = np.array(out[:dim], dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()

def _call_openai_embedding(text: str):
    # new OpenAI client usage
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def get_embedding(text: str, max_retries: int = 3) -> List[float]:
    # check cache first
    cached = get_cached_embedding(text)
    if cached:
        return cached
    # Try OpenAI if available
    last_exc = None
    if USE_OPENAI and client is not None:
        for attempt in range(max_retries):
            try:
                emb = _call_openai_embedding(text)
                set_cached_embedding(text, emb)
                return emb
            except Exception as e:
                last_exc = e
                print(f"OpenAI embedding attempt {attempt+1} failed: {e}")
                traceback.print_exc()
                # if 429 (quota) or similar, break to fallback
                time.sleep(2 ** attempt)
        print("OpenAI embedding failed after retries, falling back to deterministic local embeddings.")
    # fallback deterministic
    emb = _text_to_seed_vector(text)
    set_cached_embedding(text, emb)
    return emb
