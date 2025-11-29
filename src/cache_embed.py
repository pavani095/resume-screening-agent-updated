# src/cache_embed.py
import os, pickle, hashlib
CACHE_FILE = os.path.join(os.path.dirname(__file__), '..', '.embed_cache.pkl')
CACHE_FILE = os.path.abspath(CACHE_FILE)

def _key_for_text(text: str):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache):
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f)
    except Exception:
        pass

_cache = load_cache()

def get_cached_embedding(text):
    k = _key_for_text(text)
    return _cache.get(k)

def set_cached_embedding(text, emb):
    k = _key_for_text(text)
    _cache[k] = emb
    save_cache(_cache)
