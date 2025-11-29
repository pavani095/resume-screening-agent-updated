# src/vector_store.py
from typing import Dict, List
import chromadb

client = chromadb.Client()

class LocalVectorStore:
    def __init__(self, collection_name: str = "default"):
        try:
            self.collection = client.get_collection(collection_name)
        except Exception:
            self.collection = client.create_collection(collection_name)

    def upsert(self, id: str, embedding: List[float], metadata: Dict = None):
        if not embedding:
            return
        doc_text = metadata.get("text", "") if metadata else ""
        try:
            self.collection.add(
                ids=[id],
                embeddings=[embedding],
                metadatas=[metadata or {}],
                documents=[doc_text],
            )
        except TypeError:
            self.collection.add(
                documents=[doc_text],
                metadatas=[metadata or {}],
                ids=[id],
                embeddings=[embedding],
            )

    def query(self, query_embedding: List[float], top_k: int = 5):
        if not query_embedding:
            return []

        include_fields = ["documents", "metadatas", "distances", "embeddings"]

        try:
            res = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=include_fields
            )
        except TypeError:
            res = self.collection.query(
                query_embedding=[query_embedding],
                n_results=top_k,
                include=include_fields
            )

        results = []
        ids = None
        distances = None
        metadatas = None
        documents = None
        embeddings = None

        if isinstance(res, dict):
            ids = res.get("ids", None)
            distances = res.get("distances", None)
            metadatas = res.get("metadatas", None)
            documents = res.get("documents", None)
            embeddings = res.get("embeddings", None)
        else:
            ids = getattr(res, "ids", None)
            distances = getattr(res, "distances", None)
            metadatas = getattr(res, "metadatas", None)
            documents = getattr(res, "documents", None)
            embeddings = getattr(res, "embeddings", None)

        id_list = ids[0] if ids and isinstance(ids[0], list) else (ids or [])
        dist_list = distances[0] if distances and isinstance(distances[0], list) else (distances or [])
        meta_list = metadatas[0] if metadatas and isinstance(metadatas[0], list) else (metadatas or [])
        doc_list = documents[0] if documents and isinstance(documents[0], list) else (documents or [])
        emb_list = embeddings[0] if embeddings and isinstance(embeddings[0], list) else (embeddings or [])

        # Build results with embedding if present
        for idx in range(min(len(doc_list) or top_k, top_k)):
            rid = id_list[idx] if id_list and idx < len(id_list) else None
            distance = dist_list[idx] if idx < len(dist_list) else None
            score = 1 - distance if (isinstance(distance, (int, float))) else 0.0
            metadata = meta_list[idx] if meta_list and idx < len(meta_list) else {}
            document = doc_list[idx] if doc_list and idx < len(doc_list) else ""
            embedding = emb_list[idx] if emb_list and idx < len(emb_list) else None
            results.append({
                "id": rid or (metadata.get("id") if isinstance(metadata, dict) else None),
                "score": score,
                "metadata": metadata or {},
                "document": document or "",
                "embedding": embedding
            })

        # Fallback: if nothing found but metadatas/documents exist
        if not results and (doc_list or meta_list):
            length = max(len(doc_list), len(meta_list))
            for idx in range(min(length, top_k)):
                metadata = meta_list[idx] if meta_list and idx < len(meta_list) else {}
                document = doc_list[idx] if doc_list and idx < len(doc_list) else ""
                results.append({
                    "id": metadata.get("id") if isinstance(metadata, dict) else None,
                    "score": 0.0,
                    "metadata": metadata or {},
                    "document": document or "",
                    "embedding": None
                })
        return results

    def clear(self):
        try:
            self.collection.delete()
        except Exception:
            pass
