"""
Vector store abstraction over FAISS or Chroma with embedding utilities.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

from ..config import Settings


class EmbeddingsProvider:
    """
    Provides embeddings via sentence-transformers when available, else hash.
    """

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.backend = cfg.embed_backend
        self.model_name = cfg.embed_model
        self._model = None
        if self.backend in ("auto", "sentence-transformers"):
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(self.model_name)
                self.backend = "sentence-transformers"
            except Exception:
                self.backend = "hash"

    def embed_text(self, text: str) -> List[float]:
        if self.backend == "sentence-transformers" and self._model:
            vec = self._model.encode(text, show_progress_bar=False, normalize_embeddings=True)
            return vec.tolist()
        return _hashed_embedding(text)

    @property
    def dim(self) -> int:
        if self.backend == "sentence-transformers" and self._model:
            try:
                return int(self._model.get_sentence_embedding_dimension())  # type: ignore
            except Exception:
                return len(self.embed_text("dimension probe"))
        return 256


class VectorStore:
    """
    Thin wrapper over FAISS/Chroma for upsert and similarity search.
    Falls back to in-memory store if backends are unavailable.
    """

    def __init__(self, cfg: Settings, embedder: Optional[EmbeddingsProvider] = None) -> None:
        self.cfg = cfg
        self.embedder = embedder or EmbeddingsProvider(cfg)
        self.backend_name = cfg.vector_store
        self._faiss_index = None
        self._faiss_ids: List[str] = []
        self._id_to_meta: Dict[str, Dict[str, Any]] = {}
        self._chroma = None
        self._chroma_collection = None
        self._in_memory: List[Dict[str, Any]] = []
        self._init_backend()

    def _init_backend(self) -> None:
        if self.backend_name == "faiss" and np is not None:
            try:
                import faiss  # type: ignore

                self._faiss = faiss
                dim = self.embedder.dim
                self._faiss_index = faiss.IndexFlatIP(dim)
                return
            except Exception:
                self.backend_name = "memory"
        if self.backend_name == "chroma":
            try:
                import chromadb  # type: ignore

                client = chromadb.PersistentClient(path="data/vector_store")
                self._chroma = client
                self._chroma_collection = client.get_or_create_collection("papers")
                return
            except Exception:
                self.backend_name = "memory"

    def upsert(self, records: List[Dict[str, Any]]) -> None:
        """
        Upsert embedding records into the vector store.

        Expected record schema:
        { "id": str, "embedding": List[float], "metadata": dict }
        """
        if not records:
            return
        if self.backend_name == "faiss" and self._faiss_index is not None:
            self._upsert_faiss(records)
            return
        if self.backend_name == "chroma" and self._chroma_collection is not None:
            self._upsert_chroma(records)
            return
        self._upsert_memory(records)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Return top_k hits with scores and metadata.
        """
        if self.backend_name == "faiss" and self._faiss_index is not None:
            return self._search_faiss(query, top_k)
        if self.backend_name == "chroma" and self._chroma_collection is not None:
            return self._search_chroma(query, top_k)
        return self._search_memory(query, top_k)

    def _upsert_faiss(self, records: List[Dict[str, Any]]) -> None:
        # For simplicity, append-only; dedupe by rebuild.
        ids, vecs = [], []
        for rec in records:
            ids.append(rec["id"])
            vecs.append(rec["embedding"])
            self._id_to_meta[rec["id"]] = rec["metadata"]
        mat = np.array(vecs, dtype="float32")
        self._faiss_index.add(mat)
        self._faiss_ids.extend(ids)

    def _search_faiss(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        q_vec = np.array([self.embedder.embed_text(query)], dtype="float32")
        scores, idxs = self._faiss_index.search(q_vec, top_k)
        hits = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self._faiss_ids):
                continue
            pid = self._faiss_ids[idx]
            hits.append({"id": pid, "score": float(score), "metadata": self._id_to_meta.get(pid, {})})
        return hits

    def _upsert_chroma(self, records: List[Dict[str, Any]]) -> None:
        ids = [r["id"] for r in records]
        embeddings = [r["embedding"] for r in records]
        metadatas = [r["metadata"] for r in records]
        documents = [r["metadata"].get("text", "") for r in records]
        self._chroma_collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def _search_chroma(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        res = self._chroma_collection.query(query_embeddings=[self.embedder.embed_text(query)], n_results=top_k)
        hits = []
        for ids, scores, metas in zip(res.get("ids", [[]])[0], res.get("distances", [[]])[0], res.get("metadatas", [[]])[0]):
            hits.append({"id": ids, "score": float(scores), "metadata": metas})
        return hits

    def _upsert_memory(self, records: List[Dict[str, Any]]) -> None:
        existing_ids = {r["id"] for r in self._in_memory}
        for rec in records:
            if rec["id"] in existing_ids:
                self._in_memory = [r for r in self._in_memory if r["id"] != rec["id"]]
            self._in_memory.append(rec)

    def _search_memory(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not self._in_memory:
            return []
        q_emb = self.embedder.embed_text(query)
        scored = []
        for rec in self._in_memory:
            score = _cosine_similarity(q_emb, rec["embedding"])
            scored.append({"score": score, **rec})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


def _hashed_embedding(text: str, dim: int = 256) -> List[float]:
    """
    Lightweight hash-based embedding to keep the demo dependency-free.
    """
    vec = [0] * dim
    tokens = text.lower().split()
    for tok in tokens:
        idx = hash(tok) % dim
        vec[idx] += 1
    return vec


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    norm_b = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (norm_a * norm_b)
