"""
Researcher agent: executes searches, downloads papers, chunks, embeds, and summarizes.
"""
from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List

try:
    from pdfminer.high_level import extract_text  # type: ignore
except Exception:
    extract_text = None  # type: ignore

from ..config import Settings
from ..tools import search, downloader, vectorstore, summarizer


class ResearcherAgent:
    """
    Executes searches, downloads papers, chunks, embeds, and summarizes.
    """

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.embedder = vectorstore.EmbeddingsProvider(cfg)
        self.vector_store = vectorstore.VectorStore(cfg, embedder=self.embedder)

    def execute_plan(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Returns evidence list of paper summaries with metadata and relevance scores.
        Pipeline:
            search -> download -> chunk -> embed -> store -> summarize -> rerank.

        TODO:
        - Implement search fan-out across sources and deduplication.
        - Add PDF fetching/abstract fallback with retries.
        - Add chunking and embedding using configured model.
        - Persist embeddings and summaries to the vector store.
        - Rerank using semantic similarity or cross-encoder if enabled.
        """
        query_variants = plan.get("query_variants") or [plan.get("question", "")]
        max_results = plan.get("max_results") or self.cfg.max_results
        candidates = self._search_sources(query_variants, max_results)

        evidence: List[Dict[str, Any]] = []
        for paper in candidates:
            chunks = self._download_and_chunk(paper)
            self._embed_and_store(paper["id"], chunks, paper)
            summary = self._summarize(chunks, paper)
            evidence.append({"paper": paper, "summary": summary})

        # Basic rerank by vector similarity against the original question.
        reranked = self.vector_store.search(plan.get("question", ""), top_k=self.cfg.top_k)
        if reranked:
            # align evidence to rerank order
            rerank_ids = [r["id"] for r in reranked]
            evidence = sorted(
                evidence, key=lambda x: rerank_ids.index(x["paper"]["id"]) if x["paper"]["id"] in rerank_ids else 999
            )
        return evidence[: self.cfg.top_k]

    def _search_sources(self, query_variants: List[str], max_results: int) -> List[Dict[str, Any]]:
        """
        Execute searches across configured sources and merge/dedupe results.
        """
        seen = set()
        merged: List[Dict[str, Any]] = []
        sources = []
        if self.cfg.arxiv_enabled:
            sources.append("arxiv")
        sources.append("semanticscholar")

        for q in query_variants:
            for src in sources:
                results = search.search_papers(
                    q,
                    source=src,
                    max_results=max_results,
                    timeout=int(self.cfg.request_timeout_seconds),
                    semantic_scholar_key=self.cfg.semantic_scholar_key,
                )
                for r in results:
                    key = r.get("doi") or r.get("title")
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    merged.append(r)
        return merged

    def _download_and_chunk(self, paper: Dict[str, Any]) -> List[str]:
        """
        Download PDF or use abstract, then chunk text into manageable windows.
        """
        pdf_bytes = downloader.download_pdf(paper.get("url", ""), timeout=int(self.cfg.request_timeout_seconds))
        text = None
        if pdf_bytes and extract_text:
            try:
                text = extract_text(BytesIO(pdf_bytes))
            except Exception:
                text = None
        if not text:
            text = paper.get("abstract") or ""
        return _chunk_text(text)

    def _embed_and_store(self, paper_id: str, chunks: List[str], metadata: Dict[str, Any]) -> None:
        """
        Compute embeddings for chunks and upsert into the vector store.
        """
        records = []
        for idx, chunk in enumerate(chunks):
            emb = self.embedder.embed_text(chunk)
            records.append(
                {"id": f"{paper_id}:{idx}", "embedding": emb, "metadata": {**metadata, "chunk_id": idx, "text": chunk}}
            )
        self.vector_store.upsert(records)

    def _summarize(self, chunks: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize a paper into method/findings/limitations/citation structure.
        """
        return summarizer.summarize_paper(chunks, metadata, cfg=self.cfg)


def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 120) -> List[str]:
    """
    Simple character-based chunker to keep the prototype dependency-free.
    """
    if not text:
        return []
    chunks = []
    step = max_chars - overlap if overlap < max_chars else max_chars
    for start in range(0, len(text), step):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
    return chunks
