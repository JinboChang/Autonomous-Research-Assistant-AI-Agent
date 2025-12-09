"""
Summarization tool using an LLM.
"""
from __future__ import annotations

from typing import Dict, List

from .citation import format_citation
from .llm import LLMClient, extract_json_block
from ..config import Settings


def summarize_paper(chunks: List[str], metadata: Dict, cfg: Settings | None = None) -> Dict:
    """
    Summarizes a paper into method/findings/limitations/citation.
    """
    cfg = cfg or Settings()
    llm = LLMClient(cfg)
    text = " ".join(chunks) if chunks else metadata.get("abstract", "") or ""
    prompt = (
        "Summarize the paper in JSON with keys: method, findings, limitations. "
        "Keep it concise and factual. Use the provided context.\n\n"
        f"Title: {metadata.get('title','')}\n\n"
        f"Context:\n{text[:4000]}"
    )
    raw = llm.generate(prompt, system_prompt="You are a precise research summarizer.", max_tokens=400, temperature=0.2)
    parsed = extract_json_block(raw) or {}
    lead = (text or "").split(".")
    fallback = ". ".join(lead[:2]).strip() or "Summary unavailable."
    summary = {
        "title": metadata.get("title", "Unknown title"),
        "method": parsed.get("method") or fallback[:200],
        "findings": parsed.get("findings") or f"Key points relate to {metadata.get('title', 'the topic')}.",
        "limitations": parsed.get("limitations") or "Limitations not fully assessed.",
        "citation": format_citation(metadata, style="APA"),
        "relevance_score": 0.5,
    }
    return summary
