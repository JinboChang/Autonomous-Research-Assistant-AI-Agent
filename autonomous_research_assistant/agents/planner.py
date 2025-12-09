"""
Planner agent: derives intent, expands queries, and produces an execution plan.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..config import Settings
from ..tools.llm import LLMClient, extract_json_block


class PlannerAgent:
    """
    Derives intent, expands queries, and builds an execution plan.
    """

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        # TODO: inject LLM client and prompt templates for intent parsing.

    def create_plan(self, question: str) -> Dict[str, Any]:
        """
        Returns a plan dict:
        {
            "intent": {...},
            "query_variants": [str],
            "filters": {...},
            "max_results": int
        }

        TODO:
        - Implement intent extraction (domain, scope, time range).
        - Generate multiple query rewrites for robustness.
        - Attach filter hints (year, venue) based on heuristics/config.
        """
        llm = LLMClient(self.cfg)
        prompt = (
            "Given a research question, propose 2-3 search query variants, "
            "domain hints, and optional year filters. Respond in JSON with keys: "
            '{"query_variants": [...], "domain": "text", "year": number|null}. '
            f"Question: {question}"
        )
        raw = llm.generate(prompt, system_prompt="You create search plans for papers.", max_tokens=200)
        parsed = extract_json_block(raw) or {}

        intent = {
            "question": question,
            "domain": parsed.get("domain", "general"),
            "time_range": parsed.get("year") or self.cfg.year_filter,
        }
        query_variants = parsed.get("query_variants") or self._rewrite_queries(question)
        filters = self._infer_filters(question)
        if parsed.get("year"):
            filters["year"] = parsed["year"]
        return {
            "intent": intent,
            "query_variants": query_variants,
            "filters": filters,
            "max_results": self.cfg.max_results,
            "question": question,
        }

    def _rewrite_queries(self, question: str) -> List[str]:
        """
        Expand the user question into multiple search-friendly queries.
        TODO: call LLM or rule-based expansion.
        """
        base = question.strip()
        if not base:
            return ["latest research"]
        tokens = [t for t in base.split() if len(t) > 3]
        expanded = f"{base} review"
        return [base, expanded, " ".join(tokens)]

    def _infer_filters(self, question: str) -> Dict[str, Any]:
        """
        Infer filters such as year or domain from the question.
        TODO: add heuristics; fallback to config defaults.
        """
        filters: Dict[str, Any] = {}
        if self.cfg.year_filter:
            filters["year"] = self.cfg.year_filter
        if "recent" in question.lower():
            filters["year"] = filters.get("year") or 2021
        return filters
