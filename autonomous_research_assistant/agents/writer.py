"""
Writer agent: synthesizes final literature review from evidence.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..config import Settings
from ..tools.llm import LLMClient


class WriterAgent:
    """
    Synthesizes the final literature review from evidence and performs self-checks.
    """

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        # TODO: inject LLM client, citation formatter, and vector store handle.

    def write_report(self, question: str, evidence: List[Dict[str, Any]]) -> str:
        """
        Returns Markdown/JSON report with sections:
        - question
        - method
        - findings
        - gaps
        - limitations
        - citations

        TODO:
        - Implement RAG retrieval for richer context.
        - Add self-evaluation checklist to reduce hallucinations.
        - Support both Markdown and JSON output modes.
        """
        llm = LLMClient(self.cfg)
        summary_points: List[str] = []
        citation_lines: List[str] = []
        for item in evidence:
            paper = item.get("paper", {})
            summ = item.get("summary", {})
            summary_points.append(
                f"- {paper.get('title','Untitled')} ({paper.get('year','n.d.')}): "
                f"{summ.get('findings','')} Limitations: {summ.get('limitations','')}"
            )
            citation_str = summ.get("citation") or paper.get("title")
            if citation_str:
                citation_lines.append(f"- {citation_str}")

        context = "\n".join(summary_points)
        citations_block = "\n".join(citation_lines) if citation_lines else "None provided."

        system = "You write concise, citation-aware literature reviews. Avoid speculation."
        prompt = (
            f"Question: {question}\n\n"
            "Evidence:\n"
            f"{context}\n\n"
            "Use these citations exactly (do not invent new ones):\n"
            f"{citations_block}\n\n"
            "Write a structured literature review with sections: Overview, Key Findings, Methods, Limitations, Gaps, Citations. "
            "Ensure the Citations section lists the provided citations."
        )
        draft = llm.generate(prompt, system_prompt=system, max_tokens=600, temperature=0.3)

        if self.cfg.output_format == "json":
            import json

            return json.dumps({"question": question, "report": draft, "papers": evidence}, indent=2)

        # Append a deterministic citations section to guarantee inclusion.
        if citation_lines and "Citations" not in draft:
            draft = draft.rstrip() + "\n\n## Citations\n" + "\n".join(citation_lines)

        refined = self._self_check(draft, evidence)
        return refined

    def _self_check(self, report: str, evidence: List[Dict[str, Any]]) -> str:
        """
        Run a lightweight self-evaluation and refinement on the draft report.
        """
        # Simple check: ensure at least one citation is present.
        if "Citations" not in report and evidence:
            report += "\n\n_Caution: citations missing in this draft._"
        return report
