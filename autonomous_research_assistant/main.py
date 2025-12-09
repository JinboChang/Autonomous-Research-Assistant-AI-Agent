"""
Entry point for the Autonomous Research Assistant.
Orchestrates planner -> researcher -> writer sequence.
"""
from __future__ import annotations

import sys
from typing import Any

from .config import load_settings
from .agents.planner import PlannerAgent
from .agents.researcher import ResearcherAgent
from .agents.writer import WriterAgent


def run(question: str) -> Any:
    """
    Execute the end-to-end research workflow for a given question.
    Returns a structured report (format depends on configuration).
    """
    cfg = load_settings()
    planner = PlannerAgent(cfg)
    researcher = ResearcherAgent(cfg)
    writer = WriterAgent(cfg)

    plan = planner.create_plan(question)
    evidence = researcher.execute_plan(plan)
    report = writer.write_report(question, evidence)
    return report


if __name__ == "__main__":
    user_q = " ".join(sys.argv[1:]).strip()
    if not user_q:
        user_q = "What are recent advances in graph neural networks for molecules?"
    print(run(user_q))
