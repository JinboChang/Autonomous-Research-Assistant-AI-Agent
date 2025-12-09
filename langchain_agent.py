"""
Optional LangChain wrapper: runs the classic agent and lets an LLM reframe it.
Uses LangChain 1.1+ runnable API (no deprecated initialize_agent).
"""
from __future__ import annotations

from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from autonomous_research_assistant.main import run as run_classic
from autonomous_research_assistant.config import load_settings


def _classic_with_report(inputs: Dict[str, Any]) -> Dict[str, Any]:
    question = inputs["question"]
    report = run_classic(question)
    return {"question": question, "report": report}


def run_langchain(question: str) -> str:
    """
    Execute the classic pipeline, then ask the LLM to reframe/shorten it.
    """
    load_dotenv()
    cfg = load_settings()
    llm = ChatOpenAI(model=cfg.llm_model, temperature=0.2)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a concise research assistant. Reframe the report, keep citations if present. Do not invent sources.",
            ),
            ("human", "Question: {question}\n\nReport:\n{report}"),
        ]
    )

    # Build a chain that injects the question, runs the classic pipeline, then reframes.
    chain = (
        RunnableLambda(lambda q: {"question": q, "report": run_classic(q)})
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(question)


if __name__ == "__main__":
    print(run_langchain("graph neural networks for molecules"))
