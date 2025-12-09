"""
LangChain-first agent with tool-calling. Uses LC tools and an OpenAI tool agent.
Falls back to an internal pipeline if agent creation fails.
"""
from __future__ import annotations

import math
from io import BytesIO
from typing import Any, Dict, List

from dotenv import load_dotenv

try:
    from pdfminer.high_level import extract_text  # type: ignore
except Exception:
    extract_text = None  # type: ignore

try:
    from langchain_community.document_loaders import PyPDFLoader  # type: ignore
except Exception:
    PyPDFLoader = None  # type: ignore

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
try:
    from langchain_community.vectorstores import FAISS  # type: ignore
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
except Exception:
    FAISS = None
    HuggingFaceEmbeddings = None

try:
    from langchain.agents import create_openai_tools_agent, AgentExecutor
except Exception:
    create_openai_tools_agent = None
    AgentExecutor = None

from autonomous_research_assistant.tools.search import search_papers
from autonomous_research_assistant.tools.downloader import download_pdf
from autonomous_research_assistant.tools.vectorstore import _hashed_embedding
from autonomous_research_assistant.config import load_settings


def _build_embedder(cfg):
    if HuggingFaceEmbeddings and cfg.embed_backend in ("auto", "sentence-transformers"):
        try:
            return HuggingFaceEmbeddings(
                model_name=cfg.embed_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception:
            pass

    class HashEmbeddings:
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [_hashed_embedding(t) for t in texts]

        def embed_query(self, text: str) -> List[float]:
            return _hashed_embedding(text)

    return HashEmbeddings()


def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 120) -> List[str]:
    if not text:
        return []
    chunks = []
    step = max_chars - overlap if overlap < max_chars else max_chars
    for start in range(0, len(text), step):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
    return chunks


def _ingest_papers(papers: List[Dict[str, Any]], cfg) -> List[Document]:
    docs: List[Document] = []
    for p in papers:
        url = p.get("url", "")
        # Prefer LC PDF loader if available and URL looks like PDF; else fallback to abstract.
        if PyPDFLoader and url.lower().endswith(".pdf"):
            try:
                loader = PyPDFLoader(url)
                loaded = loader.load()
                for idx, d in enumerate(loaded):
                    for jdx, chunk in enumerate(_chunk_text(d.page_content)):
                        docs.append(
                            Document(
                                page_content=chunk,
                                metadata={**p, "chunk_id": f"{idx}-{jdx}"},
                            )
                        )
                continue
            except Exception:
                pass

        pdf_bytes = download_pdf(url, timeout=int(cfg.request_timeout_seconds))
        text = None
        if pdf_bytes and extract_text:
            try:
                text = extract_text(BytesIO(pdf_bytes))
            except Exception:
                text = None
        if not text:
            text = p.get("abstract") or ""
        for idx, chunk in enumerate(_chunk_text(text)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={**p, "chunk_id": idx},
                )
            )
    return docs


def _llm_summary(llm: ChatOpenAI, doc: Document) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise research summarizer. Summarize key methods, findings, and limitations. Keep concise; do not invent citations.",
            ),
            ("human", "Title: {title}\n\nText:\n{content}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"title": doc.metadata.get("title", ""), "content": doc.page_content[:2000]})


def _llm_synthesis(llm: ChatOpenAI, question: str, summaries: List[str], citations: List[str]) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a concise, citation-aware literature reviewer. Do not invent sources.",
            ),
            (
                "human",
                "Question: {question}\n\nEvidence summaries:\n{summaries}\n\nCitations:\n{citations}\n\n"
                "Write a structured literature review with sections: Overview, Key Findings, Methods, Limitations, Gaps, Citations.",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(
        {
            "question": question,
            "summaries": "\n".join(f"- {s}" for s in summaries),
            "citations": "\n".join(citations) if citations else "None provided",
        }
    )


def _full_pipeline(question: str) -> str:
    cfg = load_settings()
    llm_plan = ChatOpenAI(model=cfg.llm_model, temperature=0.2)
    llm_summarizer = ChatOpenAI(model=cfg.llm_model, temperature=0.2)
    llm_writer = ChatOpenAI(model=cfg.llm_model, temperature=0.3)
    embedder = _build_embedder(cfg)

    # Plan queries
    plan_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate 1-2 search queries for papers; return newline-separated."),
            ("human", "{question}"),
        ]
    )
    queries_text = (plan_prompt | llm_plan | StrOutputParser()).invoke({"question": question})
    queries = [q.strip() for q in queries_text.split("\n") if q.strip()] or [question]

    # Search and dedupe
    seen = set()
    papers: List[Dict[str, Any]] = []
    for q in queries:
        hits = search_papers(q, source="arxiv", max_results=cfg.max_results)
        for h in hits:
            key = h.get("doi") or h.get("title")
            if key and key not in seen:
                seen.add(key)
                papers.append(h)
    papers = papers[: cfg.max_results]
    if not papers:
        return "No papers found."

    # Ingest (PDF/abstract -> chunks)
    docs = _ingest_papers(papers, cfg)
    if not docs:
        return "No content retrieved."

    # Build vector store or fallback to simple scoring
    if FAISS:
        vs = FAISS.from_documents(docs, embedder)
        retriever = vs.as_retriever(search_kwargs={"k": min(cfg.top_k, len(docs))})
        retrieved = retriever.invoke(question)
    else:
        q_vec = _hashed_embedding(question)
        scored = []
        for d in docs:
            v = _hashed_embedding(d.page_content)
            dot = sum(x * y for x, y in zip(q_vec, v))
            norm_q = math.sqrt(sum(x * x for x in q_vec)) or 1e-9
            norm_v = math.sqrt(sum(x * x for x in v)) or 1e-9
            score = dot / (norm_q * norm_v)
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        retrieved = [d for _, d in scored[: min(cfg.top_k, len(scored))]]

    # Summaries
    summaries = []
    citations = []
    for d in retrieved:
        summaries.append(_llm_summary(llm_summarizer, d))
        cite = d.metadata.get("title", "")
        if cite:
            citations.append(cite)

    # Final synthesis
    report = _llm_synthesis(llm_writer, question, summaries, citations)
    return report


@tool
def search_tool(query: str, source: str = "arxiv", max_results: int = 5) -> List[Dict[str, Any]]:
    """Search papers from ArXiv or Semantic Scholar."""
    return search_papers(query=query, source=source, max_results=max_results)


@tool
def rag_report(question: str) -> str:
    """Run the full RAG pipeline (search -> ingest -> embed -> retrieve -> summarize -> synthesize)."""
    return _full_pipeline(question)


def run_langchain(question: str) -> str:
    """
    LangChain tool-calling agent; falls back to the full pipeline if agent creation fails.
    """
    load_dotenv()
    cfg = load_settings()
    llm = ChatOpenAI(model=cfg.llm_model, temperature=0.3)
    tools = [search_tool, rag_report]

    if create_openai_tools_agent and AgentExecutor:
        system = (
            "You are a research agent. Use tools to search papers and produce a final literature review. "
            "Prefer calling 'rag_report' once to generate the final answer. Do not invent citations."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{input}"),
            ]
        )
        agent = create_openai_tools_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        result = executor.invoke({"input": question})
        return result.get("output", "")

    # Fallback to direct pipeline
    return _full_pipeline(question)


if __name__ == "__main__":
    print(run_langchain("graph neural networks for molecules"))
