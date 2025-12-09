# Autonomous Research Assistant

An AI-powered literature review agent that searches ArXiv & Semantic Scholar, downloads/abstracts papers, embeds and reranks them, summarizes with an LLM, and produces a structured report with citations. Includes a FastAPI backend and a simple chat-style frontend.

## Features
- Query planning (rewrites + filters) via LLM (stub fallback for offline/dev).
- Multi-source search (ArXiv, Semantic Scholar) with merge/dedupe.
- PDF/abstract ingestion, chunking, embeddings (sentence-transformers by default, hash fallback).
- Vector search (FAISS/Chroma, fallback to in-memory).
- Per-paper LLM summarization + structured synthesis (Overview, Key Findings, Methods, Limitations, Gaps, Citations).
- Configurable output: Markdown (default) or JSON.
- FastAPI endpoint `/run` and a minimal web UI served at `/`.

## Setup
```bash
pip install -r requirements.txt
```

### Environment variables
- Copy `.env.example` to `.env` and set:
  ```
  OPENAI_API_KEY=your_key_here
  # SEMANTIC_SCHOLAR_KEY=optional
  ```
- Do **not** include quotes. The server loads `.env` automatically via `python-dotenv`.

## Running
- CLI (classic pipeline one-off):
  ```bash
  python -c "from autonomous_research_assistant.main import run; print(run('graph neural networks for molecules'))"
  ```
- API + web UI (LangChain-first by default):
  ```bash
  uvicorn app:app --reload --port 8000
  ```
  Then open http://127.0.0.1:8000 (chat UI). The `/run` endpoint uses the LangChain pipeline in `lc_agent.py`; if LangChain import fails it falls back to the classic pipeline.

## Configuration (autonomous_research_assistant/config.py)
- `llm_provider`: `"openai"` or `"stub"`.
- `llm_model`: e.g., `gpt-4o-mini`.
- `embed_backend`: `"auto"` (sentence-transformers), `"hash"` (fast/local), or `"sentence-transformers"`.
- `vector_store`: `"faiss"`, `"chroma"`, or `"memory"`.
- `max_results`, `top_k`: retrieval size and final cut.
- `arxiv_enabled`: toggle ArXiv search.
- `output_format`: `"md"` or `"json"`.

## How it works (LangChain pipeline)
1. Planner: LLM generates 1?2 search queries.
2. Search: ArXiv + Semantic Scholar (HTTP) via LC tool ? merge/dedupe.
3. Ingest: Download PDF if available (pdfminer), else abstract; chunk text.
4. Embed & store: sentence-transformers (HF) ? FAISS retriever (fallback to hash + cosine if FAISS missing).
5. Retrieve: similarity search on the question to select top-k chunks.
6. Summarize: LLM per chunk ? evidence summaries.
7. Synthesize: LLM composes structured review; citations appended.
8. Fallback: classic pipeline retained if LC tool agent creation/import fails.

## Tuning for speed vs. fidelity
- Faster/offline: `llm_provider="stub"`, `embed_backend="hash"`, `vector_store="memory"`, `arxiv_enabled=False`, smaller `max_results/top_k`, skip PDFs (use abstracts).
- Higher fidelity: `llm_provider="openai"`, `embed_backend="auto"`, `vector_store="faiss"` (or `chroma`), increase `max_results/top_k`, allow PDFs.

## Latency notes (why it can feel slow)
- First run downloads the embedding model (sentence-transformers) from Hugging Face; later runs are faster.
- Searches hit ArXiv/Semantic Scholar over the network and may retry; PDFs (if fetched) add time and pdfminer parsing is slow.
- LLM calls block on OpenAI; use a faster model (e.g., gpt-4o-mini), reduce tokens, or switch to stub for dev.
- Chunking/embedding many papers increases time; lower `max_results/top_k`, increase chunk size, or skip PDFs (use abstracts).

## Testing
```bash
pytest autonomous_research_assistant/tests/test_workflow.py
```

## Notes
- If FAISS/Chroma aren't installed, the store falls back to in-memory (works but non-persistent).
- PDF parsing uses pdfminer when a PDF is fetched; otherwise abstracts are used.
- `[stubbed-response]` means the LLM call failed or `llm_provider="stub"` is set; ensure a valid `OPENAI_API_KEY` with quota.