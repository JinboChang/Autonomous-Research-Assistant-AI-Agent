"""
Tests for search, chunking, embedding, and end-to-end stub flow.
"""
from autonomous_research_assistant.main import run
from autonomous_research_assistant.config import Settings
from autonomous_research_assistant.agents.researcher import _chunk_text
from autonomous_research_assistant.tools.vectorstore import EmbeddingsProvider
from autonomous_research_assistant.tools import search as search_tool


def test_chunk_text() -> None:
    text = "abcd" * 400  # 1600 chars
    chunks = _chunk_text(text, max_chars=500, overlap=50)
    assert len(chunks) >= 3
    assert all(len(c) <= 500 for c in chunks)


def test_embed_hash_backend() -> None:
    cfg = Settings(embed_backend="hash")
    emb = EmbeddingsProvider(cfg).embed_text("hello world")
    assert len(emb) == 256
    assert any(v != 0 for v in emb)


def test_search_arxiv_parse(monkeypatch) -> None:
    xml = """<feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Test Title</title>
        <id>http://arxiv.org/abs/1234</id>
        <summary>Test abstract</summary>
        <published>2024-01-01</published>
        <author><name>A. Author</name></author>
      </entry>
    </feed>"""

    class Resp:
        def __init__(self) -> None:
            self.text = xml

        def raise_for_status(self) -> None:
            return None

    def fake_get(url, params=None, timeout=None):
        return Resp()

    monkeypatch.setattr(search_tool.requests, "get", fake_get)
    results = search_tool.search_papers("test", source="arxiv", max_results=1)
    assert results[0]["title"] == "Test Title"
    assert results[0]["source"] == "arxiv"


def test_end_to_end_stub(monkeypatch) -> None:
    cfg = Settings(embed_backend="hash", vector_store="memory", llm_provider="stub")

    def fake_load_settings():
        return cfg

    monkeypatch.setattr("autonomous_research_assistant.main.load_settings", fake_load_settings)
    monkeypatch.setattr(
        search_tool,
        "search_papers",
        lambda *args, **kwargs: [
            {
                "id": "stub1",
                "title": "Stub Paper",
                "authors": ["A. Author"],
                "year": 2024,
                "doi": None,
                "url": "http://example.com/stub.pdf",
                "abstract": "Stub abstract for testing.",
                "source": "arxiv",
            }
        ],
    )
    monkeypatch.setattr("autonomous_research_assistant.tools.downloader.download_pdf", lambda *a, **k: None)
    report = run("sample question about AI")
    assert "review" in report.lower()
