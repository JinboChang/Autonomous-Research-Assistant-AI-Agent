"""
Configuration and settings for the Autonomous Research Assistant.
"""
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class Settings:
    """
    Central configuration model for the agent stack.

    NOTE: Values here are defaults; prefer environment variables or a CLI layer
    in production to override sensitive entries like API keys.
    """

    llm_model: str = "gpt-4o-mini"
    llm_provider: Literal["openai", "stub"] = "openai"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_backend: Literal["auto", "hash", "sentence-transformers"] = "auto"
    max_results: int = 10
    top_k: int = 5
    vector_store: Literal["faiss", "chroma", "memory"] = "faiss"
    arxiv_enabled: bool = True
    semantic_scholar_key: Optional[str] = None
    retry_attempts: int = 3
    output_format: Literal["md", "json"] = "md"
    year_filter: Optional[int] = None
    enable_cross_encoder_rerank: bool = False
    log_path: str = "logs/agent.log"
    requests_per_minute: int = 90
    burst_size: int = 20
    request_timeout_seconds: float = 12.0
    read_timeout_seconds: float = 30.0
    user_agent: str = "AutonomousResearchAssistant/0.1"
    enable_tracing: bool = False
    trace_sample_rate: float = 0.0
    request_timeout: int = 15
    # TODO: integrate env var loading (e.g., via dotenv) for keys.


def load_settings() -> Settings:
    """
    Instantiate and return default settings. Future: merge env/CLI overrides.
    """
    return Settings()
