"""
LLM client wrapper with OpenAI support and a stub fallback.
"""
from __future__ import annotations

import json
import os
from typing import Optional, List

from ..config import Settings


class LLMClient:
    """
    Lightweight wrapper to call OpenAI (or fallback stub) with guardrails.
    """

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.provider = cfg.llm_provider
        self.model = cfg.llm_model
        self._client = None
        if self.provider == "openai":
            try:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI()
            except Exception:
                # Fallback to stub if SDK not available.
                self.provider = "stub"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        """
        Generate text from the LLM or fallback stub.
        """
        if self.provider != "openai" or not self._client:
            # Deterministic stub to keep flows running offline.
            return self._stub_response(prompt, system_prompt)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            # Fail-safe to stub.
            return self._stub_response(prompt, system_prompt)

    def _stub_response(self, prompt: str, system_prompt: Optional[str]) -> str:
        """
        Predictable deterministic fallback to avoid blocking when LLM is unavailable.
        """
        summary = prompt[:400].replace("\n", " ")
        return f"[stubbed-response] {summary}"


def extract_json_block(text: str) -> Optional[dict]:
    """
    Extract the first JSON object found in the text.
    """
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return json.loads(text[start : end + 1])
    except Exception:
        return None
