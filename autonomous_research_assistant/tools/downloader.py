"""
PDF downloader tool with basic error handling.
"""
from __future__ import annotations

from typing import Optional

import requests


def download_pdf(url: str, timeout: int = 15) -> Optional[bytes]:
    """
    Download PDF bytes. Returns None on failure.
    """
    if not url:
        return None
    headers = {"User-Agent": "AutonomousResearchAssistant/0.1"}
    try:
        with requests.get(url, headers=headers, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                return None
            return resp.content
    except Exception:
        return None
