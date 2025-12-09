"""
Citation formatter utility.
"""
from __future__ import annotations

from typing import Dict


def format_citation(meta: Dict, style: str = "APA") -> str:
    """
    Format metadata into a simple citation string.

    Minimal APA-ish fallback:
    Authors (Year). Title. Source/DOI/URL.
    """
    authors = ", ".join(meta.get("authors") or [])
    year = meta.get("year") or "n.d."
    title = meta.get("title") or "Untitled"
    doi = meta.get("doi")
    url = meta.get("url")
    locator = doi or url or "No locator"
    return f"{authors} ({year}). {title}. {locator}"
