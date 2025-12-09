"""
Search tool for ArXiv and Semantic Scholar.
"""
from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional

import requests

ARXIV_API = "http://export.arxiv.org/api/query"
SEM_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"


def search_papers(
    query: str,
    source: str = "arxiv",
    max_results: int = 10,
    timeout: int = 12,
    semantic_scholar_key: Optional[str] = None,
) -> List[Dict]:
    """
    Call external search (ArXiv or Semantic Scholar) and return metadata list.

    Returns:
        List of dicts with keys: id, title, authors, year, doi, url, abstract, source.
    """
    normalized_source = source.lower()
    if normalized_source == "arxiv":
        return _search_arxiv(query, max_results=max_results, timeout=timeout)
    if normalized_source == "semanticscholar":
        return _search_semanticscholar(
            query, max_results=max_results, timeout=timeout, api_key=semantic_scholar_key
        )
    raise ValueError(f"Unsupported source: {source}")


def _search_arxiv(query: str, max_results: int, timeout: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    start = 0
    page_size = min(max_results, 50)
    while len(results) < max_results:
        params = {
            "search_query": f"all:{query}",
            "start": start,
            "max_results": page_size,
            "sortBy": "relevance",
        }
        try:
            resp = requests.get(ARXIV_API, params=params, timeout=timeout)
            resp.raise_for_status()
            feed = ET.fromstring(resp.text)
        except Exception:
            break
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in feed.findall("atom:entry", ns):
            meta = _parse_arxiv_entry(entry, ns)
            results.append(meta)
            if len(results) >= max_results:
                break
        if not feed.findall("atom:entry", ns):
            break
        start += page_size
        time.sleep(0.3)  # polite pacing
    return results


def _parse_arxiv_entry(entry: ET.Element, ns: Dict[str, str]) -> Dict[str, Any]:
    title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
    authors = [a.findtext("atom:name", default="", namespaces=ns) for a in entry.findall("atom:author", ns)]
    abstract = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
    year = (entry.findtext("atom:published", default="", namespaces=ns) or "")[:4]
    link = entry.findtext("atom:id", default="", namespaces=ns) or ""
    doi_el = entry.find("{http://arxiv.org/schemas/atom}doi")
    doi = doi_el.text if doi_el is not None else None
    return {
        "id": link,
        "title": title,
        "authors": [a for a in authors if a],
        "year": int(year) if year.isdigit() else None,
        "doi": doi,
        "url": link,
        "abstract": abstract,
        "source": "arxiv",
    }


def _search_semanticscholar(
    query: str, max_results: int, timeout: int, api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    headers = {"User-Agent": "AutonomousResearchAssistant/0.1"}
    if api_key:
        headers["x-api-key"] = api_key
    results: List[Dict[str, Any]] = []
    limit = min(20, max_results)
    offset = 0
    while len(results) < max_results:
        params = {
            "query": query,
            "offset": offset,
            "limit": limit,
            "fields": "title,abstract,year,authors,externalIds,url",
        }
        try:
            resp = requests.get(SEM_SCHOLAR_API, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            break
        papers = data.get("data", [])
        for p in papers:
            results.append(_parse_sem_scholar_entry(p))
            if len(results) >= max_results:
                break
        if not papers:
            break
        offset += limit
        time.sleep(0.2)
    return results


def _parse_sem_scholar_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    doi = None
    ext = entry.get("externalIds") or {}
    if "DOI" in ext:
        doi = ext.get("DOI")
    authors = [a.get("name", "") for a in entry.get("authors") or []]
    return {
        "id": entry.get("paperId"),
        "title": entry.get("title", ""),
        "authors": [a for a in authors if a],
        "year": entry.get("year"),
        "doi": doi,
        "url": entry.get("url"),
        "abstract": entry.get("abstract"),
        "source": "semanticscholar",
    }
