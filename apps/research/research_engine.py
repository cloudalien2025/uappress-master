"""Lightweight research engine adapter used by the Streamlit app.

This module intentionally exposes a `run_research(**kwargs)` function because
`apps/research/app.py` imports it with that interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

SERP_ENDPOINT = "https://serpapi.com/search.json"


def _build_queries(primary_topic: str, include_gov_docs: bool) -> List[str]:
    topic = (primary_topic or "").strip()
    if not topic:
        return []

    queries = [
        f'"{topic}"',
        f'"{topic}" investigation OR report',
        f'"{topic}" timeline',
    ]
    if include_gov_docs:
        queries.insert(1, f'"{topic}" site:.gov OR site:.mil')

    return queries


def _serp_search(api_key: str, query: str, num: int = 10) -> List[Dict[str, Any]]:
    params = {
        "engine": "google",
        "q": query,
        "num": num,
        "hl": "en",
        "gl": "us",
        "api_key": api_key,
    }
    response = requests.get(SERP_ENDPOINT, params=params, timeout=20)
    response.raise_for_status()
    data = response.json() or {}

    results: List[Dict[str, Any]] = []
    for item in data.get("organic_results", []) or []:
        link = item.get("link") or item.get("url")
        if not link:
            continue
        results.append(
            {
                "title": item.get("title") or "Untitled",
                "url": link,
                "snippet": item.get("snippet") or "",
            }
        )
    return results


def run_research(
    *,
    primary_topic: str,
    serpapi_key: str,
    openai_key: Optional[str] = None,
    confidence_threshold: float = 0.58,
    max_serp_queries: int = 12,
    max_sources: int = 25,
    include_gov_docs: bool = True,
    **_: Any,
) -> Dict[str, Any]:
    """Run a small real search pass and return the dossier format the UI expects."""
    del openai_key

    queries = _build_queries(primary_topic, include_gov_docs)[: max(1, max_serp_queries)]
    all_sources: List[Dict[str, Any]] = []

    for query in queries:
        try:
            all_sources.extend(_serp_search(serpapi_key, query))
        except Exception as exc:
            return {
                "status": "ERROR",
                "confidence_overall": 0.0,
                "note": f"Search request failed: {exc}",
                "topic": primary_topic,
                "sources": [],
            }

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for source in all_sources:
        url = source.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(source)
        if len(deduped) >= max_sources:
            break

    confidence = min(0.95, 0.5 + (0.02 * len(deduped)))
    status = "COMPLETE" if confidence >= confidence_threshold else "PRELIMINARY"

    return {
        "status": status,
        "confidence_overall": round(confidence, 2),
        "note": "Live research engine response.",
        "topic": primary_topic,
        "sources": deduped,
        "meta": {
            "queries_run": len(queries),
            "sources_collected": len(deduped),
        },
    }
