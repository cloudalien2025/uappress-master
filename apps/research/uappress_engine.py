from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class ResearchJob:
    primary_topic: str
    confidence_threshold: float = 0.58
    max_serp_queries: int = 12
    max_sources: int = 25
    include_gov_docs: bool = True
    openai_key: Optional[str] = None


def run_research(
    *,
    primary_topic: str,
    serpapi_key: str,
    openai_key: Optional[str] = None,
    confidence_threshold: float = 0.58,
    max_serp_queries: int = 12,
    max_sources: int = 25,
    include_gov_docs: bool = True,
) -> Dict[str, Any]:
    """Research engine entrypoint used by Streamlit app.

    This preserves the app contract while keeping API calls optional/safe.
    """
    job = ResearchJob(
        primary_topic=primary_topic,
        confidence_threshold=confidence_threshold,
        max_serp_queries=max_serp_queries,
        max_sources=max_sources,
        include_gov_docs=include_gov_docs,
        openai_key=openai_key,
    )

    return {
        "status": "PRELIMINARY",
        "confidence_overall": 0.66,
        "topic": job.primary_topic,
        "summary": "Engine import wired successfully.",
        "settings": {
            "has_serpapi_key": bool(serpapi_key),
            "has_openai_key": bool(openai_key),
            "confidence_threshold": confidence_threshold,
            "max_serp_queries": max_serp_queries,
            "max_sources": max_sources,
            "include_gov_docs": include_gov_docs,
        },
        "job": asdict(job),
        "sources": [],
    }
