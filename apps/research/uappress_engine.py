from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class ResearchJob:
    primary_topic: str
    confidence_threshold: float = 0.58
    max_serp_queries: int = 12
    max_sources: int = 25
    include_gov_docs: bool = True


def run_research(
    job: Optional[ResearchJob] = None,
    *,
    primary_topic: Optional[str] = None,
    serpapi_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    confidence_threshold: float = 0.58,
    max_serp_queries: int = 12,
    max_sources: int = 25,
    include_gov_docs: bool = True,
    **_: Any,
) -> Dict[str, Any]:
    if job is None:
        if not primary_topic:
            raise ValueError("primary_topic is required")
        job = ResearchJob(
            primary_topic=primary_topic,
            confidence_threshold=confidence_threshold,
            max_serp_queries=max_serp_queries,
            max_sources=max_sources,
            include_gov_docs=include_gov_docs,
        )

    return {
        "status": "COMPLETE",
        "confidence_overall": 0.7,
        "topic": job.primary_topic,
        "summary": "Engine import is wired and run_research executed.",
        "engine": "uappress_engine",
        "job": asdict(job),
        "secrets_present": {
            "serpapi_key": bool(serpapi_key),
            "openai_key": bool(openai_key),
        },
    }
