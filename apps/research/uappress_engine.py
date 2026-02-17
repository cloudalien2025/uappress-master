from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class ResearchJob:
    primary_topic: str
    confidence_threshold: float = 0.58
    max_serp_queries: int = 12
    max_sources: int = 25
    include_gov_docs: bool = True


def _smoke_enabled() -> bool:
    return os.getenv("UAPPRESS_SMOKE", "").strip() == "1" or os.getenv("CI", "").strip().lower() in {
        "1",
        "true",
    }


def _smoke_dossier(topic: str) -> Dict[str, Any]:
    return {
        "status": "COMPLETE",
        "confidence_overall": 0.82,
        "primary_topic": topic,
        "summary": "Deterministic smoke dossier for CI and local validation.",
        "sources": [
            {"title": "Mock Source A", "url": "https://example.com/a", "score": 0.91},
            {"title": "Mock Source B", "url": "https://example.com/b", "score": 0.84},
            {"title": "Mock Source C", "url": "https://example.com/c", "score": 0.79},
        ],
        "claim_clusters": [
            {
                "claim": "Official reporting acknowledges unexplained incidents.",
                "evidence": ["Mock Source A", "Mock Source B"],
                "confidence": 0.8,
            },
            {
                "claim": "Sensor quality and metadata consistency remain unresolved.",
                "evidence": ["Mock Source C"],
                "confidence": 0.72,
            },
        ],
        "contradictions": [
            {
                "claim_a": "Incidents show clear structured craft behavior.",
                "claim_b": "Most incidents can be explained by mundane artifacts.",
                "tension": "data quality",
                "sources": [
                    {"title": "Mock Source A", "url": "https://example.com/a"},
                    {"title": "Mock Source C", "url": "https://example.com/c"},
                ],
            }
        ],
    }


def _extract_sources(organic_results: List[Dict[str, Any]], max_sources: int) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for item in organic_results[:max_sources]:
        link = item.get("link") or item.get("url") or ""
        if not link:
            continue
        sources.append(
            {
                "title": item.get("title", "Untitled"),
                "url": link,
                "score": 0.7,
            }
        )
    return sources


def run_research(
    job: ResearchJob,
    serpapi_key: Optional[str],
    openai_key: Optional[str] = None,
) -> Dict[str, Any]:
    del openai_key
    if _smoke_enabled():
        return _smoke_dossier(job.primary_topic)

    if not serpapi_key:
        return _smoke_dossier(job.primary_topic)

    response = requests.get(
        "https://serpapi.com/search.json",
        params={
            "engine": "google",
            "q": job.primary_topic,
            "hl": "en",
            "gl": "us",
            "num": min(10, max(1, job.max_sources)),
            "api_key": serpapi_key,
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    sources = _extract_sources(data.get("organic_results", []) or [], job.max_sources)

    return {
        "status": "COMPLETE",
        "confidence_overall": 0.65 if sources else 0.45,
        "primary_topic": job.primary_topic,
        "summary": f"Collected {len(sources)} sources from SerpAPI.",
        "sources": sources,
        "claim_clusters": [
            {
                "claim": f"Search results show recurring discussion of {job.primary_topic}.",
                "evidence": [s["title"] for s in sources[:3]],
                "confidence": 0.62,
            }
        ],
        "contradictions": [],
    }


def build_documentary_blueprint(dossier: Dict[str, Any]) -> Dict[str, Any]:
    topic = str(dossier.get("primary_topic") or dossier.get("topic") or "Untitled Topic")
    sources = dossier.get("sources") if isinstance(dossier.get("sources"), list) else []
    claim_clusters = dossier.get("claim_clusters") if isinstance(dossier.get("claim_clusters"), list) else []
    contradictions = dossier.get("contradictions") if isinstance(dossier.get("contradictions"), list) else []

    lead_claim = ""
    if claim_clusters and isinstance(claim_clusters[0], dict):
        lead_claim = str(claim_clusters[0].get("claim") or "")
    if not lead_claim:
        lead_claim = f"The evidence around {topic} remains contested."

    mapped_contradictions: List[Dict[str, Any]] = []
    for item in contradictions[:5]:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or item.get("claim_a") or "Competing interpretations")
        tension = str(item.get("tension") or item.get("claim_b") or "uncertain evidence")
        item_sources = item.get("sources") if isinstance(item.get("sources"), list) else []
        normalized_sources = []
        for source in item_sources[:3]:
            if not isinstance(source, dict):
                continue
            normalized_sources.append(
                {
                    "title": str(source.get("title") or "Untitled Source"),
                    "url": str(source.get("url") or ""),
                }
            )
        mapped_contradictions.append({"claim": claim, "tension": tension, "sources": normalized_sources})

    if not mapped_contradictions:
        fallback_sources = [
            {"title": str(s.get("title") or "Untitled Source"), "url": str(s.get("url") or "")}
            for s in sources[:3]
            if isinstance(s, dict)
        ]
        mapped_contradictions.append(
            {
                "claim": f"Public narratives about {topic} diverge.",
                "tension": "evidence interpretation",
                "sources": fallback_sources,
            }
        )

    return {
        "title": f"{topic}: Signals, Claims, and Open Questions",
        "logline": f"An evidence-led breakdown of what is known, disputed, and unanswered about {topic}.",
        "cold_open": {
            "vo": f"What if the most important detail about {topic} is the one nobody can verify?",
            "beats": ["Urgent headline montage", "Key contradiction teaser", "Question-driven setup"],
        },
        "act_1_context": {
            "vo": lead_claim,
            "beats": ["Define timeline", "Introduce core sources", "Set standards of evidence"],
        },
        "act_2_contradictions": mapped_contradictions,
        "act_3_implications": {
            "vo": f"If these contradictions hold, {topic} has implications for policy, trust, and oversight.",
            "beats": ["Policy impact", "Scientific implications", "Public accountability"],
        },
        "closing_questions": [
            f"What evidence would decisively resolve the main dispute around {topic}?",
            "Who controls the highest-quality underlying data?",
            "What independent verification is still missing?",
        ],
        "thumbnail_angles": [
            f"Classified vs Public: The {topic} Gap",
            "The Footage Everyone Debates",
            "Evidence or Illusion?",
        ],
        "shorts_hooks": [
            f"One contradiction that changes the {topic} story.",
            "Three facts everyone agrees on — and one they don't.",
            "What the strongest source does (and doesn't) prove.",
        ],
    }
