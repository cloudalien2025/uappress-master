from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

SERP_ENDPOINT = "https://serpapi.com/search.json"


@dataclass
class TopicScore:
    topic: str
    authority_gap: float
    saturation: float
    contradiction_density: float
    suspense_score: float
    viability_score: float
    recommendation: str
    notes: List[str]
    signals: Dict[str, Any]


def _clamp(x: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, x))


def _round2(x: float) -> float:
    return round(_clamp(x), 2)


def _recommendation(score: float) -> str:
    if score >= 0.70:
        return "GREENLIGHT"
    if score >= 0.55:
        return "MAYBE"
    return "PASS"


def _count_term_hits(text: str, terms: List[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(term) for term in terms)


def _score_topic_smoke(topic: str) -> TopicScore:
    digest = hashlib.sha256(topic.strip().lower().encode("utf-8")).digest()

    authority_gap = _round2((digest[0] / 255.0) * 0.6 + 0.2)
    saturation = _round2((digest[1] / 255.0) * 0.7 + 0.15)
    contradiction_density = _round2((digest[2] / 255.0) * 0.6 + 0.2)
    suspense_score = _round2((digest[3] / 255.0) * 0.65 + 0.2)

    viability_raw = (
        authority_gap * 0.30
        + suspense_score * 0.30
        + contradiction_density * 0.25
        + (1.0 - saturation) * 0.15
    )
    viability_score = _round2(viability_raw)

    signals = {
        "serp_title_count": 0,
        "paa_count": int(digest[4] % 6),
        "gov_mil_count": int(digest[5] % 4),
        "wiki_like_count": int(digest[6] % 3),
        "youtube_count": int(digest[7] % 3),
        "reddit_count": int(digest[8] % 3),
        "contradiction_terms_hits": int(digest[9] % 8),
        "mystery_terms_hits": int(digest[10] % 8),
    }

    notes = [
        "Smoke mode deterministic scoring from topic hash.",
        f"Suspense {suspense_score:.2f}, contradiction {contradiction_density:.2f}, saturation {saturation:.2f}.",
    ]

    return TopicScore(
        topic=topic,
        authority_gap=authority_gap,
        saturation=saturation,
        contradiction_density=contradiction_density,
        suspense_score=suspense_score,
        viability_score=viability_score,
        recommendation=_recommendation(viability_score),
        notes=notes,
        signals=signals,
    )


def _serpapi_search(api_key: str, params: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "engine": params.get("engine", "google"),
        "q": params.get("q", ""),
        "num": params.get("num", 10),
        "hl": params.get("hl", "en"),
        "gl": params.get("gl", "us"),
        "api_key": api_key,
    }
    response = requests.get(SERP_ENDPOINT, params=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def _serp_extract_organic(result_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for item in result_json.get("organic_results", []) or []:
        link = item.get("link") or item.get("url")
        if not link:
            continue
        results.append(
            {
                "title": str(item.get("title", "")).strip(),
                "link": str(link).strip(),
                "snippet": str(item.get("snippet", "")).strip(),
            }
        )
    return results


def score_topic(
    topic: str,
    *,
    serpapi_key: Optional[str] = None,
    smoke: bool = False,
    max_results: int = 10,
) -> Dict[str, Any]:
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("Topic is required.")

    if smoke:
        return asdict(_score_topic_smoke(topic))

    if not serpapi_key:
        raise ValueError("SerpAPI key is required when smoke=False.")

    search_json = _serpapi_search(
        serpapi_key,
        {
            "engine": "google",
            "q": topic,
            "num": max(1, min(max_results, 20)),
            "hl": "en",
            "gl": "us",
        },
    )

    organic = _serp_extract_organic(search_json)[:max_results]
    paa = search_json.get("related_questions") or search_json.get("people_also_ask") or []
    related = search_json.get("related_searches") or []

    total = max(len(organic), 1)
    contradiction_terms = [
        "hoax",
        "debunked",
        "conflicting",
        "inconsistent",
        "no evidence",
        "classified",
    ]
    mystery_terms = [
        "mysterious",
        "unexplained",
        "vanished",
        "classified",
        "cover-up",
        "secret",
    ]

    institution_markers = ("nasa", "odni", "dod", "pentagon")
    saturation_markers = (
        "top 10",
        "top ten",
        "best",
        "explained",
        "wikipedia",
        "youtube",
        "reddit",
    )

    gov_mil_count = 0
    wiki_like_count = 0
    youtube_count = 0
    reddit_count = 0
    saturation_hits = 0
    authority_hits = 0

    chunks: List[str] = []
    for row in organic:
        title = str(row.get("title", ""))
        snippet = str(row.get("snippet", ""))
        link = str(row.get("link", ""))
        domain = (urlparse(link).netloc or "").lower().replace("www.", "")

        joined = f"{title} {snippet} {link}".lower()
        chunks.append(joined)

        if any(marker in joined for marker in saturation_markers):
            saturation_hits += 1
        if "wikipedia" in domain or "wikipedia" in joined:
            wiki_like_count += 1
        if "youtube.com" in domain or "youtu.be" in domain:
            youtube_count += 1
        if "reddit.com" in domain:
            reddit_count += 1

        has_gov_mil = domain.endswith(".gov") or domain.endswith(".mil")
        has_edu = domain.endswith(".edu")
        has_institution = any(m in joined for m in institution_markers)

        if has_gov_mil:
            gov_mil_count += 1
        if has_gov_mil or has_edu or has_institution:
            authority_hits += 1

    for qa in paa:
        if isinstance(qa, dict):
            chunks.append(str(qa.get("question", "")))
            chunks.append(str(qa.get("snippet", "")))
        else:
            chunks.append(str(qa))

    for rel in related:
        if isinstance(rel, dict):
            chunks.append(str(rel.get("query", "")))
        else:
            chunks.append(str(rel))

    corpus = " ".join(chunks)
    contradiction_hits = _count_term_hits(corpus, contradiction_terms)
    mystery_hits = _count_term_hits(corpus, mystery_terms)

    saturation = _round2(saturation_hits / total)
    authority_proxy = _clamp(authority_hits / total)
    authority_gap = _round2(1.0 - authority_proxy)

    contradiction_density = _round2(contradiction_hits / max(total * 3, 1))
    suspense_score = _round2(mystery_hits / max(total * 3, 1))

    viability_raw = (
        authority_gap * 0.30
        + suspense_score * 0.30
        + contradiction_density * 0.25
        + (1.0 - saturation) * 0.15
    )
    viability_score = _round2(viability_raw)

    notes: List[str] = []
    if authority_gap >= 0.65:
        notes.append("High authority gap: SERP is not dominated by institutional sources.")
    elif authority_gap <= 0.35:
        notes.append("Lower authority gap: many results are from gov/mil/edu or major institutions.")

    if saturation >= 0.60:
        notes.append("High saturation: many listicle/explainer/wiki/social style results.")
    else:
        notes.append("Moderate saturation: room for differentiated angle.")

    if contradiction_density >= 0.40:
        notes.append("Strong contradiction signal in titles/snippets and related queries.")
    if suspense_score >= 0.40:
        notes.append("Strong suspense signal from mystery-language prevalence.")

    if not notes:
        notes.append("Mixed signals; deeper source-level analysis recommended.")

    scored = TopicScore(
        topic=topic,
        authority_gap=authority_gap,
        saturation=saturation,
        contradiction_density=contradiction_density,
        suspense_score=suspense_score,
        viability_score=viability_score,
        recommendation=_recommendation(viability_score),
        notes=notes,
        signals={
            "serp_title_count": len(organic),
            "paa_count": len(paa),
            "gov_mil_count": gov_mil_count,
            "wiki_like_count": wiki_like_count,
            "youtube_count": youtube_count,
            "reddit_count": reddit_count,
            "contradiction_terms_hits": contradiction_hits,
            "mystery_terms_hits": mystery_hits,
        },
    )
    return asdict(scored)
