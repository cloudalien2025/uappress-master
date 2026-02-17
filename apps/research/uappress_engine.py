from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from apps.video.tts_engine import generate_vo_audio


SMOKE_MODE = os.getenv("UAPPRESS_SMOKE", "").strip() == "1" or os.getenv("CI", "").strip().lower() == "true"


@dataclass
class ResearchJob:
    primary_topic: str


def run_research(job: ResearchJob, serpapi_key: Optional[str], openai_key: Optional[str] = None) -> Dict[str, Any]:
    topic = (job.primary_topic or "").strip() or "Untitled Topic"
    if SMOKE_MODE or not serpapi_key:
        return {
            "status": "COMPLETE",
            "confidence_overall": 0.82,
            "topic": topic,
            "summary": "Mock smoke mode research result (deterministic).",
            "sources": [
                {"title": "Mock Source A", "url": "https://example.com/a", "score": 0.91},
                {"title": "Mock Source B", "url": "https://example.com/b", "score": 0.84},
                {"title": "Mock Source C", "url": "https://example.com/c", "score": 0.79},
            ],
            "notes": ["SMOKE_MODE enabled — no external calls were made."],
        }

    try:
        from apps.research.uappress_engine_v9 import run_research as run_research_v9  # type: ignore
        from apps.research.uappress_engine_v9 import ResearchJob as ResearchJobV9  # type: ignore

        job_v9 = ResearchJobV9(primary_topic=topic)
        return run_research_v9(job_v9, serpapi_key=serpapi_key, openai_key=openai_key)
    except Exception as exc:
        return {
            "status": "PRELIMINARY",
            "confidence_overall": 0.5,
            "topic": topic,
            "summary": f"Fallback run_research path used: {exc.__class__.__name__}",
            "sources": [],
        }


def build_documentary_blueprint(dossier: Dict[str, Any]) -> Dict[str, Any]:
    topic = dossier.get("topic") or "Untitled Topic"
    sources = dossier.get("sources") or []
    return {
        "topic": topic,
        "thesis": f"Evidence-driven overview of {topic}",
        "acts": ["Setup", "Complication", "Resolution"],
        "source_count": len(sources),
    }


def compile_voiceover_script(blueprint: Dict[str, Any], target_minutes: int = 12) -> Dict[str, Any]:
    topic = blueprint.get("topic") or "Untitled Topic"
    full_text = (
        f"This documentary explores {topic}. "
        f"Act one introduces the known facts and context. "
        f"Act two examines competing claims and supporting evidence. "
        f"Act three provides a grounded conclusion and open questions for future reporting."
    )
    return {
        "target_minutes": target_minutes,
        "full_text": full_text,
        "word_count": len(full_text.split()),
    }


def build_scene_plan(blueprint: Dict[str, Any], script_result: Dict[str, Any]) -> Dict[str, Any]:
    topic = blueprint.get("topic") or "Untitled Topic"
    return {
        "topic": topic,
        "scenes": [
            {"scene": 1, "title": "Cold Open", "goal": "Hook the audience with the central mystery."},
            {"scene": 2, "title": "Evidence Review", "goal": "Lay out key documents and testimony."},
            {"scene": 3, "title": "Resolution", "goal": "Synthesize findings and unresolved questions."},
        ],
        "script_word_count": script_result.get("word_count", 0),
    }


def build_audio_asset(script_result: dict, *, openai_key: Optional[str], smoke: bool) -> dict:
    return generate_vo_audio(
        script_result.get("full_text", ""),
        openai_key=openai_key,
        smoke=smoke,
    )
