from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from apps.video.image_engine import generate_scene_images


@dataclass
class ResearchJob:
    primary_topic: str


def run_research(
    job: ResearchJob,
    serpapi_key: Optional[str],
    openai_key: Optional[str] = None,
    progress_cb=None,
) -> Dict[str, Any]:
    _ = (serpapi_key, openai_key, progress_cb)
    topic = (job.primary_topic or "Unknown topic").strip()
    return {
        "status": "COMPLETE",
        "confidence_overall": 0.8,
        "topic": topic,
        "summary": f"Deterministic research dossier for {topic}.",
        "sources": [
            {"title": "Deterministic Source 1", "url": "https://example.com/source-1", "score": 0.9},
            {"title": "Deterministic Source 2", "url": "https://example.com/source-2", "score": 0.85},
        ],
    }


def build_documentary_blueprint(dossier: Dict[str, Any]) -> Dict[str, Any]:
    topic = str(dossier.get("topic") or "Untitled Topic")
    sections = [
        {"section_id": 1, "label": "Context", "focus": f"Historical context around {topic}"},
        {"section_id": 2, "label": "Evidence", "focus": f"Key evidence and witness claims for {topic}"},
        {"section_id": 3, "label": "Analysis", "focus": f"Competing interpretations and open questions for {topic}"},
    ]
    return {"topic": topic, "sections": sections}


def compile_voiceover_script(blueprint: Dict[str, Any], target_minutes: int = 12) -> Dict[str, Any]:
    sections = blueprint.get("sections") or []
    lines: List[str] = []
    for section in sections:
        lines.append(f"{section['label']}: {section['focus']}.")
    full_text = " ".join(lines)
    return {
        "target_minutes": int(target_minutes),
        "full_text": full_text,
        "segments": [{"section_id": s["section_id"], "text": f"{s['label']}: {s['focus']}."} for s in sections],
    }


def build_scene_plan(blueprint: Dict[str, Any], script_result: Dict[str, Any]) -> Dict[str, Any]:
    _ = script_result
    scenes = []
    for idx, section in enumerate(blueprint.get("sections") or [], start=1):
        scenes.append(
            {
                "scene_id": idx,
                "section": section.get("label"),
                "visual_prompt": f"Cinematic documentary frame about {section.get('focus')}",
            }
        )
    return {"scene_count": len(scenes), "scenes": scenes}


def build_image_assets(
    scene_plan: Dict[str, Any],
    *,
    openai_key: Optional[str],
    smoke: bool,
    max_images: int = 60,
) -> Dict[str, Any]:
    scenes = scene_plan.get("scenes") or []
    return generate_scene_images(
        scenes,
        openai_key=openai_key,
        smoke=smoke,
        max_images=max_images,
    )
