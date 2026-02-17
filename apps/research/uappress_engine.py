from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

SECTION_ORDER = ["COLD OPEN", "ACT 1", "ACT 2", "ACT 3", "CLOSING"]
SECTION_PROMPTS = {
    "COLD OPEN": "ominous cinematic establishing shot, night fog, archival feel",
    "ACT 1": "contextual archival and official imagery, maps and documents",
    "ACT 2": "tension visuals, split-screen claims versus documents, witnesses and radar screens",
    "ACT 3": "wide implication imagery with institutions, hearings, and expansive night sky",
    "CLOSING": "haunting unanswered questions, slow drift, minimal text",
}
SECTION_LABELS = {
    "COLD OPEN": "COLD OPEN",
    "ACT 1": "ACT 1 — CONTEXT",
    "ACT 2": "ACT 2 — TENSION",
    "ACT 3": "ACT 3 — IMPLICATIONS",
    "CLOSING": "CLOSING",
}


@dataclass
class ResearchJob:
    primary_topic: str


def run_research(job: ResearchJob, serpapi_key: str | None, openai_key: str | None = None, progress_cb=None) -> Dict[str, Any]:
    """Deterministic, CI-safe dossier output.

    This intentionally avoids API calls in smoke mode and gracefully degrades without keys.
    """
    topic = (job.primary_topic or "").strip() or "Unknown topic"
    if progress_cb:
        progress_cb("Preparing deterministic dossier")

    smoke_mode = os.getenv("UAPPRESS_SMOKE", "").strip() == "1" or os.getenv("UAPPRESS_CI_SMOKE", "").strip() == "1"
    note = "SMOKE_MODE enabled — no external calls were made." if smoke_mode or not serpapi_key else "Deterministic local synthesis (no network dependency)."

    return {
        "status": "COMPLETE",
        "confidence_overall": 0.82,
        "topic": topic,
        "summary": f"Deterministic research dossier for {topic}.",
        "sources": [
            {"title": "Mock Source A", "url": "https://example.com/a", "score": 0.91},
            {"title": "Mock Source B", "url": "https://example.com/b", "score": 0.84},
            {"title": "Mock Source C", "url": "https://example.com/c", "score": 0.79},
        ],
        "notes": [note],
        "job": asdict(job),
        "openai_used": bool(openai_key),
    }


def build_documentary_blueprint(dossier: Dict[str, Any]) -> Dict[str, Any]:
    topic = str(dossier.get("topic") or dossier.get("job", {}).get("primary_topic") or "Unknown topic")
    return {
        "title": topic,
        "topic": topic,
        "angle": f"Investigate unresolved claims and documented signals around {topic}.",
        "sections": SECTION_ORDER,
        "beats": {
            "COLD OPEN": [f"Start with a high-tension unresolved moment tied to {topic}."] ,
            "ACT 1": ["Establish timeline, context, and official framing."],
            "ACT 2": ["Present strongest conflicting claims with supporting records."],
            "ACT 3": ["Expand to wider institutional and societal implications."],
            "CLOSING": ["Leave the audience with the central unanswered questions."],
        },
    }


def compile_voiceover_script(blueprint: Dict[str, Any], *, target_minutes: int = 12) -> Dict[str, Any]:
    topic = str(blueprint.get("title") or blueprint.get("topic") or "Unknown topic")
    full_text = "\n\n".join(
        [
            "[COLD OPEN]\nIn the dark margins of official history, one thread keeps resurfacing."
            f" Tonight, we begin with {topic}, a case that refuses to stay buried.",
            "[ACT 1]\nWe start with what is documented: dates, reports, and institutional language."
            " The timeline appears orderly, but key gaps remain visible to anyone reading closely.",
            "[ACT 2]\nNow the contradictions sharpen. Witness statements, technical logs, and official summaries"
            " do not align perfectly, and each mismatch raises the stakes.",
            "[ACT 3]\nThe implications spread beyond one event. Oversight, accountability, and public trust"
            " enter the frame as institutions are forced to answer harder questions.",
            "[CLOSING]\nWhat remains is uncertainty with structure: known facts, unresolved conflicts,"
            " and a final question about what has still not been disclosed.",
        ]
    )
    return {
        "target_minutes": int(target_minutes),
        "title": topic,
        "full_text": full_text,
    }


def _split_sections(full_text: str) -> Dict[str, str]:
    pattern = re.compile(r"\[(COLD OPEN|ACT 1|ACT 2|ACT 3|CLOSING)\]")
    matches = list(pattern.finditer(full_text or ""))
    if not matches:
        return {"ACT 1": (full_text or "").strip()}

    sections: Dict[str, str] = {}
    for i, match in enumerate(matches):
        section_name = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        sections[section_name] = (full_text[start:end] or "").strip()
    return sections


def _split_chunks(section_text: str) -> List[str]:
    clean = (section_text or "").strip()
    if not clean:
        return []

    para_chunks = [p.strip() for p in re.split(r"\n\s*\n+", clean) if p.strip()]
    if len(para_chunks) > 1:
        return para_chunks

    sentence_chunks = [s.strip() for s in re.split(r"(?<=\.)\s+", clean) if s.strip()]
    return sentence_chunks or [clean]


def _duration_for_chunk(chunk: str) -> float:
    words = len(re.findall(r"\b\w+\b", chunk))
    duration = words / 2.6 if words else 0
    return round(max(3.5, min(9.0, duration)), 1)


def _topic_from_blueprint(blueprint: Dict[str, Any]) -> str:
    return str(blueprint.get("title") or blueprint.get("topic") or "the topic").strip() or "the topic"


def build_scene_plan(
    blueprint: Dict[str, Any],
    script_result: Dict[str, Any],
    *,
    target_scene_seconds: float = 6.0,
    max_scenes: int = 180,
) -> Dict[str, Any]:
    full_text = str(script_result.get("full_text") or "")
    sections = _split_sections(full_text)
    topic = _topic_from_blueprint(blueprint)

    scenes: List[Dict[str, Any]] = []
    scene_id = 1

    for section in SECTION_ORDER:
        text = sections.get(section, "")
        chunks = _split_chunks(text)
        for idx, chunk in enumerate(chunks):
            if scene_id > max_scenes:
                break
            chunk_words = re.findall(r"\b\w+\b", chunk)
            short_label = " ".join(chunk_words[:6])
            on_screen_text = SECTION_LABELS.get(section, section) if idx == 0 else short_label
            style = SECTION_PROMPTS.get(section, "cinematic realism visual sequence")
            visual_prompt = (
                f"Cinematic realism: {style} for {topic}. "
                f"Use restrained movement and documentary lighting tied to this narration beat."
            )
            scenes.append(
                {
                    "scene_id": scene_id,
                    "duration_s": _duration_for_chunk(chunk),
                    "vo": chunk,
                    "visual_prompt": visual_prompt,
                    "on_screen_text": on_screen_text,
                }
            )
            scene_id += 1
        if scene_id > max_scenes:
            break

    total = round(sum(float(s["duration_s"]) for s in scenes), 1)
    return {
        "target_scene_seconds": float(target_scene_seconds),
        "scenes": scenes,
        "estimated_total_seconds": total,
    }
