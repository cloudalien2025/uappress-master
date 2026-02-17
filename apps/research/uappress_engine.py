from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from apps.video.subtitles import assign_timings, split_script_into_captions, write_srt


_WORD_RE = re.compile(r"\b\w+(?:['-]\w+)?\b")


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _estimate_duration_seconds(text: str) -> float:
    words = _word_count(text)
    wpm = 145.0
    return round((words / wpm) * 60.0, 3) if words else 0.0


@dataclass
class ResearchJob:
    primary_topic: str


def run_research(job: ResearchJob, serpapi_key: str | None = None, openai_key: str | None = None) -> Dict[str, Any]:
    del serpapi_key, openai_key
    return {
        "status": "COMPLETE",
        "confidence_overall": 0.82,
        "topic": job.primary_topic,
        "summary": "Deterministic smoke dossier.",
        "sources": [],
    }


def build_documentary_blueprint(dossier: Dict[str, Any]) -> Dict[str, Any]:
    topic = str(dossier.get("topic", "Untitled Topic"))
    return {
        "topic": topic,
        "sections": [
            {"title": "Cold Open", "beats": [f"Introduce {topic}."]},
            {"title": "Act 1", "beats": [f"Context for {topic}."]},
            {"title": "Act 2", "beats": [f"Evidence around {topic}."]},
        ],
    }


def compile_voiceover_script(blueprint: Dict[str, Any], target_minutes: int = 12) -> Dict[str, Any]:
    sections = blueprint.get("sections", [])
    lines = []
    for section in sections:
        title = str(section.get("title", "Section"))
        beats = section.get("beats") or []
        beat_text = " ".join(str(b) for b in beats)
        lines.append(f"[{title.upper()}]\n{beat_text}")
    full_text = "\n\n".join(lines)
    return {
        "full_text": full_text,
        "target_minutes": target_minutes,
        "word_count": _word_count(full_text),
    }


def build_subtitles_asset(
    script_result: Dict[str, Any],
    audio_result: Dict[str, Any],
    *,
    out_dir: str = "outputs/subtitles",
    smoke: bool = False,
) -> Dict[str, Any]:
    script_text = str(script_result.get("full_text", "") or "")

    total_seconds = audio_result.get("duration_seconds")
    if total_seconds is None:
        total_seconds = _estimate_duration_seconds(script_text)
    total_seconds = round(float(total_seconds or 0.0), 3)

    captions = split_script_into_captions(script_text)
    timed = assign_timings(captions, total_seconds=total_seconds)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    script_sha = hashlib.sha256(script_text.encode("utf-8")).hexdigest()
    audio_sha = str(audio_result.get("sha256", ""))

    if smoke:
        filename = "vo_smoke.srt"
        mode = "smoke"
    else:
        basis = f"{audio_sha}:{script_sha}".encode("utf-8")
        filename = f"vo_{hashlib.sha256(basis).hexdigest()[:16]}.srt"
        mode = "real"

    srt_path = str(Path(out_dir) / filename)
    write_srt(timed, srt_path)
    srt_sha = hashlib.sha256(Path(srt_path).read_bytes()).hexdigest()

    return {
        "srt_path": srt_path,
        "total_seconds": total_seconds,
        "caption_count": len(timed),
        "sha256": srt_sha,
        "mode": mode,
    }
