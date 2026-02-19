# apps/research/app.py — UAPpress Research Engine (Production-safe + Test-deterministic)
#
# CI Contract (Playwright):
# - Always render: "TEST_HOOK:APP_LOADED"
# - In CI smoke mode (no secrets), user can:
#     * Fill "Primary Topic"
#     * Click "Run Research"
#     * See: "TEST_HOOK:RUN_DONE" and "Research Complete"
#
# IMPORTANT:
# - Do NOT short-circuit/stop in CI, because another UI test asserts the presence of
#   the Primary Topic input and Run Research button.
# - Smoke mode must be enabled in GitHub Actions where CI is typically "true".

import json
import os as _os
import time
import tempfile
import base64
import hashlib
import re
import inspect
import importlib
import shutil
import subprocess
import traceback
import wave
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict

import streamlit as st
try:
    from apps.research.ci_hooks import ci_smoke_enabled, mark_run_done
except Exception:
    from ci_hooks import ci_smoke_enabled, mark_run_done


# Import marker must always exist for Streamlit Cloud import safety.
ENGINE_IMPORT_MARKER = "TEST_HOOK:ENGINE_IMPORT_FALLBACK"
ENGINE_IMPORT_STATUS = "FALLBACK"
ENGINE_IMPORT_EXCEPTION: BaseException | None = None
ENGINE_IMPORT_EXCEPTION_TRACE = ""
_MINIMAL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAZAAAADICAIAAABJdyC1AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJ"
    "bWFnZVJlYWR5ccllPAAAABh0RVh0Q3JlYXRpb24gVGltZQAwMi8xOC8yNvVhXxkAAABBSURBVHja"
    "7MExAQAAAMKg9U9tCF8gAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB4GQAB"
    "Qf4AAQAAAABJRU5ErkJggg=="
)

TARGET_DOCUMENTARY_MINUTES = 45
WORDS_PER_MINUTE = 160
TARGET_SCRIPT_WORDS = TARGET_DOCUMENTARY_MINUTES * WORDS_PER_MINUTE
DEFAULT_SCENE_COUNT = 16
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "onyx"
PLANNING_LEAK_PATTERNS = [
    "in this section",
    "this scene",
    "scene establishes",
    "we will now",
    "our objective",
    "the narration",
    "fact pack",
    "beat sheet",
    "source quality",
    "sparse data",
    "structured fields",
    "tension shift",
    "frame the central mystery",
]


def contains_planning_language(text: str) -> bool:
    lower = str(text or "").lower()
    return any(token in lower for token in PLANNING_LEAK_PATTERNS)


def _ensure_fallback_artifacts(
    topic: str,
    scene_plan: Dict[str, Any],
    script_text: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    base_dir = (
        Path(tempfile.gettempdir())
        / "uappress_artifacts"
        / str(st.session_state.get("last_run_ts") or int(time.time()))
    )
    images_dir = base_dir / "images"
    audio_dir = base_dir / "audio"
    images_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    scenes = scene_plan.get("scenes") if isinstance(scene_plan.get("scenes"), list) else []
    scene_count = max(1, len(scenes))
    image_paths: list[str] = []

    for index in range(scene_count):
        image_path = images_dir / f"scene_{index + 1:02d}.png"
        try:
            from PIL import Image, ImageDraw  # type: ignore

            image = Image.new("RGB", (1280, 720), color=(18, 24, 38))
            draw = ImageDraw.Draw(image)
            heading = ""
            if index < len(scenes):
                heading = str(scenes[index].get("heading") or "")
            label = f"{topic}\n{heading or f'Scene {index + 1}'}"
            draw.text((40, 40), label, fill=(235, 242, 255))
            image.save(image_path)
        except Exception:
            image_path.write_bytes(base64.b64decode(_MINIMAL_PNG_BASE64))
        image_paths.append(str(image_path))

    audio_path = audio_dir / "fallback_narration.wav"
    word_count = len([token for token in script_text.split() if token.strip()])
    duration_seconds = max(2, int(round(word_count / 2.5))) if word_count else 2
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16000 * duration_seconds)

    return {
        "scene_images_dir": str(images_dir),
        "images": image_paths,
        "image_count": len(image_paths),
        "engine": "fallback",
    }, {
        "audio_path": str(audio_path),
        "duration_seconds": duration_seconds,
        "script_word_count": word_count,
        "engine": "fallback",
    }


def _consolidate_script_text(
    script_result: Dict[str, Any] | None,
    scene_plan: Dict[str, Any] | None,
) -> str:
    script_data = script_result if isinstance(script_result, dict) else {}
    scene_data = scene_plan if isinstance(scene_plan, dict) else {}

    prebuilt_text = str(script_data.get("text") or "").strip()
    if prebuilt_text:
        return prebuilt_text

    scene_lines = []
    scenes = scene_data.get("scenes") if isinstance(scene_data.get("scenes"), list) else []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        heading = str(scene.get("heading") or scene.get("scene_title") or "").strip()
        visual = str(scene.get("visual") or scene.get("visual_notes") or "").strip()
        voiceover = str(scene.get("voiceover") or "").strip()
        combined = "\n".join([part for part in [heading, visual, voiceover] if part])
        if combined:
            scene_lines.append(combined)

    narration = str(script_data.get("narration") or "").strip()
    lines = script_data.get("lines") if isinstance(script_data.get("lines"), list) else []
    script_lines = [str(line).strip() for line in lines if str(line).strip()]

    parts = []
    if scene_lines:
        parts.append("Scene Plan:\n" + "\n".join(f"- {line}" for line in scene_lines))
    if narration:
        parts.append("Narration:\n" + narration)
    elif script_lines:
        parts.append("Narration:\n" + "\n".join(script_lines))

    return "\n\n".join([part for part in parts if part]).strip()




def _generate_script_legacy(bundle: Dict[str, Any], *, overwrite: bool = False) -> Dict[str, Any] | None:
    dossier = bundle.get("dossier") if isinstance(bundle.get("dossier"), dict) else None
    if not dossier:
        return None
    script_bundle = bundle.get("script") if isinstance(bundle.get("script"), dict) else {}
    if script_bundle.get("locked") and not overwrite:
        return script_bundle
    generated = _fallback_build_documentary_script(bundle.get("blueprint") or {}, dossier)
    bundle["script"] = generated
    bundle["scene_plan"] = _fallback_build_scene_plan(bundle.get("blueprint") or {}, generated)
    return generated


def generate_scene_mp3s(bundle: Dict[str, Any], *, openai_key: str, voice: str = TTS_VOICE, model: str = TTS_MODEL, smoke: bool = False) -> Dict[str, Any]:
    if not openai_key:
        raise ValueError("OpenAI API key missing. Set OPENAI_API_KEY to generate MP3s.")

    script_bundle = bundle.get("script") if isinstance(bundle.get("script"), dict) else {}
    scenes = script_bundle.get("scenes") if isinstance(script_bundle.get("scenes"), list) else []
    validator_report = script_bundle.get("validator") if isinstance(script_bundle.get("validator"), dict) else {}
    if not scenes:
        raise ValueError("Generate Script first")
    if not validator_report.get("ok"):
        raise ValueError("Script validator must pass before generating MP3s")
    if not any(str(scene.get("voiceover") or "").strip() for scene in scenes if isinstance(scene, dict)):
        raise ValueError("Script scenes must include voiceover before generating MP3s")

    from apps.video.tts_engine import generate_vo_audio

    base_dir = Path(tempfile.gettempdir()) / "uappress_artifacts" / str(st.session_state.get("last_run_ts") or int(time.time()))
    audio_dir = base_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_bundle = bundle.get("audio") if isinstance(bundle.get("audio"), dict) else {}
    cache = audio_bundle.get("cache") if isinstance(audio_bundle.get("cache"), dict) else {}

    scene_mp3s = []
    for idx, scene in enumerate(scenes, start=1):
        voiceover = str(scene.get("voiceover") or "").strip()
        sig = hashlib.sha256(f"{voiceover}|{voice}|{model}".encode("utf-8")).hexdigest()
        target = audio_dir / f"scene_{idx:02d}.mp3"
        key = f"scene_{idx:02d}"
        if not (cache.get(key) == sig and target.exists()):
            result = generate_vo_audio(voiceover, out_dir=str(audio_dir), voice=voice, model=model, openai_key=openai_key, smoke=smoke)
            src = Path(str(result.get("mp3_path") or ""))
            if src.exists() and src != target:
                target.write_bytes(src.read_bytes())
            cache[key] = sig
        duration_sec = round(len(voiceover.split()) / WORDS_PER_MINUTE * 60.0, 2)
        scene["audio_path"] = str(target)
        scene["audio_duration_sec"] = duration_sec
        scene_mp3s.append({"scene": idx, "path": str(target), "duration_sec": duration_sec})

    merged_audio_path = ""
    ffmpeg_binary = shutil.which("ffmpeg")
    if ffmpeg_binary and scene_mp3s:
        manifest = audio_dir / "scene_audio_concat.txt"
        manifest.write_text("\n".join([f"file '{Path(item['path']).resolve().as_posix()}'" for item in scene_mp3s]) + "\n", encoding="utf-8")
        merged_target = audio_dir / "full_narration.mp3"
        run = subprocess.run([ffmpeg_binary, "-y", "-f", "concat", "-safe", "0", "-i", str(manifest), "-c", "copy", str(merged_target)], capture_output=True, text=True, check=False)
        if run.returncode == 0 and merged_target.exists():
            merged_audio_path = str(merged_target)

    total_duration = round(sum(float(item.get("duration_sec") or 0.0) for item in scene_mp3s), 2)
    audio_bundle.update({
        "scene_mp3s": scene_mp3s,
        "audio_path": merged_audio_path,
        "duration_seconds": total_duration,
        "format": "mp3",
        "voice": voice,
        "model": model,
        "cache": cache,
    })
    bundle["audio"] = audio_bundle
    bundle["script"]["scenes"] = scenes
    bundle["timing"] = build_timing_map(scenes)
    return audio_bundle


def _ffmpeg_preflight() -> Dict[str, Any]:
    ffmpeg_binary = shutil.which("ffmpeg")
    path_value = _os.getenv("PATH", "")
    details = {
        "ffmpeg_path": ffmpeg_binary,
        "PATH": path_value,
    }

    if not ffmpeg_binary:
        details["message"] = (
            "ffmpeg not found on PATH. Add packages.txt with `ffmpeg` at repo root and redeploy on Streamlit Cloud."
        )
        return {"ok": False, "details": details}

    ffmpeg_version = subprocess.run(
        [ffmpeg_binary, "-version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if ffmpeg_version.returncode != 0:
        details["message"] = (
            f"ffmpeg -version failed ({ffmpeg_version.returncode}): {ffmpeg_version.stderr.strip()}"
        )
        return {"ok": False, "details": details}

    first_line = (ffmpeg_version.stdout or "").splitlines()
    details["version"] = first_line[0] if first_line else "unknown"
    return {"ok": True, "details": details}


def _collect_missing_artifacts(
    image_result: Dict[str, Any] | None,
    audio_result: Dict[str, Any] | None,
) -> list[str]:
    missing: list[str] = []

    image_paths = []
    if isinstance(image_result, dict):
        paths = image_result.get("images")
        if isinstance(paths, list):
            image_paths = [str(path) for path in paths if str(path).strip()]

    if not image_paths:
        missing.append("images list (bundle.images)")
    else:
        missing_images = [path for path in image_paths if not Path(path).exists()]
        if missing_images:
            missing.append(f"image files not found: {', '.join(missing_images)}")

    audio_path = ""
    scene_mp3s: list[Dict[str, Any]] = []
    if isinstance(audio_result, dict):
        raw_scene_mp3s = audio_result.get("scene_mp3s")
        if isinstance(raw_scene_mp3s, list):
            scene_mp3s = [item for item in raw_scene_mp3s if isinstance(item, dict)]
        audio_path = str(
            audio_result.get("audio_path")
            or audio_result.get("audio_mp3_path")
            or ""
        ).strip()

    if not scene_mp3s:
        missing.append("scene mp3s (bundle.audio.scene_mp3s)")
    else:
        missing_scene_files = [
            str(item.get("audio_path") or item.get("path") or "")
            for item in scene_mp3s
            if not Path(str(item.get("audio_path") or item.get("path") or "")).exists()
        ]
        if missing_scene_files:
            missing.append(f"scene audio files not found: {', '.join(missing_scene_files)}")

    if not audio_path:
        missing.append("assembled narration audio path (bundle.audio.audio_path)")
    elif not Path(audio_path).exists():
        missing.append(f"assembled narration audio file not found: {audio_path}")

    return missing


def _append_bundle_error(bundle: Dict[str, Any], stage: str, exc: BaseException) -> None:
    errors = bundle.setdefault("errors", [])
    if not isinstance(errors, list):
        errors = []
        bundle["errors"] = errors
    errors.append(
        {
            "stage": stage,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    )


def _artifact_readiness(bundle: Dict[str, Any] | None) -> Dict[str, Any]:
    bundle_data = bundle if isinstance(bundle, dict) else {}

    script_result = bundle_data.get("script") if isinstance(bundle_data.get("script"), dict) else {}
    script_text = str(bundle_data.get("script_text") or script_result.get("text") or "").strip()
    script_scenes = script_result.get("scenes") if isinstance(script_result.get("scenes"), list) else []
    validator_report = bundle_data.get("validator_report") if isinstance(bundle_data.get("validator_report"), dict) else {}
    script_validator = script_result.get("validator") if isinstance(script_result.get("validator"), dict) else {}
    script_ok = bool((script_validator or validator_report).get("ok"))
    has_script = bool(script_text) and bool(script_scenes) and script_ok

    image_result = bundle_data.get("images") if isinstance(bundle_data.get("images"), dict) else {}
    image_paths_raw = image_result.get("images") if isinstance(image_result.get("images"), list) else []
    image_paths = [str(path).strip() for path in image_paths_raw if str(path).strip()]
    has_images = bool(image_paths) and all(Path(path).exists() for path in image_paths)

    audio_result = bundle_data.get("audio") if isinstance(bundle_data.get("audio"), dict) else {}
    scene_mp3s_raw = audio_result.get("scene_mp3s") if isinstance(audio_result.get("scene_mp3s"), list) else []
    scene_mp3_paths = [
        str(item.get("audio_path") or item.get("path") or "").strip()
        for item in scene_mp3s_raw
        if isinstance(item, dict)
    ]

    audio_path = str(
        audio_result.get("audio_path")
        or audio_result.get("audio_mp3_path")
        or ""
    ).strip()
    has_audio = (
        bool(scene_mp3_paths)
        and all(path and Path(path).exists() for path in scene_mp3_paths)
        and bool(audio_path)
        and Path(audio_path).exists()
    )

    ffmpeg_check = _ffmpeg_preflight()
    has_ffmpeg = bool(ffmpeg_check.get("ok"))

    missing_labels: list[str] = []
    if not has_script:
        missing_labels.append("Script scenes")
    if not has_audio:
        missing_labels.append("Scene MP3s")
    if not has_images:
        missing_labels.append("Scene images")
    if not has_ffmpeg:
        missing_labels.append("ffmpeg")

    return {
        "has_script": has_script,
        "has_images": has_images,
        "has_audio": has_audio,
        "has_ffmpeg": has_ffmpeg,
        "ready_for_video": has_script and has_images and has_audio and has_ffmpeg,
        "missing_labels": missing_labels,
        "ffmpeg_details": ffmpeg_check.get("details"),
    }


def _fallback_score_topic(topic: str, serpapi_key: str | None = None, smoke: bool = False) -> Dict[str, Any]:
    """Deterministic topic scoring fallback used when no engine helper exists."""
    normalized_topic = topic.strip()
    token_count = len([p for p in normalized_topic.split() if p])
    char_count = len(normalized_topic)

    score = min(0.95, max(0.35, 0.35 + (token_count * 0.08) + (char_count / 200.0)))
    if smoke:
        score = min(0.99, score + 0.05)

    if score >= 0.75:
        recommendation = "GREENLIGHT"
    elif score >= 0.58:
        recommendation = "MAYBE"
    else:
        recommendation = "PASS"

    reasons = [
        f"Topic length looks {'specific' if token_count >= 3 else 'broad'} ({token_count} words).",
        "Fallback heuristic score used (no external API call).",
    ]
    if serpapi_key:
        reasons.append("SerpAPI key provided but not required for fallback scoring.")

    return {
        "topic": normalized_topic,
        "score": round(score, 2),
        "recommendation": recommendation,
        "reasons": reasons,
        "engine": "fallback",
    }


def _fallback_build_documentary_blueprint(dossier: Dict[str, Any]) -> Dict[str, Any]:
    topic = str(dossier.get("topic") or dossier.get("primary_topic") or "Untitled Topic")
    summary = str(dossier.get("summary") or "")
    confidence = float(dossier.get("confidence_overall") or 0)
    sources = dossier.get("sources") if isinstance(dossier.get("sources"), list) else []

    return {
        "title": f"Documentary Blueprint: {topic}",
        "topic": topic,
        "logline": f"An evidence-led explainer on {topic}.",
        "confidence_overall": confidence,
        "summary": summary,
        "sections": [
            {"heading": "What happened", "objective": "State the core claim and context."},
            {"heading": "What evidence exists", "objective": "Summarize strongest sources."},
            {"heading": "Open questions", "objective": "List unresolved claims and next checks."},
        ],
        "source_count": len(sources),
        "engine": "fallback",
    }


def _fallback_compile_voiceover_script(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    topic = str(blueprint.get("topic") or "this case")
    lines = [
        f"Tonight we examine {topic}.",
        "We start with what is known, then contrast it with disputed claims.",
        "Finally, we map the remaining unknowns and what evidence is still needed.",
    ]
    return {
        "narration": " ".join(lines),
        "lines": lines,
        "word_count": sum(len(line.split()) for line in lines),
        "engine": "fallback",
    }


def _normalize_dossier_items(raw_value: Any) -> list[str]:
    items: list[str] = []
    if isinstance(raw_value, list):
        for value in raw_value:
            if isinstance(value, dict):
                text = str(
                    value.get("text")
                    or value.get("event")
                    or value.get("claim")
                    or value.get("summary")
                    or ""
                ).strip()
            else:
                text = str(value).strip()
            if text and text not in items:
                items.append(text)
    return items


def _source_tier(url: str) -> int:
    host = (url or "").lower()
    if ".gov" in host or ".edu" in host:
        return 1
    if any(domain in host for domain in ["reuters.com", "apnews.com", "nytimes.com", "wsj.com", "bbc.", "cnn.com"]):
        return 1
    if any(domain in host for domain in ["reddit.com", "blog", "substack.com", "medium.com"]):
        return 3
    return 2


def _source_domain(url: str) -> str:
    match = re.search(r"https?://([^/]+)", url or "")
    return (match.group(1).lower() if match else (url or "").lower()).replace("www.", "")


def _extract_timeline_candidates(dossier: Dict[str, Any], sources: list[Dict[str, Any]]) -> list[str]:
    timeline_markers = re.compile(
        r"(\b(?:19|20)\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b|"
        r"\b\d{1,2}:\d{2}\s?(?:am|pm)?\b|\bfirst\b|\bthen\b|\bnext\b|\blater\b|\bafter\b|"
        r"\bbefore\b|\bfinally\b|\bearlier\b|\bsubsequently\b)",
        re.IGNORECASE,
    )
    candidates: list[str] = []
    fields_to_scan = [
        dossier.get("timeline"),
        dossier.get("chronology"),
        dossier.get("summary"),
        dossier.get("findings"),
        dossier.get("claims"),
        dossier.get("facts"),
        dossier.get("notes"),
    ]
    for raw in fields_to_scan:
        for item in _normalize_dossier_items(raw):
            if timeline_markers.search(item) and item not in candidates:
                candidates.append(item)

    for source in sources:
        if not isinstance(source, dict):
            continue
        snippets = [
            str(source.get("title") or "").strip(),
            str(source.get("snippet") or "").strip(),
            str(source.get("summary") or "").strip(),
            str(source.get("published_at") or source.get("date") or "").strip(),
        ]
        for snippet in snippets:
            if snippet and timeline_markers.search(snippet) and snippet not in candidates:
                candidates.append(snippet)
    return candidates


def _build_narrative_spine(
    official_positions: list[Dict[str, Any]],
    witness_accounts: list[Dict[str, Any]],
    contradictions: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    spine: list[Dict[str, Any]] = []

    for item in official_positions[:3]:
        statement = str(item.get("statement") or "").strip()
        if statement:
            spine.append(
                {
                    "beat": statement,
                    "kind": "official_statement",
                    "source_idx": int(item.get("source_idx") or 1),
                }
            )

    for item in witness_accounts[:3]:
        summary = str(item.get("summary") or "").strip()
        if summary:
            spine.append(
                {
                    "beat": summary,
                    "kind": "witness_account",
                    "source_idx": int(item.get("source_idx") or 1),
                }
            )

    for item in contradictions[:3]:
        claim_a = str(item.get("claim_a") or "").strip()
        claim_b = str(item.get("claim_b") or "").strip()
        if claim_a or claim_b:
            spine.append(
                {
                    "beat": f"Contradiction: {claim_a} vs {claim_b}".strip(),
                    "kind": "contradiction",
                    "source_idx": int((item.get("source_indices") or [1])[0] or 1),
                }
            )

    deduped: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in spine:
        beat = str(item.get("beat") or "").strip()
        if beat and beat.lower() not in seen:
            deduped.append(item)
            seen.add(beat.lower())
    return deduped


def build_fact_pack(dossier: Dict[str, Any]) -> Dict[str, Any]:
    topic = str(dossier.get("topic") or dossier.get("primary_topic") or "Untitled Topic").strip()
    raw_sources = dossier.get("sources") if isinstance(dossier.get("sources"), list) else []
    unique_domains: set[str] = set()
    sources_used: list[Dict[str, Any]] = []
    gaps: Dict[str, Any] = {"items": []}

    for source in raw_sources:
        if not isinstance(source, dict):
            continue
        url = str(source.get("url") or "").strip()
        title = str(source.get("title") or "Untitled source").strip()
        domain = _source_domain(url)
        tier = _source_tier(url)
        if domain in unique_domains and tier == 3:
            continue
        unique_domains.add(domain)
        sources_used.append({"idx": len(sources_used) + 1, "title": title, "url": url, "tier": tier})

    def _pick_idx(text: str) -> int:
        if not sources_used:
            return 0
        return _pick_source_for_item(text, len(sources_used), 0)

    timeline_items = _normalize_dossier_items(dossier.get("timeline") or dossier.get("chronology"))
    extracted_candidates = _extract_timeline_candidates(dossier, raw_sources)
    if len(timeline_items) < 5:
        for candidate in extracted_candidates:
            if candidate not in timeline_items:
                timeline_items.append(candidate)
            if len(timeline_items) >= 5:
                break

    timeline_events = [
        {
            "date": "Unknown",
            "time": "Unknown",
            "location": "Unspecified",
            "event": item,
            "source_idx": _pick_idx(item),
            "confidence": 0.65,
        }
        for item in timeline_items
    ]

    claim_items = _normalize_dossier_items(dossier.get("claims") or dossier.get("facts") or dossier.get("findings"))
    key_claims = [
        {
            "claim": item,
            "claimant": "Dossier record",
            "evidence": "Primary dossier entry",
            "source_idx": _pick_idx(item),
            "confidence": 0.68,
        }
        for item in claim_items
    ]

    witness_accounts = [
        {
            "witness": "Documented witness",
            "summary": item,
            "source_idx": _pick_idx(item),
            "confidence": 0.6,
        }
        for item in _normalize_dossier_items(dossier.get("witness_accounts") or dossier.get("notes"))[:6]
    ]

    official_positions = [
        {
            "agency": "Official channel",
            "statement": item,
            "source_idx": _pick_idx(item),
            "confidence": 0.7,
        }
        for item in _normalize_dossier_items(dossier.get("official_positions") or dossier.get("summary"))[:4]
    ]

    skeptical_explanations = [
        {"explanation": item, "supporting_sources": [_pick_idx(item)], "confidence": 0.55}
        for item in _normalize_dossier_items(dossier.get("skeptical_explanations") or dossier.get("alternatives"))[:5]
    ]

    contradiction_items = _normalize_dossier_items(dossier.get("contradictions") or dossier.get("conflicts"))
    contradictions = []
    for item in contradiction_items:
        claims = [part.strip() for part in re.split(r"\bbut\b|\bhowever\b|\bvs\.?\b", item, flags=re.IGNORECASE) if part.strip()]
        claim_a = claims[0] if claims else item
        claim_b = claims[1] if len(claims) > 1 else "Counterclaim remains unresolved"
        contradictions.append(
            {
                "claim_a": claim_a,
                "claim_b": claim_b,
                "analysis": "Both claims appear in the record and require stronger corroboration.",
                "source_indices": [_pick_idx(claim_a), _pick_idx(claim_b)],
            }
        )

    quote_bank = [
        {"quote": item, "speaker": "Record excerpt", "source_idx": _pick_idx(item)}
        for item in _normalize_dossier_items(dossier.get("quotes") or dossier.get("notes"))[:10]
    ]
    unknowns = [
        {"question": item, "why_unknown": "Current source set does not close this gap."}
        for item in _normalize_dossier_items(dossier.get("unknowns") or dossier.get("open_questions"))[:8]
    ]

    spine: list[Dict[str, Any]] = []
    if not timeline_events:
        gaps["timeline_reason"] = "Timeline extraction failed from dossier timeline/chronology and source snippets."
        spine = _build_narrative_spine(official_positions, witness_accounts, contradictions)

    if not sources_used:
        gaps["items"].append("No usable sources extracted from dossier.sources")
    if not timeline_events:
        gaps["items"].append("No timeline events extracted")
    if not key_claims:
        gaps["items"].append("No key claims extracted")

    return {
        "topic": topic,
        "sources_used": sources_used,
        "timeline_events": timeline_events,
        "spine": spine,
        "key_claims": key_claims,
        "witness_accounts": witness_accounts,
        "official_positions": official_positions,
        "skeptical_explanations": skeptical_explanations,
        "contradictions": contradictions,
        "quote_bank": quote_bank,
        "unknowns": unknowns,
        "gaps": gaps,
    }


def _build_cited_sentence(statement: str, source_idx: int, lead_in: str) -> str:
    safe_statement = re.sub(r"\s+", " ", statement).strip().rstrip(".")
    return f"{lead_in} {safe_statement} (Source {source_idx})."


def _pick_source_for_item(item: str, source_count: int, offset: int) -> int:
    if source_count <= 0:
        return 1
    digest = hashlib.sha256(f"{item}|{offset}".encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % source_count) + 1


def _estimate_audio_duration_seconds(path: Path) -> float | None:
    ffprobe_binary = shutil.which("ffprobe")
    if not ffprobe_binary or not path.exists():
        return None

    run = subprocess.run(
        [
            ffprobe_binary,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0:
        return None
    try:
        return float((run.stdout or "").strip())
    except ValueError:
        return None


def _fallback_build_documentary_script(
    blueprint: Dict[str, Any],
    dossier: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    topic = str(blueprint.get("topic") or "this case").strip() or "this case"
    dossier_data = dossier if isinstance(dossier, dict) else {}
    summary = str(dossier_data.get("summary") or blueprint.get("summary") or "").strip()
    sources = dossier_data.get("sources") if isinstance(dossier_data.get("sources"), list) else []

    facts = _normalize_dossier_items(dossier_data.get("facts") or dossier_data.get("key_facts") or dossier_data.get("findings"))
    timeline = _normalize_dossier_items(dossier_data.get("timeline") or dossier_data.get("chronology"))
    claims = _normalize_dossier_items(dossier_data.get("claims"))
    contradictions = _normalize_dossier_items(dossier_data.get("contradictions") or dossier_data.get("conflicts"))
    notes = _normalize_dossier_items(dossier_data.get("notes"))

    evidence_pool = [item for item in [summary, *facts, *timeline, *claims, *contradictions, *notes] if item]
    if not evidence_pool:
        evidence_pool = [f"The dossier on {topic} is available, but structured fact fields are sparse."]

    scene_blueprint = [
        "Hook: Why this case still matters",
        "Timeline Start: First recorded events",
        "Timeline Build: Events that escalated attention",
        "What Primary Records Explicitly Show",
        "Witness and Expert Accounts",
        "Official Responses and Institutional Position",
        "Competing Explanations",
        "Evidence Quality Grading",
        "Contradictions Inside the Record",
        "What Is Supported vs. What Is Inferred",
        "Persistent Unknowns",
        "Closing: What Comes Next",
    ]

    scene_count = min(20, max(12, len(scene_blueprint)))
    target_total_words = 7200
    target_scene_words = max(360, target_total_words // scene_count)
    source_count = max(1, len(sources))

    lead_ins = [
        "The dossier states",
        "The documented record notes",
        "Research notes indicate",
        "One source summary reports",
        "A corroborating entry explains",
    ]

    scenes: list[Dict[str, Any]] = []
    pool_index = 0
    for idx in range(scene_count):
        scene_title = scene_blueprint[idx]
        scene_target = target_scene_words + ((idx % 3) - 1) * 40
        lines: list[str] = []
        sources_used: set[int] = set()

        while len(" ".join(lines).split()) < scene_target:
            datum = evidence_pool[pool_index % len(evidence_pool)]
            source_idx = _pick_source_for_item(datum, source_count, idx + pool_index)
            lead_in = lead_ins[(pool_index + idx) % len(lead_ins)]
            lines.append(_build_cited_sentence(datum, source_idx, lead_in))
            if (pool_index + idx) % 4 == 0:
                lines.append(
                    f"At this stage, the narration distinguishes direct evidence from interpretation so the audience can track certainty without losing the timeline (Source {source_idx})."
                )
            sources_used.add(source_idx)
            pool_index += 1

        voiceover = " ".join(lines).strip()
        scenes.append(
            {
                "scene_number": idx + 1,
                "scene_title": scene_title,
                "visual_notes": (
                    "Archival stills, timeline overlays, source callouts, and map/context imagery synchronized to cited dossier points."
                ),
                "on_screen_text": [
                    f"Scene {idx + 1}: {scene_title}",
                    f"Evidence references: {', '.join(f'Source {s}' for s in sorted(sources_used)[:3])}",
                ],
                "voiceover": voiceover,
                "sources_used": sorted(sources_used),
            }
        )

    narration_text = "\n\n".join(str(scene.get("voiceover") or "").strip() for scene in scenes)
    word_count = len([token for token in narration_text.split() if token.strip()])
    return {
        "topic": topic,
        "scenes": scenes,
        "text": narration_text,
        "word_count": word_count,
        "word_count_actual": word_count,
        "estimated_minutes": round(word_count / 160.0, 2),
        "scene_count": len(scenes),
        "locked": False,
        "engine": "fallback",
    }


def _generate_scene_mp3s(
    bundle: Dict[str, Any],
    openai_key: str,
    tts_model: str = "gpt-4o-mini-tts",
    tts_voice: str = "alloy",
) -> Dict[str, Any]:
    script_bundle = bundle.get("script") if isinstance(bundle.get("script"), dict) else {}
    scenes = script_bundle.get("scenes") if isinstance(script_bundle.get("scenes"), list) else []
    if not scenes:
        raise RuntimeError("Generate Script first to create scene voiceover blocks.")

    if not openai_key.strip():
        raise RuntimeError("OpenAI API key required for scene MP3 generation.")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenAI SDK is not installed; add `openai` to requirements.txt.") from exc

    client = OpenAI(api_key=openai_key)
    base_dir = (
        Path(tempfile.gettempdir())
        / "uappress_artifacts"
        / str(st.session_state.get("last_run_ts") or int(time.time()))
        / "audio"
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    scene_mp3s: list[Dict[str, Any]] = []
    for idx, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue
        voiceover = str(scene.get("voiceover") or "").strip()
        if not voiceover:
            continue

        cache_key = hashlib.sha256(f"{voiceover}|{tts_model}|{tts_voice}".encode("utf-8")).hexdigest()
        output_path = base_dir / f"scene_{idx:02d}.mp3"
        cache_path = base_dir / f"scene_{idx:02d}.sha256"

        cached_hash = cache_path.read_text(encoding="utf-8").strip() if cache_path.exists() else ""
        if not output_path.exists() or cached_hash != cache_key:
            response = client.audio.speech.create(model=tts_model, voice=tts_voice, input=voiceover)
            response.stream_to_file(str(output_path))
            cache_path.write_text(cache_key, encoding="utf-8")

        duration_sec = _estimate_audio_duration_seconds(output_path)
        scene["audio_path"] = str(output_path)
        scene["audio_duration_sec"] = duration_sec
        scene_mp3s.append(
            {
                "scene_number": idx,
                "audio_path": str(output_path),
                "audio_duration_sec": duration_sec,
            }
        )

    ffmpeg_binary = shutil.which("ffmpeg")
    merged_path = base_dir / "full_narration.mp3"
    if ffmpeg_binary and scene_mp3s:
        concat_manifest = base_dir / "scene_audio_concat.txt"
        concat_manifest.write_text(
            "\n".join(f"file '{Path(item['audio_path']).resolve().as_posix()}'" for item in scene_mp3s) + "\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                ffmpeg_binary,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_manifest),
                "-c",
                "copy",
                str(merged_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    total_duration = sum(float(item.get("audio_duration_sec") or 0.0) for item in scene_mp3s)
    audio_bundle = {
        "scene_mp3s": scene_mp3s,
        "audio_path": str(merged_path) if merged_path.exists() else "",
        "duration_seconds": total_duration,
        "engine": "openai-tts",
    }
    bundle["audio"] = audio_bundle
    return audio_bundle


def build_beat_sheet(fact_pack: Dict[str, Any], runtime_minutes: int = 45) -> Dict[str, Any]:
    contradictions = fact_pack.get("contradictions") if isinstance(fact_pack.get("contradictions"), list) else []
    strongest = contradictions[0] if contradictions else {
        "claim_a": "Evidence points one way",
        "claim_b": "Alternative reading remains plausible",
    }
    topic = str(fact_pack.get("topic") or "the case")
    beats_per_act = max(3, int(runtime_minutes / 6))

    return {
        "hook": f"A single unresolved event changed how {topic} is remembered.",
        "stakes": "Public trust, policy decisions, and witness credibility are on the line.",
        "act_1": [f"Beat {i+1}: establish record and first tension spike." for i in range(beats_per_act)],
        "act_2": [f"Beat {i+1}: escalate contradiction and pressure test claims." for i in range(beats_per_act)],
        "midpoint_reversal": f"The strongest contradiction emerges: {strongest.get('claim_a')} versus {strongest.get('claim_b')}.",
        "act_3": [f"Beat {i+1}: converge sources and test what survives scrutiny." for i in range(beats_per_act)],
        "final_synthesis": "Only claims supported by corroborated evidence are carried to the conclusion.",
        "unresolved_questions": [item.get("question") for item in fact_pack.get("unknowns", []) if isinstance(item, dict) and item.get("question")],
        "emotional_arc": ["Unease", "Curiosity", "Doubt", "Shock", "Clarity", "Measured uncertainty"],
        "escalation_interval_sec": "60-90",
        "climax_contradiction": strongest,
    }


def _scene_similarity_key(text: str) -> tuple[str, str]:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    if not normalized:
        return "", ""
    first_sentence = re.split(r"(?<=[.!?])\s+", normalized)[0]
    return first_sentence[:180], normalized[:700]


def _clean_fact_text(value: str) -> str:
    no_url = re.sub(r"https?://\S+", "", value or "")
    no_mark = re.sub(r"\[S\d+\]", "", no_url)
    return re.sub(r"\s+", " ", no_mark).strip(" .")


def plan_scenes(
    fact_pack: Dict[str, Any],
    beat_sheet: Dict[str, Any],
    runtime_minutes: int = 45,
    scene_count: int = DEFAULT_SCENE_COUNT,
) -> list[Dict[str, Any]]:
    scene_count = min(20, max(12, int(scene_count)))
    sources = fact_pack.get("sources_used") if isinstance(fact_pack.get("sources_used"), list) else []
    source_ids = [f"S{int(item.get('idx') or i + 1)}" for i, item in enumerate(sources) if isinstance(item, dict)] or ["S1", "S2"]
    timeline_events = fact_pack.get("timeline_events") if isinstance(fact_pack.get("timeline_events"), list) else []

    narrative_items: list[str] = []
    for event in timeline_events:
        if isinstance(event, dict):
            text = _clean_fact_text(str(event.get("event") or ""))
            if text:
                narrative_items.append(text)
    if not narrative_items:
        for key in ("witness_accounts", "official_positions", "contradictions", "key_claims", "skeptical_explanations"):
            records = fact_pack.get(key) if isinstance(fact_pack.get(key), list) else []
            for item in records:
                if isinstance(item, dict):
                    text = _clean_fact_text(str(item.get("summary") or item.get("statement") or item.get("claim") or item.get("reason") or item.get("beat") or ""))
                    if text:
                        narrative_items.append(text)
    if not narrative_items:
        narrative_items = ["Recorded facts remain incomplete", "Independent confirmation remains limited", "Several claims remain disputed"]

    shift_cycle = ["Reveal → Complication", "Complication → Contradiction", "Contradiction → Reversal", "Reversal → Higher stakes"]
    act_break_1 = max(4, int(scene_count * 0.3))
    act_break_2 = max(act_break_1 + 4, int(scene_count * 0.75))
    scenes: list[Dict[str, Any]] = []
    for idx in range(scene_count):
        scene_no = idx + 1
        act = "Act1" if scene_no <= act_break_1 else ("Act2" if scene_no <= act_break_2 else "Act3")
        base = narrative_items[idx % len(narrative_items)]
        evidence_ids = [source_ids[(idx + j) % len(source_ids)] for j in range(min(3, len(source_ids)))]
        anchor_facts = [
            narrative_items[(idx + step) % len(narrative_items)]
            for step in range(2)
        ]
        scenes.append(
            {
                "scene_number": scene_no,
                "scene_title": f"{['Cold Open','Record Opens','Pressure Rises','Contradiction Point','Reversal','Final Synthesis'][idx % 6]} {scene_no}",
                "goal": f"Advance the record with a new verified angle: {base}",
                "tension_shift": shift_cycle[idx % len(shift_cycle)],
                "evidence_ids": evidence_ids,
                "anchor_facts": anchor_facts,
                "open_questions": [str((beat_sheet.get("unresolved_questions") or ["What remains disputed is still unresolved."])[0])],
                "act": act,
            }
        )
    return scenes


def write_scene(
    scene_plan_item: Dict[str, Any],
    fact_pack: Dict[str, Any],
    style_cfg: Dict[str, Any],
    *,
    retry_instruction: str = "",
) -> Dict[str, Any]:
    target_words = int(style_cfg.get("target_scene_words") or 420)
    evidence_ids = scene_plan_item.get("evidence_ids") if isinstance(scene_plan_item.get("evidence_ids"), list) else []
    anchor_facts = scene_plan_item.get("anchor_facts") if isinstance(scene_plan_item.get("anchor_facts"), list) else []
    safe_facts = [_clean_fact_text(str(item)) for item in anchor_facts if _clean_fact_text(str(item))]
    if not safe_facts:
        safe_facts = ["What we can confirm is limited.", "What remains disputed is central to the case."]

    paragraphs: list[str] = []
    sentence_cursor = 0
    scene_number = int(scene_plan_item.get("scene_number") or 0)
    angle = _clean_fact_text(str(scene_plan_item.get("rewrite_angle") or ""))
    templates = [
        f"Witness accounts from phase {scene_number} sharpen the timeline around one verifiable event",
        f"The archived record in phase {scene_number} pivots when independent reports align on a single detail",
        f"In phase {scene_number}, the sequence no longer appears straightforward once multiple testimonies are compared",
        f"A contradiction in phase {scene_number} forces direct comparison of competing claims and dates",
        f"By the close of phase {scene_number}, the dispute shifts from speculation to measurable consequences",
    ]
    while len("\n\n".join(paragraphs).split()) < target_words:
        s1 = f"{templates[sentence_cursor % len(templates)]}. [{evidence_ids[sentence_cursor % len(evidence_ids)] if evidence_ids else 'S1'}]"
        fact = safe_facts[sentence_cursor % len(safe_facts)]
        s2 = f"{fact}. [{evidence_ids[(sentence_cursor + 1) % len(evidence_ids)] if evidence_ids else 'S1'}]"
        angle_line = angle or f"At marker {scene_number}-{sentence_cursor + 1}, witnesses connect the dispute to {safe_facts[(sentence_cursor + 2) % len(safe_facts)].lower()}"
        s3 = f"{angle_line}. [{evidence_ids[(sentence_cursor + 2) % len(evidence_ids)] if evidence_ids else 'S1'}]"
        aftermath = safe_facts[(sentence_cursor + 1) % len(safe_facts)].lower()
        s4 = f"Cycle {sentence_cursor + 1} reports ripple into later testimony, especially where records mention {aftermath}."
        paragraph = " ".join([s1, s2, s3, s4])
        paragraphs.append(paragraph)
        sentence_cursor += 1

    voiceover = "\n\n".join(paragraphs)
    for banned in PLANNING_LEAK_PATTERNS:
        voiceover = re.sub(re.escape(banned), "", voiceover, flags=re.IGNORECASE)
    voiceover = re.sub(r"\s+", " ", voiceover).strip() if style_cfg.get("tts_mode") else voiceover

    used = sorted(set(re.findall(r"\[(S\d+)\]", voiceover)))
    return {
        "scene_number": int(scene_plan_item.get("scene_number") or 0),
        "scene_title": str(scene_plan_item.get("scene_title") or "Untitled Scene"),
        "visual_notes": [f"Archival visual beat {scene_plan_item.get('scene_number')}", "Map overlays", "Evidence lower thirds"],
        "on_screen_text": [str(scene_plan_item.get("tension_shift") or "")],
        "voiceover": voiceover,
        "sources_used": used,
        "word_count": len(voiceover.split()),
    }


def validate_script(scenes: list[Dict[str, Any]]) -> Dict[str, Any]:
    issues: list[str] = []
    first_sentences: list[tuple[str, int]] = []
    corpus = []
    failing_scene_indexes: set[int] = set()
    seen_first_sentence: dict[str, int] = {}

    for index, scene in enumerate(scenes):
        voiceover = str(scene.get("voiceover") or "").strip()
        corpus.append(voiceover.lower())
        p = [seg.strip().lower() for seg in re.split(r"\n\s*\n", voiceover) if seg.strip()]
        if len(p) != len(set(p)):
            issues.append(f"Duplicate paragraphs detected in scene {index + 1}")
            failing_scene_indexes.add(index)
        first = re.split(r"(?<=[.!?])\s+", voiceover.strip())[0].strip().lower()
        if first:
            first_sentences.append((first, index))
            prior_idx = seen_first_sentence.get(first)
            if prior_idx is not None:
                issues.append(f"Repeated first sentence across scenes: {prior_idx + 1} and {index + 1}")
                failing_scene_indexes.add(prior_idx)
                failing_scene_indexes.add(index)
            else:
                seen_first_sentence[first] = index
        lower = voiceover.lower()
        if contains_planning_language(lower):
            issues.append(f"Planning language leakage in scene {scene.get('scene_number')}")
            failing_scene_indexes.add(index)
        citation_hits = len(re.findall(r"\[S\d+\]", voiceover))
        if citation_hits > max(22, int(len(voiceover.split()) / 8)):
            issues.append(f"Citation clustering too dense in scene {scene.get('scene_number')}")
            failing_scene_indexes.add(index)

        lower_no_citations = re.sub(r"\[s\d+\]", "", lower)
        tokens = re.findall(r"\b\w+\b", lower_no_citations)
        for n in (8, 10, 12):
            if len(tokens) < n:
                continue
            seen_ngrams: dict[str, int] = {}
            for start in range(0, len(tokens) - n + 1):
                ngram = " ".join(tokens[start:start + n])
                seen_ngrams[ngram] = seen_ngrams.get(ngram, 0) + 1
            if any(count >= 30 for count in seen_ngrams.values()):
                issues.append(f"High n-gram repetition detected in scene {index + 1}")
                failing_scene_indexes.add(index)
                break

    for i in range(len(corpus)):
        for j in range(i + 1, len(corpus)):
            a = set(corpus[i].split())
            b = set(corpus[j].split())
            if not a or not b:
                continue
            sim = len(a & b) / max(1, len(a | b))
            if sim > 0.985:
                issues.append(f"Scene-to-scene similarity too high: {i+1} vs {j+1}")
                failing_scene_indexes.add(i)
                failing_scene_indexes.add(j)
                break

    total_words = sum(len(str(scene.get("voiceover") or "").split()) for scene in scenes)
    stats = {
        "scene_count": len(scenes),
        "word_count": total_words,
        "estimated_runtime_minutes": round(total_words / WORDS_PER_MINUTE, 2),
    }
    return {
        "ok": len(issues) == 0,
        "issues": sorted(set(issues))[:12],
        "stats": stats,
        "failing_scene_indexes": sorted(failing_scene_indexes),
    }


def build_timing_map(scenes: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    timing: list[Dict[str, Any]] = []
    start_sec = 0.0
    for scene in scenes:
        scene_number = int(scene.get("scene_number") or len(timing) + 1)
        duration = float(scene.get("audio_duration_sec") or 0.0)
        if duration <= 0:
            duration = round(len(str(scene.get("voiceover") or "").split()) / WORDS_PER_MINUTE * 60.0, 2)
        end_sec = start_sec + duration
        timing.append(
            {
                "scene_number": scene_number,
                "start_sec": round(start_sec, 2),
                "end_sec": round(end_sec, 2),
                "duration_sec": round(duration, 2),
                "word_count": len(str(scene.get("voiceover") or "").split()),
            }
        )
        start_sec = end_sec
    return timing


def generate_script(bundle: Dict[str, Any], runtime_minutes: int = 45, tts_mode: bool = True) -> Dict[str, Any]:
    fact_pack = bundle.get("fact_pack") if isinstance(bundle.get("fact_pack"), dict) else None
    beat_sheet = bundle.get("beat_sheet") if isinstance(bundle.get("beat_sheet"), dict) else None
    if not fact_pack:
        raise RuntimeError("Build Fact Pack first")
    if not beat_sheet:
        raise RuntimeError("Build Beat Sheet first")

    scene_plan_items = plan_scenes(fact_pack, beat_sheet, runtime_minutes=runtime_minutes, scene_count=DEFAULT_SCENE_COUNT)
    style_cfg = {
        "tts_mode": tts_mode,
        "target_scene_words": max(280, int((runtime_minutes * WORDS_PER_MINUTE) / max(1, len(scene_plan_items)))),
    }
    scenes = [write_scene(item, fact_pack, style_cfg) for item in scene_plan_items]

    validator_report = validate_script(scenes)
    for attempt in range(2):
        if validator_report.get("ok"):
            break
        failing_indexes = set(validator_report.get("failing_scene_indexes") or [])
        if not failing_indexes:
            failing_indexes = {len(scene_plan_items) - 1}
        for idx in sorted(failing_indexes):
            current = dict(scene_plan_items[idx])
            ids = current.get("evidence_ids") if isinstance(current.get("evidence_ids"), list) else []
            if ids:
                current["evidence_ids"] = ids[1:] + ids[:1]
            current["rewrite_angle"] = f"Fresh evidentiary angle attempt {attempt + 1} for scene {idx + 1}"
            scene_plan_items[idx] = current
            scenes[idx] = write_scene(
                current,
                fact_pack,
                style_cfg,
                retry_instruction="DO NOT include any meta/planning language",
            )
        validator_report = validate_script(scenes)

    if not validator_report.get("ok"):
        bundle["validator_report"] = validator_report
        raise RuntimeError("Script validator failed. Regenerate failing scenes and try again.")

    script_text = "\n\n".join(
        f"--- Scene {int(scene.get('scene_number') or 0)}: {scene.get('scene_title') or ''} ---\n{str(scene.get('voiceover') or '').strip()}"
        for scene in scenes
    )
    total_words = sum(int(scene.get("word_count") or 0) for scene in scenes)
    source_map = {f"S{int(s.get('idx') or i+1)}": {"idx": int(s.get('idx') or i+1), "title": s.get("title"), "url": s.get("url")} for i, s in enumerate((fact_pack.get("sources_used") or [])) if isinstance(s, dict)}

    result = {
        "topic": fact_pack.get("topic"),
        "scenes": scenes,
        "scene_count": len(scenes),
        "target_words": runtime_minutes * WORDS_PER_MINUTE,
        "word_count": total_words,
        "estimated_minutes": round(total_words / WORDS_PER_MINUTE, 2),
        "tts_mode": tts_mode,
        "source_map": source_map,
        "text": script_text,
        "locked": False,
        "engine": "planner-writer-validator-v1",
        "validator": validator_report,
    }

    bundle["scene_plan"] = {
        "scene_count": len(scene_plan_items),
        "runtime_minutes": runtime_minutes,
        "scenes": scene_plan_items,
    }
    bundle["script"] = result
    bundle["script_text"] = script_text
    bundle["validator_report"] = validator_report
    bundle["timing"] = build_timing_map(scenes)
    return result


def _fallback_build_scene_plan(blueprint: Dict[str, Any], script_result: Dict[str, Any]) -> Dict[str, Any]:
    script_scenes = script_result.get("scenes") if isinstance(script_result.get("scenes"), list) else []
    if script_scenes:
        scenes = [
            {
                "scene": int(scene.get("scene_number") or (idx + 1)),
                "heading": str(scene.get("scene_title") or f"Scene {idx + 1}"),
                "visual": str(scene.get("visual_notes") or "Supporting visuals"),
            }
            for idx, scene in enumerate(script_scenes)
            if isinstance(scene, dict)
        ]
    else:
        sections = blueprint.get("sections") or []
        scenes = [
            {
                "scene": idx + 1,
                "heading": str(section.get("heading") or f"Scene {idx + 1}"),
                "visual": str(section.get("objective") or "Supporting visuals"),
            }
            for idx, section in enumerate(sections)
        ]
    return {
        "scene_count": len(scenes),
        "scenes": scenes,
        "script_word_count": script_result.get("word_count", 0),
        "engine": "fallback",
    }


def _fallback_build_video_asset(
    image_result: Dict[str, Any] | None,
    audio_result: Dict[str, Any] | None,
    subtitles_result: Dict[str, Any] | None,
    scene_plan: Dict[str, Any],
    smoke: bool = False,
) -> Dict[str, Any]:
    mode = "smoke-fallback" if smoke else "fallback"
    image_paths = []
    if isinstance(image_result, dict):
        candidate_paths = image_result.get("images")
        if isinstance(candidate_paths, list):
            image_paths = [str(path).strip() for path in candidate_paths if str(path).strip()]

    audio_paths: list[str] = []
    if isinstance(audio_result, dict):
        scene_mp3s = audio_result.get("scene_mp3s") if isinstance(audio_result.get("scene_mp3s"), list) else []
        audio_paths = [str(item.get("path") or "").strip() for item in scene_mp3s if isinstance(item, dict) and str(item.get("path") or "").strip()]

    missing_artifacts = _collect_missing_artifacts(image_result, audio_result)
    preflight_errors: list[str] = []

    ffmpeg_check = _ffmpeg_preflight()
    ffmpeg_binary = str((ffmpeg_check.get("details") or {}).get("ffmpeg_path") or "")
    if not ffmpeg_check.get("ok"):
        preflight_errors.append(str((ffmpeg_check.get("details") or {}).get("message") or "ffmpeg not available"))

    if missing_artifacts:
        preflight_errors.extend(missing_artifacts)

    video_result: Dict[str, Any] = {
        "mode": mode,
        "mp4_path": None,
        "sha256": None,
        "scene_count": int(scene_plan.get("scene_count") or 0),
        "has_images": bool(image_result),
        "has_audio": bool(audio_result),
        "has_subtitles": bool(subtitles_result),
    }

    if preflight_errors:
        video_result["error"] = "Preflight failed: " + " | ".join(preflight_errors)
        video_result["error_details"] = {
            "preflight_errors": preflight_errors,
            "ffmpeg": ffmpeg_check.get("details"),
        }
        return video_result

    first_audio = Path(audio_paths[0]).resolve()
    base_video_dir = first_audio.parent.parent / "video"
    base_video_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = base_video_dir / "fallback_assembly.mp4"
    concat_file = base_video_dir / "fallback_images.txt"

    # Keep each scene on screen long enough to cover the audio duration.
    per_scene_duration = 2.0
    if isinstance(audio_result, dict):
        duration_seconds = float(audio_result.get("duration_seconds") or 0)
        if duration_seconds > 0 and image_paths:
            per_scene_duration = max(0.5, duration_seconds / len(image_paths))

    concat_lines: list[str] = []
    for image_path in image_paths:
        resolved = Path(image_path).resolve()
        concat_lines.append(f"file '{resolved.as_posix()}'")
        concat_lines.append(f"duration {per_scene_duration:.3f}")
    # Repeat final frame per ffmpeg concat demuxer requirements.
    concat_lines.append(f"file '{Path(image_paths[-1]).resolve().as_posix()}'")
    concat_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    audio_concat = base_video_dir / "fallback_audio.txt"
    merged_audio_path = base_video_dir / "merged_audio.mp3"
    audio_concat.write_text("\n".join([f"file '{Path(p).resolve().as_posix()}'" for p in audio_paths]) + "\n", encoding="utf-8")
    audio_merge_cmd = [str(ffmpeg_binary), "-y", "-f", "concat", "-safe", "0", "-i", str(audio_concat), "-c", "copy", str(merged_audio_path)]
    audio_merge_run = subprocess.run(audio_merge_cmd, capture_output=True, text=True, check=False)
    if audio_merge_run.returncode != 0:
        raise RuntimeError(audio_merge_run.stderr.strip() or "ffmpeg failed merging scene mp3 files")

    ffmpeg_cmd = [
        str(ffmpeg_binary),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-i",
        str(merged_audio_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(mp4_path),
    ]

    ffmpeg_run = subprocess.run(
        ffmpeg_cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if ffmpeg_run.returncode != 0:
        stderr = ffmpeg_run.stderr.strip() or "ffmpeg returned non-zero exit status"
        video_result["error"] = stderr
        video_result["error_details"] = {
            "ffmpeg_cmd": ffmpeg_cmd,
            "stderr": ffmpeg_run.stderr,
            "stdout": ffmpeg_run.stdout,
        }
        raise RuntimeError(stderr)

    if not mp4_path.exists():
        error_message = f"ffmpeg reported success but output file missing: {mp4_path}"
        video_result["error"] = error_message
        raise RuntimeError(error_message)

    digest = hashlib.sha256(mp4_path.read_bytes()).hexdigest()
    video_result["mp4_path"] = str(mp4_path)
    video_result["sha256"] = digest
    return video_result


@dataclass
class EngineSurface:
    run_research: Any
    score_topic: Any
    build_documentary_blueprint: Any
    build_documentary_script: Any
    compile_voiceover_script: Any
    build_scene_plan: Any
    build_video_asset: Any

# ------------------------------------------------------------------------------
# Import-time safe research function
# ------------------------------------------------------------------------------
ENGINE_IMPORT_OK = False
ENGINE_IMPORT_TARGET = ""

_engine_module = None
_engine_run_research = None
_engine_research_job = None
_import_failures: list[str] = []

for module_name in (
    "apps.research.uappress_engine_v9",
    "apps.research.research_engine",
    "uappress_engine_v9",
    "research_engine",
):
    try:
        candidate = importlib.import_module(module_name)
        candidate_run = getattr(candidate, "run_research")
    except Exception as exc:
        _import_failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
        continue

    _engine_module = candidate
    _engine_run_research = candidate_run
    _engine_research_job = getattr(candidate, "ResearchJob", None)
    ENGINE_IMPORT_OK = True
    ENGINE_IMPORT_TARGET = module_name
    break

if ENGINE_IMPORT_OK:
    run_research = _engine_run_research  # type: ignore[assignment]

    if _engine_research_job is None:
        @dataclass
        class ResearchJob:  # type: ignore
            primary_topic: str
    else:
        ResearchJob = _engine_research_job  # type: ignore[assignment]

    def _run_research_adapter(**kwargs: Any) -> Dict[str, Any]:
        call_kwargs = dict(kwargs)
        if "job" in call_kwargs and "primary_topic" not in call_kwargs:
            job_value = call_kwargs.pop("job")
            call_kwargs["primary_topic"] = str(
                getattr(job_value, "primary_topic", "")
            ).strip()
        return _engine_run_research(**call_kwargs)  # type: ignore[misc]

    run_research = _run_research_adapter
    ENGINE_IMPORT_STATUS = "PRIMARY"
    ENGINE_IMPORT_MARKER = "TEST_HOOK:ENGINE_IMPORT_PRIMARY"
else:
    ENGINE_IMPORT_EXCEPTION_TRACE = " | ".join(_import_failures) or "No import attempts recorded."

if not ENGINE_IMPORT_OK:
    @st.cache_data(show_spinner=False)
    def _fallback_job_type() -> str:
        return "fallback"

    @dataclass
    class ResearchJob:  # type: ignore
        primary_topic: str

    def run_research(**kwargs) -> Dict[str, Any]:
        # Safe placeholder: never crashes UI
        return {
            "status": "PRELIMINARY",
            "confidence_overall": 0.62,
            "note": "run_research import not wired yet (fallback stub).",
            "args": {k: ("***" if "key" in k.lower() else v) for k, v in kwargs.items()},
            "job_type": _fallback_job_type(),
        }


engine = EngineSurface(
    run_research=run_research,
    score_topic=_fallback_score_topic,
    build_documentary_blueprint=_fallback_build_documentary_blueprint,
    build_documentary_script=_fallback_build_documentary_script,
    compile_voiceover_script=_fallback_compile_voiceover_script,
    build_scene_plan=_fallback_build_scene_plan,
    build_video_asset=_fallback_build_video_asset,
)


def _to_jsonable(value: Any) -> Any:
    """Recursively convert values into JSON-serializable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, BaseException):
        return {
            "type": value.__class__.__name__,
            "message": str(value),
        }

    if is_dataclass(value):
        return _to_jsonable(asdict(value))

    if isinstance(value, dict):
        return {
            str(k): _to_jsonable(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]

    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))

    return str(value)


def _create_research_job(primary_topic: str) -> Any:
    attempted_kwargs = {"primary_topic": primary_topic}
    try:
        return ResearchJob(**attempted_kwargs)
    except TypeError as exc:
        module_name = getattr(ResearchJob, "__module__", "<unknown>")
        try:
            ctor_sig = str(inspect.signature(ResearchJob))
        except Exception:
            ctor_sig = "<unavailable>"
        raise RuntimeError(
            "ResearchJob constructor mismatch "
            f"(loaded={module_name}.ResearchJob, signature={ctor_sig}, "
            f"attempted_kwargs={list(attempted_kwargs.keys())})"
        ) from exc


def _assert_run_research_signature() -> None:
    """Fail fast with a clearer error if callback wiring/signature drifts."""
    try:
        sig = inspect.signature(engine.run_research)
    except Exception:
        return

    params = sig.parameters
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if accepts_kwargs or "job" in params:
        return

    raise RuntimeError(
        "run_research signature mismatch: expected a 'job' parameter or **kwargs "
        f"but got {sig}"
    )


# ------------------------------------------------------------------------------
# Deterministic modes
# ------------------------------------------------------------------------------
# Explicit envs (optional) + robust CI detection via ci_smoke_enabled()
SMOKE_MODE = (
    _os.getenv("UAPPRESS_SMOKE", "").strip() == "1"
    or _os.getenv("UAPPRESS_CI_SMOKE", "").strip() == "1"
    or ci_smoke_enabled()
)


# ------------------------------------------------------------------------------
# Page Config
# ------------------------------------------------------------------------------
st.set_page_config(page_title="UAPpress Research Engine", layout="wide")
st.title("UAPpress Research Engine")
st.caption("DEPLOY_STAMP: 2026-02-17-B")

ui_version = _os.getenv("UAPPRESS_UI_VERSION", "dev-local").strip() or "dev-local"
st.caption(f"UI Version: {ui_version}")

# Stable marker for Playwright to know Streamlit hydrated
st.caption("TEST_HOOK:APP_LOADED")
st.caption(ENGINE_IMPORT_MARKER)
st.caption(f"TEST_HOOK:RESEARCHJOB_MODULE:{getattr(ResearchJob, '__module__', '<unknown>')}")


# ------------------------------------------------------------------------------
# Sidebar — Connections + Mode
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("Connections")

    serpapi_key = st.text_input(
        "SerpAPI Key",
        type="password",
        help="Bring Your Own SerpAPI key",
        key="serpapi_key_input",
    )

    openai_key = st.text_input(
        "OpenAI Key (optional)",
        type="password",
        help="Optional — only required if LLM refinement is enabled",
        key="openai_key_input",
    )

    st.divider()
    st.header("Mode")

    mode = st.radio(
        "Select Mode",
        ["Simple", "Pro"],
        index=0,
        help="Simple = optimized defaults. Pro = full control.",
        key="mode_radio",
    )

    if mode == "Pro":
        st.divider()
        st.subheader("Pro Settings")

        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.4,
            0.9,
            0.60,
            0.05,
            key="confidence_threshold_slider",
        )

        max_serp_queries = st.slider(
            "Max SERP Queries",
            5,
            30,
            12,
            key="max_serp_queries_slider",
        )

        max_sources = st.slider(
            "Max Sources Ingested",
            10,
            50,
            25,
            key="max_sources_slider",
        )

        include_gov_docs = st.checkbox(
            "Include .gov/.mil focus",
            value=True,
            key="include_gov_docs_checkbox",
        )
    else:
        # Simple mode defaults (keep stable, deterministic)
        confidence_threshold = 0.58
        max_serp_queries = 12
        max_sources = 25
        include_gov_docs = True

    st.divider()
    st.caption(f"ENGINE_IMPORT: {ENGINE_IMPORT_STATUS}")
    if ENGINE_IMPORT_STATUS == "FALLBACK":
        st.caption(
            "ENGINE_IMPORT_EXCEPTION: "
            + (
                ENGINE_IMPORT_EXCEPTION_TRACE
                or "Unavailable (no exception details captured)."
            )
        )
    st.caption(f"APP_FILE: {__file__}")

    st.divider()
    if SMOKE_MODE:
        st.success("Smoke mode enabled — no API keys required.")
        st.caption("TEST_HOOK:SMOKE_MODE")


# ------------------------------------------------------------------------------
# Main Page — Minimal Inputs (stable render order)
# ------------------------------------------------------------------------------
st.subheader("Topic Intelligence")

if "last_topic_score" not in st.session_state:
    st.session_state["last_topic_score"] = None

topic_idea = st.text_input(
    "Topic Idea",
    placeholder="Example: Shag Harbor Incident",
    key="topic_idea_input",
)

if st.button("Score Topic", key="btn_topic_score"):
    if not topic_idea.strip():
        st.warning("Please enter a topic idea to score.")
    else:
        try:
            st.session_state["last_topic_score"] = engine.score_topic(
                topic_idea,
                serpapi_key=serpapi_key or None,
                smoke=SMOKE_MODE,
            )
        except Exception as exc:
            st.error(f"Topic scoring error: {str(exc)}")

topic_score = st.session_state.get("last_topic_score")
if topic_score:
    recommendation = topic_score.get("recommendation", "MAYBE")
    badge_type = {
        "GREENLIGHT": "success",
        "MAYBE": "warning",
        "PASS": "error",
    }.get(recommendation, "info")
    getattr(st, badge_type)(f"Recommendation: {recommendation}")
    st.json(topic_score)

if st.button("Use as Primary Topic", key="btn_topic_use_as_primary"):
    if topic_idea.strip():
        st.session_state["primary_topic_input"] = topic_idea
        st.success("Primary Topic updated from Topic Idea.")
    else:
        st.warning("Enter a Topic Idea first.")

st.caption("Topic scoring ready")

st.subheader("Research Topic")

# Initialize session state for deterministic outputs across reruns
if "last_dossier" not in st.session_state:
    st.session_state["last_dossier"] = None
if "last_run_ts" not in st.session_state:
    st.session_state["last_run_ts"] = None
if "run_status" not in st.session_state:
    st.session_state["run_status"] = "IDLE"
if "last_video" not in st.session_state:
    st.session_state["last_video"] = None
if "last_bundle" not in st.session_state:
    st.session_state["last_bundle"] = None
if "script_text" not in st.session_state:
    st.session_state["script_text"] = ""
if "generated_script_text" not in st.session_state:
    st.session_state["generated_script_text"] = ""
if "script_editor_text" not in st.session_state:
    st.session_state["script_editor_text"] = ""
if "last_fact_pack" not in st.session_state:
    st.session_state["last_fact_pack"] = None
if "last_beat_sheet" not in st.session_state:
    st.session_state["last_beat_sheet"] = None
if "last_timing" not in st.session_state:
    st.session_state["last_timing"] = None
if "last_validator_report" not in st.session_state:
    st.session_state["last_validator_report"] = None


# Atomic submit to prevent rerun races that break Playwright clicks
with st.form("research_form", clear_on_submit=False):
    primary_topic = st.text_input(
        "Primary Topic",
        placeholder="Example: ODNI UAP Report 2023",
        key="primary_topic_input",
    )

    run_button = st.form_submit_button(
        "Run Research",
        use_container_width=True,
        key="submit_research_run",
    )


# ------------------------------------------------------------------------------
# Deterministic mock dossier (CI smoke fixture)
# ------------------------------------------------------------------------------
def _mock_dossier(topic: str) -> Dict[str, Any]:
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


# ------------------------------------------------------------------------------
# Run Logic
# ------------------------------------------------------------------------------
if run_button:
    st.session_state["run_status"] = "RUNNING"
    st.session_state["last_run_ts"] = int(time.time())
    st.session_state["last_images"] = None
    st.session_state["last_audio"] = None
    st.session_state["last_video"] = None

    run_bundle: Dict[str, Any] = {
        "mode": mode,
        "topic": primary_topic,
        "errors": [],
    }
    st.session_state["last_bundle"] = run_bundle

    if not primary_topic:
        st.warning("Please enter a topic.")
        st.session_state["run_status"] = "ERROR"
        st.stop()

    # In non-smoke mode, require SerpAPI key
    if (not serpapi_key) and (not SMOKE_MODE):
        st.warning("SerpAPI key required (or enable smoke mode).")
        st.session_state["run_status"] = "ERROR"
        st.stop()

    st.info("Running research...")
    st.caption("TEST_HOOK:RUN_STARTED")

    try:
        if SMOKE_MODE:
            dossier = _mock_dossier(primary_topic)
        else:
            try:
                job = _create_research_job(primary_topic)
                _assert_run_research_signature()
                dossier = engine.run_research(
                    job=job,
                    serpapi_key=serpapi_key,
                    openai_key=openai_key or None,
                )
            except Exception as exc:
                _append_bundle_error(run_bundle, "research", exc)
                raise

        run_bundle["dossier"] = dossier

        try:
            blueprint = engine.build_documentary_blueprint(dossier)
            run_bundle["blueprint"] = blueprint
        except Exception as exc:
            _append_bundle_error(run_bundle, "blueprint", exc)
            raise

        run_bundle["fact_pack"] = {}
        run_bundle["beat_sheet"] = {}
        run_bundle["script"] = {"text": "", "scenes": [], "locked": False}
        run_bundle["script_text"] = ""
        run_bundle["scene_plan"] = {"scene_count": 0, "scenes": [], "engine": "pending"}
        run_bundle["validator_report"] = {"ok": False, "issues": ["Script not generated yet"], "stats": {"scene_count": 0, "word_count": 0, "estimated_runtime_minutes": 0}}
        run_bundle["timing"] = []
        run_bundle["images"] = {"images": [], "image_count": 0, "engine": "pending"}
        run_bundle["audio"] = {"scene_mp3s": [], "audio_path": "", "duration_seconds": 0, "engine": "pending"}

        run_bundle["readiness"] = _artifact_readiness(run_bundle)

        st.session_state["last_dossier"] = dossier
        st.session_state["last_blueprint"] = blueprint
        st.session_state["last_script"] = run_bundle.get("script")
        st.session_state["script_text"] = ""
        st.session_state["generated_script_text"] = ""
        st.session_state["script_editor_text"] = ""
        st.session_state["last_fact_pack"] = run_bundle.get("fact_pack")
        st.session_state["last_beat_sheet"] = run_bundle.get("beat_sheet")
        st.session_state["last_scene_plan"] = run_bundle.get("scene_plan")
        st.session_state["last_timing"] = run_bundle.get("timing")
        st.session_state["last_validator_report"] = run_bundle.get("validator_report")
        st.session_state["last_images"] = run_bundle.get("images")
        st.session_state["last_audio"] = run_bundle.get("audio")
        st.session_state["last_bundle"] = run_bundle
        st.session_state["run_status"] = "DONE"

    except Exception as e:
        st.session_state["run_status"] = "ERROR"
        run_bundle["readiness"] = _artifact_readiness(run_bundle)
        st.session_state["last_bundle"] = run_bundle
        st.error(f"Unexpected error: {str(e)}")
        st.caption("TEST_HOOK:RUN_ERROR")
        st.stop()


# ------------------------------------------------------------------------------
# Output Rendering (stable; always renders if we have data)
# ------------------------------------------------------------------------------
dossier = st.session_state.get("last_dossier")
blueprint = st.session_state.get("last_blueprint")
script_result = st.session_state.get("last_script")
scene_plan = st.session_state.get("last_scene_plan")

if dossier:
    score = float(dossier.get("confidence_overall", 0) or 0)

    if score >= 0.75:
        quality = "High"
    elif score >= 0.60:
        quality = "Medium"
    else:
        quality = "Preliminary"

    st.success(f"Research Complete — Quality: {quality} ({round(score, 2)})")

    # This is the marker Playwright waits for after clicking Run Research
    mark_run_done()

    st.subheader("Dossier Output")
    st.json(dossier)

    sources = dossier.get("sources") or []
    if isinstance(sources, list) and sources:
        st.subheader("Top Sources")
        for i, s in enumerate(sources, start=1):
            title = str(s.get("title", f"Source {i}"))
            url = str(s.get("url", ""))
            st.markdown(f"{i}. **{title}** — {url}")

    image_result = st.session_state.get("last_images")
    audio_result = st.session_state.get("last_audio")
    subtitles_result = st.session_state.get("last_subtitles")
    scene_plan = st.session_state.get("last_scene_plan") or {}
    bundle = st.session_state.get("last_bundle") or {
        "dossier": dossier,
        "blueprint": blueprint,
        "fact_pack": st.session_state.get("last_fact_pack") or {},
        "beat_sheet": st.session_state.get("last_beat_sheet") or {},
        "script": script_result,
        "script_text": st.session_state.get("script_text") or "",
        "scene_plan": scene_plan,
        "validator_report": st.session_state.get("last_validator_report") or {"ok": False, "issues": ["Script not generated yet"], "stats": {}},
        "timing": st.session_state.get("last_timing") or [],
        "images": image_result,
        "audio": audio_result,
        "video": st.session_state.get("last_video"),
        "errors": [],
    }

    script_bundle = bundle.get("script") if isinstance(bundle.get("script"), dict) else {}
    script_scenes = script_bundle.get("scenes") if isinstance(script_bundle.get("scenes"), list) else []
    generated_script_text = str(bundle.get("script_text") or "").strip()

    if "script_text" not in st.session_state:
        st.session_state["script_text"] = generated_script_text
    if "generated_script_text" not in st.session_state:
        st.session_state["generated_script_text"] = generated_script_text
    if "script_editor_text" not in st.session_state:
        st.session_state["script_editor_text"] = st.session_state.get("script_text", "")

    st.subheader("Documentary Writer v3")
    fact_col, beat_col, script_col, regen_col, save_col, mp3_col = st.columns(6)

    if fact_col.button("Build Fact Pack", use_container_width=True, key="btn_topic_build_fact_pack"):
        bundle["fact_pack"] = build_fact_pack(dossier)
        st.session_state["last_fact_pack"] = bundle["fact_pack"]
        st.success("Fact pack built.")

    fact_pack = bundle.get("fact_pack") if isinstance(bundle.get("fact_pack"), dict) else None
    if fact_pack:
        st.caption(f"Fact Pack: {len(fact_pack.get('sources_used') or [])} sources, {len(fact_pack.get('timeline_events') or [])} timeline events")

    if beat_col.button(
        "Build Beat Sheet",
        use_container_width=True,
        disabled=not bool(fact_pack),
        key="btn_topic_build_beat_sheet",
    ):
        bundle["beat_sheet"] = build_beat_sheet(fact_pack or {}, runtime_minutes=TARGET_DOCUMENTARY_MINUTES)
        st.session_state["last_beat_sheet"] = bundle["beat_sheet"]
        st.success("Beat sheet built.")

    beat_sheet = bundle.get("beat_sheet") if isinstance(bundle.get("beat_sheet"), dict) else None
    can_generate_script = bool(fact_pack and beat_sheet)
    if not (openai_key or _os.getenv("OPENAI_API_KEY")):
        st.warning("LLM not enabled; using deterministic fallback script.")

    if script_col.button(
        "Generate Script",
        use_container_width=True,
        disabled=not can_generate_script,
        key="btn_topic_generate_script",
    ):
        timeline_events = fact_pack.get("timeline_events") if isinstance((fact_pack or {}).get("timeline_events"), list) else []
        spine = fact_pack.get("spine") if isinstance((fact_pack or {}).get("spine"), list) else []
        if len(timeline_events) == 0 and len(spine) == 0:
            st.error("Cannot generate cinematic script: no timeline_events or narrative spine. Rebuild Fact Pack.")
        else:
            try:
                script_generated = generate_script(bundle, runtime_minutes=TARGET_DOCUMENTARY_MINUTES, tts_mode=True)
                script_text = str(script_generated.get("text") or "").strip()
                st.session_state["last_script"] = script_generated
                st.session_state["last_scene_plan"] = bundle.get("scene_plan")
                st.session_state["last_timing"] = bundle.get("timing")
                st.session_state["last_validator_report"] = bundle.get("validator_report")
                st.session_state["generated_script_text"] = script_text
                st.session_state["script_text"] = script_text
                st.session_state["script_editor_text"] = script_text
                updated_images, _ = _ensure_fallback_artifacts(primary_topic, bundle.get("scene_plan") or {}, script_text)
                bundle["images"] = updated_images
                st.session_state["last_images"] = updated_images
                bundle["audio"] = {"scene_mp3s": [], "audio_path": "", "duration_seconds": 0, "engine": "pending"}
                st.session_state["last_audio"] = bundle["audio"]
                st.success("Generated cinematic script.")
            except Exception as exc:
                _append_bundle_error(bundle, "script", exc)
                st.error(str(exc))

    if regen_col.button("Regenerate failing scenes", use_container_width=True, key="btn_topic_regen_failing_scenes"):
        try:
            script_generated = generate_script(bundle, runtime_minutes=TARGET_DOCUMENTARY_MINUTES, tts_mode=True)
            script_text = str(script_generated.get("text") or "").strip()
            st.session_state["last_script"] = script_generated
            st.session_state["last_scene_plan"] = bundle.get("scene_plan")
            st.session_state["last_timing"] = bundle.get("timing")
            st.session_state["last_validator_report"] = bundle.get("validator_report")
            st.session_state["generated_script_text"] = script_text
            st.session_state["script_text"] = script_text
            st.session_state["script_editor_text"] = script_text
            st.success("Regenerated failing scenes.")
        except Exception as exc:
            _append_bundle_error(bundle, "script_regen", exc)
            st.error(str(exc))

    validator_report = bundle.get("validator_report") if isinstance(bundle.get("validator_report"), dict) else {}
    script_validator = bundle.get("script", {}).get("validator") if isinstance(bundle.get("script", {}), dict) else {}
    scenes = bundle.get("script", {}).get("scenes") if isinstance(bundle.get("script", {}), dict) and isinstance(bundle.get("script", {}).get("scenes"), list) else []
    scenes_exist = any(str(scene.get("voiceover") or "").strip() for scene in scenes if isinstance(scene, dict))
    script_pass = bool((script_validator or validator_report).get("ok"))
    if mp3_col.button(
        "Generate MP3s (per scene)",
        use_container_width=True,
        disabled=not (scenes_exist and script_pass),
        key="btn_audio_generate_mp3s_per_scene",
    ):
        try:
            updated_audio = generate_scene_mp3s(bundle, openai_key=openai_key or "", voice=TTS_VOICE, model=TTS_MODEL, smoke=SMOKE_MODE)
            bundle["timing"] = build_timing_map(bundle.get("script", {}).get("scenes") or [])
            st.session_state["last_audio"] = updated_audio
            st.session_state["last_timing"] = bundle.get("timing")
            st.success("Per-scene MP3s generated.")
        except Exception as exc:
            _append_bundle_error(bundle, "audio_tts", exc)
            st.error(str(exc))

    st.text_area("Script (validated)", value=str(bundle.get("script_text") or ""), height=400, disabled=True)

    validator_report = bundle.get("validator_report") if isinstance(bundle.get("validator_report"), dict) else {}
    quality_stats = validator_report.get("stats") if isinstance(validator_report.get("stats"), dict) else {}
    status_label = "PASS" if validator_report.get("ok") else "FAIL"
    st.subheader("Script Quality")
    st.write({
        "status": status_label,
        "scene_count": quality_stats.get("scene_count", len(script_scenes)),
        "word_count": quality_stats.get("word_count", len(str(bundle.get("script_text") or "").split())),
        "estimated_runtime_minutes": quality_stats.get("estimated_runtime_minutes", round(len(str(bundle.get("script_text") or "").split()) / WORDS_PER_MINUTE, 2)),
    })
    if not validator_report.get("ok"):
        st.warning("; ".join((validator_report.get("issues") or ["Validator has unresolved issues"])[:3]))

    timing_map = bundle.get("timing") if isinstance(bundle.get("timing"), list) else []
    if timing_map:
        st.subheader("Timing Map")
        st.json(timing_map)

    current_audio = bundle.get("audio") if isinstance(bundle.get("audio"), dict) else {}
    scene_mp3s = current_audio.get("scene_mp3s") if isinstance(current_audio.get("scene_mp3s"), list) else []
    if scene_mp3s:
        st.caption(f"Audio status: {len(scene_mp3s)} scene mp3s")
        for item in scene_mp3s[:3]:
            path = ""
            if isinstance(item, dict):
                path = str(item.get("path") or item.get("audio_path") or "").strip()
            if path and Path(path).exists():
                st.audio(path)

    if not bundle.get("script", {}).get("scenes"):
        st.warning("Script generation is disabled until fact pack and beat sheet exist, and validator passes for downstream audio/video.")

    st.subheader("Video Assembly")
    ffmpeg_check = _ffmpeg_preflight()
    if not ffmpeg_check.get("ok"):
        ffmpeg_message = str((ffmpeg_check.get("details") or {}).get("message") or "ffmpeg not found on PATH")
        video_bundle = bundle.get("video") if isinstance(bundle.get("video"), dict) else {}
        video_bundle["error"] = "Preflight failed: ffmpeg not found on PATH"
        video_bundle["error_details"] = ffmpeg_check.get("details")
        bundle["video"] = video_bundle
        st.error(ffmpeg_message)

    bundle["readiness"] = _artifact_readiness(bundle)
    readiness = bundle["readiness"]
    missing_labels = readiness.get("missing_labels") or []

    if missing_labels:
        st.warning("Video Assembly is not ready. Missing: " + ", ".join(missing_labels))

    image_result = st.session_state.get("last_images")
    audio_result = st.session_state.get("last_audio")

    if st.button(
        "Assemble Video (MP4)",
        disabled=not readiness.get("ready_for_video", False),
        key="btn_video_assemble_mp4",
    ):
        try:
            st.session_state["last_video"] = engine.build_video_asset(
                image_result=image_result,
                audio_result=audio_result,
                subtitles_result=subtitles_result,
                scene_plan=scene_plan,
                smoke=SMOKE_MODE,
            )
            bundle["video"] = st.session_state["last_video"]
            st.success("Video assembly complete.")
        except Exception as e:
            _append_bundle_error(bundle, "video_assembly", e)
            previous_video = st.session_state.get("last_video")
            last_error = str(e)
            error_details = {"PATH": _os.getenv("PATH", "")}
            if isinstance(previous_video, dict):
                prior_error = str(previous_video.get("error") or "").strip()
                if prior_error:
                    last_error = prior_error
                prior_details = previous_video.get("error_details")
                if isinstance(prior_details, dict):
                    error_details = prior_details
            st.session_state["last_video"] = {
                "mode": "smoke-fallback" if SMOKE_MODE else "fallback",
                "mp4_path": None,
                "sha256": None,
                "error": last_error,
                "error_details": error_details,
            }
            bundle["video"] = st.session_state["last_video"]
            st.error(f"Video assembly failed: {last_error}")

    video_result = st.session_state.get("last_video")
    if video_result:
        st.json(
            {
                "mode": video_result.get("mode"),
                "mp4_path": video_result.get("mp4_path"),
                "sha256": video_result.get("sha256"),
                "error": video_result.get("error"),
                "error_details": video_result.get("error_details"),
            }
        )
        mp4_path = str(video_result.get("mp4_path") or "").strip()
        if mp4_path and Path(mp4_path).exists():
            st.video(mp4_path)

    bundle["video"] = video_result
    bundle["readiness"] = _artifact_readiness(bundle)
    st.session_state["last_bundle"] = bundle

    bundle_payload = bundle
    st.download_button(
        "Download Bundle",
        data=json.dumps(_to_jsonable(bundle_payload), indent=2),
        file_name="uappress_bundle.json",
        mime="application/json",
    )

else:
    st.info("Enter a topic and click Run Research to generate a dossier.")
    st.caption("TEST_HOOK:EMPTY_STATE")
