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
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "onyx"


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




def generate_script(bundle: Dict[str, Any], *, overwrite: bool = False) -> Dict[str, Any] | None:
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
    if not scenes:
        raise ValueError("Generate Script first")

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

    audio_bundle.update({"scene_mp3s": scene_mp3s, "format": "mp3", "voice": voice, "model": model, "cache": cache})
    bundle["audio"] = audio_bundle
    bundle["script"]["scenes"] = scenes
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

    scene_mp3s = []
    if isinstance(audio_result, dict):
        raw = audio_result.get("scene_mp3s")
        if isinstance(raw, list):
            scene_mp3s = [item for item in raw if isinstance(item, dict)]

    if not scene_mp3s:
        missing.append("scene mp3 list (bundle.audio.scene_mp3s)")
    else:
        missing_audio = [str(item.get("path") or "") for item in scene_mp3s if not Path(str(item.get("path") or "")).exists()]
        if missing_audio:
            missing.append(f"scene mp3 files not found: {', '.join(missing_audio)}")

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
    script_text = str(script_result.get("text") or "").strip()
    script_scenes = script_result.get("scenes") if isinstance(script_result.get("scenes"), list) else []
    has_script = bool(script_text) and bool(script_scenes)

    image_result = bundle_data.get("images") if isinstance(bundle_data.get("images"), dict) else {}
    image_paths_raw = image_result.get("images") if isinstance(image_result.get("images"), list) else []
    image_paths = [str(path).strip() for path in image_paths_raw if str(path).strip()]
    has_images = bool(image_paths) and all(Path(path).exists() for path in image_paths)

    audio_result = bundle_data.get("audio") if isinstance(bundle_data.get("audio"), dict) else {}
    scene_mp3s = audio_result.get("scene_mp3s") if isinstance(audio_result.get("scene_mp3s"), list) else []
    has_audio = bool(scene_mp3s) and all(Path(str(item.get("path") or "")).exists() for item in scene_mp3s if isinstance(item, dict))

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


def _fallback_build_documentary_script(
    blueprint: Dict[str, Any],
    dossier: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    topic = str(blueprint.get("topic") or "this case").strip() or "this case"
    dossier_data = dossier if isinstance(dossier, dict) else {}

    sources = dossier_data.get("sources") if isinstance(dossier_data.get("sources"), list) else []
    source_refs = []
    for idx, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            continue
        source_refs.append({
            "index": idx,
            "title": str(source.get("title") or f"Source {idx}"),
            "url": str(source.get("url") or "").strip(),
        })
    if not source_refs:
        source_refs = [{"index": 1, "title": "Source 1", "url": ""}]

    pool: list[str] = []
    for key in ("summary", "facts", "key_facts", "timeline", "claims", "contradictions", "open_questions", "notes"):
        value = dossier_data.get(key)
        if isinstance(value, list):
            for item in value:
                text = str(item.get("text") if isinstance(item, dict) else item).strip()
                if text and text not in pool:
                    pool.append(text)
        elif isinstance(value, str):
            text = value.strip()
            if text and text not in pool:
                pool.append(text)
    if not pool:
        pool = [
            "The available dossier entries are limited, so all conclusions in this draft remain provisional.",
            "Where details are incomplete, this narration labels them as unconfirmed rather than definitive.",
            "Future updates should replace placeholders with direct quotations and dated records.",
        ]

    scene_count = 14
    target_words_per_scene = max(320, TARGET_SCRIPT_WORDS // scene_count)
    scenes: list[Dict[str, Any]] = []

    for idx in range(scene_count):
        scene_number = idx + 1
        focus = [pool[(idx + o) % len(pool)] for o in range(4)]
        citation_indices = sorted({((idx + o) % len(source_refs)) + 1 for o in range(3)})
        source_cues = [source_refs[c - 1].get("url") or f"Source #{c}" for c in citation_indices]

        paragraphs: list[str] = []
        while len(" ".join(paragraphs).split()) < target_words_per_scene:
            cidx = citation_indices[len(paragraphs) % len(citation_indices)]
            point = focus[len(paragraphs) % len(focus)]
            paragraphs.append(
                f"For this chapter on {topic}, we anchor every claim to the dossier and identified records. "
                f"The current evidence emphasizes: {point}. (Source {cidx}) "
                "If a detail cannot be independently verified, we label it as unconfirmed and avoid escalating language."
            )
            paragraphs.append(
                f"We then compare that entry against other claims, timeline notes, and contradictions in the same dossier. "
                f"When records disagree, we preserve the disagreement instead of forcing certainty, and we keep the burden of proof visible. (Source {cidx})"
            )

        voiceover = "\n\n".join(paragraphs).strip()
        scenes.append(
            {
                "scene_number": scene_number,
                "scene_title": f"{topic} — Scene {scene_number}",
                "visual_notes": "Archival pages, timeline graphics, map context, and labeled evidence overlays.",
                "on_screen_text": [f"Scene {scene_number}", "Dossier-grounded narrative"],
                "voiceover": voiceover,
                "source_cues": source_cues,
            }
        )

    narration_text = "\n\n".join(str(scene.get("voiceover") or "") for scene in scenes).strip()
    word_count = len([token for token in narration_text.split() if token.strip()])
    return {
        "topic": topic,
        "scenes": scenes,
        "text": narration_text,
        "target_minutes": TARGET_DOCUMENTARY_MINUTES,
        "target_words": TARGET_SCRIPT_WORDS,
        "word_count_actual": word_count,
        "based_on_sources": source_refs,
        "scene_count": len(scenes),
        "locked": False,
        "engine": "fallback",
    }


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

if st.button("Score Topic", key="score_topic_button"):
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

if st.button("Use as Primary Topic", key="use_as_primary_topic_button"):
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

        try:
            if getattr(engine, "build_documentary_script", None) is not None:
                script_result = engine.build_documentary_script(blueprint, dossier)
            else:
                script_result = engine.compile_voiceover_script(blueprint)

            scene_plan = engine.build_scene_plan(blueprint, script_result)
            generated_script_text = _consolidate_script_text(script_result, scene_plan)
            script_bundle = script_result if isinstance(script_result, dict) else {}
            script_bundle["text"] = generated_script_text
            script_bundle["scenes"] = (
                script_bundle.get("scenes")
                if isinstance(script_bundle.get("scenes"), list)
                else []
            )
            script_bundle["locked"] = False
            run_bundle["script"] = script_bundle
            run_bundle["scene_plan"] = scene_plan
        except Exception as exc:
            _append_bundle_error(run_bundle, "script", exc)
            raise

        try:
            image_result, audio_result = _ensure_fallback_artifacts(primary_topic, scene_plan, generated_script_text)
            run_bundle["images"] = image_result
            run_bundle["audio"] = audio_result
        except Exception as exc:
            _append_bundle_error(run_bundle, "assets", exc)
            raise

        run_bundle["readiness"] = _artifact_readiness(run_bundle)

        st.session_state["last_dossier"] = dossier
        st.session_state["last_blueprint"] = blueprint
        st.session_state["last_script"] = run_bundle.get("script")
        st.session_state["script_text"] = generated_script_text
        st.session_state["generated_script_text"] = generated_script_text
        st.session_state["script_editor_text"] = generated_script_text
        st.session_state["last_scene_plan"] = scene_plan
        st.session_state["last_images"] = image_result
        st.session_state["last_audio"] = audio_result
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

    if (
        st.session_state.get("last_script_result") is None
        and engine.build_documentary_blueprint is not None
    ):
        blueprint = engine.build_documentary_blueprint(dossier)
        if getattr(engine, "build_documentary_script", None) is not None:
            st.session_state["last_script_result"] = engine.build_documentary_script(blueprint, dossier)
        elif engine.compile_voiceover_script is not None:
            st.session_state["last_script_result"] = engine.compile_voiceover_script(blueprint)

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
        "script": script_result,
        "scene_plan": scene_plan,
        "images": image_result,
        "audio": audio_result,
        "video": st.session_state.get("last_video"),
        "errors": [],
    }

    script_bundle = bundle.get("script") if isinstance(bundle.get("script"), dict) else {}
    generated_script_text = str(script_bundle.get("text") or st.session_state.get("generated_script_text") or "").strip()
    if not generated_script_text:
        generated_script_text = _consolidate_script_text(script_bundle, scene_plan)

    if not st.session_state.get("script_text"):
        st.session_state["script_text"] = generated_script_text
    if not st.session_state.get("generated_script_text"):
        st.session_state["generated_script_text"] = generated_script_text
    if not st.session_state.get("script_editor_text"):
        st.session_state["script_editor_text"] = st.session_state.get("script_text", "")

    current_script_text = str(st.session_state.get("script_text") or "").strip()
    script_locked = bool(script_bundle.get("locked"))
    script_bundle["text"] = current_script_text
    script_bundle["scenes"] = (
        script_bundle.get("scenes")
        if isinstance(script_bundle.get("scenes"), list)
        else []
    )
    script_bundle["locked"] = script_locked
    bundle["script"] = script_bundle

    st.subheader("Script")
    gen_col, overwrite_col = st.columns(2)
    if gen_col.button("Generate Script", use_container_width=True):
        generated = generate_script(bundle, overwrite=False)
        if generated is None:
            st.warning("Run Research first")
        elif bool(bundle.get("script", {}).get("locked")):
            st.warning("Script is locked. Use Overwrite Generated Script to regenerate.")
        else:
            generated_text = str(generated.get("text") or "")
            st.session_state["last_script"] = generated
            st.session_state["script_text"] = generated_text
            st.session_state["generated_script_text"] = generated_text
            st.session_state["script_editor_text"] = generated_text
            st.session_state["last_scene_plan"] = bundle.get("scene_plan")
            st.success("Script generated from dossier.")

    if overwrite_col.button("Overwrite Generated Script", use_container_width=True):
        generated = generate_script(bundle, overwrite=True)
        if generated is None:
            st.warning("Run Research first")
        else:
            generated_text = str(generated.get("text") or "")
            st.session_state["last_script"] = generated
            st.session_state["script_text"] = generated_text
            st.session_state["generated_script_text"] = generated_text
            st.session_state["script_editor_text"] = generated_text
            st.session_state["last_scene_plan"] = bundle.get("scene_plan")
            st.success("Script overwritten and regenerated.")

    script_bundle = bundle.get("script") if isinstance(bundle.get("script"), dict) else {}
    if script_bundle.get("text"):
        word_count_actual = int(script_bundle.get("word_count_actual") or len(str(script_bundle.get("text") or "").split()))
        runtime_min = round(word_count_actual / WORDS_PER_MINUTE, 1)
        st.caption(f"Word count: {word_count_actual} | Estimated runtime: {runtime_min} min")

    st.text_area("Script (editable)", key="script_editor_text", height=400)

    script_save_col, script_reset_col = st.columns(2)
    if script_save_col.button("Save Script Changes", use_container_width=True):
        edited_script = str(st.session_state.get("script_editor_text") or "").strip()
        st.session_state["script_text"] = edited_script
        bundle["script"]["text"] = edited_script
        bundle["script"]["word_count_actual"] = len(edited_script.split())
        bundle["script"]["locked"] = True
        st.success("Script changes saved. Script locked.")

    if script_reset_col.button("Reset to Generated Script", use_container_width=True):
        reset_text = str(st.session_state.get("generated_script_text") or generated_script_text)
        st.session_state["script_text"] = reset_text
        st.session_state["script_editor_text"] = reset_text
        bundle["script"]["text"] = reset_text
        bundle["script"]["locked"] = False
        st.info("Script reset to generated version.")

    bundle["script"]["text"] = str(st.session_state.get("script_text") or "").strip()
    bundle["script"]["locked"] = bool(bundle["script"].get("locked"))

    st.subheader("Audio (Per-Scene TTS)")
    if st.button("Generate MP3s (per scene)", use_container_width=True):
        try:
            updated_audio = generate_scene_mp3s(bundle, openai_key=openai_key or "", voice=TTS_VOICE, model=TTS_MODEL, smoke=SMOKE_MODE)
            st.session_state["last_audio"] = updated_audio
            st.success("Per-scene MP3s generated.")
        except Exception as exc:
            _append_bundle_error(bundle, "audio_tts", exc)
            st.error(str(exc))

    current_audio = bundle.get("audio") if isinstance(bundle.get("audio"), dict) else {}
    scene_mp3s = current_audio.get("scene_mp3s") if isinstance(current_audio.get("scene_mp3s"), list) else []
    for item in scene_mp3s[:3]:
        path = str(item.get("path") or "").strip() if isinstance(item, dict) else ""
        if path and Path(path).exists():
            st.caption(f"Scene {item.get('scene')}: {path}")
            st.audio(path)
    if len(scene_mp3s) > 3:
        st.caption("Additional MP3 files")
        st.json(scene_mp3s[3:])

    if not bundle.get("script", {}).get("scenes"):
        st.warning("Image generation is disabled until script scenes exist.")

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

    if st.button("Assemble Video (MP4)", disabled=not readiness.get("ready_for_video", False)):
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
