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
import inspect
import importlib
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


def _ensure_fallback_artifacts(
    topic: str,
    scene_plan: Dict[str, Any],
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
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16000 * 2)

    return {
        "scene_images_dir": str(images_dir),
        "images": image_paths,
        "image_count": len(image_paths),
        "engine": "fallback",
    }, {
        "audio_path": str(audio_path),
        "duration_seconds": 2,
        "engine": "fallback",
    }


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
    if isinstance(audio_result, dict):
        audio_path = str(
            audio_result.get("audio_path")
            or audio_result.get("audio_mp3_path")
            or ""
        ).strip()

    if not audio_path:
        missing.append(
            "audio path (bundle.audio.audio_path or bundle.audio.audio_mp3_path)"
        )
    elif not Path(audio_path).exists():
        missing.append(f"audio file not found: {audio_path}")

    return missing


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


def _fallback_build_scene_plan(blueprint: Dict[str, Any], script_result: Dict[str, Any]) -> Dict[str, Any]:
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
    return {
        "mode": "smoke-fallback" if smoke else "fallback",
        "mp4_path": None,
        "sha256": None,
        "scene_count": int(scene_plan.get("scene_count") or 0),
        "has_images": bool(image_result),
        "has_audio": bool(audio_result),
        "has_subtitles": bool(subtitles_result),
    }


@dataclass
class EngineSurface:
    run_research: Any
    score_topic: Any
    build_documentary_blueprint: Any
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
            job = _create_research_job(primary_topic)
            _assert_run_research_signature()
            dossier = engine.run_research(
                job=job,
                serpapi_key=serpapi_key,
                openai_key=openai_key or None,
            )

        blueprint = engine.build_documentary_blueprint(dossier)
        script_result = engine.compile_voiceover_script(blueprint)
        scene_plan = engine.build_scene_plan(blueprint, script_result)

        st.session_state["last_dossier"] = dossier
        st.session_state["last_blueprint"] = blueprint
        st.session_state["last_script"] = script_result
        st.session_state["last_scene_plan"] = scene_plan
        image_result, audio_result = _ensure_fallback_artifacts(primary_topic, scene_plan)
        st.session_state["last_images"] = image_result
        st.session_state["last_audio"] = audio_result
        st.session_state["run_status"] = "DONE"

    except Exception as e:
        st.session_state["run_status"] = "ERROR"
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
        and engine.compile_voiceover_script is not None
    ):
        blueprint = engine.build_documentary_blueprint(dossier)
        st.session_state["last_script_result"] = engine.compile_voiceover_script(blueprint)

    sources = dossier.get("sources") or []
    if isinstance(sources, list) and sources:
        st.subheader("Top Sources")
        for i, s in enumerate(sources, start=1):
            title = str(s.get("title", f"Source {i}"))
            url = str(s.get("url", ""))
            st.markdown(f"{i}. **{title}** — {url}")

    st.subheader("Video Assembly")

    image_result = st.session_state.get("last_images")
    audio_result = st.session_state.get("last_audio")
    subtitles_result = st.session_state.get("last_subtitles")
    scene_plan = st.session_state.get("last_scene_plan") or {}

    if st.button("Assemble Video (MP4)"):
        missing_artifacts = _collect_missing_artifacts(image_result, audio_result)
        if missing_artifacts:
            st.warning(
                "Video assembly prerequisites missing: "
                + " | ".join(missing_artifacts)
            )
        else:
            try:
                st.session_state["last_video"] = engine.build_video_asset(
                    image_result=image_result,
                    audio_result=audio_result,
                    subtitles_result=subtitles_result,
                    scene_plan=scene_plan,
                    smoke=SMOKE_MODE,
                )
                st.success("Video assembly complete.")
            except Exception as e:
                st.error(f"Video assembly failed: {str(e)}")

    video_result = st.session_state.get("last_video")
    if video_result:
        st.json(
            {
                "mode": video_result.get("mode"),
                "mp4_path": video_result.get("mp4_path"),
                "sha256": video_result.get("sha256"),
            }
        )

    bundle_payload = {
        "dossier": dossier,
        "blueprint": blueprint,
        "script": script_result,
        "scene_plan": scene_plan,
        "images": image_result,
        "audio": audio_result,
        "video": video_result,
    }
    st.download_button(
        "Download Bundle",
        data=json.dumps(_to_jsonable(bundle_payload), indent=2),
        file_name="uappress_bundle.json",
        mime="application/json",
    )

else:
    st.info("Enter a topic and click Run Research to generate a dossier.")
    st.caption("TEST_HOOK:EMPTY_STATE")
