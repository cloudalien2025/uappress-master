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
import os
import time
import json
from typing import Any, Dict

import streamlit as st
from apps.research.ci_hooks import ci_smoke_enabled, mark_run_done
from apps.research.uappress_engine import score_topic


# ------------------------------------------------------------------------------
# Import-time safe research function
# ------------------------------------------------------------------------------
try:
    from apps.research.uappress_engine import (  # type: ignore
        ResearchJob,
        run_research,
        build_documentary_blueprint,
        compile_voiceover_script,
        build_scene_plan,
        build_audio_asset,
    )
except Exception:
    try:
        from .research_engine import run_research  # type: ignore
    except Exception:
        def run_research(**kwargs) -> Dict[str, Any]:
            # Safe placeholder: never crashes UI
            return {
                "status": "PRELIMINARY",
                "confidence_overall": 0.62,
                "note": "run_research import not wired yet (fallback stub).",
                "args": {k: ("***" if "key" in k.lower() else v) for k, v in kwargs.items()},
            }


try:
    from uappress_engine import (
        build_documentary_blueprint,
        build_image_assets,
        build_scene_plan,
        compile_voiceover_script,
    )
except Exception:
    try:
        from .uappress_engine import (
            build_documentary_blueprint,
            build_image_assets,
            build_scene_plan,
            compile_voiceover_script,
        )
    except Exception:
        build_documentary_blueprint = None
        compile_voiceover_script = None
        build_scene_plan = None
        build_image_assets = None

    def build_documentary_blueprint(dossier: Dict[str, Any]) -> Dict[str, Any]:
        return {"topic": dossier.get("topic", "Unknown"), "acts": ["Setup", "Complication", "Resolution"]}

    def compile_voiceover_script(blueprint: Dict[str, Any], target_minutes: int = 12) -> Dict[str, Any]:
        return {"target_minutes": target_minutes, "full_text": f"Voiceover script for {blueprint.get('topic', 'Unknown')}"}

    def build_scene_plan(blueprint: Dict[str, Any], script_result: Dict[str, Any]) -> Dict[str, Any]:
        return {"topic": blueprint.get("topic", "Unknown"), "scenes": []}

    def build_audio_asset(script_result: dict, *, openai_key: str | None, smoke: bool) -> dict:
        return {"mode": "smoke" if smoke else "real", "mp3_path": "", "duration_seconds": 0.0, "voice": "onyx", "model": "gpt-4o-mini-tts", "sha256": ""}


# ------------------------------------------------------------------------------
# Deterministic modes
# ------------------------------------------------------------------------------
# Explicit envs (optional) + robust CI detection via ci_smoke_enabled()
SMOKE_MODE = (
    os.getenv("UAPPRESS_SMOKE", "").strip() == "1"
    or os.getenv("UAPPRESS_CI_SMOKE", "").strip() == "1"
    or ci_smoke_enabled()
)


# ------------------------------------------------------------------------------
# Page Config
# ------------------------------------------------------------------------------
st.set_page_config(page_title="UAPpress Research Engine", layout="wide")
st.title("UAPpress Research Engine")

# Stable marker for Playwright to know Streamlit hydrated
st.caption("TEST_HOOK:APP_LOADED")
st.caption(ENGINE_IMPORT_MARKER)


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
            st.session_state["last_topic_score"] = score_topic(
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
            dossier = run_research(
                ResearchJob(primary_topic=primary_topic),
                serpapi_key=serpapi_key,
                openai_key=openai_key or None,
            )

        blueprint = build_documentary_blueprint(dossier)
        script_result = compile_voiceover_script(blueprint)
        scene_plan = build_scene_plan(blueprint, script_result)

        st.session_state["last_dossier"] = dossier
        st.session_state["last_blueprint"] = blueprint
        st.session_state["last_script"] = script_result
        st.session_state["last_scene_plan"] = scene_plan
        st.session_state["last_audio"] = None
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
        and build_documentary_blueprint is not None
        and compile_voiceover_script is not None
    ):
        blueprint = build_documentary_blueprint(dossier)
        st.session_state["last_script_result"] = compile_voiceover_script(blueprint)

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
        if not image_result or not audio_result:
            st.warning("Images and audio artifacts are required before assembly.")
        else:
            try:
                st.session_state["last_video"] = build_video_asset(
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
        "video": video_result,
    }
    st.download_button(
        "Download Bundle",
        data=json.dumps(bundle_payload, indent=2),
        file_name="uappress_bundle.json",
        mime="application/json",
    )

else:
    st.info("Enter a topic and click Run Research to generate a dossier.")
    st.caption("TEST_HOOK:EMPTY_STATE")
