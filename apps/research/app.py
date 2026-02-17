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
try:
    from ci_hooks import ci_smoke_enabled, mark_run_done
except Exception:
    from apps.research.ci_hooks import ci_smoke_enabled, mark_run_done


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
    class ResearchJob:  # type: ignore
        def __init__(self, primary_topic: str):
            self.primary_topic = primary_topic

    def run_research(*args, **kwargs) -> Dict[str, Any]:
        # Safe placeholder: never crashes UI
        return {
            "status": "PRELIMINARY",
            "confidence_overall": 0.62,
            "note": "run_research import not wired yet (fallback stub).",
            "args": {k: ("***" if "key" in k.lower() else v) for k, v in kwargs.items()},
        }

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
st.subheader("Research Topic")

# Initialize session state for deterministic outputs across reruns
if "last_dossier" not in st.session_state:
    st.session_state["last_dossier"] = None
if "last_run_ts" not in st.session_state:
    st.session_state["last_run_ts"] = None
if "run_status" not in st.session_state:
    st.session_state["run_status"] = "IDLE"
if "last_blueprint" not in st.session_state:
    st.session_state["last_blueprint"] = None
if "last_script" not in st.session_state:
    st.session_state["last_script"] = None
if "last_scene_plan" not in st.session_state:
    st.session_state["last_scene_plan"] = None
if "last_audio" not in st.session_state:
    st.session_state["last_audio"] = None


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

    blueprint = build_documentary_blueprint(dossier)
    script_result = compile_voiceover_script(blueprint, target_minutes=12)
    scene_plan = build_scene_plan(blueprint, script_result, target_scene_seconds=6.0, max_scenes=180)

    st.subheader("Voiceover Script")
    st.json(script_result)

    st.subheader("Scene Plan")
    st.json(scene_plan)

    bundle = {
        "dossier": dossier,
        "blueprint": blueprint,
        "script": script_result,
        "scene_plan": scene_plan,
    }
    st.download_button(
        "Download Bundle (JSON)",
        data=json.dumps(bundle, sort_keys=True, indent=2),
        file_name="uappress_bundle.json",
        mime="application/json",
    )

    sources = dossier.get("sources") or []
    if isinstance(sources, list) and sources:
        st.subheader("Top Sources")
        for i, s in enumerate(sources, start=1):
            title = str(s.get("title", f"Source {i}"))
            url = str(s.get("url", ""))
            st.markdown(f"{i}. **{title}** — {url}")

    if blueprint:
        st.subheader("Documentary Blueprint")
        st.json(blueprint)

    if script_result:
        st.subheader("Voiceover Script")
        st.json(script_result)

    if scene_plan:
        st.subheader("Scene Plan")
        st.json(scene_plan)

    if st.button("Generate Audio (MP3)", key="generate_audio_button", use_container_width=True):
        try:
            audio_result = build_audio_asset(
                script_result or {"full_text": ""},
                openai_key=openai_key or None,
                smoke=SMOKE_MODE,
            )
            st.session_state["last_audio"] = audio_result
        except Exception as e:
            st.error(f"Audio generation failed: {str(e)}")

    st.subheader("Audio Asset")
    st.json(st.session_state.get("last_audio"))

    bundle = {
        "dossier": dossier,
        "blueprint": blueprint,
        "script": script_result,
        "scene_plan": scene_plan,
        "audio": st.session_state.get("last_audio"),
    }
    st.download_button(
        "Download Audio Bundle",
        data=json.dumps(bundle, indent=2),
        file_name="uappress_audio_bundle.json",
        mime="application/json",
        key="download_audio_bundle_button",
    )

else:
    st.info("Enter a topic and click Run Research to generate a dossier.")
    st.caption("TEST_HOOK:EMPTY_STATE")
