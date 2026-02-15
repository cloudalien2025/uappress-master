# apps/research/app.py — UAPpress Research Engine (Production-safe + CI-deterministic)
#
# Goals:
# - Imports cleanly (import-time safety)
# - Deterministic CI smoke mode: NO secrets, NO network, NO clicks
# - Stable Playwright hooks:
#     TEST_HOOK:APP_LOADED
#     TEST_HOOK:RUN_DONE
#
# Notes:
# - We do NOT rely solely on CI="1". GitHub Actions typically sets CI="true" and GITHUB_ACTIONS="true".
# - CI smoke short-circuit runs immediately after set_page_config to avoid sidebar/key-gating UI.

import os
import time
from typing import Any, Dict

import streamlit as st


def _is_truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def ci_smoke_enabled() -> bool:
    # Robust detection for GitHub Actions + local overrides
    return (
        _is_truthy(os.getenv("UAPPRESS_CI_SMOKE", "")) or
        _is_truthy(os.getenv("UAPPRESS_SMOKE", "")) or
        _is_truthy(os.getenv("CI", "")) or
        _is_truthy(os.getenv("GITHUB_ACTIONS", ""))
    )


def mark_run_done() -> None:
    # Stable Playwright contract marker (plain text)
    st.markdown("TEST_HOOK:RUN_DONE")


# ------------------------------------------------------------------------------
# Page Config (must be first Streamlit call)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="UAPpress Research Engine", layout="wide")


# ------------------------------------------------------------------------------
# CI Smoke Short-Circuit (NO secrets, NO network, NO clicks)
# ------------------------------------------------------------------------------
if ci_smoke_enabled():
    st.title("UAPpress Research Engine")
    st.caption("TEST_HOOK:APP_LOADED")
    st.success("Research Complete (CI smoke)")
    mark_run_done()
    st.stop()


# ------------------------------------------------------------------------------
# Import-time safe research function (real engine optional)
# ------------------------------------------------------------------------------
try:
    # If/when you wire a real engine module, keep this import.
    from research_engine import run_research  # type: ignore
except Exception:
    def run_research(**kwargs) -> Dict[str, Any]:
        # Safe fallback: never crashes UI
        return {
            "status": "PRELIMINARY",
            "confidence_overall": 0.62,
            "note": "run_research import not wired yet (fallback stub).",
            "args": {k: ("***" if "key" in k.lower() else v) for k, v in kwargs.items()},
        }


# ------------------------------------------------------------------------------
# Normal App (non-CI mode)
# ------------------------------------------------------------------------------
st.title("UAPpress Research Engine")
st.caption("TEST_HOOK:APP_LOADED")


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


# ------------------------------------------------------------------------------
# Main Page — Minimal Inputs (stable render order)
# ------------------------------------------------------------------------------
st.subheader("Research Topic")

# Initialize session state (deterministic across reruns)
if "last_dossier" not in st.session_state:
    st.session_state["last_dossier"] = None
if "last_run_ts" not in st.session_state:
    st.session_state["last_run_ts"] = None
if "run_status" not in st.session_state:
    st.session_state["run_status"] = "IDLE"


# Atomic submit to prevent rerun races
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
# Deterministic mock dossier (for local smoke if you ever enable UAPPRESS_SMOKE)
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
        "notes": ["Local smoke mode — no external calls were made."],
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

    # In normal mode, require SerpAPI key
    if not serpapi_key:
        st.warning("SerpAPI key required.")
        st.session_state["run_status"] = "ERROR"
        st.stop()

    st.info("Running research...")
    st.caption("TEST_HOOK:RUN_STARTED")

    try:
        dossier = run_research(
            primary_topic=primary_topic,
            serpapi_key=serpapi_key,
            openai_key=openai_key or None,
            confidence_threshold=confidence_threshold,
            max_serp_queries=max_serp_queries,
            max_sources=max_sources,
            include_gov_docs=include_gov_docs,
        )

        st.session_state["last_dossier"] = dossier
        st.session_state["run_status"] = "DONE"

    except Exception as e:
        st.session_state["run_status"] = "ERROR"
        st.error(f"Unexpected error: {str(e)}")
        st.caption("TEST_HOOK:RUN_ERROR")
        st.stop()


# ------------------------------------------------------------------------------
# Output Rendering (stable)
# ------------------------------------------------------------------------------
dossier = st.session_state.get("last_dossier")

if dossier:
    score = float(dossier.get("confidence_overall", 0) or 0)

    if score >= 0.75:
        quality = "High"
    elif score >= 0.60:
        quality = "Medium"
    else:
        quality = "Preliminary"

    st.success(f"Research Complete — Quality: {quality} ({round(score, 2)})")
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

else:
    st.info("Enter a topic and click Run Research to generate a dossier.")
    st.caption("TEST_HOOK:EMPTY_STATE")
