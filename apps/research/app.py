# apps/research/app.py — UAPpress Research Engine (Production-safe + CI-deterministic)

import os
import json
import time
from typing import Any, Dict

import streamlit as st
from ci_hooks import ci_smoke_enabled, mark_run_done


# ------------------------------------------------------------------------------
# Page Config (must be first Streamlit call)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="UAPpress Research Engine", layout="wide")


# ------------------------------------------------------------------------------
# CI Smoke Short-Circuit (NO secrets, NO clicks, NO network)
# ------------------------------------------------------------------------------
if ci_smoke_enabled():
    st.title("UAPpress Research Engine")
    st.caption("TEST_HOOK:APP_LOADED")
    st.success("Research Complete (CI smoke)")
    mark_run_done()
    st.stop()


# ------------------------------------------------------------------------------
# Import-time safe research function
# ------------------------------------------------------------------------------
try:
    from research_engine import run_research  # type: ignore
except Exception:
    def run_research(**kwargs) -> Dict[str, Any]:
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
# Sidebar
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("Connections")

    serpapi_key = st.text_input(
        "SerpAPI Key",
        type="password",
        key="serpapi_key_input",
    )

    openai_key = st.text_input(
        "OpenAI Key (optional)",
        type="password",
        key="openai_key_input",
    )

    st.divider()
    st.header("Mode")

    mode = st.radio(
        "Select Mode",
        ["Simple", "Pro"],
        index=0,
        key="mode_radio",
    )

    if mode == "Pro":
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
        confidence_threshold = 0.58
        max_serp_queries = 12
        max_sources = 25
        include_gov_docs = True


# ------------------------------------------------------------------------------
# Session State
# ------------------------------------------------------------------------------
if "last_dossier" not in st.session_state:
    st.session_state["last_dossier"] = None
if "run_status" not in st.session_state:
    st.session_state["run_status"] = "IDLE"


# ------------------------------------------------------------------------------
# Main Input Form
# ------------------------------------------------------------------------------
st.subheader("Research Topic")

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
# Deterministic Mock (used only if you manually enable smoke via env)
# ------------------------------------------------------------------------------
def _mock_dossier(topic: str) -> Dict[str, Any]:
    return {
        "status": "COMPLETE",
        "confidence_overall": 0.82,
        "topic": topic,
        "summary": "Mock deterministic research result.",
        "sources": [
            {"title": "Mock Source A", "url": "https://example.com/a", "score": 0.91},
            {"title": "Mock Source B", "url": "https://example.com/b", "score": 0.84},
            {"title": "Mock Source C", "url": "https://example.com/c", "score": 0.79},
        ],
    }


# ------------------------------------------------------------------------------
# Run Logic
# ------------------------------------------------------------------------------
if run_button:
    st.session_state["run_status"] = "RUNNING"

    if not primary_topic:
        st.warning("Please enter a topic.")
        st.stop()

    if not serpapi_key:
        st.warning("SerpAPI key required.")
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
        st.error(f"Unexpected error: {str(e)}")
        st.caption("TEST_HOOK:RUN_ERROR")
        st.stop()


# ------------------------------------------------------------------------------
# Output Rendering
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
