# apps/research/app.py (FIXED) — robust engine loader
# Fixes ModuleNotFoundError: uappress_engine_v9
#
# This loader will find and import the engine even if you uploaded it as:
# - apps/research/uappress_engine_v9.py
# - apps/research/uappress_engine_v9.txt
# - repo root uappress_engine_v9.py / .txt
#
# If the engine file is missing, the app will show an explicit error telling you what to upload.

import os
import sys
import json
import re
import importlib.util
import streamlit as st

# -----------------------------
# Robust engine import
# -----------------------------

ENGINE_MODULE_NAME = "uappress_engine_v9"

def _try_load_engine_from_path(engine_path: str):
    try:
        spec = importlib.util.spec_from_file_location(ENGINE_MODULE_NAME, engine_path)
        if spec is None or spec.loader is None:
            return None, f"spec_load_failed: {engine_path}"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod, ""
    except Exception as e:
        return None, f"engine_import_error: {engine_path} :: {e}"

def load_engine():
    # 1) Normal import if module is on sys.path
    try:
        import uappress_engine_v9 as eng  # type: ignore
        return eng, ""
    except Exception:
        pass

    # 2) Try common file locations relative to this app
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))

    candidates = [
        os.path.join(here, "uappress_engine_v9.py"),
        os.path.join(here, "uappress_engine_v9.txt"),
        os.path.join(repo_root, "uappress_engine_v9.py"),
        os.path.join(repo_root, "uappress_engine_v9.txt"),
        os.path.join(repo_root, "apps", "research", "uappress_engine_v9.py"),
        os.path.join(repo_root, "apps", "research", "uappress_engine_v9.txt"),
    ]

    errors = []
    for p in candidates:
        if os.path.exists(p):
            mod, err = _try_load_engine_from_path(p)
            if mod is not None:
                return mod, ""
            errors.append(err)

    return None, " | ".join(errors) if errors else "engine_file_not_found"

eng, eng_err = load_engine()

# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="UAPpress Research Engine", layout="wide")
st.title("UAPpress Research Engine")
st.caption("Engineering mode: macro→micro query strategy, source ingestion, authority scoring, dossier JSON output.")

if eng is None:
    st.error("Engine module not found: uappress_engine_v9")
    st.write("✅ Fix: upload the engine file into your repo as one of the following:")
    st.code(
        "apps/research/uappress_engine_v9.py\n"
        "OR apps/research/uappress_engine_v9.txt\n"
        "OR (repo root) uappress_engine_v9.py\n"
        "OR (repo root) uappress_engine_v9.txt\n",
        language="text"
    )
    st.write("Details:")
    st.code(eng_err, language="text")
    st.stop()

# ---- Sidebar controls ----
with st.sidebar:
    st.header("API Keys (BYO)")
    serpapi_key = st.text_input("SERPAPI_API_KEY", type="password", value=os.getenv("SERPAPI_API_KEY", ""))
    openai_key = st.text_input("OPENAI_API_KEY (optional)", type="password", value=os.getenv("OPENAI_API_KEY", ""))

    st.header("Job Inputs")
    primary_topic = st.text_input("Primary topic", value="Phoenix Lights 1997")
    time_scope = st.text_input("Time scope (optional)", value="March 13, 1997")
    geo_scope = st.text_input("Geo scope (optional)", value="Arizona / Nevada")
    event_focus = st.text_area(
        "Event focus (optional)",
        value="two events V-formation over Arizona AND later stationary lights near Sierra Estrella; military flares explanation; Maryland Air National Guard; Governor Fife Symington press conference; later admission; timeline",
        height=120,
    )

    st.header("Budgets")
    max_sources = st.slider("Max sources", 10, 60, 30, 5)
    max_serp_queries = st.slider("Max SerpAPI queries", 4, 20, 10, 1)
    max_fulltext_fetches = st.slider("Max fulltext fetches", 0, 30, 12, 1)
    per_domain_fetch_cap = st.slider("Per-domain fetch cap", 1, 6, 3, 1)
    request_timeout_s = st.slider("Request timeout (seconds)", 4, 20, 8, 1)

    st.header("PDF Handling")
    max_pdf_downloads = st.slider("Max PDF downloads per run", 0, 15, 4, 1)
    per_domain_pdf_cap = st.slider("Per-domain PDF cap", 1, 5, 2, 1)

    st.header("Playwright (JS-heavy)")
    playwright_fallback = st.checkbox("Enable Playwright fallback", value=False)
    max_playwright_renders = st.slider("Max Playwright renders per run", 0, 12, 4, 1)
    per_domain_playwright_cap = st.slider("Per-domain Playwright cap", 1, 5, 2, 1)
    playwright_timeout_s = st.slider("Playwright timeout (seconds)", 6, 25, 12, 1)

    st.header("Flags")
    primary_source_mode = st.checkbox("Primary-source mode", value=True)
    include_gov_docs = st.checkbox("Include gov/mil docs", value=True)
    include_archival = st.checkbox("Include archival/news", value=True)
    include_patents = st.checkbox("Include patents", value=False)
    include_forums = st.checkbox("Include forums", value=False)
    prefer_tier_1_3 = st.checkbox("Prefer Tier 1–3", value=True)

    run_btn = st.button("Run Research", type="primary", use_container_width=True)


colA, colB = st.columns([1, 1], gap="large")

with colA:
    st.subheader("Run")
    if not serpapi_key:
        st.warning("Add SERPAPI_API_KEY (sidebar) to run.")

    if run_btn and serpapi_key:
        job = eng.ResearchJob(
            primary_topic=primary_topic.strip(),
            time_scope=time_scope.strip() or None,
            geo_scope=geo_scope.strip() or None,
            event_focus=event_focus.strip() or None,

            confidence_threshold=0.70,
            max_sources=int(max_sources),
            max_serp_queries=int(max_serp_queries),

            include_gov_docs=bool(include_gov_docs),
            include_archival=bool(include_archival),
            include_patents=bool(include_patents),
            include_forums=bool(include_forums),

            fulltext_hybrid=True,
            max_fulltext_fetches=int(max_fulltext_fetches),
            per_domain_fetch_cap=int(per_domain_fetch_cap),
            request_timeout_s=int(request_timeout_s),

            max_pdf_downloads=int(max_pdf_downloads),
            per_domain_pdf_cap=int(per_domain_pdf_cap),

            allowlist_domains=None,
            blocklist_domains=None,
            prefer_tier_1_3=bool(prefer_tier_1_3),

            llm_refine_top_sources=False,
            primary_source_mode=bool(primary_source_mode),

            micro_min_tier_1_3_to_skip=3,
            micro_min_candidates_to_skip=14,

            playwright_fallback=bool(playwright_fallback),
            max_playwright_renders=int(max_playwright_renders),
            per_domain_playwright_cap=int(per_domain_playwright_cap),
            playwright_timeout_s=int(playwright_timeout_s),
        )

        with st.spinner("Running research…"):
            dossier = eng.run_research(
                job,
                serpapi_key=serpapi_key.strip(),
                openai_key=openai_key.strip() or None,
                progress_cb=None
            )

        st.success("Research complete.")
        st.json({
            "status": dossier.get("status"),
            "confidence_overall": dossier.get("confidence_overall"),
            "sources": len(dossier.get("sources") or []),
            "claim_clusters": len(((dossier.get("evidence_graph") or {}).get("claim_clusters") or [])),
            "contradictions": len(((dossier.get("evidence_graph") or {}).get("contradictions") or [])),
            "pdf_extracted": ((dossier.get("metrics") or {}).get("pdf_extracted")),
            "playwright_rendered": ((dossier.get("metrics") or {}).get("playwright_rendered")),
        })

        st.download_button(
            "Download dossier JSON",
            data=json.dumps(dossier, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name=f"uappress_dossier_{dossier.get('job_id','job')}.json",
            mime="application/json",
            use_container_width=True,
        )

with colB:
    st.subheader("Notes")
    st.write(
        "- This error happens when the engine file is not present in the repo (or named differently).\n"
        "- Upload **uappress_engine_v9.py** into `apps/research/` to make this bulletproof.\n"
        "- Playwright on Streamlit Cloud may fail unless Chromium is installed (expected).\n"
    )
