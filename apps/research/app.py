# app.py (v9)
# Streamlit UI for UAPpress Research Engine
# BYO APIs (SerpAPI + optional OpenAI)
#
# Requires: uappress_engine_v9.py in same directory.

import os
import re
import json
import streamlit as st

import uappress_engine_v9 as eng


st.set_page_config(page_title="UAPpress Research Engine (SerpAPI)", layout="wide")

st.title("UAPpress Research Engine — SerpAPI (Two-Pass)")
st.caption("Engineering mode: macro→micro query strategy, source ingestion, authority scoring, deterministic dossier JSON.")


# -----------------------------
# Playwright/Chromium Readiness Check
# -----------------------------
with st.expander("Runtime Readiness (Playwright/Chromium)", expanded=False):
    pw_ok = True
    pw_err = ""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
        with sync_playwright() as p:
            try:
                b = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
                b.close()
            except Exception as e:
                pw_ok = False
                pw_err = str(e)
    except Exception as e:
        pw_ok = False
        pw_err = str(e)

    if pw_ok:
        st.success("Playwright + Chromium look ready.")
    else:
        st.error("Playwright/Chromium not ready.")
        st.code(pw_err, language="text")
        st.markdown(
            "- Ensure `playwright` is installed.\n"
            "- Ensure Chromium is installed: `playwright install chromium`.\n"
            "- On Streamlit Cloud, you may need a post-install step to install browser binaries.\n"
        )


with st.sidebar:
    st.header("Job Inputs")

    primary_topic = st.text_input("Primary topic (required)", value="", placeholder="e.g., 'Nimitz Tic Tac 2004'")
    time_scope = st.text_input("Time scope (optional)", value="", placeholder="e.g., '2004' or '2017–2020'")
    geo_scope = st.text_input("Geo scope (optional)", value="", placeholder="e.g., 'United States' or 'Pacific'")
    event_focus = st.text_input("Event focus (optional)", value="", placeholder="e.g., 'DoD statements'")

    st.divider()
    st.subheader("Budgets")
    confidence_threshold = st.slider("Confidence threshold", 0.55, 0.95, 0.75, 0.01)
    max_serp_queries = st.slider("Max SERP queries", 6, 30, 18, 1)
    max_sources = st.slider("Max sources ingested", 12, 50, 24, 1)

    st.divider()
    st.subheader("Inclusion Toggles")
    primary_source_mode = st.checkbox("Primary-source mode (bias to .gov/.mil/.edu/PDF)", value=False)
    include_gov_docs = st.checkbox("Include gov docs focus (.gov/.mil)", value=True)
    include_archival = st.checkbox("Include archival (pdf/press releases)", value=True)
    include_patents = st.checkbox("Include patents", value=True)
    include_forums = st.checkbox("Include forums/social (low authority)", value=False)

    st.divider()
    st.subheader("Domain Controls")
    prefer_tier_1_3 = st.checkbox("Prefer Tier 1–3 sources (quality bias)", value=True)
    allowlist_raw = st.text_area("Allowlist domains (optional, one per line or comma-separated)", value="", placeholder="e.g. nasa.gov\n.dni.gov\nreuters.com")
    blocklist_raw = st.text_area("Blocklist domains (optional, one per line or comma-separated)", value="", placeholder="e.g. example.com, lowqualitysite.net")

    st.divider()
    st.subheader("Full-text Fetching")
    fulltext_hybrid = st.checkbox("Hybrid full-text fetch (guarded)", value=True)
    max_fulltext_fetches = st.slider("Max full-text fetches per run", 0, 25, 12, 1)
    per_domain_fetch_cap = st.slider("Per-domain fetch cap", 1, 6, 3, 1)
    request_timeout_s = st.slider("Fetch timeout (seconds)", 4, 15, 8, 1)

    st.subheader("PDF Handling")
    max_pdf_downloads = st.slider("Max PDF downloads per run", 0, 15, 6, 1)
    per_domain_pdf_cap = st.slider("Per-domain PDF cap", 1, 5, 2, 1)

    st.subheader("Playwright/Chromium Fallback (JS-heavy)")
    playwright_fallback = st.checkbox("Enable Playwright fallback for JS-heavy pages", value=False)
    max_playwright_renders = st.slider("Max Playwright renders per run", 0, 12, 6, 1)
    per_domain_playwright_cap = st.slider("Per-domain Playwright cap", 1, 5, 2, 1)
    playwright_timeout_s = st.slider("Playwright timeout (seconds)", 6, 25, 12, 1)

    st.divider()
    st.subheader("BYO API Keys")
    serp_key_env = os.getenv("SERPAPI_API_KEY", "")
    serpapi_key = st.text_input("SerpAPI key (SERPAPI_API_KEY)", value=serp_key_env, type="password", placeholder="Paste for session, or set env var")

    llm_refine = st.checkbox("LLM refine top sources (optional)", value=False)
    openai_key = ""
    if llm_refine:
        openai_key_env = os.getenv("OPENAI_API_KEY", "")
        openai_key = st.text_input("OpenAI key (OPENAI_API_KEY)", value=openai_key_env, type="password", placeholder="Used only if refine enabled")
        llm_top_n = st.slider("Refine top N sources", 3, 15, 8, 1)
    else:
        llm_top_n = 8

    st.divider()
    st.subheader("Adaptive Micro-pass (Cost Control)")
    micro_skip_tier = st.slider("Skip micro if macro Tier 1–3 results ≥", 1, 8, 3, 1)
    micro_skip_cands = st.slider("Skip micro if macro candidates ≥", 8, 40, 16, 1)

    st.divider()
    run_btn = st.button("Run Research", type="primary", use_container_width=True)



tab_research, tab_diag, tab_smoke = st.tabs(["Research Run", "Diagnostics", "Smoke Tests"])

with tab_research:
    colA, colB = st.columns([1, 1], gap="large")

    with colA:
        st.subheader("Run Status")
        status_box = st.empty()
        log_box = st.empty()

    with colB:
        st.subheader("Dossier Output")
        out_box = st.empty()


    def _progress_cb(msg: str):
        status_box.info(msg)
        if "log_lines" not in st.session_state:
            st.session_state.log_lines = []
        st.session_state.log_lines.append(f"{eng._now_iso()}  {msg}")
        st.session_state.log_lines = st.session_state.log_lines[-140:]
        log_box.code("\n".join(st.session_state.log_lines), language="text")


    def _validate_topic(t: str) -> Tuple[bool, str]:
        t = (t or "").strip()
        if not t:
            return False, "Primary topic is required."
        if len(t) < 6:
            return False, "Primary topic is too short."
        vague = {"aliens", "ufo", "uap", "ufos", "disclosure"}
        if t.lower().strip() in vague:
            return False, "Topic is too vague. Add a specific event, location, person, program, or date range."
        return True, ""


    if run_btn:
        ok, err = _validate_topic(primary_topic)
        if not ok:
            status_box.error(err)
            st.stop()
        if not serpapi_key:
            status_box.error("Missing SerpAPI key. Set SERPAPI_API_KEY or paste it for this session.")
            st.stop()

        job = eng.ResearchJob(
            primary_topic=primary_topic.strip(),
            time_scope=time_scope.strip() or None,
            geo_scope=geo_scope.strip() or None,
            event_focus=event_focus.strip() or None,
            confidence_threshold=float(confidence_threshold),
            max_sources=int(max_sources),
            max_serp_queries=int(max_serp_queries),
            include_archival=bool(include_archival),
            include_forums=bool(include_forums),
            include_patents=bool(include_patents),
            include_gov_docs=bool(include_gov_docs),
            primary_source_mode=bool(primary_source_mode),
            fulltext_hybrid=bool(fulltext_hybrid),
            max_fulltext_fetches=int(max_fulltext_fetches),
            per_domain_fetch_cap=int(per_domain_fetch_cap),
            request_timeout_s=int(request_timeout_s),
            max_pdf_downloads=int(max_pdf_downloads),
            per_domain_pdf_cap=int(per_domain_pdf_cap),
            playwright_fallback=bool(playwright_fallback),
            max_playwright_renders=int(max_playwright_renders),
            per_domain_playwright_cap=int(per_domain_playwright_cap),
            playwright_timeout_s=int(playwright_timeout_s),
            llm_refine_top_sources=bool(llm_refine),
            llm_refine_top_n=int(llm_top_n),
            allowlist_domains=[d for d in re.split(r"[\n,]+", (allowlist_raw or "")) if d.strip()] or None,
            blocklist_domains=[d for d in re.split(r"[\n,]+", (blocklist_raw or "")) if d.strip()] or None,
            prefer_tier_1_3=bool(prefer_tier_1_3),
            micro_min_tier_1_3_to_skip=int(micro_skip_tier),
            micro_min_candidates_to_skip=int(micro_skip_cands),
        )

        st.session_state.log_lines = []
        _progress_cb("Initializing run…")

        try:
            dossier = eng.run_research(job, serpapi_key=serpapi_key, openai_key=(openai_key or None), progress_cb=_progress_cb)

            status = dossier.get("status", "UNKNOWN")
            if status == "COMPLETED":
                status_box.success("Completed — dossier ready.")
            elif status == "FAILED":
                status_box.warning("Failed completion criteria — dossier includes reasons and next actions.")
            else:
                status_box.info(f"Run finished with status: {status}")

            metrics = dossier.get("metrics", {})
            out_box.json({
                "status": dossier.get("status"),
                "confidence_overall": dossier.get("confidence_overall"),
                "metrics": metrics,
                "high_authority_sources_count": len(dossier.get("high_authority_sources", []) or []),
                "core_claims_count": len(dossier.get("core_claims", []) or []),
                "claim_clusters_count": len((dossier.get("evidence_graph", {}) or {}).get("claim_clusters", []) or []),
                "key_entities_count": len(dossier.get("key_entities", []) or []),
                "runtime_seconds": dossier.get("runtime_seconds"),
            })

            # Observability Panel
            with st.expander("Debug / Telemetry (Queries • Scoring • Rejections)", expanded=True):
                tele = dossier.get("telemetry", {}) or {}
                qrows = tele.get("queries", []) or []
                notes = tele.get("notes", []) or []

                st.markdown("**Adaptive Notes**")
                if notes:
                    st.write(notes)
                else:
                    st.write("—")

                st.markdown("**Queries Executed**")
                if qrows:
                    st.dataframe(qrows, use_container_width=True, hide_index=True)
                else:
                    st.write("—")

                st.markdown("**Rejected Reasons (counts)**")
                rcounts = (metrics.get("rejected_reason_counts") or {})
                if rcounts:
                    st.dataframe([{"reason": k, "count": v} for k, v in sorted(rcounts.items(), key=lambda kv: kv[1], reverse=True)],
                                 use_container_width=True, hide_index=True)
                else:
                    st.write("—")

                st.markdown("**Top Source Score Breakdown (top 12)**")
                srcs = dossier.get("sources", []) or []
                top = []
                for s in srcs[:12]:
                    top.append({
                        "authority_score": s.get("authority_score"),
                        "tier": s.get("tier"),
                        "domain": s.get("domain"),
                        "fetched": s.get("fetched"),
                        "evidence_density": s.get("evidence_density"),
                        "citation_depth": s.get("citation_depth"),
                        "named_attribution": s.get("named_attribution"),
                        "recency": s.get("recency"),
                        "corroboration_count": s.get("corroboration_count"),
                        "url": s.get("url"),
                        "pdf_extracted": s.get("pdf_extracted"),
                        "pdf_text_len": s.get("pdf_text_len"),
                    "playwright_rendered": s.get("playwright_rendered"),
                    "playwright_title": s.get("playwright_title"),
                    })
                if top:
                    st.dataframe(top, use_container_width=True, hide_index=True)
                else:
                    st.write("—")

                st.markdown("**Evidence Graph: Claim Clusters (top 12)**")
                eg = dossier.get("evidence_graph", {}) or {}
                clusters = (eg.get("claim_clusters") or [])[:12]
                if clusters:
                    st.dataframe([{
                        "id": c.get("id"),
                        "corroboration_count": c.get("corroboration_count"),
                        "max_source_authority": c.get("max_source_authority"),
                        "confidence_score": c.get("confidence_score"),
                        "canonical_claim": c.get("canonical_claim"),
                        "supporting_domains": ", ".join((c.get("supporting_domains") or [])[:4]),
                    } for c in clusters], use_container_width=True, hide_index=True)
                else:
                    st.write("—")

                st.markdown("**Narrative Blueprint (preview)**")
                nb = dossier.get("narrative_blueprint", {}) or {}
                if nb:
                    st.json({
                        "cold_open_options": nb.get("cold_open_options", [])[:3],
                        "acts": nb.get("acts", {}),
                        "scene_beats": nb.get("scene_beats", [])[:6],
                    })
                else:
                    st.write("—")

            with st.expander("View Full Dossier JSON", expanded=False):
                st.json(dossier)

            dossier_bytes = json.dumps(dossier, indent=2, ensure_ascii=False).encode("utf-8")
            st.download_button(
                label="Download dossier JSON",
                data=dossier_bytes,
                file_name=f"uappress_dossier_{dossier.get('job_id','job')}.json",
                mime="application/json",
                use_container_width=True
            )

            # v5: Source Pack exports (CSV)
            def _to_csv_bytes(rows: List[Dict[str, Any]], fieldnames: List[str]) -> bytes:
                buf = io.StringIO()
                w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow(r)
                return buf.getvalue().encode("utf-8")

            # sources.csv
            src_rows = []
            for s in (dossier.get("sources") or []):
                src_rows.append({
                    "url": s.get("url",""),
                    "domain": s.get("domain",""),
                    "tier": s.get("tier"),
                    "authority_score": s.get("authority_score"),
                    "fetched": s.get("fetched"),
                    "pdf_extracted": s.get("pdf_extracted"),
                    "pdf_text_len": s.get("pdf_text_len"),
                    "publication_date": s.get("publication_date",""),
                    "title": s.get("title",""),
                })
            sources_csv = _to_csv_bytes(src_rows, ["url","domain","tier","authority_score","fetched","pdf_extracted","pdf_text_len","playwright_rendered","playwright_title","publication_date","title"])

            st.download_button(
                label="Download sources.csv",
                data=sources_csv,
                file_name=f"uappress_sources_{dossier.get('job_id','job')}.csv",
                mime="text/csv",
                use_container_width=True
            )

            # claim_clusters.csv
            eg = dossier.get("evidence_graph", {}) or {}
            clusters = eg.get("claim_clusters") or []
            cl_rows = []
            for c in clusters:
                cl_rows.append({
                    "cluster_id": c.get("id"),
                    "corroboration_count": c.get("corroboration_count"),
                    "confidence_score": c.get("confidence_score"),
                    "max_source_authority": c.get("max_source_authority"),
                    "canonical_claim": c.get("canonical_claim"),
                    "supporting_sources": " | ".join((c.get("supporting_sources") or [])[:8]),
                    "supporting_domains": " | ".join((c.get("supporting_domains") or [])[:8]),
                })
            clusters_csv = _to_csv_bytes(cl_rows, ["cluster_id","corroboration_count","confidence_score","max_source_authority","canonical_claim","supporting_sources","supporting_domains"])

            st.download_button(
                label="Download claim_clusters.csv",
                data=clusters_csv,
                file_name=f"uappress_claim_clusters_{dossier.get('job_id','job')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        except requests.HTTPError as e:
            status_box.error(f"HTTP error while calling SerpAPI or fetching sources: {e}")
        except Exception as e:
            status_box.error(f"Unexpected error: {e}")


    st.divider()
    st.caption("Tip: Start with a specific event/person/date range. Adaptive micro-pass can reduce SerpAPI cost when macro results are already strong.")



with tab_diag:
    st.subheader("Single-URL Diagnostic (Requests vs Playwright)")
    st.caption("Use this to troubleshoot extraction issues without spending SerpAPI queries.")
    test_url = st.text_input("Test URL", value="", placeholder="Paste an article or PDF URL")

    c1, c2, c3 = st.columns([1,1,1])
    do_req = c1.button("Run Requests Fetch", use_container_width=True)
    do_pw = c2.button("Run Playwright Render", use_container_width=True)
    do_pdf = c3.button("Run PDF Extract", use_container_width=True)

    if test_url:
        if do_req:
            st.info("Fetching via requests…")
            res = eng.fetch_url_text_cached(test_url, timeout_s=10)
            st.json({
                "ok": res.get("ok"),
                "status": res.get("status"),
                "content_type": res.get("content_type"),
                "text_len": len(res.get("text") or ""),
            })
            preview = (res.get("text") or "")[:800]
            if preview:
                st.code(preview, language="text")

        if do_pw:
            st.info("Rendering via Playwright/Chromium…")
            res = eng.render_url_text_playwright_cached(test_url, timeout_ms=12000)
            st.json({
                "ok": res.get("ok"),
                "title": res.get("title"),
                "text_len": len(res.get("text") or ""),
                "error": res.get("error"),
            })
            preview = (res.get("text") or "")[:800]
            if preview:
                st.code(preview, language="text")

        if do_pdf:
            st.info("Extracting PDF text…")
            res = eng.fetch_pdf_text_cached(test_url, timeout_s=12)
            st.json({
                "ok": res.get("ok"),
                "status": res.get("status"),
                "bytes": res.get("bytes"),
                "text_len": len(res.get("text") or ""),
                "error": res.get("error"),
            })
            preview = (res.get("text") or "")[:800]
            if preview:
                st.code(preview, language="text")
    else:
        st.write("Paste a URL above, then run a diagnostic.")

    with st.expander("Deployment Notes (Streamlit Cloud)", expanded=False):
        st.markdown(
            "- If Playwright fails on Streamlit Cloud, you likely need Playwright + Chromium installed.\n"
            "- Ensure dependencies include: `playwright`, `beautifulsoup4`, `pypdf` (or `PyPDF2`).\n"
            "- Streamlit Cloud often needs a post-install step to run `playwright install chromium`.\n"
        )




with tab_smoke:
    st.subheader("Smoke Tests (Benchmark Topics)")
    st.caption("Run these after code changes to confirm we didn't break research, PDFs, claim clustering, or narrative blueprint.")

    # Presets: deterministic, minimal input
    presets = [
        ("Nimitz Tic Tac 2004 (Primary-source)", {
            "primary_topic": "Nimitz Tic Tac 2004",
            "time_scope": "2004",
            "geo_scope": "Pacific",
            "event_focus": "DoD statements hearing report",
            "primary_source_mode": True,
        }),
        ("Phoenix Lights 1997 (Primary-source)", {
            "primary_topic": "Phoenix Lights 1997",
            "time_scope": "1997",
            "geo_scope": "Arizona",
            "event_focus": "official statements investigation report",
            "primary_source_mode": True,
        }),
        ("Roswell 1947 (Primary-source)", {
            "primary_topic": "Roswell 1947",
            "time_scope": "1947",
            "geo_scope": "New Mexico",
            "event_focus": "US Army Air Forces press release report",
            "primary_source_mode": True,
        }),
    ]

    preset_name = st.selectbox("Choose a smoke test", [p[0] for p in presets], index=0)
    preset = next(p[1] for p in presets if p[0] == preset_name)

    st.markdown("**Budget defaults for smoke tests** (kept small so it runs fast):")
    st.write({
        "max_serp_queries": 10,
        "max_sources": 20,
        "max_fulltext_fetches": 10,
        "max_pdf_downloads": 4,
        "playwright_fallback": False,
    })

    run_smoke = st.button("Run Smoke Test", type="primary", use_container_width=True)

    serp_key_env = os.getenv("SERPAPI_API_KEY", "")
    if not serp_key_env:
        st.warning("Set SERPAPI_API_KEY in your environment to run smoke tests.")

    if run_smoke:
        if not serp_key_env:
            st.stop()

        job = eng.ResearchJob(
            primary_topic=preset["primary_topic"],
            time_scope=preset.get("time_scope"),
            geo_scope=preset.get("geo_scope"),
            event_focus=preset.get("event_focus"),
            primary_source_mode=bool(preset.get("primary_source_mode", False)),

            confidence_threshold=0.70,
            max_sources=20,
            max_serp_queries=10,

            include_gov_docs=True,
            include_archival=True,
            include_patents=False,
            include_forums=False,

            fulltext_hybrid=True,
            max_fulltext_fetches=10,
            per_domain_fetch_cap=3,
            request_timeout_s=8,

            max_pdf_downloads=4,
            per_domain_pdf_cap=2,

            allowlist_domains=None,
            blocklist_domains=None,
            prefer_tier_1_3=True,

            llm_refine_top_sources=False,

            micro_min_tier_1_3_to_skip=3,
            micro_min_candidates_to_skip=14,

            playwright_fallback=False,
            max_playwright_renders=0,
            per_domain_playwright_cap=0,
            playwright_timeout_s=12,
        )

        status = st.empty()
        status.info("Running smoke test…")

        dossier = eng.run_research(job, serpapi_key=serp_key_env, openai_key=None, progress_cb=None)

        metrics = dossier.get("metrics", {}) or {}
        clusters = ((dossier.get("evidence_graph", {}) or {}).get("claim_clusters") or [])
        beats = ((dossier.get("narrative_blueprint", {}) or {}).get("scene_beats") or [])

        # Pass/fail gates
        gates = {
            "tier_1_3_sources>=3": sum(1 for s in dossier.get("sources", []) if s.get("tier") in (1,2,3)) >= 3,
            "pdf_extracted>=1": (metrics.get("pdf_extracted") or 0) >= 1,
            "claim_clusters>=8": len(clusters) >= 8,
            "scene_beats>=6": len(beats) >= 6,
        }
        passed = all(gates.values())

        if passed:
            status.success("Smoke test PASSED ✅")
        else:
            status.error("Smoke test FAILED ❌ (see gates below)")

        st.dataframe([{"gate": k, "pass": v} for k, v in gates.items()], use_container_width=True, hide_index=True)
        st.json({
            "status": dossier.get("status"),
            "confidence_overall": dossier.get("confidence_overall"),
            "metrics": metrics,
            "claim_clusters_count": len(clusters),
            "scene_beats_count": len(beats),
            "runtime_seconds": dossier.get("runtime_seconds"),
        })

        st.download_button(
            "Download smoke dossier JSON",
            data=json.dumps(dossier, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name=f"uappress_smoke_{dossier.get('job_id','job')}.json",
            mime="application/json",
            use_container_width=True
        )


