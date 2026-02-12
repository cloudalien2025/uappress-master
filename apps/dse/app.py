# UAPpress Documentary Script Engine (DSE v1) — Streamlit-first
# Target: 35–45 min long-form documentary scripts (single narrator)
# Input: UAPpress Research Engine dossier JSON (machine-readable)
# Output: Script TXT + Script JSON + Citation Index
#
# BYO APIs: Optional OpenAI key for "polish" layer. Deterministic structure always works without OpenAI.

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import requests


# -----------------------------
# Utilities
# -----------------------------

def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _now_iso() -> str:
    import datetime as dt
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""

def _is_gov_mil_edu(dom: str) -> bool:
    dom = dom.lower().strip()
    return dom.endswith(".gov") or dom.endswith(".mil") or dom.endswith(".edu")


# -----------------------------
# Citation Index
# -----------------------------

def build_citation_index(dossier: Dict[str, Any], max_n: int = 40) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Returns:
      citation_index: [{id:'S01', url, domain, tier, authority_score, title}]
      url_to_id: {url: 'S01'}
    """
    sources = dossier.get("sources") or []
    # Sort by authority_score desc then tier asc
    def _key(s):
        return (-float(s.get("authority_score", 0) or 0), int(s.get("tier", 9) or 9))
    sources_sorted = sorted(sources, key=_key)[:max_n]

    citation_index = []
    url_to_id = {}
    for i, s in enumerate(sources_sorted, start=1):
        sid = f"S{i:02d}"
        url = s.get("url", "")
        url_to_id[url] = sid
        citation_index.append({
            "id": sid,
            "url": url,
            "domain": s.get("domain") or _domain(url),
            "tier": s.get("tier"),
            "authority_score": s.get("authority_score"),
            "title": s.get("title") or "",
        })
    return citation_index, url_to_id


# -----------------------------
# Deterministic Act Builder (no OpenAI)
# -----------------------------

@dataclass
class Beat:
    beat_id: str
    cluster_id: str
    beat_summary: str
    citations: List[str]  # ['S01','S05']


def _cluster_tags(canonical_claim: str, domains: List[str], tier_mix: Dict[str, Any]) -> Dict[str, bool]:
    txt = (canonical_claim or "").lower()

    witness = any(k in txt for k in [
        "witness", "pilot", "radar", "operator", "crew", "airman", "marine", "sailor",
        "reported", "saw", "observed", "encounter", "testified", "account", "eyewitness"
    ])

    official = any(k in txt for k in [
        "department", "dod", "pentagon", "navy", "air force", "cia", "fbi", "dni", "aaaro",
        "hearing", "testimony", "report", "memo", "briefing", "statement", "press release",
        "declassified", "foia", "congress", "senate", "house"
    ]) or any(_is_gov_mil_edu(d) for d in domains) or (tier_mix or {}).get("tier_1_3", 0) >= 2

    incident = any(k in txt for k in [
        "incident", "event", "occurred", "on ", "at ", "near ", "off the coast", "over ", "above ",
        "timeline", "date", "time", "minutes", "seconds"
    ])

    fracture = any(k in txt for k in [
        "contradict", "inconsistent", "dispute", "denied", "retracted", "redacted",
        "conflicting", "changed", "revised", "missing", "unverified", "cannot confirm"
    ])

    return {"witness": witness, "official": official, "incident": incident, "fracture": fracture}


def build_acts_from_dossier(
    dossier: Dict[str, Any],
    runtime_target_min: int = 40,
    suspense: str = "Medium",
    citation_density: str = "Standard",
) -> Dict[str, Any]:
    eg = dossier.get("evidence_graph", {}) or {}
    clusters = eg.get("claim_clusters") or []
    contradictions = eg.get("contradictions") or []
    topic = dossier.get("topic") or dossier.get("primary_topic") or "Untitled Topic"

    citation_index, url_to_id = build_citation_index(dossier, max_n=50)

    # Build cluster -> citations mapping
    cluster_citations: Dict[str, List[str]] = {}
    for c in clusters:
        cids: List[str] = []
        for u in (c.get("supporting_sources") or [])[:10]:
            if u in url_to_id:
                cids.append(url_to_id[u])
        # De-dupe, keep order
        seen = set()
        cids2 = []
        for x in cids:
            if x in seen:
                continue
            seen.add(x)
            cids2.append(x)
        cluster_citations[c.get("id")] = cids2

    # Score clusters for selection (deterministic)
    def _score(c):
        return (
            float(c.get("corroboration_count", 0) or 0) * 3.0
            + float(c.get("max_source_authority", 0) or 0) * 0.06
            + float(c.get("confidence_score", 0) or 0) * 10.0
        )

    clusters_sorted = sorted(clusters, key=_score, reverse=True)

    # Tag clusters
    tagged = []
    for c in clusters_sorted:
        tags = _cluster_tags(
            c.get("canonical_claim", ""),
            c.get("supporting_domains") or [],
            c.get("tier_mix") or {}
        )
        tagged.append((c, tags))

    act1: List[Dict[str, Any]] = []
    act2: List[Dict[str, Any]] = []
    act3: List[Dict[str, Any]] = []
    act4: List[Dict[str, Any]] = []
    act5: List[Dict[str, Any]] = []

    # Act 1: strongest, incident-ish
    for c, tags in tagged:
        if len(act1) >= 3:
            break
        if tags["incident"] or (not tags["witness"] and not tags["official"]):
            act1.append(c)

    # Act 2: witness
    for c, tags in tagged:
        if len(act2) >= 3:
            break
        if c in act1:
            continue
        if tags["witness"]:
            act2.append(c)

    # Act 3: official narrative
    for c, tags in tagged:
        if len(act3) >= 3:
            break
        if c in act1 or c in act2:
            continue
        if tags["official"]:
            act3.append(c)

    # Act 4: fracture / contradictions
    used_cluster_ids = set([x.get("id") for x in act1 + act2 + act3])
    conflict_pairs = []
    for con in contradictions[:6]:
        a = con.get("cluster_a")
        b = con.get("cluster_b")
        if not a or not b:
            continue
        conflict_pairs.append({"pair": [a, b], "dispute_type": con.get("dispute_type", "conflict")})

    for c, tags in tagged:
        if len(act4) >= 3:
            break
        if c.get("id") in used_cluster_ids:
            continue
        if tags["fracture"] or (c.get("corroboration_count", 0) <= 2 and c.get("max_source_authority", 0) >= 70):
            act4.append(c)
            used_cluster_ids.add(c.get("id"))

    # Act 5: unresolved / open loops
    for c, tags in tagged:
        if len(act5) >= 3:
            break
        if c.get("id") in used_cluster_ids:
            continue
        if 0.45 <= float(c.get("confidence_score", 0) or 0) <= 0.78:
            act5.append(c)
            used_cluster_ids.add(c.get("id"))

    # Fallback fill
    def _fill(act_list: List[Dict[str, Any]], target_n: int):
        for c, _tags in tagged:
            if len(act_list) >= target_n:
                break
            if c.get("id") in used_cluster_ids:
                continue
            act_list.append(c)
            used_cluster_ids.add(c.get("id"))

    _fill(act1, 3)
    _fill(act2, 3)
    _fill(act3, 3)
    _fill(act4, 3)
    _fill(act5, 3)

    def beats_for(act_prefix: str, selected: List[Dict[str, Any]]) -> List[Beat]:
        beats: List[Beat] = []
        cap = 6 if citation_density == "Heavy" else 4 if citation_density == "Standard" else 2
        for i, c in enumerate(selected, start=1):
            bid = f"{act_prefix}_B{i:02d}"
            beats.append(Beat(
                beat_id=bid,
                cluster_id=c.get("id"),
                beat_summary=c.get("canonical_claim") or "",
                citations=(cluster_citations.get(c.get("id")) or [])[:cap],
            ))
        return beats

    acts = {
        "ACT_1_INCIDENT": beats_for("A1", act1),
        "ACT_2_WITNESS_LAYER": beats_for("A2", act2),
        "ACT_3_OFFICIAL_NARRATIVE": beats_for("A3", act3),
        "ACT_4_FRACTURE": beats_for("A4", act4),
        "ACT_5_UNRESOLVED": beats_for("A5", act5),
    }

    return {
        "engine": "DSE_v1",
        "generated_at": _now_iso(),
        "topic": topic,
        "runtime_target_minutes": int(runtime_target_min),
        "controls": {
            "suspense": suspense,
            "citation_density": citation_density,
            "single_narrator": True,
        },
        "contradiction_pairs": conflict_pairs,
        "citation_index": citation_index,
        "acts": {k: [b.__dict__ for b in v] for k, v in acts.items()},
    }


# -----------------------------
# OpenAI Polish Layer (optional; BYO)
# -----------------------------

def openai_expand_act(
    api_key: str,
    model: str,
    topic: str,
    act_name: str,
    beats: List[Dict[str, Any]],
    contradiction_pairs: List[Dict[str, Any]],
    suspense: str,
) -> str:
    sys = (
        "You are a documentary narration writer. You must obey strict evidence constraints.\n"
        "ABSOLUTE RULES:\n"
        "1) Use ONLY the provided beats as factual material. Do NOT add facts, names, dates, or claims not present.\n"
        "2) Preserve citations exactly as given, like [S01] [S12]. Do not invent new citations.\n"
        "3) If a beat is uncertain, use qualifying language: 'reportedly', 'according to', 'unverified', 'disputed'.\n"
        "4) Single narrator voice. No dialogue.\n"
        "5) Suspense comes from structure: withhold confirmation, end segments with evidence-based questions.\n"
    )

    user = {
        "topic": topic,
        "act": act_name,
        "beats": beats,
        "contradiction_pairs": contradiction_pairs[:4],
        "suspense_intensity": suspense,
        "length_guidance": "Write ~7–10 minutes of narration for this act, unless the act is ACT_5_UNRESOLVED (write ~5–7 minutes).",
        "format": "Return plain text narration with section breaks and keep citations inline.",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "temperature": 0.7 if suspense == "High" else 0.5 if suspense == "Medium" else 0.35,
        "max_tokens": 2000,
    }

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI API error {r.status_code}: {r.text[:400]}")

    data = r.json()
    return (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "").strip()


def build_full_script_text(structured: Dict[str, Any]) -> str:
    topic = structured.get("topic") or "Untitled Topic"
    acts = structured.get("acts") or {}
    suspense = (structured.get("controls") or {}).get("suspense", "Medium")

    lines = []
    lines.append(topic)
    lines.append("")
    lines.append(f"Runtime Target: {structured.get('runtime_target_minutes', 40)} minutes")
    lines.append(f"Suspense: {suspense}")
    lines.append("Single Narrator: Yes")
    lines.append("")
    lines.append("=== COLD OPEN (draft) ===")
    lines.append("Open with the most compelling, evidence-backed moment. Do not overstate beyond sources.")
    lines.append("")
    for act_name, beats in acts.items():
        lines.append(f"=== {act_name} ===")
        for b in beats:
            cite = " ".join(b.get("citations") or [])
            summary = (b.get("beat_summary") or "").strip()
            lines.append(f"- {summary} {cite}".strip())
        lines.append("")
        lines.append("End this act with an evidence-based question that pulls the viewer forward.")
        lines.append("")

    lines.append("=== OUTRO ===")
    lines.append("Summarize what is verified, what is disputed, and what remains unknown. No speculation.")
    return "\n".join(lines)


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="UAPpress Documentary Script Engine (DSE v1)", layout="wide")
st.title("UAPpress Documentary Script Engine (DSE v1)")
st.caption("Engineering mode: deterministic 5-act structure + optional OpenAI polish. Long-form (35–45 min), single narrator.")

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("1) Upload Dossier JSON")
    up = st.file_uploader("Upload a dossier JSON from the UAPpress Research Engine", type=["json"])

    st.subheader("2) Controls")
    runtime_target = st.slider("Runtime target (minutes)", 35, 45, 40, 1)
    suspense = st.selectbox("Suspense intensity", ["Low", "Medium", "High"], index=1)
    citation_density = st.selectbox("Citation density", ["Light", "Standard", "Heavy"], index=1)

    st.subheader("3) Optional OpenAI Polish (BYO)")
    use_openai = st.checkbox("Enable OpenAI narration expansion (polish layer)", value=False)
    openai_key = st.text_input("OpenAI API Key", type="password", value="")
    model = st.text_input("Model name", value="gpt-4o-mini")

    build_btn = st.button("Build Documentary Script", type="primary", use_container_width=True, disabled=(up is None))

with right:
    st.subheader("Output")
    if up is None:
        st.info("Upload a dossier JSON to begin.")
    elif build_btn:
        dossier = json.loads(up.read().decode("utf-8"))
        structured = build_acts_from_dossier(
            dossier,
            runtime_target_min=runtime_target,
            suspense=suspense,
            citation_density=citation_density,
        )

        draft_script = build_full_script_text(structured)

        expanded_script = ""
        if use_openai:
            if not openai_key.strip():
                st.error("OpenAI key is required when polish is enabled.")
            else:
                st.info("Generating narration via OpenAI (act-by-act)…")
                topic = structured.get("topic", "")
                contradiction_pairs = structured.get("contradiction_pairs") or []
                acts = structured.get("acts") or {}
                act_texts = []
                order = [
                    "ACT_1_INCIDENT",
                    "ACT_2_WITNESS_LAYER",
                    "ACT_3_OFFICIAL_NARRATIVE",
                    "ACT_4_FRACTURE",
                    "ACT_5_UNRESOLVED",
                ]
                for act_name in order:
                    beats = acts.get(act_name) or []
                    try:
                        txt = openai_expand_act(
                            api_key=openai_key.strip(),
                            model=model.strip(),
                            topic=topic,
                            act_name=act_name,
                            beats=beats,
                            contradiction_pairs=contradiction_pairs,
                            suspense=suspense,
                        )
                        act_texts.append(f"=== {act_name} ===\n{txt}\n")
                    except Exception as e:
                        act_texts.append(f"=== {act_name} ===\n[OPENAI_ERROR] {e}\n")
                expanded_script = "\n".join(act_texts)

        final = expanded_script if (use_openai and expanded_script.strip() and "[OPENAI_ERROR]" not in expanded_script) else draft_script

        st.markdown("### Script Preview")
        st.text_area("Script", value=final, height=520)

        st.markdown("### Downloads")
        script_bytes = final.encode("utf-8")
        json_bytes = json.dumps(structured, indent=2, ensure_ascii=False).encode("utf-8")

        st.download_button("Download script.txt", data=script_bytes, file_name="uappress_documentary_script.txt", mime="text/plain", use_container_width=True)
        st.download_button("Download script.json", data=json_bytes, file_name="uappress_documentary_script.json", mime="application/json", use_container_width=True)

        st.markdown("### Citation Index")
        cindex = structured.get("citation_index") or []
        if cindex:
            md = []
            for c in cindex:
                md.append(f"- **[{c.get('id')}]** {c.get('domain')} (Tier {c.get('tier')}, Score {c.get('authority_score')}): {c.get('url')}")
            st.markdown("\n".join(md))
        else:
            st.write("No citations available in dossier.")

        with st.expander("Structured Acts (debug)", expanded=False):
            st.json(structured.get("acts"))
    else:
        dossier = json.loads(up.read().decode("utf-8"))
        st.success("Dossier loaded.")
        st.write({
            "topic": dossier.get("topic") or dossier.get("primary_topic"),
            "sources": len(dossier.get("sources") or []),
            "claim_clusters": len(((dossier.get("evidence_graph") or {}).get("claim_clusters") or [])),
            "confidence_overall": dossier.get("confidence_overall"),
        })
        st.caption("Click **Build Documentary Script** to generate the 5-act structure and script outputs.")
