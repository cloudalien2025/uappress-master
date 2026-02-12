# uappress_research_engine_app.py
# Streamlit-first, engine-in-one-file. BYO APIs (SerpAPI + optional OpenAI).
# Deterministic JSON dossier output with two-pass (macro/micro) research + authority scoring.
#
# ENV VARS (recommended):
#   SERPAPI_API_KEY
#   OPENAI_API_KEY (only used if "LLM refine" is enabled)
#
# v2 additions:
# - Observability/Telemetry panel: queries executed, per-query tier mix, rejection reasons, score breakdown table
# - Query budgeter + early-stop: adaptive micro-pass (skip/limit micro when macro already yields sufficient Tier 1–3 + candidate volume)

from __future__ import annotations

import os
import re
import json
import time
import math
import hashlib
import datetime as dt
import io
import csv
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Set
from urllib.parse import urlparse

import streamlit as st
import requests


# -----------------------------
# Utilities
# -----------------------------

def _now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()

def _clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""

def _is_pdf(url: str) -> bool:
    return url.lower().split("?")[0].endswith(".pdf")

def _canonical_url(url: str) -> str:
    try:
        p = urlparse(url)
        query = p.query
        if query:
            kept = []
            for kv in query.split("&"):
                k = kv.split("=")[0].lower()
                if k.startswith("utm_") or k in {"gclid", "fbclid", "mc_cid", "mc_eid"}:
                    continue
                kept.append(kv)
            query = "&".join(kept)
        rebuilt = p._replace(fragment="", query=query).geturl()
        return rebuilt
    except Exception:
        return url

def _is_probable_paywall_domain(dom: str) -> bool:
    return dom in {"wsj.com", "ft.com", "economist.com"} or dom.endswith(".pressreader.com")


# -----------------------------
# SerpAPI Client (cached)
# -----------------------------

SERP_ENDPOINT = "https://serpapi.com/search.json"

@st.cache_data(show_spinner=False, ttl=60 * 60)  # 1 hour
def serpapi_search_cached(api_key: str, params: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "engine": params.get("engine", "google"),
        "q": params.get("q", ""),
        "num": params.get("num", 10),
        "hl": params.get("hl", "en"),
        "gl": params.get("gl", "us"),
        "api_key": api_key,
    }
    for k, v in params.items():
        if k in payload:
            continue
        payload[k] = v

    r = requests.get(SERP_ENDPOINT, params=payload, timeout=20)
    r.raise_for_status()
    return r.json()


def serp_extract_organic(result_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for item in result_json.get("organic_results", []) or []:
        link = item.get("link") or item.get("url")
        if not link:
            continue
        out.append({
            "title": _clean_whitespace(item.get("title", "")),
            "link": _canonical_url(link),
            "snippet": _clean_whitespace(item.get("snippet", "")),
            "position": item.get("position"),
            "source": item.get("source"),
            "date": item.get("date") or item.get("published_date") or "",
        })
    return out



# -----------------------------
# PDF fetching + extraction (guarded, cached)
# -----------------------------

@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)  # 24 hours
def fetch_pdf_text_cached(url: str, timeout_s: int = 12, max_bytes: int = 12_000_000) -> Dict[str, Any]:
    """Download and extract text from a PDF (best-effort).
    Guardrails:
      - caps bytes
      - timeouts
    Returns {"ok": bool, "status": int|None, "text": str, "bytes": int}
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; UAPpressResearchEngine/1.0)"}
        r = requests.get(url, headers=headers, timeout=timeout_s)
        status = r.status_code
        if status != 200:
            return {"ok": False, "status": status, "text": "", "bytes": 0}
        data = r.content or b""
        if len(data) > max_bytes:
            return {"ok": False, "status": status, "text": "", "bytes": len(data), "error": "pdf_too_large"}

        text = pdf_bytes_to_text(data)
        return {"ok": True, "status": status, "text": text, "bytes": len(data)}
    except Exception as e:
        return {"ok": False, "status": None, "text": "", "bytes": 0, "error": str(e)}


def pdf_bytes_to_text(data: bytes) -> str:
    """Best-effort PDF text extraction. Tries pypdf/PyPDF2 if installed."""
    if not data:
        return ""
    # Try pypdf
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for page in reader.pages[:50]:  # cap pages
            parts.append(page.extract_text() or "")
        return _clean_whitespace(" ".join(parts))
    except Exception:
        pass
    # Try PyPDF2
    try:
        from PyPDF2 import PdfReader  # type: ignore
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for page in reader.pages[:50]:
            parts.append(page.extract_text() or "")
        return _clean_whitespace(" ".join(parts))
    except Exception:
        return ""


def keyword_snippets(text: str, keywords: List[str], max_snips: int = 5, window: int = 140) -> List[str]:
    """Extract short snippets around keyword hits (deterministic)."""
    if not text or not keywords:
        return []
    low = text.lower()
    hits = []
    for kw in keywords:
        kw2 = kw.strip().lower()
        if not kw2 or len(kw2) < 4:
            continue
        pos = low.find(kw2)
        if pos >= 0:
            start = max(0, pos - window)
            end = min(len(text), pos + len(kw2) + window)
            snip = _clean_whitespace(text[start:end])
            hits.append(snip)
        if len(hits) >= max_snips:
            break
    # Dedup
    out, seen = [], set()
    for s in hits:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out[:max_snips]



# -----------------------------
# Playwright/Chromium fallback (guarded, cached)
# -----------------------------

@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)  # 24 hours
def render_url_text_playwright_cached(url: str, timeout_ms: int = 12000) -> Dict[str, Any]:
    """Render a URL with Playwright (Chromium) and extract visible text.
    Best-effort; if Playwright isn't installed/available, returns ok=False with error.
    """
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as e:
        return {"ok": False, "text": "", "title": "", "error": f"playwright_not_available: {e}"}

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = browser.new_page()
            page.set_default_timeout(timeout_ms)
            page.goto(url, wait_until="domcontentloaded")
            try:
                page.wait_for_timeout(750)
            except Exception:
                pass

            title = (page.title() or "").strip()
            text = page.evaluate("""() => {
                const pick = (el) => el ? (el.innerText || '').trim() : '';
                const candidates = [];
                candidates.push(pick(document.querySelector('article')));
                candidates.push(pick(document.querySelector('main')));
                candidates.push(pick(document.body));
                let best = '';
                for (const t of candidates) {
                    if (t && t.length > best.length) best = t;
                }
                return best;
            }""") or ""

            browser.close()

            text = (text or "").strip()
            if len(text) > 120000:
                text = text[:120000]
            return {"ok": True, "text": _clean_whitespace(text), "title": title}
    except Exception as e:
        return {"ok": False, "text": "", "title": "", "error": str(e)}


# -----------------------------
# Full-text fetching (hybrid) with guardrails (cached)
# -----------------------------

@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)  # 24 hours
def fetch_url_text_cached(url: str, timeout_s: int = 8) -> Dict[str, Any]:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; UAPpressResearchEngine/1.0)"}
        r = requests.get(url, headers=headers, timeout=timeout_s)
        ct = (r.headers.get("Content-Type") or "").lower()
        status = r.status_code
        if status != 200:
            return {"ok": False, "status": status, "text": "", "content_type": ct}
        if "pdf" in ct or _is_pdf(url):
            return {"ok": True, "status": status, "text": "", "content_type": ct}
        html = r.text or ""
        text = html_to_text(html)
        return {"ok": True, "status": status, "text": text, "content_type": ct}
    except Exception as e:
        return {"ok": False, "status": None, "text": "", "content_type": "", "error": str(e)}


def html_to_text(html: str) -> str:
    html = html or ""
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        return _clean_whitespace(text)
    except Exception:
        text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html)
        text = re.sub(r"(?is)<.*?>", " ", text)
        text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
        return _clean_whitespace(text)


# -----------------------------
# Authority Scoring
# -----------------------------

MAJOR_MEDIA_DOMAINS = {
    "nytimes.com", "bbc.com", "reuters.com", "apnews.com", "theguardian.com",
    "washingtonpost.com", "cnn.com", "foxnews.com", "nbcnews.com", "cbsnews.com",
    "abcnews.go.com", "npr.org", "time.com", "newsweek.com", "nationalgeographic.com",
    "scientificamerican.com"
}

INVESTIGATIVE_DOMAINS = {"propublica.org", "theintercept.com", "bellingcat.com"}

ACADEMIC_HINTS = (".edu", "arxiv.org", "nature.com", "science.org", "sciencedirect.com", "springer.com", "wiley.com")

def domain_tier_and_base(dom: str) -> Tuple[int, int]:
    if not dom:
        return 6, 30
    if dom.endswith(".gov") or dom.endswith(".mil") or dom in {"nasa.gov", "dni.gov"} or dom.endswith(".gov.uk"):
        return 1, 95
    if dom in MAJOR_MEDIA_DOMAINS:
        return 2, 85
    if dom.endswith(".edu") or dom in ACADEMIC_HINTS or any(dom.endswith(x) for x in ACADEMIC_HINTS):
        return 3, 80
    if dom in INVESTIGATIVE_DOMAINS:
        return 4, 70
    if dom.endswith("medium.com") or dom.endswith("substack.com") or dom.endswith("blogspot.com"):
        return 5, 50
    if dom in {"reddit.com", "twitter.com", "x.com", "facebook.com", "tiktok.com", "youtube.com"}:
        return 6, 30
    return 5, 50


def evidence_density_score(text: str, snippet: str) -> int:
    t = (text or "")[:20000]
    s = (snippet or "")
    joined = (t + " " + s).lower()
    score = 0
    for kw in ["foia", "declass", "report", "hearing", "testimony", "transcript", "memorandum", "directive", "briefing"]:
        if kw in joined:
            score += 10
    score += min(20, len(re.findall(r"\b\d{4}\b", joined)) * 3)
    score += min(20, joined.count('"') * 2 + joined.count("“") * 2 + joined.count("”") * 2)
    for kw in ["according to", "cited", "citation", "references", "source:"]:
        if kw in joined:
            score += 8
    return max(0, min(100, score))


def citation_depth_score(text: str) -> int:
    t = (text or "")[:40000].lower()
    score = 0
    if "references" in t or "bibliography" in t:
        score += 25
    if "footnote" in t or "endnote" in t:
        score += 10
    score += min(40, len(re.findall(r"\[\d+\]", t)) * 4)
    score += min(25, t.count("http") * 2)
    return max(0, min(100, score))


def named_attribution_score(text: str, snippet: str) -> int:
    joined = ((text or "")[:20000] + " " + (snippet or ""))
    hits = len(re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2}\s(?:said|stated|testified|wrote|reported)\b", joined))
    return min(100, hits * 12)


def recency_factor_score(pub_date: str) -> int:
    if not pub_date:
        return 40
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            d = dt.datetime.strptime(pub_date.strip(), fmt).date()
            days = (dt.date.today() - d).days
            if days < 0:
                days = 0
            return int(max(20, min(100, 100 - math.log1p(days) * 10)))
        except Exception:
            pass
    return 40


def authority_score(dom_tier_base, evidence_density, citation_depth, named_attr, recency, corroboration_count) -> int:
    tier, base = dom_tier_base
    domain_component = base
    cor = max(0, min(5, corroboration_count))
    corroboration_component = int((cor / 5.0) * 100)
    score = (
        domain_component * 0.30 +
        evidence_density * 0.20 +
        citation_depth * 0.15 +
        named_attr * 0.15 +
        recency * 0.10 +
        corroboration_component * 0.10
    )
    return int(max(0, min(100, round(score))))


# -----------------------------
# Entity + Claim extraction (deterministic)
# -----------------------------

STOPWORDS = set("""
The A An And Or But If Then Of In On At By For With From Into Over Under To As Is Are Was Were Be Been Being
This That These Those It Its Their His Her They Them We Us You Your I
""".split())

def extract_entities_from_text(title: str, snippet: str, max_entities: int = 18) -> List[str]:
    text = f"{title} {snippet}"
    text = re.sub(r"[\(\)\[\]\{\}]", " ", text)
    candidates = re.findall(r"\b(?:[A-Z][a-zA-Z\-]+(?:\s|$)){2,6}", text)
    cleaned: List[str] = []
    for c in candidates:
        c = _clean_whitespace(c)
        if not c or len(c) < 4:
            continue
        parts = c.split()
        if any(p in STOPWORDS for p in parts):
            continue
        parts = [p for p in parts if len(p) > 1]
        if len(parts) < 2:
            continue
        c2 = " ".join(parts)
        if c2.lower().startswith(("read more", "click here")):
            continue
        cleaned.append(c2)

    seen = set()
    out = []
    for c in cleaned:
        k = c.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
        if len(out) >= max_entities:
            break
    return out


def extract_claim_candidates(text: str, snippet: str, max_claims: int = 10) -> List[str]:
    src = _clean_whitespace(snippet or "")
    body = _clean_whitespace((text or "")[:25000])
    blob = (src + ". " + body)
    sents = re.split(r"(?<=[\.\!\?])\s+", blob)
    keep = []
    for s in sents:
        s2 = s.strip()
        if len(s2) < 35 or len(s2) > 220:
            continue
        low = s2.lower()
        if any(k in low for k in ["according to", "reported", "testified", "said", "stated", "claimed"]):
            keep.append(s2)
        elif re.search(r"\b(19\d{2}|20\d{2})\b", s2):
            keep.append(s2)
        if len(keep) >= max_claims:
            break
    out, seen = [], set()
    for s in keep:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out[:max_claims]


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class ResearchJob:
    primary_topic: str
    time_scope: Optional[str] = None
    geo_scope: Optional[str] = None
    event_focus: Optional[str] = None

    confidence_threshold: float = 0.75
    max_sources: int = 24
    max_serp_queries: int = 18

    include_archival: bool = True
    include_forums: bool = False
    include_patents: bool = True
    include_gov_docs: bool = True

    fulltext_hybrid: bool = True
    max_fulltext_fetches: int = 12
    per_domain_fetch_cap: int = 3
    request_timeout_s: int = 8

    serp_num: int = 10
    serp_gl: str = "us"
    serp_hl: str = "en"

    llm_refine_top_sources: bool = False
    llm_refine_top_n: int = 8

    # v5: primary-source mode + PDF budgets
    # v7: Playwright fallback (JS-heavy pages)
    playwright_fallback: bool = False
    max_playwright_renders: int = 6
    per_domain_playwright_cap: int = 2
    playwright_timeout_s: int = 12

    primary_source_mode: bool = False
    max_pdf_downloads: int = 6
    per_domain_pdf_cap: int = 2

    # v3: domain controls
    allowlist_domains: Optional[List[str]] = None  # if set, only these domains pass
    blocklist_domains: Optional[List[str]] = None  # always rejected
    prefer_tier_1_3: bool = True

    # v2: adaptive micro-pass
    micro_min_tier_1_3_to_skip: int = 3
    micro_min_candidates_to_skip: int = 16


@dataclass
class IngestedSource:
    url: str
    title: str
    snippet: str
    publisher: str
    publication_date: str
    domain: str
    tier: int
    tier_base: int
    fetched: bool
    content_type: str
    text_len: int


    # v5: PDF extraction
    pdf_extracted: bool
    pdf_text_len: int
    pdf_key_snippets: List[str]
    pdf_sha1: str

    # v7: Playwright
    playwright_rendered: bool
    playwright_title: str


    evidence_density: int
    citation_depth: int
    named_attribution: int
    recency: int
    corroboration_count: int
    authority_score: int

    entities: List[str]
    claim_candidates: List[str]


# -----------------------------
# Query Strategy
# -----------------------------

def build_macro_queries(job: ResearchJob) -> List[str]:
    t = job.primary_topic.strip()
    geo = f" {job.geo_scope.strip()}" if job.geo_scope else ""
    time_scope = f" {job.time_scope.strip()}" if job.time_scope else ""
    focus = f" {job.event_focus.strip()}" if job.event_focus else ""
    q_base = f"{t}{geo}{time_scope}{focus}".strip()

        qs = [
        f'"{q_base}" site:.gov' if job.include_gov_docs else f'"{q_base}"',
        f'"{q_base}" report OR hearing OR testimony',
        f'"{q_base}" declassified OR FOIA',
        f'"{q_base}" timeline',
        f'"{q_base}" investigation OR analysis',
        f'"{q_base}" history',
    ]
    if job.include_patents:
        qs.append(f'"{t}" patent OR uspto')
    if job.include_archival:
        qs.append(f'"{t}" archive OR pdf OR "press release"')

    out, seen = [], set()
    for q in qs:
        q2 = _clean_whitespace(q)
        if q2.lower() in seen:
            continue
        seen.add(q2.lower())
        out.append(q2)
    return out


def build_micro_queries(job: ResearchJob, entities: List[str]) -> List[str]:
    t = job.primary_topic.strip()
    qs = []
    for e in entities:
        e2 = e.strip()
        if not e2:
            continue
        qs.extend([
            # v5: in primary-source mode, bias micro queries to official/academic where possible
            pass

            f'"{e2}" "{t}"',
            f'"{e2}" testimony OR hearing OR statement',
            f'"{e2}" declassified OR FOIA OR report',
        ])
        if job.include_gov_docs:
            qs.append(f'"{e2}" site:.gov "{t}"')
    out, seen = [], set()
    for q in qs:
        q2 = _clean_whitespace(q)
        k = q2.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(q2)
    return out


# -----------------------------
# Ingestion Rules
# -----------------------------

SEO_SPAM_HINTS = ("best-", "top-", "review", "coupon", "promo", "discount", "casino", "slots", "loan", "payday", "forex", "crypto-news")

def should_reject_domain(dom: str, include_forums: bool, allowlist: Optional[List[str]] = None, blocklist: Optional[List[str]] = None) -> Tuple[bool, str]:
    if not dom:
        return True, "missing_domain"
    # v3 allow/block lists
    if blocklist and dom in set(d.lower().strip() for d in blocklist if d and d.strip()):
        return True, "blocked_domain"
    if allowlist:
        allowed = set(d.lower().strip() for d in allowlist if d and d.strip())
        if allowed and dom not in allowed:
            return True, "not_in_allowlist"
    if dom in {"reddit.com", "twitter.com", "x.com", "facebook.com", "tiktok.com", "youtube.com"}:
        return (False, "allowed_forum") if include_forums else (True, "forum_social_disallowed")
    for h in SEO_SPAM_HINTS:
        if h in dom:
            return True, "seo_spam_domain"
    return False, ""


def should_fetch_fulltext(dom: str, tier: int, job: ResearchJob, url: str) -> bool:
    if not job.fulltext_hybrid:
        return False
    if _is_pdf(url):
        return True
    if tier in (1, 2, 3) or (not job.primary_source_mode and tier == 4):
        if _is_probable_paywall_domain(dom):
            return False
        return True
    return False


def qualifies_minimal(title: str, snippet: str) -> bool:
    return len(title.strip()) >= 8 and (len(snippet.strip()) >= 40 or len(title.strip()) >= 20)


# -----------------------------
# Corroboration logic (simple clustering)
# -----------------------------

def claim_fingerprint(claim: str) -> str:
    c = claim.lower()
    c = re.sub(r"[^a-z0-9\s]", " ", c)
    c = re.sub(r"\s+", " ", c).strip()
    words = [w for w in c.split() if w not in {"the","a","an","and","or","to","of","in","on","at","for","with","from","by","that","this","it","was","were","is","are"}]
    return " ".join(words[:12])


def compute_corroboration(sources: List[IngestedSource]) -> Dict[str, int]:
    fp_to_domains: Dict[str, Set[str]] = {}
    for src in sources:
        for claim in src.claim_candidates:
            fp = claim_fingerprint(claim)
            if not fp:
                continue
            fp_to_domains.setdefault(fp, set()).add(src.domain)
    return {fp: len(domains) for fp, domains in fp_to_domains.items()}




# -----------------------------
# Cross-Run Knowledge Memory (lightweight persistent store)
# -----------------------------

KNOWLEDGE_MEMORY_FILE = "uappress_knowledge_memory.json"

def load_knowledge_memory() -> Dict[str, Any]:
    try:
        if os.path.exists(KNOWLEDGE_MEMORY_FILE):
            with open(KNOWLEDGE_MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"clusters": {}, "entities": {}}

def save_knowledge_memory(memory: Dict[str, Any]) -> None:
    try:
        with open(KNOWLEDGE_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2)
    except Exception:
        pass

def update_knowledge_memory(memory: Dict[str, Any], claim_clusters: List[Dict[str, Any]]) -> None:
    for c in claim_clusters:
        fp = c.get("fingerprint")
        if not fp:
            continue
        existing = memory["clusters"].get(fp)
        if not existing:
            memory["clusters"][fp] = {
                "first_seen": _now_iso(),
                "times_seen": 1,
                "last_corroboration": c.get("corroboration_count", 0),
            }
        else:
            existing["times_seen"] += 1
            existing["last_corroboration"] = c.get("corroboration_count", 0)
    save_knowledge_memory(memory)

# -----------------------------
# Evidence Graph + Narrative Blueprint (deterministic)
# -----------------------------

def build_claim_clusters(sources: List[IngestedSource], max_sources_scan: int = 18, max_clusters: int = 30) -> List[Dict[str, Any]]:
    """Cluster claims by fingerprint. Deterministic, uses existing claim_fingerprint.
    Returns clusters sorted by corroboration_count then max_source_authority.
    """
    clusters: Dict[str, Dict[str, Any]] = {}
    for s in sources[:max_sources_scan]:
        for c in s.claim_candidates:
            fp = claim_fingerprint(c)
            if not fp:
                continue
            cl = clusters.get(fp)
            if not cl:
                cl = {
                    "fingerprint": fp,
                    "canonical_claim": c,
                    "canonical_source": s.url,
                    "supporting_sources": set(),
                    "supporting_domains": set(),
                    "supporting_tiers": [],
                    "max_source_authority": s.authority_score,
                }
                clusters[fp] = cl
            cl["supporting_sources"].add(s.url)
            cl["supporting_domains"].add(s.domain)
            cl["supporting_tiers"].append(s.tier)
            if s.authority_score > cl["max_source_authority"]:
                cl["max_source_authority"] = s.authority_score
                cl["canonical_claim"] = c
                cl["canonical_source"] = s.url

    out = []
    for idx, (fp, cl) in enumerate(clusters.items()):
        doms = sorted(list(cl["supporting_domains"]))
        tiers = cl["supporting_tiers"] or []
        tier_mix = {
            "tier_1_3": sum(1 for t in tiers if t in (1,2,3)),
            "tier_4": sum(1 for t in tiers if t == 4),
            "tier_5_6": sum(1 for t in tiers if t in (5,6)),
        }
        corroboration = len(doms)
        out.append({
            "id": f"CC{idx+1:03d}",
            "fingerprint": fp,
            "canonical_claim": cl["canonical_claim"],
            "canonical_source": cl["canonical_source"],
            "supporting_sources": sorted(list(cl["supporting_sources"])),
            "supporting_domains": doms,
            "corroboration_count": corroboration,
            "tier_mix": tier_mix,
            "max_source_authority": cl["max_source_authority"],
            "confidence_score": round(min(0.98, 0.30 + (cl["max_source_authority"]/140.0) + (min(5, corroboration)/20.0)), 2),
        })

    out.sort(key=lambda x: (x["corroboration_count"], x["max_source_authority"], x["confidence_score"]), reverse=True)
    return out[:max_clusters]


def map_claim_to_cluster_id(clusters: List[Dict[str, Any]]) -> Dict[str, str]:
    return {c["fingerprint"]: c["id"] for c in clusters}


def build_contradictions_from_conflicts(conflict_matrix: List[Dict[str, Any]], cluster_map: Dict[str, str]) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for item in conflict_matrix or []:
        a = item.get("claim_a", "")
        b = item.get("claim_b", "")
        fa = claim_fingerprint(a)
        fb = claim_fingerprint(b)
        ida = cluster_map.get(fa)
        idb = cluster_map.get(fb)
        if not ida or not idb or ida == idb:
            continue
        key = tuple(sorted([ida, idb]) + [item.get("dispute_type","")])
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "cluster_a": ida,
            "cluster_b": idb,
            "dispute_type": item.get("dispute_type",""),
            "example_claim_a": a,
            "example_claim_b": b,
        })
    return out[:12]


def build_narrative_blueprint(job: ResearchJob, clusters: List[Dict[str, Any]], contradictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    top = clusters[:6]
    cold_open_options = []
    for c in top[:3]:
        cold_open_options.append({
            "cluster_id": c["id"],
            "hook": c["canonical_claim"],
            "required_citations": c["supporting_sources"][:3],
        })

    # Act structure: deterministic mapping from ranked clusters
    act1 = [c["id"] for c in top[:2]]
    act2 = [c["id"] for c in top[2:4]]
    act3 = [c["id"] for c in top[4:6]]

    scene_beats = []
    beat_id = 1
    for act_name, ids in [("ACT_1_SETUP", act1), ("ACT_2_COMPLICATION", act2), ("ACT_3_RESOLUTION", act3)]:
        for cid in ids:
            cl = next((x for x in clusters if x["id"] == cid), None)
            if not cl:
                continue
            scene_beats.append({
                "beat_id": f"B{beat_id:02d}",
                "act": act_name,
                "cluster_id": cid,
                "beat_summary": cl["canonical_claim"],
                "required_citations": cl["supporting_sources"][:4],
            })
            beat_id += 1

    # Conflict beats: include up to 2 contradiction pairs as explicit beats
    conflict_beats = []
    for k, con in enumerate(contradictions[:2], start=1):
        conflict_beats.append({
            "beat_id": f"X{k:02d}",
            "act": "ACT_2_COMPLICATION",
            "cluster_pair": [con["cluster_a"], con["cluster_b"]],
            "beat_summary": f"Evaluate conflict between {con['cluster_a']} and {con['cluster_b']}: {con.get('dispute_type','conflict')}.",
            "required_citations": [],
        })

    blueprint = {
        "topic": job.primary_topic,
        "cold_open_options": cold_open_options,
        "acts": {
            "ACT_1_SETUP": act1,
            "ACT_2_COMPLICATION": act2,
            "ACT_3_RESOLUTION": act3,
        },
        "scene_beats": scene_beats + conflict_beats,
        "open_loops": [
            "Which cluster is supported by the highest-tier sources (Tier 1–3)?",
            "Which contradiction can be resolved by primary documentation?",
        ],
    }
    return blueprint

# -----------------------------
# Engine Core (with telemetry + adaptive micro)
# -----------------------------

def run_research(job: ResearchJob, serpapi_key: str, openai_key: Optional[str] = None, progress_cb=None) -> Dict[str, Any]:
    started = time.time()
    job_id = _sha1(json.dumps(asdict(job), sort_keys=True))[:12]

    telemetry: Dict[str, Any] = {
        "job_id": job_id,
        "queries": [],  # list of {pass, q, results, tier1_3, domains}
        "notes": [],
    }
telemetry["notes"].append(f"Domain controls: prefer_tier_1_3={job.prefer_tier_1_3}, allowlist={len(job.allowlist_domains or [])}, blocklist={len(job.blocklist_domains or [])}.")

telemetry["notes"].append(f"PDF budgets: max_pdf_downloads={job.max_pdf_downloads}, per_domain_pdf_cap={job.per_domain_pdf_cap}. Primary-source mode={job.primary_source_mode}.")

telemetry["notes"].append(f"Playwright: enabled={job.playwright_fallback}, max_renders={job.max_playwright_renders}, per_domain_cap={job.per_domain_playwright_cap}, timeout_s={job.playwright_timeout_s}.")


    def progress(msg: str):
        if progress_cb:
            progress_cb(msg)

    max_queries = max(1, job.max_serp_queries)
    max_sources = max(8, job.max_sources)

    used_queries = 0
    all_org: List[Dict[str, Any]] = []

    # 1) Macro pass
    macro_qs = build_macro_queries(job)[:max_queries]
    progress("Running MACRO pass…")
    for q in macro_qs:
        if used_queries >= max_queries:
            break
        used_queries += 1
        params = {"engine": "google", "q": q, "num": job.serp_num, "hl": job.serp_hl, "gl": job.serp_gl}
        try:
            j = serpapi_search_cached(serpapi_key, params)
            org = serp_extract_organic(j)
            for o in org:
                o["query"] = q
                o["pass"] = "MACRO"
            all_org.extend(org)

            # telemetry for this query
            doms = [_domain(o.get("link","")) for o in org]
            tiers = [domain_tier_and_base(d)[0] for d in doms if d]
            telemetry["queries"].append({
                "pass": "MACRO",
                "q": q,
                "results": len(org),
                "tier1_3": sum(1 for t in tiers if t in (1,2,3)),
                "unique_domains": len(set(d for d in doms if d)),
            })
        except Exception as e:
            telemetry["queries"].append({"pass": "MACRO", "q": q, "results": 0, "tier1_3": 0, "unique_domains": 0, "error": str(e)})
            all_org.append({"title": "", "link": "", "snippet": "", "error": str(e), "query": q, "pass": "MACRO"})

    # 2) Preliminary entity discovery (from macro)
    discovered_entities: List[str] = []
    for o in all_org[:40]:
        discovered_entities.extend(extract_entities_from_text(o.get("title", ""), o.get("snippet", ""), max_entities=8))

    ent_seen, ent_out = set(), []
    for e in discovered_entities:
        k = e.lower()
        if k in ent_seen:
            continue
        ent_seen.add(k)
        ent_out.append(e)
    discovered_entities = ent_out[:8]

    # 3) Candidate preview to decide whether to run micro-pass (cost control)
    # Build a lightweight candidate set from macro results only.
    macro_candidates = []
    macro_seen = set()
    for o in all_org:
        if o.get("pass") != "MACRO":
            continue
        url = o.get("link") or ""
        if not url or url in macro_seen:
            continue
        macro_seen.add(url)
        dom = _domain(url)
        reject, _ = should_reject_domain(dom, job.include_forums, job.allowlist_domains, job.blocklist_domains)
        if reject:
            continue
        if not qualifies_minimal(o.get("title",""), o.get("snippet","")):
            continue
        macro_candidates.append(o)

    macro_tier1_3 = 0
    for o in macro_candidates[:60]:
        dom = _domain(o.get("link",""))
        tier, _ = domain_tier_and_base(dom)
        if tier in (1,2,3):
            macro_tier1_3 += 1

    skip_micro = (macro_tier1_3 >= job.micro_min_tier_1_3_to_skip) and (len(macro_candidates) >= job.micro_min_candidates_to_skip)
    if skip_micro:
        telemetry["notes"].append(f"Adaptive micro-pass skipped: macro_tier1_3={macro_tier1_3}, macro_candidates={len(macro_candidates)}.")
        progress("Adaptive micro-pass skipped (macro already strong).")
    else:
        telemetry["notes"].append(f"Adaptive micro-pass enabled: macro_tier1_3={macro_tier1_3}, macro_candidates={len(macro_candidates)}.")
        # 4) Micro pass
        remaining = max_queries - used_queries
        micro_qs = build_micro_queries(job, discovered_entities)[:max(0, remaining)]
        progress("Running MICRO pass…")
        for q in micro_qs:
            if used_queries >= max_queries:
                break
            used_queries += 1
            params = {"engine": "google", "q": q, "num": job.serp_num, "hl": job.serp_hl, "gl": job.serp_gl}
            try:
                j = serpapi_search_cached(serpapi_key, params)
                org = serp_extract_organic(j)
                for o in org:
                    o["query"] = q
                    o["pass"] = "MICRO"
                all_org.extend(org)

                doms = [_domain(o.get("link","")) for o in org]
                tiers = [domain_tier_and_base(d)[0] for d in doms if d]
                telemetry["queries"].append({
                    "pass": "MICRO",
                    "q": q,
                    "results": len(org),
                    "tier1_3": sum(1 for t in tiers if t in (1,2,3)),
                    "unique_domains": len(set(d for d in doms if d)),
                })
            except Exception as e:
                telemetry["queries"].append({"pass": "MICRO", "q": q, "results": 0, "tier1_3": 0, "unique_domains": 0, "error": str(e)})
                all_org.append({"title": "", "link": "", "snippet": "", "error": str(e), "query": q, "pass": "MICRO"})

    # 5) Dedup + ingestion filtering (both passes)
    progress("Filtering & ingesting sources…")

    seen_urls = set()
    candidates: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for o in all_org:
        url = o.get("link") or ""
        title = o.get("title") or ""
        snippet = o.get("snippet") or ""
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        dom = _domain(url)
        reject, reason = should_reject_domain(dom, job.include_forums, job.allowlist_domains, job.blocklist_domains)
        if reject:
            rejected.append({"url": url, "domain": dom, "reason": reason, "query": o.get("query",""), "pass": o.get("pass","")})
            continue
        if not qualifies_minimal(title, snippet):
            rejected.append({"url": url, "domain": dom, "reason": "too_thin", "query": o.get("query",""), "pass": o.get("pass","")})
            continue

        candidates.append(o)
        if len(candidates) >= max_sources * 2:
            break

        # v3: prefer Tier 1–3 domains when selecting which candidates to ingest (cost-effective quality)
    if job.prefer_tier_1_3 and candidates:
        def _cand_key(o):
            d = _domain(o.get("link",""))
            t, base = domain_tier_and_base(d)
            # lower tier number preferred; higher base preferred; keep original order as tiebreaker using pass (MACRO before MICRO)
            p = o.get("pass","")
            pass_rank = 0 if p == "MACRO" else 1
            return (t, -base, pass_rank)
        candidates.sort(key=_cand_key)

# 6) Fulltext hybrid fetching with caps
    fulltext_fetch_budget = max(0, job.max_fulltext_fetches)
    per_domain_fetch: Dict[str, int] = {}
    ingested: List[IngestedSource] = []

    pdf_fetch_budget = max(0, job.max_pdf_downloads)
    per_domain_pdf: Dict[str, int] = {}

    playwright_budget = max(0, job.max_playwright_renders)
    per_domain_pw: Dict[str, int] = {}

    for o in candidates:
        if len(ingested) >= max_sources:
            break

        url = o.get("link") or ""
        title = o.get("title") or ""
        snippet = o.get("snippet") or ""
        pub_date = o.get("date") or ""
        publisher = o.get("source") or ""
        dom = _domain(url)
        tier, base = domain_tier_and_base(dom)

        do_fetch = should_fetch_fulltext(dom, tier, job, url)
        fetched = False
        content_type = ""
        text = ""

        if do_fetch and fulltext_fetch_budget > 0 and per_domain_fetch.get(dom, 0) < job.per_domain_fetch_cap:
            res = fetch_url_text_cached(url, timeout_s=job.request_timeout_s)
            fetched = bool(res.get("ok"))
            content_type = res.get("content_type") or ""
            text = res.get("text") or ""
            per_domain_fetch[dom] = per_domain_fetch.get(dom, 0) + 1
            fulltext_fetch_budget -= 1

        # v5: PDF extraction (guarded)
        pdf_extracted = False
        pdf_text = ""
        pdf_text_len = 0
        pdf_key_snips: List[str] = []
        pdf_sha1 = ""

        if _is_pdf(url) and pdf_fetch_budget > 0 and per_domain_pdf.get(dom, 0) < job.per_domain_pdf_cap:
            pres = fetch_pdf_text_cached(url, timeout_s=max(10, job.request_timeout_s + 2))
            if pres.get("ok"):
                pdf_extracted = True
                pdf_text = pres.get("text") or ""
                pdf_text_len = len(pdf_text)
                pdf_sha1 = hashlib.sha1(pdf_text.encode("utf-8", "ignore")).hexdigest() if pdf_text else ""
                # Keywords from primary topic + event focus (split into tokens)
                kws = []
                for part in [job.primary_topic or "", job.event_focus or ""]:
                    kws += [w for w in re.split(r"[^A-Za-z0-9]+", part) if w]
                kws = [k for k in kws if len(k) >= 4][:12]
                pdf_key_snips = keyword_snippets(pdf_text, kws, max_snips=5, window=150)
            per_domain_pdf[dom] = per_domain_pdf.get(dom, 0) + 1
            pdf_fetch_budget -= 1

        # v7: Playwright fallback if JS-heavy / thin content (guarded)
        pw_rendered = False
        pw_title = ""
        if job.playwright_fallback and playwright_budget > 0 and per_domain_pw.get(dom, 0) < job.per_domain_playwright_cap:
            trigger = (do_fetch and (not fetched)) or (fetched and (not _is_pdf(url)) and len(text or "") < 800) or ((not fetched) and tier in (1,2,3))
            if trigger and (not _is_pdf(url)):
                pw = render_url_text_playwright_cached(url, timeout_ms=int(job.playwright_timeout_s * 1000))
                if pw.get("ok") and (pw.get("text") or ""):
                    text = pw.get("text") or ""
                    content_type = (content_type or "") + "|playwright"
                    fetched = True
                    pw_rendered = True
                    pw_title = pw.get("title") or ""
                    per_domain_pw[dom] = per_domain_pw.get(dom, 0) + 1
                    playwright_budget -= 1

        text_len = len(text)
        if fetched and (not _is_pdf(url)) and text_len < 800:
            fetched = False
            text = ""
            text_len = 0

        base_text_for_scoring = pdf_text if pdf_extracted else text
        ev = evidence_density_score(base_text_for_scoring, snippet)
        cd = citation_depth_score(base_text_for_scoring) if (fetched or pdf_extracted) else 10
        na = named_attribution_score(text, snippet)
        rc = recency_factor_score(pub_date)

        entities = extract_entities_from_text(title, snippet, max_entities=12)
        claims = extract_claim_candidates(base_text_for_scoring, snippet, max_claims=8)

        ingested.append(IngestedSource(
            url=url,
            title=title,
            snippet=snippet,
            publisher=publisher,
            publication_date=pub_date,
            domain=dom,
            tier=tier,
            tier_base=base,
            fetched=fetched,
            content_type=content_type,
            text_len=text_len,
            pdf_extracted=pdf_extracted,
            pdf_text_len=pdf_text_len,
            pdf_key_snippets=pdf_key_snips,
            pdf_sha1=pdf_sha1,
            playwright_rendered=pw_rendered,
            playwright_title=pw_title,
            evidence_density=ev,
            citation_depth=cd,
            named_attribution=na,
            recency=rc,
            corroboration_count=0,
            authority_score=0,
            entities=entities,
            claim_candidates=claims,
        ))

    # 7) Corroboration + final scores
    cor_map = compute_corroboration(ingested)
    for src in ingested:
        cor_counts = []
        for c in src.claim_candidates:
            fp = claim_fingerprint(c)
            if fp in cor_map:
                cor_counts.append(cor_map[fp])
        src.corroboration_count = max(cor_counts) if cor_counts else 0
        src.authority_score = authority_score(
            (src.tier, src.tier_base),
            src.evidence_density,
            src.citation_depth,
            src.named_attribution,
            src.recency,
            src.corroboration_count,
        )

    ingested.sort(key=lambda s: (s.authority_score, s.tier_base, s.text_len), reverse=True)

    # 8) Build dossier + completion checks
    dossier = build_dossier(job, ingested, rejected, used_queries, started, job_id, telemetry)
    status = evaluate_completion(job, ingested, dossier)
    dossier["status"] = status["status"]
    dossier["completion"] = status
    dossier["runtime_seconds"] = round(time.time() - started, 2)
    return dossier


def evaluate_completion(job: ResearchJob, sources: List[IngestedSource], dossier: Dict[str, Any]) -> Dict[str, Any]:
    qualified = len(sources)
    tier_1_3 = sum(1 for s in sources if s.tier in (1,2,3))
    core_claims = dossier.get("core_claims", []) or []
    max_cor = max((_safe_int(c.get("corroboration_count", 0), 0) for c in core_claims), default=0)

    min_sources = min(12, job.max_sources)
    min_tier = 3

    ok_sources = qualified >= min_sources
    ok_tier = tier_1_3 >= min_tier
    ok_cor = (max_cor >= 2) if core_claims else False
    conf = _safe_float(dossier.get("confidence_overall", 0.0), 0.0)
    ok_conf = conf >= job.confidence_threshold

    if ok_sources and ok_tier and ok_conf:
        return {
            "status": "COMPLETED",
            "criteria": {
                "qualified_sources": {"value": qualified, "min": min_sources, "ok": ok_sources},
                "tier_1_3_sources": {"value": tier_1_3, "min": min_tier, "ok": ok_tier},
                "max_corroboration": {"value": max_cor, "min": 2, "ok": ok_cor},
                "confidence_overall": {"value": conf, "min": job.confidence_threshold, "ok": ok_conf},
            },
            "notes": "Completed based on source volume, authority mix, and confidence threshold."
        }

    reasons = []
    if qualified < 5:
        reasons.append("Too few qualified sources found (<5).")
    if tier_1_3 == 0:
        reasons.append("No Tier 1–3 sources found.")
    if qualified > 0 and all(s.tier >= 5 for s in sources):
        reasons.append("Only low-tier domains found (Tier 5–6).")
    if core_claims and max_cor == 0:
        reasons.append("No corroboration detected for primary claims (corroboration_count=0).")
    if not reasons:
        reasons.append("Completion criteria not met within query/source budget.")

    return {
        "status": "FAILED",
        "criteria": {
            "qualified_sources": {"value": qualified, "min": min_sources, "ok": ok_sources},
            "tier_1_3_sources": {"value": tier_1_3, "min": min_tier, "ok": ok_tier},
            "max_corroboration": {"value": max_cor, "min": 2, "ok": ok_cor},
            "confidence_overall": {"value": conf, "min": job.confidence_threshold, "ok": ok_conf},
        },
        "reasons": reasons,
        "recommended_next_action": recommend_next_action(job, sources, tier_1_3)
    }


def recommend_next_action(job: ResearchJob, sources: List[IngestedSource], tier_1_3: int) -> str:
    if not sources:
        return "Broaden the topic or add a specific named entity/event focus; increase max_serp_queries slightly."
    if tier_1_3 == 0:
        return "Add targeted site:.gov and academic queries; include specific program names or official report titles."
    return "Increase max_serp_queries slightly or tighten allow/blocklists (next); refine event_focus or time_scope."



def calibrated_confidence_overall(sources: List[IngestedSource], clusters: List[Dict[str, Any]]) -> float:
    if not sources:
        return 0.0
    avg_authority = sum(s.authority_score for s in sources[:10]) / min(10, len(sources))
    tier_1_3 = sum(1 for s in sources if s.tier in (1,2,3))
    pdf_bonus = sum(1 for s in sources if s.pdf_extracted)
    cluster_strength = sum(int(c.get("corroboration_count", 0)) for c in (clusters or [])[:6])

    base = (avg_authority / 100.0) * 0.50
    tier_factor = min(1.0, tier_1_3 / 8.0) * 0.20
    pdf_factor = min(1.0, pdf_bonus / 5.0) * 0.15
    cluster_factor = min(1.0, cluster_strength / 15.0) * 0.15

    conf = base + tier_factor + pdf_factor + cluster_factor
    return round(min(0.99, max(0.1, conf)), 2)



def build_dossier(
    job: ResearchJob,
    sources: List[IngestedSource],
    rejected: List[Dict[str, Any]],
    used_queries: int,
    started_ts: float,
    job_id: str,
    telemetry: Dict[str, Any],
) -> Dict[str, Any]:
    top_sources = sources[:8]
    top_domains = sorted({s.domain for s in top_sources})

    timeline = []
    for s in top_sources:
        for c in s.claim_candidates:
            m = re.search(r"\b(19\d{2}|20\d{2})\b", c)
            if m:
                year = m.group(1)
                timeline.append({
                    "date": year,
                    "event": c,
                    "confidence_score": round(min(0.95, 0.5 + (s.authority_score / 200.0)), 2)
                })
            if len(timeline) >= 12:
                break
        if len(timeline) >= 12:
            break

    ent_scores: Dict[str, float] = {}
    for s in sources[:16]:
        for e in s.entities:
            ent_scores[e] = ent_scores.get(e, 0.0) + (s.authority_score / 100.0)

    key_entities = []
    for e, sc in sorted(ent_scores.items(), key=lambda kv: kv[1], reverse=True)[:14]:
        key_entities.append({"name": e, "type": "unknown", "role": "", "credibility_score": round(min(1.0, 0.4 + sc / 6.0), 2)})

    fp_best: Dict[str, Dict[str, Any]] = {}
    cor_map = compute_corroboration(sources)
    for s in sources[:16]:
        for c in s.claim_candidates:
            fp = claim_fingerprint(c)
            if not fp:
                continue
            cor = cor_map.get(fp, 0)
            cand = {
                "claim": c,
                "first_source": s.url,
                "corroboration_count": cor,
                "evidence_strength": "high" if (s.tier in (1,2,3) and s.authority_score >= 75) else "medium" if s.authority_score >= 55 else "low",
                "confidence_score": round(min(0.98, 0.35 + (s.authority_score / 140.0) + (min(5, cor)/20.0)), 2),
                "_score": s.authority_score,
            }
            existing = fp_best.get(fp)
            if not existing or (cand["corroboration_count"], cand["_score"]) > (existing["corroboration_count"], existing["_score"]):
                fp_best[fp] = cand

    core_claims = sorted(fp_best.values(), key=lambda x: (x["corroboration_count"], x["_score"], x["confidence_score"]), reverse=True)
    for c in core_claims:
        c.pop("_score", None)
    core_claims = core_claims[:12]

    conflict_matrix = []
    for i in range(min(8, len(core_claims))):
        for j in range(i+1, min(8, len(core_claims))):
            a = core_claims[i]["claim"].lower()
            b = core_claims[j]["claim"].lower()
            pairs = [("no evidence", "evidence"), ("denied", "confirmed"), ("hoax", "authentic"), ("weather balloon", "craft")]
            for x, y in pairs:
                if (x in a and y in b) or (y in a and x in b):
                    conflict_matrix.append({"claim_a": core_claims[i]["claim"], "claim_b": core_claims[j]["claim"], "dispute_type": f"keyword_conflict:{x}<->{y}"})
                    break
            if len(conflict_matrix) >= 8:
                break
        if len(conflict_matrix) >= 8:
            break

    # v5: Primary-source mode biases toward official/academic PDFs and proceedings
    if job.primary_source_mode:
        qs = [
            f'"{q_base}" site:.gov OR site:.mil',
            f'"{q_base}" pdf OR report OR "press release"',
            f'"{q_base}" hearing OR testimony OR transcript',
            f'"{q_base}" FOIA OR declassified',
            f'"{q_base}" site:.edu OR arxiv OR "journal"',
            f'"{q_base}" timeline',
        ]
    else:
    high_authority_sources = [s.url for s in sources[:16] if s.tier in (1,2,3) and s.authority_score >= 70][:12]

    # v4: Evidence Graph (claim clusters) + contradictions + narrative blueprint
    # v6: Cross-run memory integration
    memory = load_knowledge_memory()
    update_knowledge_memory(memory, claim_clusters)

    claim_clusters = build_claim_clusters(sources, max_sources_scan=18, max_clusters=30)
    cluster_map = map_claim_to_cluster_id(claim_clusters)
    contradictions = build_contradictions_from_conflicts(conflict_matrix, cluster_map)
    narrative_blueprint = build_narrative_blueprint(job, claim_clusters, contradictions)

    open_questions = []
    if not high_authority_sources:
        open_questions.append("Which official documents (Tier 1) directly address the primary topic?")
    if core_claims and all(c.get("corroboration_count", 0) < 2 for c in core_claims):
        open_questions.append("Which claims can be corroborated by at least two independent Tier 1–3 sources?")
    open_questions.append("Which named entities/programs should be queried next for deeper primary-source coverage?")
    open_questions = open_questions[:6]

    pressure = []
    if core_claims:
        pressure.append("Identify the single most corroborated claim and build the narrative spine around it.")
        pressure.append("Surface the strongest conflicting claims and isolate their evidence chains.")
        pressure.append("Pinpoint timeline inflection points where official statements changed.")
    pressure = pressure[:6]

    confidence_overall = calibrated_confidence_overall(sources, claim_clusters)

    macro_summary = _clean_whitespace(
        f"Collected {len(sources)} sources across {len({s.domain for s in sources}) if sources else 0} domains; "
        f"top domains: {', '.join(top_domains[:6]) if top_domains else 'none'}."
    )

    rej_counts = Counter([r.get("reason","unknown") for r in rejected])

    dossier = {
        "job_id": job_id,
        "created_at": _now_iso(),
        "topic": job.primary_topic,
        "inputs": asdict(job),
        "macro_summary": macro_summary,
        "timeline": timeline,
        "key_entities": key_entities,
        "core_claims": core_claims,
        "conflict_matrix": conflict_matrix,
        "high_authority_sources": high_authority_sources,
        "open_questions": open_questions,
        "narrative_pressure_points": pressure,
        "confidence_overall": confidence_overall,
        "metrics": {
            "serp_queries_used": used_queries,
            "sources_ingested": len(sources),
            "sources_rejected": len(rejected),
            "unique_domains": len({s.domain for s in sources}) if sources else 0,
            "fulltext_fetched": sum(1 for s in sources if s.fetched),
            "pdf_extracted": sum(1 for s in sources if s.pdf_extracted),
            "playwright_rendered": sum(1 for s in sources if getattr(s, "playwright_rendered", False)),
            "knowledge_memory_clusters_total": len(load_knowledge_memory().get("clusters", {})),
            "rejected_reason_counts": dict(rej_counts),
        },
        "evidence_graph": {
            "claim_clusters": claim_clusters,
            "contradictions": contradictions,
        },
        "narrative_blueprint": narrative_blueprint,
        "telemetry": telemetry,
        "sources": [asdict(s) for s in sources],
        "rejected": rejected[:60],
    }
    return dossier

# End of engine module
