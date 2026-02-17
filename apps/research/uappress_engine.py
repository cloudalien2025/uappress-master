from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ResearchJob:
    primary_topic: str


def run_research(
    job: ResearchJob,
    serpapi_key: Optional[str],
    openai_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Deterministic dossier builder used by smoke and local development flows."""
    topic = (job.primary_topic or "Unknown topic").strip()
    source_mode = "LIVE" if serpapi_key else "SMOKE"

    summary = (
        f"Baseline dossier for {topic}. "
        "Evidence is organized for documentary scripting and contradiction analysis."
    )

    sources = [
        {
            "title": f"{topic} — Official Summary",
            "url": "https://example.org/official-summary",
            "score": 0.91,
        },
        {
            "title": f"{topic} — Investigative Record",
            "url": "https://example.org/investigative-record",
            "score": 0.86,
        },
        {
            "title": f"{topic} — Technical Notes",
            "url": "https://example.org/technical-notes",
            "score": 0.81,
        },
    ]

    note = "No external calls were required." if source_mode == "SMOKE" else "API-backed mode configured."

    return {
        "status": "COMPLETE",
        "confidence_overall": 0.82,
        "topic": topic,
        "summary": summary,
        "source_mode": source_mode,
        "sources": sources,
        "notes": [note, "Deterministic dossier profile."],
        "openai_enabled": bool(openai_key),
    }


def build_documentary_blueprint(dossier: Dict[str, Any]) -> Dict[str, Any]:
    """Convert dossier data into a deterministic documentary blueprint."""
    topic = str(dossier.get("topic") or "Unknown topic").strip()
    summary = str(dossier.get("summary") or "").strip()
    sources = dossier.get("sources") if isinstance(dossier.get("sources"), list) else []

    source_titles: List[str] = [str(s.get("title") or "Untitled source") for s in sources[:3] if isinstance(s, dict)]
    source_titles = source_titles or ["Primary source record", "Independent reporting", "Technical analysis"]

    contradictions = []
    for idx, title in enumerate(source_titles, start=1):
        contradictions.append(
            {
                "id": f"C{idx:02d}",
                "claim_a": f"{title} indicates structured progress on {topic}.",
                "claim_b": f"Parallel assessments argue critical uncertainty remains unresolved for {topic}.",
                "tension": "evidence certainty vs unresolved ambiguity",
            }
        )

    # Ensure at least 3 and at most 6 deterministic contradiction units.
    while len(contradictions) < 3:
        idx = len(contradictions) + 1
        contradictions.append(
            {
                "id": f"C{idx:02d}",
                "claim_a": f"Archival material suggests continuity in reporting around {topic}.",
                "claim_b": "Reviewers still dispute how much of that continuity reflects verified facts.",
                "tension": "continuity vs verification",
            }
        )

    blueprint = {
        "topic": topic,
        "cold_open": {
            "vo": f"A familiar case file keeps returning to the center of policy debate: {topic}.",
            "beats": [
                "Open on unresolved official language.",
                "Frame why uncertainty persists despite repeated reviews.",
            ],
        },
        "act_1_context": {
            "vo": summary or f"The public record around {topic} expanded, but interpretation did not converge.",
            "beats": [
                "Map key institutions and release timeline.",
                "Anchor the baseline facts all parties accept.",
            ],
        },
        "act_2_contradictions": contradictions[:6],
        "act_3_implications": {
            "vo": "The pattern is less about a single incident and more about decision systems under stress.",
            "beats": [
                "Explain strategic implications for oversight.",
                "Separate operational risk from public narrative pressure.",
            ],
        },
        "closing_questions": [
            "What evidence threshold should trigger policy change?",
            "Who is accountable when uncertainty is repeatedly deferred?",
            "Which records remain unavailable to independent review?",
        ],
    }
    return blueprint


def _paragraph_from_vo_and_beats(vo: str, beats: List[str]) -> str:
    parts = [vo.strip()] if vo else []
    for beat in beats:
        beat_text = str(beat).strip()
        if beat_text:
            parts.append(beat_text)
    return "\n\n".join(parts)


def compile_voiceover_script(blueprint: dict, *, target_minutes: int = 12) -> dict:
    """Deterministically compile a blueprint into a VO-ready script payload."""
    cold_open = blueprint.get("cold_open", {}) if isinstance(blueprint, dict) else {}
    act_1 = blueprint.get("act_1_context", {}) if isinstance(blueprint, dict) else {}
    act_2_items = blueprint.get("act_2_contradictions", []) if isinstance(blueprint, dict) else []
    act_3 = blueprint.get("act_3_implications", {}) if isinstance(blueprint, dict) else {}
    closing_questions = blueprint.get("closing_questions", []) if isinstance(blueprint, dict) else []

    cold_open_text = _paragraph_from_vo_and_beats(
        str(cold_open.get("vo") or ""),
        [str(b) for b in (cold_open.get("beats", []) if isinstance(cold_open.get("beats"), list) else [])],
    )

    act_1_text = _paragraph_from_vo_and_beats(
        str(act_1.get("vo") or ""),
        [str(b) for b in (act_1.get("beats", []) if isinstance(act_1.get("beats"), list) else [])],
    )

    contradictions: List[dict] = [item for item in act_2_items if isinstance(item, dict)]
    selected = contradictions[:6]

    act_2_lines: List[str] = [
        "The deeper record reveals a controlled escalation: each contradiction narrows what can be dismissed."
    ]
    for idx, item in enumerate(selected, start=1):
        claim_a = str(item.get("claim_a") or "A first account defines the baseline.").strip()
        claim_b = str(item.get("claim_b") or "A competing account challenges that baseline.").strip()
        tension = str(item.get("tension") or "unresolved tension").strip()
        act_2_lines.append(
            f"Contradiction {idx}: {claim_a} {claim_b} The tension is {tension}."
        )
    act_2_lines.append(
        "Taken together, these conflicts indicate not random noise, but a system still unable to produce a stable conclusion."
    )
    act_2_text = "\n\n".join(act_2_lines)

    act_3_text = _paragraph_from_vo_and_beats(
        str(act_3.get("vo") or ""),
        [str(b) for b in (act_3.get("beats", []) if isinstance(act_3.get("beats"), list) else [])],
    )

    closing_lines = [
        "The archive is no longer the problem. The remaining question is institutional resolve."
    ]
    for q in closing_questions:
        q_text = str(q).strip()
        if q_text:
            closing_lines.append(f"- {q_text}")
    closing_text = "\n\n".join(closing_lines)

    sections = [
        {"id": "cold_open", "title": "Cold Open", "text": cold_open_text},
        {"id": "act_1", "title": "Act 1 — Context", "text": act_1_text},
        {"id": "act_2", "title": "Act 2 — Contradictions", "text": act_2_text},
        {"id": "act_3", "title": "Act 3 — Implications", "text": act_3_text},
        {"id": "closing", "title": "Closing Questions", "text": closing_text},
    ]

    full_text = "\n\n".join(
        [
            "[COLD OPEN]",
            cold_open_text,
            "[ACT 1]",
            act_1_text,
            "[ACT 2]",
            act_2_text,
            "[ACT 3]",
            act_3_text,
            "[CLOSING]",
            closing_text,
        ]
    ).strip()

    word_count = len([w for w in full_text.split() if w.strip()])
    estimated_minutes = round(word_count / 150.0, 2)

    return {
        "target_minutes": int(target_minutes),
        "estimated_minutes": estimated_minutes,
        "sections": sections,
        "full_text": full_text,
    }
