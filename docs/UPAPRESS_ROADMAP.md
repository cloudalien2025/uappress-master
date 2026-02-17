# UAPpress Master Roadmap (Research → Video)

## Goals
- Define a milestone-based path from research input to video output.
- Preserve CI stability for Streamlit + Playwright hooks and deterministic smoke behavior.
- Keep each milestone independently shippable.

## Cross-Cutting Contracts
- **Deterministic smoke mode:** enabled by `UAPPRESS_SMOKE=1` or `CI=1`; no secrets required.
- **UI hooks:** preserve all existing markers, including `TEST_HOOK:APP_LOADED`, `TEST_HOOK:RUN_STARTED`, `TEST_HOOK:RUN_DONE`.
- **Import safety:** app must render even if engine import fails; emit explicit import marker.
- **BYO API behavior:** SerpAPI required only in real mode; OpenAI optional and only for real-mode enhancements.

## Stage Contracts

### Stage 1 — Topic Intelligence Engine
**Input**
```json
{ "idea": "string" }
```
**Output**
```json
{
  "idea": "string",
  "viability": {
    "authority_gap": 0.0,
    "suspense_score": 0.0,
    "contradiction_density": 0.0,
    "saturation": 0.0,
    "overall": 0.0
  }
}
```

### Stage 2 — Research Engine (Dossier Builder)
**Input**
```json
{
  "primary_topic": "string",
  "confidence_threshold": 0.58,
  "max_serp_queries": 12,
  "max_sources": 25,
  "include_gov_docs": true
}
```
**Output**
```json
{
  "status": "COMPLETE|PRELIMINARY|ERROR",
  "primary_topic": "string",
  "confidence_overall": 0.0,
  "summary": "string",
  "sources": [{"title": "string", "url": "string", "score": 0.0}],
  "claim_clusters": [{"claim": "string", "evidence": ["string"], "confidence": 0.0}],
  "contradictions": [{"claim_a": "string", "claim_b": "string", "tension": "string", "sources": [{"title": "string", "url": "string"}]}]
}
```

### Stage 3 — Documentary Blueprint Generator
**Input:** dossier (Stage 2).

**Output**
```json
{
  "title": "string",
  "logline": "string",
  "cold_open": {"vo": "string", "beats": ["string"]},
  "act_1_context": {"vo": "string", "beats": ["string"]},
  "act_2_contradictions": [{"claim": "string", "tension": "string", "sources": [{"title": "string", "url": "string"}]}],
  "act_3_implications": {"vo": "string", "beats": ["string"]},
  "closing_questions": ["string"],
  "thumbnail_angles": ["string"],
  "shorts_hooks": ["string"]
}
```

### Stage 4 — Script Compiler
**Input:** blueprint (Stage 3).

**Output**
```text
# COLD_OPEN
...
# ACT_1_CONTEXT
...
# ACT_2_CONTRADICTIONS
...
# ACT_3_IMPLICATIONS
...
# CLOSING
...
```

### Stage 5 — Scene Plan Generator
**Input:** VO script + blueprint.

**Output**
```json
[
  {"scene_id": 1, "duration_s": 6.5, "vo": "string", "visual_prompt": "string", "on_screen_text": "string"}
]
```

### Stage 6 — MP3 / Audio Generation (TTS)
**Input:** VO script.

**Output**
```json
{
  "audio_path": "artifacts/voice.mp3",
  "duration_s": 0.0,
  "segments": [{"section": "string", "start_s": 0.0, "end_s": 0.0}]
}
```
- Smoke mode: deterministic placeholder output.
- Real mode: OpenAI TTS when key supplied.

### Stage 7 — Visual Generation
**Input:** scene plan.

**Output**
```json
{
  "images": [{"scene_id": 1, "path": "artifacts/scene_001.png", "prompt": "string"}]
}
```
- Smoke mode: placeholders.

### Stage 8 — Video Assembly
**Input:** images + MP3 + subtitles.

**Output**
```json
{ "video_path": "artifacts/final.mp4", "status": "COMPLETE|SKIPPED_SMOKE" }
```

## Milestones
1. **Milestone 1 (this iteration):** package/import reliability + Stage 3 blueprint generation + UI rendering of blueprint.
2. **Milestone 2:** Stage 1 topic intelligence implementation + UI tab.
3. **Milestone 3:** Stage 4 script compiler + deterministic smoke fixture.
4. **Milestone 4:** Stage 5 scene planner + timing heuristics.
5. **Milestone 5:** Stage 6 TTS (smoke placeholder + real OpenAI path).
6. **Milestone 6:** Stage 7 visuals + Stage 8 smoke-safe assembly skeleton.
