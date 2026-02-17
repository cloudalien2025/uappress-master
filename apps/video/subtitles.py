from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List


_SECTION_MARKER_RE = re.compile(r"\[(?:[A-Z][A-Z0-9 _\-]{1,40})\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\b\w+(?:['-]\w+)?\b")


def _normalize_script(script_text: str) -> str:
    text = script_text or ""
    text = _SECTION_MARKER_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _wrap_chunk(words: List[str], max_chars: int, max_lines: int) -> str:
    lines: List[str] = []
    current: List[str] = []
    for word in words:
        candidate = " ".join(current + [word]).strip()
        if current and len(candidate) > max_chars:
            lines.append(" ".join(current))
            current = [word]
            if len(lines) >= max_lines:
                break
        else:
            current.append(word)

    if current and len(lines) < max_lines:
        lines.append(" ".join(current))

    return "\n".join(line for line in lines if line)


def split_script_into_captions(script_text: str, *, max_chars: int = 72, max_lines: int = 2) -> List[Dict[str, int | str]]:
    normalized = _normalize_script(script_text)
    if not normalized:
        return []

    captions: List[Dict[str, int | str]] = []
    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]

    for paragraph in paragraphs:
        raw_sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]
        sentences = raw_sentences if raw_sentences else [paragraph]

        for sentence in sentences:
            words = sentence.split()
            cursor = 0
            per_caption_budget = max_chars * max_lines
            while cursor < len(words):
                take = 1
                while cursor + take <= len(words):
                    segment = " ".join(words[cursor : cursor + take])
                    if len(segment) <= per_caption_budget:
                        take += 1
                        continue
                    break
                take = max(1, take - 1)
                chunk_words = words[cursor : cursor + take]
                chunk_text = _wrap_chunk(chunk_words, max_chars=max_chars, max_lines=max_lines)
                cursor += take
                if not chunk_text:
                    continue
                captions.append({"text": chunk_text, "words": _word_count(chunk_text)})

    return captions


def assign_timings(captions: List[Dict[str, int | str]], *, total_seconds: float) -> List[Dict[str, int | float | str]]:
    if not captions:
        return []

    total = max(0.0, float(total_seconds or 0.0))
    total_words = sum(max(1, int(cap.get("words", 0) or 0)) for cap in captions)
    if total_words <= 0:
        total_words = len(captions)

    timed: List[Dict[str, int | float | str]] = []
    raw_durations: List[float] = []
    for cap in captions:
        words = max(1, int(cap.get("words", 0) or 0))
        base = total * (words / total_words) if total > 0 else 0.0
        raw_durations.append(max(1.2, min(6.0, base)))

    cursor = 0.0
    for i, cap in enumerate(captions, start=1):
        remaining_slots = len(captions) - i
        remaining_total = max(0.0, total - cursor)
        proposed = raw_durations[i - 1]

        if remaining_slots == 0:
            duration = remaining_total
        else:
            min_remaining = 0.0
            if remaining_total > 0:
                min_remaining = min_remaining
            duration = min(proposed, max(0.0, remaining_total - min_remaining))

        start = round(cursor, 3)
        end = round(min(total, cursor + duration), 3)
        if end < start:
            end = start

        timed.append(
            {
                "index": i,
                "start": start,
                "end": end,
                "text": str(cap.get("text", "")),
            }
        )
        cursor = end

    if timed:
        timed[-1]["end"] = round(total, 3)

    return timed


def _srt_timestamp(seconds: float) -> str:
    ms_total = int(round(max(0.0, seconds) * 1000))
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_srt(timed: List[Dict[str, int | float | str]], out_path: str) -> str:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    blocks: List[str] = []
    for item in timed:
        idx = int(item.get("index", 0) or 0)
        start = _srt_timestamp(float(item.get("start", 0.0) or 0.0))
        end = _srt_timestamp(float(item.get("end", 0.0) or 0.0))
        text = str(item.get("text", "")).strip()
        blocks.append(f"{idx}\n{start} --> {end}\n{text}\n")

    path.write_text("\n".join(blocks), encoding="utf-8")
    return str(path)
