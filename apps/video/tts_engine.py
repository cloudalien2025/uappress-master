from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional


def _duration_estimate(script_text: str) -> float:
    words = len((script_text or "").split())
    return round(words / 150.0 * 60.0, 2)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate_vo_audio(
    script_text: str,
    *,
    out_dir: str = "outputs/audio",
    voice: str = "onyx",
    model: str = "gpt-4o-mini-tts",
    openai_key: Optional[str] = None,
    smoke: bool = False,
) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if smoke:
        payload = b"UAPPRESS_SMOKE_MP3_PLACEHOLDER_V1\n"
        mp3_path = out_path / "vo_smoke.mp3"
        mp3_path.write_bytes(payload)
        return {
            "mode": "smoke",
            "mp3_path": str(mp3_path),
            "duration_seconds": _duration_estimate(script_text),
            "voice": voice,
            "model": model,
            "sha256": _sha256_bytes(payload),
        }

    if not openai_key:
        raise ValueError("openai_key is required for real TTS generation")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenAI package is required for real TTS generation") from exc

    text_hash = hashlib.sha256((script_text or "").encode("utf-8")).hexdigest()[:12]
    mp3_path = out_path / f"vo_{text_hash}.mp3"

    client = OpenAI(api_key=openai_key)
    speech = client.audio.speech.create(
        model=model,
        voice=voice,
        input=script_text,
        format="mp3",
    )

    audio_bytes: bytes
    if hasattr(speech, "content"):
        audio_bytes = speech.content  # type: ignore[assignment]
    else:  # pragma: no cover
        audio_bytes = bytes(speech.read())

    mp3_path.write_bytes(audio_bytes)

    return {
        "mode": "real",
        "mp3_path": str(mp3_path),
        "duration_seconds": _duration_estimate(script_text),
        "voice": voice,
        "model": model,
        "sha256": _sha256_bytes(audio_bytes),
    }
