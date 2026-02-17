from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import urllib.error
import urllib.request
import zlib
from typing import Dict, List, Optional, Tuple


def _scene_prompt(scene: Dict, fallback_index: int) -> str:
    prompt = str(scene.get("visual_prompt") or "").strip()
    if prompt:
        return prompt
    labels = [
        str(scene.get("section") or "").strip(),
        str(scene.get("section_label") or "").strip(),
        str(scene.get("title") or "").strip(),
    ]
    label = " | ".join([x for x in labels if x])
    if not label:
        label = f"Scene {fallback_index}"
    return f"Documentary cinematic still frame: {label}"


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def _deterministic_png_bytes(width: int, height: int, seed: int) -> bytes:
    rows = []
    for y in range(height):
        row = bytearray()
        row.append(0)
        for x in range(width):
            r = (x * 13 + seed * 17) % 256
            g = (y * 7 + seed * 29) % 256
            b = ((x + y) * 5 + seed * 11) % 256
            if (x // 64 + y // 64 + seed) % 7 == 0:
                r, g, b = 240, 240, 240
            row.extend((r, g, b))
        rows.append(bytes(row))

    raw = b"".join(rows)
    compressed = zlib.compress(raw, level=9)
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    idat = _png_chunk(b"IDAT", compressed)
    iend = _png_chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


def _openai_image_bytes(prompt: str, model: str, size: Tuple[int, int], openai_key: str) -> bytes:
    width, height = size
    payload = {
        "model": model,
        "prompt": prompt,
        "size": f"{width}x{height}",
        "response_format": "b64_json",
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/images/generations",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"OpenAI image request failed ({exc.code}): {detail[:240]}") from exc

    parsed = json.loads(body.decode("utf-8"))
    data = parsed.get("data") or []
    if not data:
        raise RuntimeError("OpenAI image response missing data.")

    first = data[0]
    if first.get("b64_json"):
        return base64.b64decode(first["b64_json"])

    image_url = first.get("url")
    if image_url:
        with urllib.request.urlopen(image_url, timeout=90) as response:
            return response.read()

    raise RuntimeError("OpenAI image response did not include b64_json or url.")


def generate_scene_images(
    scenes: List[Dict],
    *,
    out_dir: str = "outputs/images",
    size: Tuple[int, int] = (1280, 720),
    openai_key: Optional[str] = None,
    model: str = "gpt-image-1",
    smoke: bool = False,
    max_images: int = 60,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    mode = "smoke" if smoke else "real"
    selected_scenes = list(scenes or [])[: max(0, int(max_images))]
    images = []

    for idx, scene in enumerate(selected_scenes, start=1):
        scene_id = int(scene.get("scene_id") or idx)
        prompt = _scene_prompt(scene, idx)
        filename = f"scene_{idx:04d}.png"
        path = os.path.join(out_dir, filename)

        if smoke:
            image_bytes = _deterministic_png_bytes(int(size[0]), int(size[1]), scene_id)
        else:
            if not openai_key:
                raise ValueError("OpenAI key is required for real image generation mode.")
            image_bytes = _openai_image_bytes(prompt, model, size, openai_key)

        with open(path, "wb") as f:
            f.write(image_bytes)

        images.append(
            {
                "scene_id": scene_id,
                "path": path,
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
                "prompt": prompt,
            }
        )

    return {
        "mode": mode,
        "out_dir": out_dir,
        "image_count": len(images),
        "images": images,
    }
