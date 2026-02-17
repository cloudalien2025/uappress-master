from __future__ import annotations

import os
from typing import Optional

from apps.video.assemble import assemble_video


def _resolve_images_dir(image_result: dict) -> str:
    if image_result.get("images_dir"):
        return str(image_result["images_dir"])

    for key in ("image_paths", "images", "scene_images"):
        paths = image_result.get(key)
        if isinstance(paths, list) and paths:
            return os.path.dirname(str(paths[0])) or "."

    raise ValueError("Unable to resolve images_dir from image_result")


def build_video_asset(
    *,
    image_result: dict,
    audio_result: dict,
    subtitles_result: Optional[dict],
    scene_plan: dict,
    smoke: bool,
) -> dict:
    images_dir = _resolve_images_dir(image_result)

    mp3_path = audio_result.get("mp3_path") or audio_result.get("audio_path")
    if not mp3_path:
        raise ValueError("audio_result missing mp3_path")

    srt_path = None
    if subtitles_result:
        srt_path = subtitles_result.get("srt_path")

    video_result = assemble_video(
        images_dir=str(images_dir),
        audio_mp3_path=str(mp3_path),
        subtitles_srt_path=str(srt_path) if srt_path else None,
        smoke=smoke,
    )

    if scene_plan and isinstance(scene_plan, dict):
        scene_count = len(scene_plan.get("scenes", []) or [])
        video_result.setdefault("notes", []).append(f"scene_plan_scenes={scene_count}")

    return video_result
