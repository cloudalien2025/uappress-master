from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import tempfile
from glob import glob
from typing import Optional


SMOKE_FILENAME = "uappress_final_smoke.mp4"
SMOKE_BYTES = b"UAPPRESS_SMOKE_MP4_PLACEHOLDER_V1\n"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _escape_filter_path(path: str) -> str:
    # ffmpeg filter parser escaping for subtitles path in a single filter string
    return path.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")


def assemble_video(
    *,
    images_dir: str,
    audio_mp3_path: str,
    subtitles_srt_path: Optional[str] = None,
    out_dir: str = "outputs/video",
    out_name: str = "uappress_final.mp4",
    fps: int = 30,
    resolution: tuple[int, int] = (1280, 720),
    smoke: bool = False,
    ffmpeg_path: str = "ffmpeg",
) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    if smoke:
        smoke_path = os.path.join(out_dir, SMOKE_FILENAME)
        with open(smoke_path, "wb") as f:
            f.write(SMOKE_BYTES)
        return {
            "mode": "smoke",
            "mp4_path": smoke_path,
            "sha256": hashlib.sha256(SMOKE_BYTES).hexdigest(),
            "fps": int(fps),
            "resolution": [int(resolution[0]), int(resolution[1])],
            "notes": [
                "Smoke mode placeholder artifact written.",
                "FFmpeg was intentionally skipped for CI-safe determinism.",
            ],
        }

    version_check = subprocess.run(
        [ffmpeg_path, "-version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if version_check.returncode != 0:
        stderr = (version_check.stderr or "").strip()
        raise RuntimeError(f"ffmpeg unavailable: {stderr or 'missing binary or execution failed'}")

    image_paths = sorted(glob(os.path.join(images_dir, "scene_*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No scene images found in {images_dir}")
    if not os.path.exists(audio_mp3_path):
        raise FileNotFoundError(f"Audio file not found: {audio_mp3_path}")

    width, height = int(resolution[0]), int(resolution[1])
    out_path = os.path.join(out_dir, out_name)
    notes = [f"Assembled from {len(image_paths)} image(s) at fixed 6s/image slideshow."]

    with tempfile.TemporaryDirectory(prefix="uappress_video_") as tdir:
        concat_path = os.path.join(tdir, "images.txt")
        with open(concat_path, "w", encoding="utf-8") as f:
            for path in image_paths:
                f.write(f"file {shlex.quote(os.path.abspath(path))}\n")
                f.write("duration 6\n")
            # concat demuxer expects the final file listed again
            f.write(f"file {shlex.quote(os.path.abspath(image_paths[-1]))}\n")

        vf = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        if subtitles_srt_path:
            if not os.path.exists(subtitles_srt_path):
                raise FileNotFoundError(f"Subtitle file not found: {subtitles_srt_path}")
            vf = f"{vf},subtitles='{_escape_filter_path(os.path.abspath(subtitles_srt_path))}'"
            notes.append("Burned subtitles into final video.")
        else:
            notes.append("No subtitle track provided; skipping subtitle burn-in.")

        cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_path,
            "-i",
            audio_mp3_path,
            "-vf",
            vf,
            "-r",
            str(int(fps)),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {(result.stderr or '').strip()}")

    return {
        "mode": "real",
        "mp4_path": out_path,
        "sha256": _sha256_file(out_path),
        "fps": int(fps),
        "resolution": [width, height],
        "notes": notes,
    }
