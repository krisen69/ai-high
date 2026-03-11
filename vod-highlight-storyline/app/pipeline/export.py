from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.utils.io import ensure_dir, read_json, write_json
from app.utils.timecode import sec_to_tc
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _write_srt(job_dir: Path) -> None:
    transcript = read_json(job_dir / "transcript.json", [])
    lines: list[str] = []
    for idx, segment in enumerate(transcript, start=1):
        start = sec_to_tc(float(segment["start_sec"]))
        end = sec_to_tc(float(segment["end_sec"]))
        lines.extend([str(idx), f"{start.replace('.', ',')} --> {end.replace('.', ',')}", segment.get("text", ""), ""])
    (job_dir / "transcript.srt").write_text("\n".join(lines), encoding="utf-8")


def _extract_clip(video_path: Path, output_path: Path, start_sec: float, end_sec: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_sec),
        "-to",
        str(end_sec),
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        str(output_path),
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def run_export(job_dir: Path, extract_clips: bool = False, top_n: int = 5) -> None:
    highlights = read_json(job_dir / "highlights.json", [])
    pd.DataFrame(highlights).to_csv(job_dir / "highlights.csv", index=False)

    markers_path = job_dir / "markers.csv"
    with markers_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "start_tc", "end_tc", "score", "label", "note"])
        writer.writeheader()
        for event in highlights:
            writer.writerow(
                {
                    "name": event.get("suggested_title", event["id"]),
                    "start_tc": event["start_tc"],
                    "end_tc": event["end_tc"],
                    "score": event["score"],
                    "label": event.get("label", "mixed"),
                    "note": " | ".join(event.get("reasons", [])),
                }
            )

    _write_srt(job_dir)

    if extract_clips:
        cfg = read_json(job_dir / "job_config.json", {})
        video_path = Path(cfg.get("video_path", ""))
        if video_path.exists():
            clips_dir = ensure_dir(job_dir / "exports" / "clips")
            ranked = sorted(highlights, key=lambda row: row.get("score", 0.0), reverse=True)[:top_n]
            for event in ranked:
                out_path = clips_dir / f"{event['id']}.mp4"
                _extract_clip(video_path, out_path, float(event["start_sec"]), float(event["end_sec"]))
        else:
            logger.warning("export clips skipped: video path missing %s", video_path)

    metadata = {
        "job_dir": str(job_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": read_json(job_dir / "job_config.json", {}),
    }
    write_json(job_dir / "export_metadata.json", metadata)
