from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import pandas as pd

from app.utils.io import ensure_dir, read_json, write_json
from app.utils.timecode import sec_to_tc


def _write_srt(job_dir: Path) -> None:
    transcript = read_json(job_dir / "transcript.json", [])
    lines = []
    for i, seg in enumerate(transcript, start=1):
        start = sec_to_tc(float(seg["start_sec"]))
        end = sec_to_tc(float(seg["end_sec"]))
        lines += [str(i), f"{start.replace('.', ',')} --> {end.replace('.', ',')}", seg.get("text", ""), ""]
    (job_dir / "transcript.srt").write_text("\n".join(lines), encoding="utf-8")


def _extract_clip(video: Path, out_path: Path, start: float, end: float) -> None:
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", str(video), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", str(out_path)]
    subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def run_export(job_dir: Path, extract_clips: bool = False, top_n: int = 5) -> None:
    highlights = read_json(job_dir / "highlights.json", [])
    pd.DataFrame(highlights).to_csv(job_dir / "highlights.csv", index=False)

    markers = []
    for h in highlights:
        markers.append({
            "name": h.get("suggested_title", h["id"]),
            "start_tc": h["start_tc"],
            "end_tc": h["end_tc"],
            "score": h["score"],
            "label": h.get("label", "mixed"),
            "note": "; ".join(h.get("reasons", [])),
        })
    with (job_dir / "markers.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "start_tc", "end_tc", "score", "label", "note"])
        w.writeheader()
        w.writerows(markers)

    _write_srt(job_dir)

    if extract_clips:
        video_path = Path(read_json(job_dir / "job_config.json", {}).get("video_path", ""))
        clips_dir = ensure_dir(job_dir / "exports" / "clips")
        for h in sorted(highlights, key=lambda x: x.get("score", 0), reverse=True)[:top_n]:
            out = clips_dir / f"{h['id']}.mp4"
            _extract_clip(video_path, out, float(h["start_sec"]), float(h["end_sec"]))

    write_json(job_dir / "export_metadata.json", {"exported": True})
