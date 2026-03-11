from __future__ import annotations

from pathlib import Path


def stage_done(job_dir: Path, stage: str) -> bool:
    return (job_dir / ".cache" / f"{stage}.done").exists()


def mark_stage_done(job_dir: Path, stage: str) -> None:
    p = job_dir / ".cache"
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{stage}.done").write_text("ok", encoding="utf-8")
