from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.analyzers.chat_parser import compute_chat_features, parse_chat
from app.analyzers.feature_fusion import fuse_features
from app.analyzers.highlight_detection import DetectConfig, detect_highlights, score_signal
from app.config import AppConfig, load_preset
from app.schemas import ChatMessage
from app.utils.io import read_json, write_df, write_json
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _load_chat_messages(job_dir: Path, cfg: AppConfig) -> list[ChatMessage]:
    job_cfg = read_json(job_dir / "job_config.json", {})
    candidates: list[Path] = []

    if job_cfg.get("chat_path"):
        candidates.append(Path(job_cfg["chat_path"]))

    chat_filename = job_cfg.get("chat_filename")
    if chat_filename:
        candidates.append(job_dir / f"chat_source{Path(chat_filename).suffix.lower() or '.txt'}")
    candidates.extend([job_dir / "chat_source.txt", job_dir / "chat_source.csv"])

    for path in candidates:
        if path.exists():
            logger.info("score: using chat source %s", path)
            return parse_chat(path, chat_offset_seconds=cfg.chat_offset_seconds)

    logger.warning("score: no original/raw chat source found, using cached normalized chat")
    base = [ChatMessage(**row) for row in read_json(job_dir / "chat_normalized.json", [])]
    return [
        ChatMessage(
            timestamp_sec=float(row.timestamp_sec) + cfg.chat_offset_seconds,
            raw_timestamp=row.raw_timestamp,
            username=row.username,
            message=row.message,
        )
        for row in base
    ]


def run_score(job_dir: Path, preset_path: Path, cfg: AppConfig) -> list[dict]:
    metadata = read_json(job_dir / "metadata.json", {})
    duration_sec = float(metadata.get("format", {}).get("duration", 0.0))
    if duration_sec <= 0:
        raise ValueError(f"Invalid or missing metadata duration in {job_dir / 'metadata.json'}")

    messages = _load_chat_messages(job_dir, cfg)
    chat_df = compute_chat_features(messages, cfg.bin_size_sec, duration_sec)

    audio_df = pd.read_csv(job_dir / "audio_features.csv")
    transcript_segments = read_json(job_dir / "transcript.json", [])
    scene_cuts = read_json(job_dir / "scene_cuts.json", [])

    fused = fuse_features(
        chat_df=chat_df,
        audio_df=audio_df,
        transcript_segments=transcript_segments,
        scene_cuts=scene_cuts,
        bin_size=cfg.bin_size_sec,
        smoothing_window=cfg.smoothing_window_bins,
    )
    weights = load_preset(preset_path)
    scored = score_signal(fused, weights)
    write_df(job_dir / "fused_features.csv", scored)

    detect_cfg = DetectConfig(
        pre_roll_sec=cfg.pre_roll_sec,
        post_roll_sec=cfg.post_roll_sec,
        peak_quantile=cfg.peak_quantile,
    )
    events = detect_highlights(scored, duration_sec, detect_cfg, transcript_segments, messages)

    write_json(job_dir / "chat_normalized.json", [m.model_dump() for m in messages])
    write_json(job_dir / "highlights.json", [e.model_dump() for e in events])
    logger.info("score: generated %d highlights", len(events))
    return [e.model_dump() for e in events]
