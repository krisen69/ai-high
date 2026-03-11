from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.analyzers.chat_parser import compute_chat_features, parse_chat
from app.analyzers.feature_fusion import fuse_features
from app.analyzers.highlight_detection import DetectConfig, detect_highlights, score_signal
from app.config import AppConfig, load_preset
from app.schemas import ChatMessage
from app.utils.io import read_json, write_df, write_json


def run_score(job_dir: Path, preset_path: Path, cfg: AppConfig) -> list[dict]:
    metadata = read_json(job_dir / "metadata.json", {})
    duration_sec = float(metadata.get("format", {}).get("duration", 0.0))

    job_cfg = read_json(job_dir / "job_config.json", {})
    chat_path = Path(job_cfg.get("chat_path", "")) if job_cfg.get("chat_path") else None

    if chat_path and chat_path.exists():
        msgs = parse_chat(chat_path, chat_offset_seconds=cfg.chat_offset_seconds)
    else:
        msgs = [ChatMessage(**m) for m in read_json(job_dir / "chat_normalized.json", [])]
        for m in msgs:
            m.timestamp_sec += cfg.chat_offset_seconds

    chat_df = compute_chat_features(msgs, cfg.bin_size_sec, duration_sec)
    audio_df = pd.read_csv(job_dir / "audio_features.csv")
    transcript_segments = read_json(job_dir / "transcript.json", [])
    scene_cuts = read_json(job_dir / "scene_cuts.json", [])

    fused = fuse_features(chat_df, audio_df, transcript_segments, scene_cuts, cfg.bin_size_sec, cfg.smoothing_window_bins)
    weights = load_preset(preset_path)
    scored = score_signal(fused, weights)
    write_df(job_dir / "fused_features.csv", scored)

    events = detect_highlights(scored, duration_sec, DetectConfig(cfg.pre_roll_sec, cfg.post_roll_sec, cfg.peak_quantile))

    normalized_chat = [m.model_dump() for m in msgs]
    for event in events:
        in_range = [m for m in msgs if event.start_sec <= m.timestamp_sec <= event.end_sec]
        event.representative_chat = [f"{m.username}: {m.message}" for m in in_range[:5]]

    write_json(job_dir / "chat_normalized.json", normalized_chat)
    write_json(job_dir / "highlights.json", [e.model_dump() for e in events])
    return [e.model_dump() for e in events]
