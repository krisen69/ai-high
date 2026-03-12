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


def _resolve_bin_size(job_dir: Path, cfg: AppConfig) -> float:
    job_cfg = read_json(job_dir / "job_config.json", {})
    prepared_bin = float(job_cfg.get("bin_size_sec", 5.0))
    if cfg.bin_size_sec is None:
        return prepared_bin
    if abs(cfg.bin_size_sec - prepared_bin) > 1e-6:
        raise ValueError(
            f"Bin size mismatch: prepare used {prepared_bin}, score requested {cfg.bin_size_sec}. "
            "Use matching bin size or rerun prepare."
        )
    return cfg.bin_size_sec


def _apply_offset_once(
    cached_rows: list[dict],
    stored_offset_seconds: float,
    target_offset_seconds: float,
) -> list[ChatMessage]:
    delta = target_offset_seconds - stored_offset_seconds
    messages = [ChatMessage(**row) for row in cached_rows]
    return [
        ChatMessage(
            timestamp_sec=message.timestamp_sec + delta,
            raw_timestamp=message.raw_timestamp,
            username=message.username,
            message=message.message,
        )
        for message in messages
    ]


def _load_chat_messages(job_dir: Path, cfg: AppConfig) -> list[ChatMessage]:
    job_cfg = read_json(job_dir / "job_config.json", {})
    candidates: list[Path] = []

    if job_cfg.get("chat_path"):
        candidates.append(Path(job_cfg["chat_path"]))

    chat_filename = job_cfg.get("chat_filename")
    if chat_filename:
        suffix = Path(chat_filename).suffix.lower() or ".txt"
        candidates.append(job_dir / f"chat_source{suffix}")

    candidates.extend([job_dir / "chat_source.txt", job_dir / "chat_source.csv"])

    for path in candidates:
        if path.exists():
            logger.info("score: using chat source %s", path)
            return parse_chat(path, chat_offset_seconds=cfg.chat_offset_seconds)

    base_rows = read_json(job_dir / "chat_normalized_base.json", None)
    if base_rows is not None:
        logger.warning("score: raw chat unavailable, using chat_normalized_base.json fallback")
        return _apply_offset_once(base_rows, stored_offset_seconds=0.0, target_offset_seconds=cfg.chat_offset_seconds)

    chat_state = read_json(job_dir / "chat_state.json", {})
    stored_offset = float(chat_state.get("last_chat_offset_seconds", 0.0))
    normalized_rows = read_json(job_dir / "chat_normalized.json", [])
    logger.warning(
        "score: base chat missing, using chat_normalized.json with stored offset=%s",
        stored_offset,
    )
    return _apply_offset_once(
        normalized_rows,
        stored_offset_seconds=stored_offset,
        target_offset_seconds=cfg.chat_offset_seconds,
    )


def _validate_audio_bin_alignment(audio_df: pd.DataFrame, bin_size_sec: float) -> None:
    if audio_df.empty:
        return

    first = audio_df.iloc[0]
    observed = float(first["t_end"] - first["t_start"])
    if abs(observed - bin_size_sec) > 0.2:
        raise ValueError(
            f"Audio feature bin mismatch: expected ~{bin_size_sec}s, observed {observed:.3f}s. "
            "Rerun prepare or use matching bin size."
        )


def run_score(job_dir: Path, preset_path: Path, cfg: AppConfig) -> list[dict]:
    metadata = read_json(job_dir / "metadata.json", {})
    duration_sec = float(metadata.get("format", {}).get("duration", 0.0))
    if duration_sec <= 0:
        raise ValueError(f"Invalid or missing metadata duration in {job_dir / 'metadata.json'}")

    bin_size_sec = _resolve_bin_size(job_dir, cfg)
    messages = _load_chat_messages(job_dir, cfg)
    chat_df = compute_chat_features(messages, bin_size_sec, duration_sec)

    audio_df = pd.read_csv(job_dir / "audio_features.csv")
    _validate_audio_bin_alignment(audio_df, bin_size_sec)

    transcript_segments = read_json(job_dir / "transcript.json", [])
    scene_cuts = read_json(job_dir / "scene_cuts.json", [])

    fused = fuse_features(
        chat_df=chat_df,
        audio_df=audio_df,
        transcript_segments=transcript_segments,
        scene_cuts=scene_cuts,
        bin_size=bin_size_sec,
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
    write_json(job_dir / "chat_state.json", {"last_chat_offset_seconds": cfg.chat_offset_seconds})
    write_json(job_dir / "highlights.json", [e.model_dump() for e in events])
    logger.info("score: generated %d highlights", len(events))
    return [e.model_dump() for e in events]
