from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from app.analyzers.audio_features import compute_audio_features
from app.analyzers.chat_parser import parse_chat
from app.analyzers.scene_detection import detect_scene_cuts
from app.analyzers.transcript import transcribe_audio
from app.utils.cache import mark_stage_done, stage_done
from app.utils.io import ensure_dir, write_df, write_json
from app.utils.logging import get_logger

logger = get_logger(__name__)


def ffprobe_metadata(video: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video),
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def extract_audio(video: Path, audio_path: Path) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(video), "-ac", "1", "-ar", "16000", "-vn", str(audio_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def run_prepare(
    video: Path,
    chat: Path,
    job_dir: Path,
    chat_offset_seconds: float = 0.0,
    bin_size_sec: float = 5.0,
) -> None:
    ensure_dir(job_dir)
    started = time.time()

    job_config = {
        "video_path": str(video.resolve()),
        "chat_path": str(chat.resolve()),
        "chat_filename": chat.name,
        "bin_size_sec": bin_size_sec,
    }
    write_json(job_dir / "job_config.json", job_config)

    raw_chat_copy = job_dir / f"chat_source{chat.suffix.lower() or '.txt'}"
    shutil.copy2(chat, raw_chat_copy)

    if not stage_done(job_dir, "metadata"):
        logger.info("prepare: probing metadata")
        write_json(job_dir / "metadata.json", ffprobe_metadata(video))
        mark_stage_done(job_dir, "metadata")

    audio_path = job_dir / "audio.wav"
    if not stage_done(job_dir, "audio"):
        logger.info("prepare: extracting audio")
        extract_audio(video, audio_path)
        mark_stage_done(job_dir, "audio")

    if not stage_done(job_dir, "transcript"):
        logger.info("prepare: transcribing")
        segments = transcribe_audio(audio_path)
        write_json(job_dir / "transcript.json", [s.model_dump() for s in segments])
        mark_stage_done(job_dir, "transcript")

    if not stage_done(job_dir, "scene"):
        logger.info("prepare: scene detection")
        scene_cuts = detect_scene_cuts(video)
        write_json(job_dir / "scene_cuts.json", scene_cuts)
        mark_stage_done(job_dir, "scene")

    if not stage_done(job_dir, "audio_features"):
        logger.info("prepare: audio feature extraction")
        audio_df = compute_audio_features(audio_path, bin_size_sec=bin_size_sec)
        write_df(job_dir / "audio_features.csv", audio_df)
        mark_stage_done(job_dir, "audio_features")

    messages = parse_chat(chat, chat_offset_seconds=chat_offset_seconds)
    write_json(job_dir / "chat_normalized.json", [m.model_dump() for m in messages])
    logger.info("prepare finished in %.2fs", time.time() - started)
