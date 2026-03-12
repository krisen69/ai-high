from __future__ import annotations

from pathlib import Path

from app.schemas import TranscriptSegment
from app.utils.logging import get_logger

logger = get_logger(__name__)


def transcribe_audio(audio_path: Path, model_size: str = "small") -> list[TranscriptSegment]:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:
        logger.warning("faster-whisper unavailable: %s. Continuing without transcript.", exc)
        return []

    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(audio_path), vad_filter=True)
        return [
            TranscriptSegment(start_sec=float(seg.start), end_sec=float(seg.end), text=seg.text.strip())
            for seg in segments
            if seg.text.strip()
        ]
    except Exception as exc:
        logger.warning("transcription failed: %s. Continuing with empty transcript.", exc)
        return []


def transcript_excitement(text: str) -> float:
    lowered = text.lower()
    score = 0.0
    score += lowered.count("!") * 0.5
    score += lowered.count("?") * 0.25
    keywords = ["wow", "omg", "let's go", "대박", "미쳤", "와", "레전드", "실화냐"]
    score += sum(lowered.count(word) for word in keywords)
    token_count = len(lowered.split())
    if 1 <= token_count <= 4:
        score += 0.3
    return score
