from __future__ import annotations

from pathlib import Path

from app.schemas import TranscriptSegment


def transcribe_audio(audio_path: Path, out_json: Path, model_size: str = "small") -> list[TranscriptSegment]:
    try:
        from faster_whisper import WhisperModel  # type: ignore

        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(audio_path), vad_filter=True)
        result = [TranscriptSegment(start_sec=s.start, end_sec=s.end, text=s.text.strip()) for s in segments]
    except Exception:
        # fallback for environments without whisper model/deps
        result = [TranscriptSegment(start_sec=0.0, end_sec=5.0, text="")]
    return result


def transcript_excitement(text: str) -> float:
    t = text.lower()
    score = 0.0
    score += t.count("!") * 0.5
    score += t.count("?") * 0.3
    keywords = ["wow", "omg", "let's go", "대박", "미쳤", "와", "레전드"]
    score += sum(t.count(k) for k in keywords)
    if len(t.split()) <= 3 and t.strip():
        score += 0.5
    return score
