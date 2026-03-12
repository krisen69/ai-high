from pathlib import Path

import pandas as pd

from app.analyzers.storyline import generate_storyline
from app.config import AppConfig
from app.pipeline.export import run_export
from app.pipeline.prepare import run_prepare
from app.pipeline.score import run_score
from app.schemas import TranscriptSegment
from app.utils.io import read_json, write_json
from app.utils.review import load_reviewed_highlights


def test_smoke_prepare_score_storyline(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "video.mp4"
    chat = tmp_path / "chat.txt"
    job = tmp_path / "job"

    video.write_bytes(b"fake")
    chat.write_text("[00:00:01] u1: wow\n[00:00:02] u2: ㅋㅋ\n", encoding="utf-8")

    monkeypatch.setattr("app.pipeline.prepare.ffprobe_metadata", lambda _: {"format": {"duration": "60"}})
    monkeypatch.setattr("app.pipeline.prepare.extract_audio", lambda _v, out: out.write_bytes(b"wav"))
    monkeypatch.setattr(
        "app.pipeline.prepare.compute_audio_features",
        lambda _a, bin_size_sec=5.0: pd.DataFrame(
            [
                {
                    "t_start": 0.0,
                    "t_end": 5.0,
                    "audio_rms": 0.1,
                    "audio_peak": 0.2,
                    "audio_delta": 0.1,
                    "silence_break_score": 0.0,
                    "speech_activity_ratio": 0.5,
                },
                {
                    "t_start": 5.0,
                    "t_end": 10.0,
                    "audio_rms": 0.6,
                    "audio_peak": 0.8,
                    "audio_delta": 0.5,
                    "silence_break_score": 1.0,
                    "speech_activity_ratio": 0.9,
                },
            ]
        ),
    )
    monkeypatch.setattr(
        "app.pipeline.prepare.transcribe_audio",
        lambda _a: [TranscriptSegment(start_sec=0, end_sec=7, text="hello wow")],
    )
    monkeypatch.setattr("app.pipeline.prepare.detect_scene_cuts", lambda _v: [2.0, 7.0])

    run_prepare(video=video, chat=chat, job_dir=job, bin_size_sec=5.0)
    first = run_score(job, Path("app/presets/game.yaml"), AppConfig(chat_offset_seconds=0.0))
    second = run_score(job, Path("app/presets/game.yaml"), AppConfig(chat_offset_seconds=3.0))

    assert (job / "metadata.json").exists()
    assert (job / "chat_source.txt").exists()
    assert (job / "chat_normalized_base.json").exists()
    assert first
    assert second

    feedback = [{"id": second[0]["id"], "accepted": True, "start_sec": 12.0, "end_sec": 24.0}]
    write_json(job / "feedback.json", feedback)
    reviewed = load_reviewed_highlights(job)
    assert reviewed[0].start_sec == 12.0
    assert reviewed[0].end_sec == 24.0

    storyline = generate_storyline(reviewed, 60)
    write_json(job / "storyline.json", storyline.model_dump())
    assert storyline.items
    assert storyline.items[0].start_tc

    run_export(job, extract_clips=False)
    reviewed_json = read_json(job / "reviewed_highlights.json", [])
    assert reviewed_json[0]["start_sec"] == 12.0
    assert (job / "markers.csv").exists()
