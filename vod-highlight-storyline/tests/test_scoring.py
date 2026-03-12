import pandas as pd
import pytest

from app.analyzers.highlight_detection import DetectConfig, detect_highlights, score_signal
from app.config import AppConfig
from app.pipeline import score as score_pipeline
from app.schemas import ChatMessage
from app.utils.io import write_json


def test_scoring_and_merge() -> None:
    df = pd.DataFrame(
        {
            "t_start": [0, 5, 10, 15, 20],
            "t_end": [5, 10, 15, 20, 25],
            "chat_volume_z": [0, 1.5, 2.2, 0.2, 0],
            "unique_users_z": [0, 1.0, 1.2, 0.0, 0],
            "audio_rms_z": [0, 0.9, 1.3, 0.1, 0],
            "audio_delta_z": [0, 0.7, 1.1, 0.1, 0],
            "scene_cut_density": [0, 0.05, 0.10, 0.0, 0],
            "transcript_excitement_score": [0, 0.5, 1.2, 0.1, 0],
            "laughter_token_rate": [0, 0.1, 0.2, 0, 0],
        }
    )
    scored = score_signal(df, {"chat_volume_z": 0.6, "audio_rms_z": 0.4})
    transcript = [{"start_sec": 8, "end_sec": 14, "text": "wow great play"}]
    messages = [ChatMessage(timestamp_sec=11, raw_timestamp="00:00:11", username="u", message="omg")]

    events = detect_highlights(
        scored=scored,
        duration_sec=30,
        cfg=DetectConfig(pre_roll_sec=5, post_roll_sec=5, peak_quantile=0.7),
        transcript_segments=transcript,
        messages=messages,
    )
    assert events
    assert events[0].transcript_excerpt
    assert events[0].representative_chat


def test_bin_size_mismatch_raises(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    write_json(job / "job_config.json", {"bin_size_sec": 5.0})
    with pytest.raises(ValueError):
        score_pipeline._resolve_bin_size(job, AppConfig(bin_size_sec=4.0))
