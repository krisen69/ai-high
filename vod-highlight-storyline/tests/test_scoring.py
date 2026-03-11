import pytest

pd = pytest.importorskip("pandas")

from app.analyzers.highlight_detection import DetectConfig, detect_highlights, score_signal


def test_scoring_and_merge():
    df = pd.DataFrame(
        {
            "t_start": [0, 5, 10, 15],
            "t_end": [5, 10, 15, 20],
            "chat_volume_z": [0, 2, 2.5, 0],
            "audio_rms_z": [0, 1, 1.2, 0],
        }
    )
    scored = score_signal(df, {"chat_volume_z": 0.8, "audio_rms_z": 0.2})
    events = detect_highlights(scored, 30, DetectConfig(pre_roll_sec=5, post_roll_sec=5, peak_quantile=0.75))
    assert len(events) == 1
    assert events[0].start_sec <= 5
