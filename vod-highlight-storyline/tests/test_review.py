from app.schemas import HighlightEvent
from app.utils.review import apply_feedback_to_highlights


def test_apply_feedback_accept_and_retime() -> None:
    highlights = [
        HighlightEvent(
            id="h1",
            rank=1,
            score=1.0,
            start_sec=10,
            end_sec=20,
            start_tc="00:00:10.000",
            end_tc="00:00:20.000",
        ),
        HighlightEvent(
            id="h2",
            rank=2,
            score=0.5,
            start_sec=30,
            end_sec=40,
            start_tc="00:00:30.000",
            end_tc="00:00:40.000",
        ),
    ]
    feedback = [
        {"id": "h2", "accepted": True, "start_sec": 31.0, "end_sec": 42.0},
        {"id": "h1", "accepted": False},
    ]

    reviewed = apply_feedback_to_highlights(highlights, feedback)
    assert len(reviewed) == 1
    assert reviewed[0].id == "h2"
    assert reviewed[0].start_sec == 31.0
    assert reviewed[0].end_tc == "00:00:42.000"
