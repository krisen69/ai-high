from app.analyzers.storyline import generate_storyline
from app.schemas import HighlightEvent


def test_storyline_fallback_roles():
    events = [
        HighlightEvent(id="a", start_sec=10, end_sec=20, start_tc="00:00:10.000", end_tc="00:00:20.000", score=0.4),
        HighlightEvent(id="b", start_sec=40, end_sec=50, start_tc="00:00:40.000", end_tc="00:00:50.000", score=0.9),
        HighlightEvent(id="c", start_sec=70, end_sec=80, start_tc="00:01:10.000", end_tc="00:01:20.000", score=0.6),
    ]
    story = generate_storyline(events, 100)
    roles = [i.role for i in story.items]
    assert "setup" in roles
    assert "climax" in roles
