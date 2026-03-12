from __future__ import annotations

from pathlib import Path

from app.schemas import HighlightEvent
from app.utils.io import read_json
from app.utils.timecode import sec_to_tc


def load_highlights(job_dir: Path) -> list[HighlightEvent]:
    rows = read_json(job_dir / "highlights.json", [])
    return [HighlightEvent(**row) for row in rows]


def apply_feedback_to_highlights(
    highlights: list[HighlightEvent],
    feedback_rows: list[dict],
) -> list[HighlightEvent]:
    feedback_map = {row.get("id"): row for row in feedback_rows if row.get("id")}
    explicitly_accepted = [
        highlight for highlight in highlights if feedback_map.get(highlight.id, {}).get("accepted") is True
    ]
    selected = explicitly_accepted if explicitly_accepted else highlights

    reviewed: list[HighlightEvent] = []
    for highlight in selected:
        feedback = feedback_map.get(highlight.id, {})
        start_sec = float(feedback.get("start_sec", highlight.start_sec))
        end_sec = float(feedback.get("end_sec", highlight.end_sec))
        if end_sec < start_sec:
            end_sec = start_sec

        payload = highlight.model_dump()
        payload["start_sec"] = start_sec
        payload["end_sec"] = end_sec
        payload["start_tc"] = sec_to_tc(start_sec)
        payload["end_tc"] = sec_to_tc(end_sec)
        if "accepted" in feedback:
            payload["accepted"] = bool(feedback["accepted"])

        reviewed.append(HighlightEvent(**payload))

    reviewed.sort(key=lambda item: item.score, reverse=True)
    for rank, item in enumerate(reviewed, start=1):
        item.rank = rank
    return reviewed


def load_reviewed_highlights(job_dir: Path) -> list[HighlightEvent]:
    highlights = load_highlights(job_dir)
    feedback_rows = read_json(job_dir / "feedback.json", [])
    return apply_feedback_to_highlights(highlights, feedback_rows)
