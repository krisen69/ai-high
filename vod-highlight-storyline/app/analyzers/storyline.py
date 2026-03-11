from __future__ import annotations

import os

from app.schemas import HighlightEvent, StorylineItem, StorylineOutput
from app.utils.timecode import sec_to_tc


def infer_language(events: list[HighlightEvent]) -> str:
    text = " ".join((e.transcript_excerpt + " " + " ".join(e.representative_chat)) for e in events)
    korean = sum(1 for ch in text if "가" <= ch <= "힣")
    return "ko" if korean > max(5, len(text) * 0.02) else "en"


def generate_storyline(events: list[HighlightEvent], duration_sec: float) -> StorylineOutput:
    top = sorted(events, key=lambda e: e.score, reverse=True)[:8]
    if not top:
        return StorylineOutput(title="No highlights", one_line_summary="No significant events detected.", language="en", items=[], arc={})

    lang = infer_language(top)
    # optional LLM
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI()
            prompt = "Create concise storyline JSON for events with timestamp citations."
            events_text = "\n".join(f"{e.start_tc}-{e.end_tc}: {e.suggested_summary}" for e in top)
            resp = client.responses.create(model="gpt-4.1-mini", input=f"{prompt}\n{events_text}")
            _ = resp.output_text
        except Exception:
            pass

    chronological = sorted(top, key=lambda e: e.start_sec)
    setup = next((e for e in chronological if e.start_sec <= duration_sec * 0.25), chronological[0])
    climax = max(top, key=lambda e: e.score)
    later = [e for e in chronological if e.start_sec > climax.start_sec]
    resolution = later[-1] if later else climax
    turning = next((e for e in chronological if e.start_sec > setup.start_sec and e.score >= climax.score * 0.7), climax)
    buildup = [e for e in chronological if setup.start_sec < e.start_sec < turning.start_sec][:2]

    def mk(role: str, e: HighlightEvent) -> StorylineItem:
        return StorylineItem(
            role=role,
            start_sec=e.start_sec,
            end_sec=e.end_sec,
            summary=e.suggested_summary,
            evidence=[f"time {e.start_tc}-{e.end_tc}", *e.reasons[:2]],
        )

    items = [mk("setup", setup)] + [mk("buildup", e) for e in buildup] + [mk("turning_point", turning), mk("climax", climax)]
    if resolution.id != climax.id:
        items.append(mk("resolution", resolution))

    title = "스토리 하이라이트" if lang == "ko" else "Highlight Storyline"
    summary = (
        f"{sec_to_tc(setup.start_sec)}에서 시작해 {sec_to_tc(climax.start_sec)}에서 절정에 도달합니다."
        if lang == "ko"
        else f"Builds from {sec_to_tc(setup.start_sec)} and peaks at {sec_to_tc(climax.start_sec)}."
    )
    arc = {"setup": [setup.id], "buildup": [e.id for e in buildup], "turning_point": [turning.id], "climax": [climax.id], "resolution": [resolution.id] if resolution.id != climax.id else []}
    return StorylineOutput(title=title, one_line_summary=summary, language=lang, items=items, arc=arc)
