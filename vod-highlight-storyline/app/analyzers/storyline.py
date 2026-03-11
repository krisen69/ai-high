from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, ValidationError

from app.schemas import HighlightEvent, StorylineItem, StorylineOutput
from app.utils.timecode import sec_to_tc
from app.utils.logging import get_logger

logger = get_logger(__name__)


class LLMStorylineResponse(BaseModel):
    title: str
    one_line_summary: str
    items: list[dict[str, Any]]


def infer_language(events: list[HighlightEvent]) -> str:
    combined = " ".join(
        f"{event.transcript_excerpt} {' '.join(event.representative_chat)}" for event in events
    )
    korean_count = sum(1 for char in combined if "가" <= char <= "힣")
    return "ko" if korean_count > max(5, int(len(combined) * 0.02)) else "en"


def _heuristic_storyline(events: list[HighlightEvent], duration_sec: float) -> StorylineOutput:
    if not events:
        return StorylineOutput(
            title="No highlights",
            one_line_summary="No significant events detected.",
            language="en",
            items=[],
            arc={},
        )

    ordered = sorted(events, key=lambda event: event.start_sec)
    top = sorted(events, key=lambda event: event.score, reverse=True)
    setup = next((event for event in ordered if event.start_sec <= duration_sec * 0.25), ordered[0])
    climax = top[0]
    turning = next(
        (event for event in ordered if event.start_sec >= setup.start_sec and event.score >= climax.score * 0.7),
        climax,
    )
    buildup = [event for event in ordered if setup.start_sec < event.start_sec < turning.start_sec][:2]
    resolution_candidates = [event for event in ordered if event.start_sec > climax.start_sec]
    resolution = resolution_candidates[-1] if resolution_candidates else climax

    def as_item(role: str, event: HighlightEvent) -> StorylineItem:
        return StorylineItem(
            role=role,
            start_sec=event.start_sec,
            end_sec=event.end_sec,
            start_tc=event.start_tc,
            end_tc=event.end_tc,
            summary=event.suggested_summary or event.transcript_excerpt[:140] or event.label,
            evidence=[f"time={event.start_tc}-{event.end_tc}", *event.reasons[:2], event.transcript_excerpt[:120]],
            editor_note=f"Use this for {role.replace('_', ' ')} because score={event.score:.2f} and label={event.label}.",
        )

    items = [as_item("setup", setup)]
    items.extend(as_item("buildup", event) for event in buildup)
    items.append(as_item("turning_point", turning))
    items.append(as_item("climax", climax))
    if resolution.id != climax.id:
        items.append(as_item("resolution", resolution))

    language = infer_language(events)
    title = "스토리 하이라이트" if language == "ko" else "Highlight Storyline"
    one_line = (
        f"{sec_to_tc(setup.start_sec)} 시작, {sec_to_tc(climax.start_sec)} 절정."
        if language == "ko"
        else f"Starts at {sec_to_tc(setup.start_sec)} and peaks at {sec_to_tc(climax.start_sec)}."
    )

    return StorylineOutput(
        title=title,
        one_line_summary=one_line,
        language=language,
        items=items,
        arc={
            "setup": [setup.id],
            "buildup": [event.id for event in buildup],
            "turning_point": [turning.id],
            "climax": [climax.id],
            "resolution": [resolution.id] if resolution.id != climax.id else [],
        },
    )


def _llm_refine(base: StorylineOutput, events: list[HighlightEvent]) -> StorylineOutput:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return base

    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        events_payload = [
            {
                "id": event.id,
                "time": f"{event.start_tc}-{event.end_tc}",
                "score": event.score,
                "label": event.label,
                "summary": event.suggested_summary,
                "reasons": event.reasons,
                "transcript_excerpt": event.transcript_excerpt,
                "representative_chat": event.representative_chat,
            }
            for event in sorted(events, key=lambda item: item.rank)
        ]
        prompt = {
            "task": "Refine storyline while staying faithful to evidence.",
            "requirements": [
                "Return JSON only",
                "Use only provided events",
                "Each item must include role,start_tc,end_tc,summary,evidence,editor_note",
                "Roles allowed: setup,buildup,turning_point,climax,resolution,chronological",
            ],
            "events": events_payload,
            "base": base.model_dump(),
        }
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=json.dumps(prompt, ensure_ascii=False),
            response_format={"type": "json_object"},
        )
        payload = LLMStorylineResponse.model_validate_json(response.output_text)

        items: list[StorylineItem] = []
        for raw in payload.items:
            start_tc = str(raw.get("start_tc"))
            end_tc = str(raw.get("end_tc"))
            start_sec = next((event.start_sec for event in events if event.start_tc == start_tc), 0.0)
            end_sec = next((event.end_sec for event in events if event.end_tc == end_tc), 0.0)
            items.append(
                StorylineItem(
                    role=raw.get("role", "chronological"),
                    start_sec=start_sec,
                    end_sec=end_sec,
                    start_tc=start_tc,
                    end_tc=end_tc,
                    summary=str(raw.get("summary", "")),
                    evidence=list(raw.get("evidence", [])) or [f"time={start_tc}-{end_tc}"],
                    editor_note=str(raw.get("editor_note", "")),
                )
            )

        return StorylineOutput(
            title=payload.title,
            one_line_summary=payload.one_line_summary,
            language=base.language,
            items=items,
            arc=base.arc,
        )
    except (ValidationError, Exception) as exc:
        logger.warning("LLM storyline refinement failed, using heuristic fallback: %s", exc)
        return base


def generate_storyline(events: list[HighlightEvent], duration_sec: float) -> StorylineOutput:
    if len(events) < 2:
        ordered = sorted(events, key=lambda event: event.start_sec)
        simple = [
            StorylineItem(
                role="chronological",
                start_sec=event.start_sec,
                end_sec=event.end_sec,
                start_tc=event.start_tc,
                end_tc=event.end_tc,
                summary=event.suggested_summary,
                evidence=[f"time={event.start_tc}-{event.end_tc}", event.transcript_excerpt[:120]],
                editor_note="Not enough major events, using chronological fallback.",
            )
            for event in ordered
        ]
        base = StorylineOutput(
            title="Chronological Highlights",
            one_line_summary="Limited events detected; ordered timeline provided.",
            language=infer_language(events) if events else "en",
            items=simple,
            arc={"setup": [], "buildup": [], "turning_point": [], "climax": [], "resolution": []},
        )
        return _llm_refine(base, events)

    base_story = _heuristic_storyline(events, duration_sec)
    return _llm_refine(base_story, events)
