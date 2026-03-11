from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    timestamp_sec: float
    raw_timestamp: str
    username: str = "unknown"
    message: str


class TranscriptSegment(BaseModel):
    start_sec: float
    end_sec: float
    text: str


class FeatureBin(BaseModel):
    t_start: float
    t_end: float
    values: dict[str, float] = Field(default_factory=dict)


class HighlightEvent(BaseModel):
    id: str
    start_sec: float
    end_sec: float
    start_tc: str
    end_tc: str
    score: float
    rank: int = 0
    label: Literal["funny", "hype", "surprise", "clutch", "emotional", "chat-explosion", "mixed"] = "mixed"
    reasons: list[str] = Field(default_factory=list)
    transcript_excerpt: str = ""
    representative_chat: list[str] = Field(default_factory=list)
    suggested_title: str = ""
    suggested_summary: str = ""
    accepted: bool | None = None


class StorylineItem(BaseModel):
    role: Literal["setup", "buildup", "turning_point", "climax", "resolution", "chronological"]
    start_sec: float
    end_sec: float
    summary: str
    evidence: list[str]


class StorylineOutput(BaseModel):
    title: str
    one_line_summary: str
    language: str
    items: list[StorylineItem]
    arc: dict[str, list[str]]


class ExportMetadata(BaseModel):
    job_dir: str
    created_at: str
    config: dict[str, Any]
