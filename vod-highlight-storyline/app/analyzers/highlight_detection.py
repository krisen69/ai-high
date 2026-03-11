from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.schemas import ChatMessage, HighlightEvent
from app.utils.timecode import sec_to_tc


@dataclass
class DetectConfig:
    pre_roll_sec: float = 15.0
    post_roll_sec: float = 25.0
    peak_quantile: float = 0.9
    min_peak_distance_bins: int = 2


def score_signal(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    score = np.zeros(len(df), dtype=float)
    for feature, weight in weights.items():
        if feature in df.columns:
            score += df[feature].to_numpy(dtype=float) * float(weight)
    out = df.copy()
    out["score"] = score
    out["score_smoothed"] = out["score"].rolling(window=3, min_periods=1, center=True).mean()
    return out


def _pick_peaks(signal: np.ndarray, threshold: float, min_distance: int) -> list[int]:
    peaks: list[int] = []
    for idx in range(1, len(signal) - 1):
        is_peak = signal[idx] >= signal[idx - 1] and signal[idx] >= signal[idx + 1] and signal[idx] >= threshold
        if not is_peak:
            continue
        if peaks and idx - peaks[-1] < min_distance:
            if signal[idx] > signal[peaks[-1]]:
                peaks[-1] = idx
            continue
        peaks.append(idx)
    return peaks


def _classify_label(row: pd.Series) -> str:
    if row.get("chat_volume_z", 0.0) > 1.3:
        return "chat-explosion"
    if row.get("laughter_token_rate", 0.0) > 0.35:
        return "funny"
    if row.get("audio_delta_z", 0.0) > 1.2 and row.get("scene_cut_density", 0.0) > 0.05:
        return "surprise"
    if row.get("audio_rms_z", 0.0) > 1.0:
        return "hype"
    if row.get("transcript_excitement_score", 0.0) > 1.8:
        return "clutch"
    return "mixed"


def _top_reasons(row: pd.Series) -> list[str]:
    candidates = [
        ("Chat volume spike", float(row.get("chat_volume_z", 0.0))),
        ("Unique users increased", float(row.get("unique_users_z", 0.0))),
        ("Audio intensity", float(row.get("audio_rms_z", 0.0))),
        ("Sudden audio change", float(row.get("audio_delta_z", 0.0))),
        ("Scene cut density", float(row.get("scene_cut_density", 0.0))),
        ("Transcript excitement", float(row.get("transcript_excitement_score", 0.0))),
    ]
    ranked = sorted(candidates, key=lambda x: x[1], reverse=True)
    return [f"{name}: {value:.2f}" for name, value in ranked[:3]]


def _transcript_excerpt(transcript_segments: list[dict], start_sec: float, end_sec: float, max_chars: int = 260) -> str:
    texts = [
        seg.get("text", "")
        for seg in transcript_segments
        if float(seg.get("start_sec", 0.0)) < end_sec and float(seg.get("end_sec", 0.0)) > start_sec
    ]
    joined = " ".join(text.strip() for text in texts if text.strip())
    return joined[:max_chars].strip()


def _chat_excerpt(messages: list[ChatMessage], start_sec: float, end_sec: float, limit: int = 5) -> list[str]:
    selected = [m for m in messages if start_sec <= m.timestamp_sec <= end_sec]
    return [f"{m.username}: {m.message}" for m in selected[:limit]]


def detect_highlights(
    scored: pd.DataFrame,
    duration_sec: float,
    cfg: DetectConfig,
    transcript_segments: list[dict],
    messages: list[ChatMessage],
) -> list[HighlightEvent]:
    if scored.empty:
        return []

    signal = scored["score_smoothed"].to_numpy(dtype=float)
    threshold = float(np.quantile(signal, cfg.peak_quantile))
    peaks = _pick_peaks(signal, threshold, max(1, cfg.min_peak_distance_bins))

    if not peaks and len(signal):
        peaks = [int(np.argmax(signal))]

    windows: list[tuple[float, float, int]] = []
    for idx in peaks:
        row = scored.iloc[idx]
        center = float((row.t_start + row.t_end) / 2)
        start = max(0.0, center - cfg.pre_roll_sec)
        end = min(duration_sec, center + cfg.post_roll_sec)
        windows.append((start, end, idx))

    windows.sort(key=lambda item: item[0])
    merged: list[tuple[float, float, list[int]]] = []
    for start, end, idx in windows:
        if not merged or start > merged[-1][1]:
            merged.append((start, end, [idx]))
            continue
        prev_start, prev_end, indexes = merged[-1]
        merged[-1] = (prev_start, max(prev_end, end), indexes + [idx])

    events: list[HighlightEvent] = []
    for i, (start, end, idxs) in enumerate(merged, start=1):
        best_idx = max(idxs, key=lambda j: signal[j])
        row = scored.iloc[best_idx]
        reasons = _top_reasons(row)
        label = _classify_label(row)
        excerpt = _transcript_excerpt(transcript_segments, start, end)
        chats = _chat_excerpt(messages, start, end)
        center_tc = sec_to_tc((start + end) / 2)

        events.append(
            HighlightEvent(
                id=f"hl_{i:03d}",
                rank=0,
                score=float(row["score_smoothed"]),
                start_sec=float(start),
                end_sec=float(end),
                start_tc=sec_to_tc(start),
                end_tc=sec_to_tc(end),
                label=label,
                reasons=reasons,
                transcript_excerpt=excerpt,
                representative_chat=chats,
                suggested_title=f"{label.title()} @ {center_tc}",
                suggested_summary=reasons[0] if reasons else "Detected combined highlight signals",
            )
        )

    events.sort(key=lambda event: event.score, reverse=True)
    for rank, event in enumerate(events, start=1):
        event.rank = rank
    return events
