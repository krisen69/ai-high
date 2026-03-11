from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.schemas import HighlightEvent
from app.utils.timecode import sec_to_tc


@dataclass
class DetectConfig:
    pre_roll_sec: float = 15.0
    post_roll_sec: float = 25.0
    peak_quantile: float = 0.9


def score_signal(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    s = np.zeros(len(df))
    for k, w in weights.items():
        if k in df.columns:
            s = s + df[k].to_numpy(dtype=float) * w
    out = df.copy()
    out["score"] = s
    return out


def _classify_label(row: pd.Series) -> str:
    if row.get("chat_volume_z", 0) > 1.2:
        return "chat-explosion"
    if row.get("laughter_token_rate", 0) > 0.3:
        return "funny"
    if row.get("audio_delta_z", 0) > 1.0:
        return "surprise"
    if row.get("audio_rms_z", 0) > 1.0:
        return "hype"
    return "mixed"


def detect_highlights(scored: pd.DataFrame, duration_sec: float, cfg: DetectConfig) -> list[HighlightEvent]:
    threshold = float(scored["score"].quantile(cfg.peak_quantile)) if len(scored) else 0.0
    peaks = scored[scored["score"] >= threshold]
    candidates: list[tuple[float, float, pd.Series]] = []
    for _, row in peaks.iterrows():
        center = (row.t_start + row.t_end) / 2
        start = max(0.0, center - cfg.pre_roll_sec)
        end = min(duration_sec, center + cfg.post_roll_sec)
        candidates.append((start, end, row))
    candidates.sort(key=lambda x: x[0])

    merged: list[tuple[float, float, list[pd.Series]]] = []
    for s, e, r in candidates:
        if not merged or s > merged[-1][1]:
            merged.append((s, e, [r]))
        else:
            ps, pe, rows = merged[-1]
            merged[-1] = (ps, max(pe, e), rows + [r])

    events: list[HighlightEvent] = []
    for i, (s, e, rows) in enumerate(merged, start=1):
        best = max(rows, key=lambda x: float(x.score))
        reasons = sorted(
            [
                ("chat spike", float(best.get("chat_volume_z", 0))),
                ("audio intensity", float(best.get("audio_rms_z", 0))),
                ("audio change", float(best.get("audio_delta_z", 0))),
                ("scene activity", float(best.get("scene_cut_density", 0))),
                ("transcript excitement", float(best.get("transcript_excitement_score", 0))),
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        reason_text = [f"{k}: {v:.2f}" for k, v in reasons]
        label = _classify_label(best)
        events.append(
            HighlightEvent(
                id=f"hl_{i:03d}",
                start_sec=s,
                end_sec=e,
                start_tc=sec_to_tc(s),
                end_tc=sec_to_tc(e),
                score=float(best.score),
                label=label,
                reasons=reason_text,
                suggested_title=f"{label.title()} moment around {sec_to_tc((s+e)/2)}",
                suggested_summary=f"Likely {label} moment triggered by {reason_text[0] if reason_text else 'signals'}",
            )
        )

    events.sort(key=lambda x: x.score, reverse=True)
    for idx, event in enumerate(events, start=1):
        event.rank = idx
    return events
