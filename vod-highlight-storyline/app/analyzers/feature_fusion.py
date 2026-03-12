from __future__ import annotations

import numpy as np
import pandas as pd

from app.analyzers.transcript import transcript_excitement


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)))
    return (series - series.mean()) / std


def fuse_features(
    chat_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    transcript_segments: list[dict],
    scene_cuts: list[float],
    bin_size: float,
    smoothing_window: int = 3,
) -> pd.DataFrame:
    merged = chat_df.merge(audio_df, on=["t_start", "t_end"], how="outer").fillna(0.0)
    merged = merged.sort_values("t_start").reset_index(drop=True)

    densities: list[float] = []
    transcript_scores: list[float] = []
    for _, row in merged.iterrows():
        cut_count = sum(1 for cut in scene_cuts if row.t_start <= cut < row.t_end)
        densities.append(cut_count / max(bin_size, 1.0))

        texts = [
            seg.get("text", "")
            for seg in transcript_segments
            if float(seg.get("start_sec", 0.0)) < float(row.t_end)
            and float(seg.get("end_sec", 0.0)) > float(row.t_start)
        ]
        transcript_scores.append(transcript_excitement(" ".join(texts)))

    merged["scene_cut_density"] = densities
    merged["transcript_excitement_score"] = transcript_scores

    merged["chat_volume_z"] = _zscore(merged["message_count"])
    merged["unique_users_z"] = _zscore(merged["unique_users"])
    merged["audio_rms_z"] = _zscore(merged["audio_rms"])
    merged["audio_delta_z"] = _zscore(merged["audio_delta"])

    denominator = merged["message_count"].replace(0, 1)
    merged["excitement_token_rate"] = merged["excitement_token_count"] / denominator
    merged["laughter_token_rate"] = merged["laughter_token_count"] / denominator

    smooth_cols = [
        "chat_volume_z",
        "unique_users_z",
        "audio_rms_z",
        "audio_delta_z",
        "scene_cut_density",
        "transcript_excitement_score",
        "excitement_token_rate",
        "laughter_token_rate",
        "silence_break_score",
    ]
    for col in smooth_cols:
        if col in merged.columns:
            merged[col] = merged[col].rolling(window=smoothing_window, min_periods=1, center=True).mean()

    return merged
