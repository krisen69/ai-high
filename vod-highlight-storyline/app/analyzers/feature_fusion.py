from __future__ import annotations

import numpy as np
import pandas as pd

from app.analyzers.transcript import transcript_excitement


def _z(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series([0.0] * len(series))
    return (series - series.mean()) / std


def fuse_features(
    chat_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    transcript_segments: list[dict],
    scene_cuts: list[float],
    bin_size: float,
    smoothing_window: int = 3,
) -> pd.DataFrame:
    df = chat_df.merge(audio_df, on=["t_start", "t_end"], how="outer").fillna(0)
    df = df.sort_values("t_start").reset_index(drop=True)

    # scene density
    dens = []
    for _, row in df.iterrows():
        cnt = sum(1 for c in scene_cuts if row.t_start <= c < row.t_end)
        dens.append(cnt / max(bin_size, 1.0))
    df["scene_cut_density"] = dens

    tx_scores = []
    for _, row in df.iterrows():
        joined = " ".join(
            seg["text"]
            for seg in transcript_segments
            if seg["start_sec"] < row.t_end and seg["end_sec"] >= row.t_start
        )
        tx_scores.append(transcript_excitement(joined))
    df["transcript_excitement_score"] = tx_scores

    df["chat_volume_z"] = _z(df["message_count"])
    df["unique_users_z"] = _z(df["unique_users"])
    df["audio_rms_z"] = _z(df["audio_rms"])
    df["audio_delta_z"] = _z(df["audio_delta"])

    df["excitement_token_rate"] = df["excitement_token_count"] / (df["message_count"].replace(0, 1))
    df["laughter_token_rate"] = df["laughter_token_count"] / (df["message_count"].replace(0, 1))
    smooth_cols = [
        "chat_volume_z",
        "unique_users_z",
        "audio_rms_z",
        "audio_delta_z",
        "scene_cut_density",
        "transcript_excitement_score",
        "excitement_token_rate",
        "laughter_token_rate",
    ]
    for c in smooth_cols:
        df[c] = df[c].rolling(window=smoothing_window, min_periods=1, center=True).mean()
    return df
