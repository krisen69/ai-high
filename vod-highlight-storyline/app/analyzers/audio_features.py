from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd


def compute_audio_features(audio_path: Path, bin_size_sec: float = 5.0) -> pd.DataFrame:
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    hop = int(sr * 0.5)
    frame = int(sr * 1.0)
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    peak = np.abs(y)
    bins = int(times.max() // bin_size_sec) + 1 if len(times) else 1
    rows = []
    for i in range(bins):
        s = i * bin_size_sec
        e = s + bin_size_sec
        idx = (times >= s) & (times < e)
        if idx.sum() == 0:
            rv = np.array([0.0])
        else:
            rv = rms[idx]
        rms_mean = float(rv.mean())
        rms_peak = float(rv.max())
        delta = float(np.abs(np.diff(rv)).mean()) if len(rv) > 1 else 0.0
        silence_break = float((rv[: max(1, len(rv)//2)].mean() < 0.02) and (rv[max(1, len(rv)//2):].max() > 0.06))
        speech_ratio = float((rv > 0.03).mean())
        rows.append({
            "t_start": s,
            "t_end": e,
            "audio_rms": rms_mean,
            "audio_peak": rms_peak,
            "audio_delta": delta,
            "silence_break_score": silence_break,
            "speech_activity_ratio": speech_ratio,
        })
    return pd.DataFrame(rows)
