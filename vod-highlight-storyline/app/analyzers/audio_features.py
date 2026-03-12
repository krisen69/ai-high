from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def _chunk_rms(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))


def _chunk_peak(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return 0.0
    return float(np.max(np.abs(chunk)))


def compute_audio_features(audio_path: Path, bin_size_sec: float = 5.0) -> pd.DataFrame:
    with sf.SoundFile(audio_path) as f:
        sample_rate = f.samplerate
        channels = f.channels
        frames_per_bin = max(1, int(sample_rate * bin_size_sec))

        rows: list[dict[str, float]] = []
        previous_rms = 0.0
        bin_index = 0

        while True:
            frames = f.read(frames_per_bin, dtype="float32", always_2d=True)
            if frames.size == 0:
                break

            mono = np.mean(frames, axis=1) if channels > 1 else frames[:, 0]
            rms = _chunk_rms(mono)
            peak = _chunk_peak(mono)
            delta = abs(rms - previous_rms)
            silence_break = 1.0 if previous_rms < 0.01 and rms > 0.05 else 0.0
            speech_activity_ratio = float(np.mean(np.abs(mono) > 0.02))

            start = bin_index * bin_size_sec
            end = start + (len(mono) / sample_rate)
            rows.append(
                {
                    "t_start": float(start),
                    "t_end": float(end),
                    "audio_rms": rms,
                    "audio_peak": peak,
                    "audio_delta": float(delta),
                    "silence_break_score": silence_break,
                    "speech_activity_ratio": speech_activity_ratio,
                }
            )

            previous_rms = rms
            bin_index += 1

    if not rows:
        return pd.DataFrame(
            columns=[
                "t_start",
                "t_end",
                "audio_rms",
                "audio_peak",
                "audio_delta",
                "silence_break_score",
                "speech_activity_ratio",
            ]
        )

    return pd.DataFrame(rows)
