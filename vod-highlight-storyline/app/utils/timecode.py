from __future__ import annotations


def sec_to_tc(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def tc_to_sec(tc: str) -> float:
    parts = tc.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timecode: {tc}")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s
