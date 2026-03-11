from __future__ import annotations

import csv
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


from app.schemas import ChatMessage

RE_BRACKET = re.compile(r"^\[(?P<ts>[^\]]+)\]\s*(?P<user>[^:]+):\s*(?P<msg>.+)$")
RE_COLON = re.compile(r"^(?P<ts>\d{2}:\d{2}:\d{2})\s*(?P<user>[^:]+):\s*(?P<msg>.+)$")
RE_DATE = re.compile(r"^(?P<dt>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*(?P<rest>.+)$")

KO_EXCITE = ["ㅋㅋ", "ㅎㅎ", "와", "대박", "미쳤", "레전드", "실화냐"]
EN_EXCITE = ["lol", "lmao", "omg", "wow", "insane", "no way", "let's go"]
LAUGH_RE = re.compile(r"(k{2,}|h{2,}|ㅋ+|ㅎ+|😂|🤣|lol)", re.IGNORECASE)


def _hms_to_sec(text: str) -> float:
    h, m, s = text.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_chat(path: Path, chat_offset_seconds: float = 0.0) -> list[ChatMessage]:
    if path.suffix.lower() == ".csv":
        return _parse_csv(path, chat_offset_seconds)
    rows: list[ChatMessage] = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    abs_times: list[datetime] = []
    pending: list[tuple[datetime, str, str, str]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = RE_BRACKET.match(line) or RE_COLON.match(line)
        if m:
            ts = m.group("ts")
            rows.append(ChatMessage(timestamp_sec=_hms_to_sec(ts) + chat_offset_seconds, raw_timestamp=ts, username=m.group("user").strip(), message=m.group("msg").strip()))
            continue
        if "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 3 and re.match(r"\d{2}:\d{2}:\d{2}", parts[0]):
                ts, user, msg = parts[0], parts[1], "\t".join(parts[2:])
                rows.append(ChatMessage(timestamp_sec=_hms_to_sec(ts) + chat_offset_seconds, raw_timestamp=ts, username=user.strip(), message=msg.strip()))
                continue
        md = RE_DATE.match(line)
        if md:
            dt = datetime.strptime(md.group("dt"), "%Y-%m-%d %H:%M:%S")
            rest = md.group("rest")
            if ":" in rest:
                user, msg = rest.split(":", 1)
            else:
                user, msg = "unknown", rest
            abs_times.append(dt)
            pending.append((dt, md.group("dt"), user.strip(), msg.strip()))

    if pending:
        t0 = min(abs_times)
        for dt, raw, user, msg in pending:
            rows.append(ChatMessage(timestamp_sec=(dt - t0).total_seconds() + chat_offset_seconds, raw_timestamp=raw, username=user, message=msg))

    rows.sort(key=lambda x: x.timestamp_sec)
    return rows


def _parse_csv(path: Path, offset: float) -> list[ChatMessage]:
    import pandas as pd

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    ts_col = cols.get("timestamp") or cols.get("time")
    user_col = cols.get("user") or cols.get("username")
    msg_col = cols.get("message") or cols.get("msg") or cols.get("text")
    if not ts_col or not msg_col:
        raise ValueError("CSV must include timestamp/time and message columns")

    rows: list[ChatMessage] = []
    parsed_dt: list[datetime] = []
    raw_rows: list[tuple[str, str, str]] = []
    for _, row in df.iterrows():
        raw_ts = str(row[ts_col])
        user = str(row[user_col]) if user_col else "unknown"
        msg = str(row[msg_col])
        if re.match(r"\d{2}:\d{2}:\d{2}", raw_ts):
            sec = _hms_to_sec(raw_ts)
            rows.append(ChatMessage(timestamp_sec=sec + offset, raw_timestamp=raw_ts, username=user, message=msg))
        else:
            dt = datetime.fromisoformat(raw_ts.replace("Z", ""))
            parsed_dt.append(dt)
            raw_rows.append((raw_ts, user, msg))

    if parsed_dt:
        t0 = min(parsed_dt)
        for raw_ts, user, msg in raw_rows:
            dt = datetime.fromisoformat(raw_ts.replace("Z", ""))
            rows.append(ChatMessage(timestamp_sec=(dt - t0).total_seconds() + offset, raw_timestamp=raw_ts, username=user, message=msg))
    rows.sort(key=lambda x: x.timestamp_sec)
    return rows


def compute_chat_features(messages: list[ChatMessage], bin_size_sec: float, duration_sec: float):
    import pandas as pd
    bins = int(duration_sec // bin_size_sec) + 1
    data = []
    for i in range(bins):
        s = i * bin_size_sec
        e = s + bin_size_sec
        bucket = [m for m in messages if s <= m.timestamp_sec < e]
        msgs = [m.message for m in bucket]
        users = {m.username for m in bucket}
        cnt = len(bucket)
        dup_rate = 0.0
        if cnt:
            counts = Counter(msgs)
            dup_rate = sum(v for v in counts.values() if v > 1) / cnt
        text = " ".join(msgs).lower()
        excite = sum(text.count(t) for t in KO_EXCITE + EN_EXCITE)
        laugh = len(LAUGH_RE.findall(text))
        data.append({
            "t_start": s,
            "t_end": e,
            "message_count": cnt,
            "unique_users": len(users),
            "repeated_message_rate": dup_rate,
            "excitement_token_count": excite,
            "laughter_token_count": laugh,
        })
    return pd.DataFrame(data)
