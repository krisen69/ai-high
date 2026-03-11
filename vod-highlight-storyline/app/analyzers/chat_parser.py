from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

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
    absolute_rows: list[tuple[datetime, str, str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text:
            continue

        matched = RE_BRACKET.match(text) or RE_COLON.match(text)
        if matched:
            ts = matched.group("ts")
            rows.append(
                ChatMessage(
                    timestamp_sec=_hms_to_sec(ts) + chat_offset_seconds,
                    raw_timestamp=ts,
                    username=matched.group("user").strip(),
                    message=matched.group("msg").strip(),
                )
            )
            continue

        if "\t" in text:
            parts = text.split("\t")
            if len(parts) >= 3 and re.match(r"\d{2}:\d{2}:\d{2}", parts[0]):
                rows.append(
                    ChatMessage(
                        timestamp_sec=_hms_to_sec(parts[0]) + chat_offset_seconds,
                        raw_timestamp=parts[0],
                        username=parts[1].strip(),
                        message="\t".join(parts[2:]).strip(),
                    )
                )
                continue

        date_match = RE_DATE.match(text)
        if date_match:
            timestamp = datetime.strptime(date_match.group("dt"), "%Y-%m-%d %H:%M:%S")
            rest = date_match.group("rest")
            username, message = (rest.split(":", 1) + [""])[:2] if ":" in rest else ("unknown", rest)
            absolute_rows.append((timestamp, date_match.group("dt"), username.strip(), message.strip()))

    if absolute_rows:
        first_time = min(row[0] for row in absolute_rows)
        for timestamp, raw_ts, username, message in absolute_rows:
            rows.append(
                ChatMessage(
                    timestamp_sec=(timestamp - first_time).total_seconds() + chat_offset_seconds,
                    raw_timestamp=raw_ts,
                    username=username,
                    message=message,
                )
            )

    rows.sort(key=lambda row: row.timestamp_sec)
    return rows


def _parse_csv(path: Path, offset: float) -> list[ChatMessage]:
    df = pd.read_csv(path)
    columns = {col.lower(): col for col in df.columns}
    ts_col = columns.get("timestamp") or columns.get("time")
    user_col = columns.get("user") or columns.get("username")
    msg_col = columns.get("message") or columns.get("msg") or columns.get("text")
    if not ts_col or not msg_col:
        raise ValueError("CSV must include timestamp/time and message columns")

    rows: list[ChatMessage] = []
    absolute: list[tuple[datetime, str, str]] = []
    for _, row in df.iterrows():
        raw_ts = str(row[ts_col])
        username = str(row[user_col]) if user_col else "unknown"
        message = str(row[msg_col])
        if re.match(r"\d{2}:\d{2}:\d{2}", raw_ts):
            rows.append(
                ChatMessage(
                    timestamp_sec=_hms_to_sec(raw_ts) + offset,
                    raw_timestamp=raw_ts,
                    username=username,
                    message=message,
                )
            )
        else:
            absolute.append((datetime.fromisoformat(raw_ts.replace("Z", "")), username, message))

    if absolute:
        first_time = min(row[0] for row in absolute)
        for timestamp, username, message in absolute:
            rows.append(
                ChatMessage(
                    timestamp_sec=(timestamp - first_time).total_seconds() + offset,
                    raw_timestamp=timestamp.isoformat(),
                    username=username,
                    message=message,
                )
            )

    rows.sort(key=lambda row: row.timestamp_sec)
    return rows


def compute_chat_features(messages: list[ChatMessage], bin_size_sec: float, duration_sec: float) -> pd.DataFrame:
    bins = int(duration_sec // bin_size_sec) + 1
    records: list[dict[str, float]] = []

    for idx in range(bins):
        start = idx * bin_size_sec
        end = start + bin_size_sec
        bucket = [msg for msg in messages if start <= msg.timestamp_sec < end]

        all_messages = [msg.message for msg in bucket]
        counts = Counter(all_messages)
        count = len(all_messages)
        repeated_rate = sum(value for value in counts.values() if value > 1) / count if count else 0.0
        users = len({msg.username for msg in bucket})

        text = " ".join(all_messages).lower()
        excitement_count = sum(text.count(token) for token in KO_EXCITE + EN_EXCITE)
        laughter_count = len(LAUGH_RE.findall(text))

        records.append(
            {
                "t_start": float(start),
                "t_end": float(end),
                "message_count": float(count),
                "unique_users": float(users),
                "repeated_message_rate": float(repeated_rate),
                "excitement_token_count": float(excitement_count),
                "laughter_token_count": float(laughter_count),
            }
        )

    return pd.DataFrame(records)
