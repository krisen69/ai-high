from pathlib import Path

from app.analyzers.chat_parser import compute_chat_features, parse_chat


def test_parse_multiple_formats(tmp_path: Path) -> None:
    chat = tmp_path / "chat.txt"
    chat.write_text("[00:00:01] u1: hi\n00:00:02\tu2\tlol\n", encoding="utf-8")
    rows = parse_chat(chat)
    assert len(rows) == 2
    assert rows[1].username == "u2"


def test_parse_absolute_datetime(tmp_path: Path) -> None:
    chat = tmp_path / "chat.txt"
    chat.write_text("2024-01-01 00:00:10 a: hey\n2024-01-01 00:00:13 b: wow\n", encoding="utf-8")
    rows = parse_chat(chat)
    assert rows[0].timestamp_sec == 0
    assert rows[1].timestamp_sec == 3


def test_chat_features(tmp_path: Path) -> None:
    chat = tmp_path / "chat.txt"
    chat.write_text("[00:00:01] u1: ㅋㅋ\n[00:00:02] u2: lol\n[00:00:03] u2: lol\n", encoding="utf-8")
    rows = parse_chat(chat)
    df = compute_chat_features(rows, 5, 10)
    assert int(df.iloc[0]["message_count"]) == 3
    assert int(df.iloc[0]["unique_users"]) == 2
