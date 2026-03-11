from __future__ import annotations

import argparse
from pathlib import Path

from app.analyzers.storyline import generate_storyline
from app.config import AppConfig
from app.pipeline.export import run_export
from app.pipeline.prepare import run_prepare
from app.pipeline.score import run_score
from app.schemas import HighlightEvent
from app.utils.io import read_json, write_json
from app.utils.timecode import sec_to_tc
from app.utils.logging import get_logger

logger = get_logger(__name__)


def cmd_prepare(args: argparse.Namespace) -> None:
    logger.info("CLI prepare started")
    run_prepare(
        video=Path(args.video),
        chat=Path(args.chat),
        job_dir=Path(args.out),
        chat_offset_seconds=args.chat_offset,
        bin_size_sec=args.bin_size,
    )


def cmd_score(args: argparse.Namespace) -> None:
    logger.info("CLI score started")
    cfg = AppConfig(chat_offset_seconds=args.chat_offset)
    run_score(Path(args.job), Path(args.preset), cfg)


def _apply_feedback(highlights: list[HighlightEvent], feedback_rows: list[dict]) -> list[HighlightEvent]:
    feedback = {row["id"]: row for row in feedback_rows}
    accepted = [event for event in highlights if feedback.get(event.id, {}).get("accepted") is True]
    selected = accepted if accepted else highlights

    updated: list[HighlightEvent] = []
    for event in selected:
        fb = feedback.get(event.id, {})
        if "start_sec" in fb:
            event.start_sec = float(fb["start_sec"])
        if "end_sec" in fb:
            event.end_sec = float(fb["end_sec"])
        event.start_tc = sec_to_tc(event.start_sec)
        event.end_tc = sec_to_tc(event.end_sec)
        event.accepted = fb.get("accepted")
        updated.append(event)
    return updated


def cmd_storyline(args: argparse.Namespace) -> None:
    logger.info("CLI storyline started")
    job = Path(args.job)
    highlights = [HighlightEvent(**row) for row in read_json(job / "highlights.json", [])]
    if not highlights:
        raise ValueError(f"No highlights found at {job / 'highlights.json'}. Run score first.")

    feedback_rows = read_json(job / "feedback.json", []) if (job / "feedback.json").exists() else []
    selected = _apply_feedback(highlights, feedback_rows)

    duration_sec = float(read_json(job / "metadata.json", {}).get("format", {}).get("duration", 0.0))
    storyline = generate_storyline(selected, duration_sec)
    write_json(job / "storyline.json", storyline.model_dump())

    lines = [f"# {storyline.title}", "", storyline.one_line_summary, ""]
    for item in storyline.items:
        lines.extend(
            [
                f"## {item.role}",
                f"- time: {item.start_tc} - {item.end_tc}",
                f"- summary: {item.summary}",
                "- evidence:",
                *[f"  - {evidence}" for evidence in item.evidence],
                f"- editor_note: {item.editor_note}",
                "",
            ]
        )
    (job / "storyline.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("storyline saved: %s", job / "storyline.md")


def cmd_export(args: argparse.Namespace) -> None:
    logger.info("CLI export started")
    run_export(Path(args.job), extract_clips=args.extract_clips)


def main() -> None:
    parser = argparse.ArgumentParser(prog="vod-highlight-storyline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--video", required=True)
    prepare.add_argument("--chat", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--chat-offset", type=float, default=0.0)
    prepare.add_argument("--bin-size", type=float, default=5.0)
    prepare.set_defaults(func=cmd_prepare)

    score = sub.add_parser("score")
    score.add_argument("--job", required=True)
    score.add_argument("--preset", default="app/presets/game.yaml")
    score.add_argument("--chat-offset", type=float, default=0.0)
    score.set_defaults(func=cmd_score)

    storyline = sub.add_parser("storyline")
    storyline.add_argument("--job", required=True)
    storyline.set_defaults(func=cmd_storyline)

    export = sub.add_parser("export")
    export.add_argument("--job", required=True)
    export.add_argument("--extract-clips", action="store_true")
    export.set_defaults(func=cmd_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
