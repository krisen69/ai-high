from __future__ import annotations

import argparse
from pathlib import Path

from app.analyzers.storyline import generate_storyline
from app.config import AppConfig
from app.pipeline.export import run_export
from app.pipeline.prepare import run_prepare
from app.pipeline.score import run_score
from app.utils.io import read_json, write_json
from app.utils.logging import get_logger
from app.utils.review import load_reviewed_highlights

logger = get_logger(__name__)


def _write_storyline_files(job_dir: Path, storyline: dict) -> None:
    write_json(job_dir / "storyline.json", storyline)

    lines = [f"# {storyline['title']}", "", storyline["one_line_summary"], ""]
    for item in storyline["items"]:
        lines.extend(
            [
                f"## {item['role']}",
                f"- time: {item['start_tc']} - {item['end_tc']}",
                f"- summary: {item['summary']}",
                "- evidence:",
                *[f"  - {evidence}" for evidence in item["evidence"]],
                f"- editor_note: {item['editor_note']}",
                "",
            ]
        )
    (job_dir / "storyline.md").write_text("\n".join(lines), encoding="utf-8")


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
    config = AppConfig(chat_offset_seconds=args.chat_offset, bin_size_sec=args.bin_size)
    run_score(Path(args.job), Path(args.preset), config)


def cmd_storyline(args: argparse.Namespace) -> None:
    logger.info("CLI storyline started")
    job_dir = Path(args.job)
    reviewed = load_reviewed_highlights(job_dir)
    if not reviewed:
        raise ValueError(f"No highlights found at {job_dir / 'highlights.json'}. Run score first.")

    duration_sec = float(read_json(job_dir / "metadata.json", {}).get("format", {}).get("duration", 0.0))
    storyline = generate_storyline(reviewed, duration_sec)
    _write_storyline_files(job_dir, storyline.model_dump())
    logger.info("storyline saved: %s", job_dir / "storyline.md")


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
    score.add_argument("--bin-size", type=float, default=None)
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
