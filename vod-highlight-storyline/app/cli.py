from __future__ import annotations

import argparse
from pathlib import Path

from app.analyzers.storyline import generate_storyline
from app.config import AppConfig
from app.pipeline.export import run_export
from app.pipeline.prepare import run_prepare
from app.pipeline.score import run_score
from app.utils.io import read_json, write_json


def cmd_prepare(args: argparse.Namespace) -> None:
    out = Path(args.out)
    run_prepare(Path(args.video), Path(args.chat), out, chat_offset_seconds=args.chat_offset, bin_size_sec=args.bin_size)
    cfg = {
        "video_path": str(Path(args.video).resolve()),
        "chat_path": str(Path(args.chat).resolve()),
        "bin_size_sec": args.bin_size,
    }
    write_json(out / "job_config.json", cfg)
    # store raw chat copy for rescoring offsets
    (out / "chat_source.txt").write_text(Path(args.chat).read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")


def cmd_score(args: argparse.Namespace) -> None:
    cfg = AppConfig(chat_offset_seconds=args.chat_offset)
    run_score(Path(args.job), Path(args.preset), cfg)


def cmd_storyline(args: argparse.Namespace) -> None:
    job = Path(args.job)
    highlights = read_json(job / "highlights.json", [])
    duration = float(read_json(job / "metadata.json", {}).get("format", {}).get("duration", 0.0))
    from app.schemas import HighlightEvent

    events = [HighlightEvent(**h) for h in highlights]
    if (job / "feedback.json").exists():
        fb = {x["id"]: x for x in read_json(job / "feedback.json", [])}
        accepted = [e for e in events if fb.get(e.id, {}).get("accepted")]
        if accepted:
            events = accepted
    story = generate_storyline(events, duration)
    write_json(job / "storyline.json", story.model_dump())

    md = [f"# {story.title}", "", story.one_line_summary, ""]
    for item in story.items:
        md += [f"## {item.role}", f"- time: {item.start_sec:.1f}s - {item.end_sec:.1f}s", f"- summary: {item.summary}", "- evidence:"]
        md += [f"  - {e}" for e in item.evidence]
        md.append("")
    (job / "storyline.md").write_text("\n".join(md), encoding="utf-8")


def cmd_export(args: argparse.Namespace) -> None:
    run_export(Path(args.job), extract_clips=args.extract_clips)


def main() -> None:
    p = argparse.ArgumentParser(prog="vod-highlight-storyline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare")
    pp.add_argument("--video", required=True)
    pp.add_argument("--chat", required=True)
    pp.add_argument("--out", required=True)
    pp.add_argument("--chat-offset", type=float, default=0.0)
    pp.add_argument("--bin-size", type=float, default=5.0)
    pp.set_defaults(func=cmd_prepare)

    ps = sub.add_parser("score")
    ps.add_argument("--job", required=True)
    ps.add_argument("--preset", default="app/presets/game.yaml")
    ps.add_argument("--chat-offset", type=float, default=0.0)
    ps.set_defaults(func=cmd_score)

    pst = sub.add_parser("storyline")
    pst.add_argument("--job", required=True)
    pst.set_defaults(func=cmd_storyline)

    pe = sub.add_parser("export")
    pe.add_argument("--job", required=True)
    pe.add_argument("--extract-clips", action="store_true")
    pe.set_defaults(func=cmd_export)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
