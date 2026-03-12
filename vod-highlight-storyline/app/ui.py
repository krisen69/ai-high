from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.analyzers.storyline import generate_storyline
from app.config import AppConfig
from app.pipeline.export import run_export
from app.pipeline.prepare import run_prepare
from app.pipeline.score import run_score
from app.utils.io import read_json, write_json
from app.utils.review import load_highlights, load_reviewed_highlights

st.set_page_config(page_title="vod-highlight-storyline", layout="wide")
st.title("vod-highlight-storyline")

job_dir = Path(st.text_input("Job directory", "./jobs/ui_job"))
video_path = Path(st.text_input("Video path", ""))
chat_path = Path(st.text_input("Chat path", ""))
preset = st.selectbox("Preset", ["game", "talk", "variety"], index=0)
chat_offset = st.number_input("chat_offset_seconds", value=0.0, step=0.5)

col1, col2, col3 = st.columns(3)
if col1.button("Run prepare"):
    run_prepare(video_path, chat_path, job_dir, chat_offset_seconds=float(chat_offset))
    st.success("prepare complete")

if col2.button("Run score"):
    events = run_score(
        job_dir,
        Path(f"app/presets/{preset}.yaml"),
        AppConfig(chat_offset_seconds=float(chat_offset)),
    )
    st.success(f"score complete ({len(events)} highlights)")

if col3.button("Export outputs"):
    run_export(job_dir, extract_clips=False)
    st.success("export complete")

fused_path = job_dir / "fused_features.csv"
if fused_path.exists():
    fused_df = pd.read_csv(fused_path)
    if "score_smoothed" in fused_df:
        st.subheader("Score-over-time")
        st.line_chart(fused_df.set_index("t_start")["score_smoothed"])

highlights_raw = [item.model_dump() for item in load_highlights(job_dir)]
if highlights_raw:
    highlights_df = pd.DataFrame(highlights_raw)
    st.subheader("Highlights")
    st.dataframe(
        highlights_df[
            [
                "id",
                "rank",
                "start_tc",
                "end_tc",
                "score",
                "label",
                "reasons",
                "transcript_excerpt",
                "representative_chat",
            ]
        ]
    )

    st.subheader("Review")
    existing_feedback = {row["id"]: row for row in read_json(job_dir / "feedback.json", [])}
    feedback_rows: list[dict] = []

    for row in highlights_raw:
        st.markdown(f"**{row['id']}** `{row['start_tc']} - {row['end_tc']}`")
        st.caption(f"Reasons: {' | '.join(row.get('reasons', []))}")
        st.caption(f"Transcript: {row.get('transcript_excerpt', '')}")
        st.caption(f"Chat: {' / '.join(row.get('representative_chat', []))}")

        c1, c2, c3 = st.columns(3)
        accepted = c1.checkbox(
            "accept",
            value=existing_feedback.get(row["id"], {}).get("accepted", False),
            key=f"acc_{row['id']}",
        )
        start_sec = c2.number_input(
            "start_sec",
            value=float(existing_feedback.get(row["id"], {}).get("start_sec", row["start_sec"])),
            key=f"start_{row['id']}",
        )
        end_sec = c3.number_input(
            "end_sec",
            value=float(existing_feedback.get(row["id"], {}).get("end_sec", row["end_sec"])),
            key=f"end_{row['id']}",
        )
        feedback_rows.append(
            {
                "id": row["id"],
                "accepted": accepted,
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
        )

    if st.button("Save feedback"):
        write_json(job_dir / "feedback.json", feedback_rows)
        st.success("feedback saved")

    if st.button("Generate storyline"):
        reviewed = load_reviewed_highlights(job_dir)
        duration_sec = float(read_json(job_dir / "metadata.json", {}).get("format", {}).get("duration", 0.0))
        storyline = generate_storyline(reviewed, duration_sec)
        payload = storyline.model_dump()
        write_json(job_dir / "storyline.json", payload)

        lines = [f"# {payload['title']}", "", payload["one_line_summary"], ""]
        for item in payload["items"]:
            lines += [
                f"## {item['role']}",
                f"- time: {item['start_tc']} - {item['end_tc']}",
                f"- summary: {item['summary']}",
                "- evidence:",
                *[f"  - {evidence}" for evidence in item["evidence"]],
                f"- editor_note: {item['editor_note']}",
                "",
            ]
        (job_dir / "storyline.md").write_text("\n".join(lines), encoding="utf-8")
        st.success("storyline generated")
        st.json(payload)
