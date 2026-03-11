from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.analyzers.storyline import generate_storyline
from app.config import AppConfig
from app.pipeline.prepare import run_prepare
from app.pipeline.score import run_score
from app.utils.io import read_json, write_json

st.set_page_config(page_title="vod-highlight-storyline", layout="wide")
st.title("vod-highlight-storyline")

job_dir = Path(st.text_input("Job directory", "./jobs/ui_job"))
video = Path(st.text_input("Video path", ""))
chat = Path(st.text_input("Chat path", ""))
preset = st.selectbox("Preset", ["game", "talk", "variety"])
chat_offset = st.number_input("chat_offset_seconds", value=0.0)

if st.button("Run prepare"):
    run_prepare(video, chat, job_dir, chat_offset_seconds=float(chat_offset))
    write_json(job_dir / "job_config.json", {"video_path": str(video), "chat_path": str(chat)})
    st.success("Prepare complete")

if st.button("Run score"):
    events = run_score(job_dir, Path(f"app/presets/{preset}.yaml"), AppConfig(chat_offset_seconds=float(chat_offset)))
    st.success(f"Score complete: {len(events)} highlights")

highlights = read_json(job_dir / "highlights.json", [])
if highlights:
    df = pd.DataFrame(highlights)
    st.subheader("Highlights")
    st.dataframe(df[["id", "start_tc", "end_tc", "score", "label", "reasons"]])

    st.subheader("Review decisions")
    feedback = []
    for h in highlights:
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.write(f"{h['id']} {h['start_tc']} - {h['end_tc']} | {h['suggested_summary']}")
        accepted = c2.checkbox("accept", key=f"acc_{h['id']}")
        start = c3.number_input("start", value=float(h["start_sec"]), key=f"st_{h['id']}")
        end = st.number_input("end", value=float(h["end_sec"]), key=f"en_{h['id']}")
        feedback.append({"id": h["id"], "accepted": accepted, "start_sec": start, "end_sec": end})
    if st.button("Save feedback"):
        write_json(job_dir / "feedback.json", feedback)
        st.success("Saved feedback.json")

    if st.button("Generate storyline"):
        from app.schemas import HighlightEvent

        fb = {x["id"]: x for x in read_json(job_dir / "feedback.json", [])}
        events = []
        for h in highlights:
            if fb and not fb.get(h["id"], {}).get("accepted", False):
                continue
            if h["id"] in fb:
                h["start_sec"] = fb[h["id"]]["start_sec"]
                h["end_sec"] = fb[h["id"]]["end_sec"]
            events.append(HighlightEvent(**h))
        if not events:
            events = [HighlightEvent(**h) for h in highlights[:5]]
        duration = float(read_json(job_dir / "metadata.json", {}).get("format", {}).get("duration", 0.0))
        story = generate_storyline(events, duration)
        write_json(job_dir / "storyline.json", story.model_dump())
        st.json(story.model_dump())
