# vod-highlight-storyline

Local-first MVP for editors: **video + chat → highlights → storyline → exports**.

## Key behavior
- Works fully local with a job-folder workflow.
- Keeps expensive `prepare` stage separate from fast `score` stage.
- Supports rescoring with new `chat_offset_seconds` **without rerunning transcription/scene/audio prep**.
- Optional LLM storyline refinement (`OPENAI_API_KEY`) with strict fallback to deterministic heuristics.

## Install
```bash
cd vod-highlight-storyline
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
# optional extras
pip install -e .[ml]
```

## Prerequisites
`ffmpeg` and `ffprobe` must be available in PATH:
```bash
ffmpeg -version
ffprobe -version
```

## CLI
```bash
python -m app.cli prepare --video /path/stream.mp4 --chat /path/chat.txt --out ./jobs/job1
python -m app.cli score --job ./jobs/job1 --preset app/presets/game.yaml --chat-offset 0
python -m app.cli storyline --job ./jobs/job1
python -m app.cli export --job ./jobs/job1 --extract-clips
```

## Streamlit
```bash
streamlit run app/ui.py
```

UI supports:
- input video/chat paths
- preset + chat offset
- run `prepare` / `score`
- highlight review table (reasons, transcript excerpt, representative chat)
- accept/reject + manual start/end edits
- save `feedback.json`
- generate storyline from accepted highlights (or fallback top highlights)
- export outputs

## Job folder artifacts
`jobs/job1/` typically contains:
- `metadata.json`
- `audio.wav`
- `transcript.json`
- `scene_cuts.json`
- `audio_features.csv`
- `chat_source.txt` or `chat_source.csv`
- `chat_normalized.json`
- `fused_features.csv`
- `highlights.json`, `highlights.csv`
- `markers.csv`
- `feedback.json`
- `storyline.json`, `storyline.md`
- `transcript.srt`
- `exports/clips/*.mp4` (optional)
- `.cache/*.done`

## Caching + rescoring behavior
- `prepare` writes heavy artifacts once and marks stages complete.
- `score` reloads cached artifacts and recomputes fusion/highlights quickly.
- Chat source priority in `score`:
  1. original chat path from `job_config.json` (if still present)
  2. local job copy `chat_source.*`
  3. fallback `chat_normalized.json`
- `chat_offset_seconds` is applied exactly once.

## Long VOD handling
Audio feature extraction is chunked with `soundfile.SoundFile.read(...)` blocks, so the full audio is not loaded into RAM.

## Limitations (current V1)
- Transcription quality depends on local faster-whisper availability/model.
- Scene detection is cut-based only (no frame-by-frame CV analysis).
- Streamlit UI is functional but intentionally simple.

## Typical workflow
1. Run `prepare` once on long VOD + chat.
2. Run `score` with preset and offset.
3. Review highlights in UI, save `feedback.json`.
4. Run/generate `storyline`.
5. Run `export` for editor deliverables.
