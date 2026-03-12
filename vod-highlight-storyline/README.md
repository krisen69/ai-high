# vod-highlight-storyline

Local-first V1 for editors: **video + chat → highlights → review → storyline → exports**.

## Install
```bash
cd vod-highlight-storyline
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
# Optional model/API extras
pip install -e .[ml]
```

## System requirement
`ffmpeg` and `ffprobe` must exist in PATH:
```bash
ffmpeg -version
ffprobe -version
```

## CLI commands
```bash
python -m app.cli prepare --video /path/stream.mp4 --chat /path/chat.txt --out ./jobs/job1 --bin-size 5
python -m app.cli score --job ./jobs/job1 --preset app/presets/game.yaml --chat-offset 0
python -m app.cli storyline --job ./jobs/job1
python -m app.cli export --job ./jobs/job1 --extract-clips
```

Notes:
- `score` defaults to the prepared `bin_size_sec` from `job_config.json`.
- `score --bin-size` can be passed only with the same value; mismatch fails fast.

## Streamlit UI
```bash
streamlit run app/ui.py
```

UI supports:
- video/chat paths, preset, chat offset
- run `prepare` and `score`
- review table with reasons, transcript excerpt, representative chat
- accept/reject + manual retiming
- save `feedback.json`
- generate storyline from reviewed highlights
- export outputs

## Job folder outputs
Typical `jobs/job1/` artifacts:
- `job_config.json`
- `metadata.json`
- `audio.wav`
- `transcript.json`
- `scene_cuts.json`
- `audio_features.csv`
- `chat_source.txt` or `chat_source.csv`
- `chat_normalized_base.json` (zero-offset baseline)
- `chat_normalized.json` (latest scored offset)
- `fused_features.csv`
- `highlights.json` / `highlights.csv` (raw model output)
- `reviewed_highlights.json` / `reviewed_highlights.csv` (feedback-applied)
- `markers.csv` (reviewed timings)
- `feedback.json`
- `storyline.json` / `storyline.md`
- `transcript.srt`
- `exports/clips/*.mp4` (optional)
- `.cache/*.done`

## Caching and rescoring
- Heavy `prepare` stages are cached (`metadata`, `audio`, `transcript`, `scene`, `audio_features`).
- Changing `chat_offset_seconds` only needs `score`; no retranscription or re-scene-detection.
- Chat source priority in `score`:
  1. original path from `job_config.json` (if still exists)
  2. job-local `chat_source.*`
  3. `chat_normalized_base.json` fallback
- Offset is applied exactly once during scoring.

## Feedback consistency
The same feedback application logic is used by:
- CLI storyline generation
- UI storyline generation
- export pipeline

Rules:
- if any highlights are explicitly accepted, only accepted highlights are used
- otherwise all highlights are used
- manual start/end edits update timecodes and are reflected in storyline/export outputs

## Long VOD safety
Audio feature extraction is chunked with `soundfile.SoundFile.read(...)` blocks and does not load entire audio into RAM.

## Failure behavior
- Transcription unavailable/failure: warning logged, transcript remains empty.
- Scene detection failure: warning logged, scene cuts empty.
- LLM refinement failure: warning logged, deterministic storyline fallback used.
- Clip extraction failure: warning logged per clip.

## Limitations
- Transcription quality depends on local model/hardware.
- Scene signal is cut-density only (no expensive frame-by-frame CV).
- UI is intentionally simple for V1.
