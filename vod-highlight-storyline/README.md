# vod-highlight-storyline

Local-first MVP that finds livestream highlights from a long video + chat timeline, then builds an editor-friendly storyline.

## What it does
- Parses `.txt`/`.csv` chat logs (Korean/English-friendly).
- Runs heavy preparation once (`ffprobe`, audio extraction, transcription, scene cuts, audio features).
- Re-scores quickly with different presets or `chat_offset_seconds` without retranscribing.
- Produces ranked highlight events with reasons, representative chat, and suggested titles.
- Generates storyline in two modes:
  - Heuristic fallback (default, no API key required)
  - Optional LLM-assisted refinement when `OPENAI_API_KEY` is present
- Exports JSON/CSV/Markdown/SRT and optional preview clips.
- Includes Streamlit review UI (accept/reject, manual timing adjustments, feedback saved locally).

## Assumptions (MVP)
- `ffmpeg`/`ffprobe` are installed and in PATH.
- Input chat is roughly aligned with the video timeline; manual `chat_offset_seconds` corrects drift.
- For absolute datetime chats, first message timestamp becomes t=0 unless offset overrides.
- Processing remains local by default.
- CPU fallback is used if GPU is unavailable.

## Install
```bash
cd vod-highlight-storyline
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
# Optional ML extras
pip install -e .[ml]
```

## Required external dependency
Install ffmpeg:
```bash
ffmpeg -version
ffprobe -version
```

## CLI usage
```bash
python -m app.cli prepare --video /path/video.mp4 --chat /path/chat.txt --out ./jobs/job1
python -m app.cli score --job ./jobs/job1 --preset app/presets/game.yaml --chat-offset 0
python -m app.cli storyline --job ./jobs/job1
python -m app.cli export --job ./jobs/job1 --extract-clips
```

Run UI:
```bash
streamlit run app/ui.py
```

## Job folder structure
Typical outputs in `jobs/job1/`:
- `metadata.json`
- `audio.wav`
- `transcript.json`
- `scene_cuts.json`
- `audio_features.csv`
- `chat_normalized.json`
- `fused_features.csv`
- `highlights.json`
- `highlights.csv`
- `markers.csv`
- `storyline.md`
- `storyline.json`
- `feedback.json`
- `transcript.srt`
- `exports/clips/*.mp4` (optional)
- `.cache/*.done`

## Caching and resumability
- Heavy steps are checkpointed with `.cache/<stage>.done`.
- `prepare` skips completed heavy artifacts.
- `score` is lightweight and can be rerun quickly.
- Changing `chat_offset_seconds` only requires re-running `score`, not transcription/scene detection.

## Presets
Weights are YAML-defined in `app/presets/`:
- `game.yaml`
- `talk.yaml`
- `variety.yaml`

Edit these files to customize scoring behavior.

## With and without API key
- Without key: storyline generation uses deterministic heuristics.
- With key (`OPENAI_API_KEY`): optional LLM call is attempted; failures safely fall back.

## Limitations (MVP)
- Whisper transcription quality depends on local model and hardware.
- Scene detection is basic content cut detection.
- No advanced per-frame visual understanding.
- UI is intentionally simple.

## Next steps
- Better diarization and speaker-turn features.
- Multi-language lexicons and auto-calibration of excitement tokens.
- Improved clip preview playback in UI.
- Optional NLE EDL/XML export.
