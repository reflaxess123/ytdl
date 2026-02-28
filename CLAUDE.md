# CLAUDE.md

## Project overview

YouTube downloader tool built on yt-dlp. Downloads video, audio (MP3), subtitles (SRT), and exports video lists (CSV) from YouTube videos, playlists, and channels.

## Key files

- `download.py` — main CLI tool (video/audio/subs/list download)
- `dub_video.py` — video dubbing pipeline (Whisper + GLM + Qwen3-TTS)
- `.env` — local proxy config (YTDL_PROXY), gitignored

## Architecture

- Single-file CLI (`download.py`) using yt-dlp Python API
- `.env` auto-loaded for proxy config (no external deps like python-dotenv)
- `DEFAULT_PROXY` read from `YTDL_PROXY` env var
- Downloads go to `./downloads/` with playlist subdirectories
- `.archive.txt` tracks downloaded video IDs to skip on re-run

## Commands

```bash
uv run python download.py URL                    # video 1080p
uv run python download.py URL --mp3 --ru         # audio via proxy
uv run python download.py URL --subs --ru        # subtitles
uv run python download.py URL --list --ru        # CSV list
```

## Conventions

- Language: Russian for user-facing text, English for code/comments
- Run with: `uv run python download.py`
- Proxy IP must never be committed — always read from .env/env var
- `.gitignore` excludes: downloads/, .venv/, .env, __pycache__, transcript.json, tts_output.log
