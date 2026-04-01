from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe video files to timestamped text and SRT subtitles"
    )
    parser.add_argument(
        "video_path", help="Path to video file (mp4, mkv, avi, mov, webm)"
    )
    parser.add_argument(
        "--model", default="large-v3",
        help="Whisper model size (default: large-v3)",
    )
    parser.add_argument(
        "--language", default="fi",
        help="Language code for transcription (default: fi)",
    )
    parser.add_argument(
        "--output", help="Output file path (default: <video_name>_transcript.txt)"
    )
    parser.add_argument(
        "--srt", action="store_true", help="Also output SRT subtitle file"
    )
    return parser.parse_args(argv)
