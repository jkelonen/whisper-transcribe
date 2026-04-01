from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transcriber import (
    Segment, validate_video_file, extract_audio, load_model,
    transcribe_audio, format_txt, format_srt, write_output,
)


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        video_path = validate_video_file(args.video_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    txt_output = (
        Path(args.output) if args.output
        else video_path.with_name(f"{video_path.stem}_transcript.txt")
    )
    audio_path = None

    try:
        model = load_model(args.model)

        print(f"Extracting audio from {video_path.name}...")
        audio_path = extract_audio(video_path)

        print(f"Transcribing (language: {args.language})...")
        segments = transcribe_audio(model, audio_path, language=args.language)

        txt_content = format_txt(segments)
        write_output(txt_content, txt_output)
        print(f"Transcript saved: {txt_output}")

        if args.srt:
            srt_output = txt_output.with_suffix(".srt")
            srt_content = format_srt(segments)
            write_output(srt_content, srt_output)
            print(f"SRT saved: {srt_output}")

        print(f"Done — {len(segments)} segments transcribed.")
        return 0

    except (RuntimeError, Exception) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Cleaning up...")
        return 130
    finally:
        if audio_path and audio_path.exists():
            audio_path.unlink()


if __name__ == "__main__":
    sys.exit(main())
