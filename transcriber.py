from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel
from tqdm import tqdm


@dataclass
class Segment:
    start: float
    end: float
    text: str


def format_timestamp_txt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_timestamp_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


SUPPORTED_FORMATS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def validate_video_file(path: str) -> Path:
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    if video_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {video_path.suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    return video_path


def format_txt(segments: list[Segment]) -> str:
    lines = []
    for seg in segments:
        start = format_timestamp_txt(seg.start)
        end = format_timestamp_txt(seg.end)
        lines.append(f"[{start} --> {end}] {seg.text}")
    return "\n".join(lines)


def format_srt(segments: list[Segment]) -> str:
    if not segments:
        return ""
    blocks = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp_srt(seg.start)
        end = format_timestamp_srt(seg.end)
        blocks.append(f"{i}\n{start} --> {end}\n{seg.text}")
    return "\n\n".join(blocks) + "\n"


def extract_audio(video_path: Path, output_path: Path | None = None) -> Path:
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = Path(tmp.name)
        tmp.close()
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn", "-ar", "16000", "-ac", "1", "-f", "wav",
        "-y", str(output_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install it: https://ffmpeg.org/download.html"
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else "Unknown error"
        raise RuntimeError(f"ffmpeg failed: {stderr}")
    return output_path


def load_model(model_size: str = "large-v3") -> WhisperModel:
    return WhisperModel(model_size, device="auto", compute_type="auto")


def transcribe_audio(
    model: WhisperModel,
    audio_path: Path,
    language: str = "fi",
) -> list[Segment]:
    segments_gen, info = model.transcribe(str(audio_path), language=language)
    results: list[Segment] = []
    with tqdm(total=int(info.duration), unit="s", desc="Transcribing") as pbar:
        last_end = 0.0
        for seg in segments_gen:
            results.append(Segment(
                start=seg.start, end=seg.end, text=seg.text.strip()
            ))
            pbar.update(int(seg.end - last_end))
            last_end = seg.end
    return results


def write_output(content: str, output_path: Path) -> None:
    output_path.write_text(content, encoding="utf-8")
