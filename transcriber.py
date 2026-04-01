from __future__ import annotations

import os
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

# Suppress noisy HuggingFace symlink warnings on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message=".*huggingface_hub.*cache-system uses symlinks.*")

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
    total_ms = int(round(seconds * 1000))
    h, remainder = divmod(total_ms, 3_600_000)
    m, remainder = divmod(remainder, 60_000)
    s, ms = divmod(remainder, 1000)
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
    created_temp = False
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = Path(tmp.name)
        tmp.close()
        created_temp = True
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn", "-ar", "16000", "-ac", "1", "-f", "wav",
        "-y", str(output_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError:
        if created_temp and output_path.exists():
            output_path.unlink()
        raise RuntimeError(
            "ffmpeg not found. Install it: https://ffmpeg.org/download.html"
        )
    except subprocess.CalledProcessError as e:
        if created_temp and output_path.exists():
            output_path.unlink()
        stderr = e.stderr.decode(errors="replace") if e.stderr else "Unknown error"
        raise RuntimeError(f"ffmpeg failed: {stderr}")
    return output_path


def _detect_device() -> tuple[str, str]:
    """Detect best available device, falling back to CPU if CUDA fails."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            # Verify CUDA actually works with a small operation
            ctranslate2.StorageView.from_array([0], "cuda")
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


MODEL_SIZES = {
    "tiny": "~75MB",
    "base": "~150MB",
    "small": "~500MB",
    "medium": "~1.5GB",
    "large-v3": "~3GB",
}


def load_model(model_size: str = "large-v3") -> WhisperModel:
    device, compute_type = _detect_device()
    size_hint = MODEL_SIZES.get(model_size, "unknown size")
    print(
        f"Note: If this is the first run with '{model_size}', "
        f"the model ({size_hint}) will be downloaded. This may take a few minutes."
    )
    return WhisperModel(model_size, device=device, compute_type=compute_type)


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
