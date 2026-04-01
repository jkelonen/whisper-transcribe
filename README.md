# Finnish Video Transcriber

Transcribe video files to timestamped text and SRT subtitles using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Default language: Finnish.

## Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) installed and on PATH
- (Optional) NVIDIA GPU with CUDA 12 + cuDNN 9 for GPU acceleration

## Install

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies (~200MB download, includes CTranslate2 engine)
pip install -r requirements.txt
```

## Usage

```bash
python cli.py <video_path> [--model large-v3] [--language fi] [--output <path>] [--srt]
```

### Examples

```bash
# Basic — outputs video_transcript.txt next to the video (Finnish)
python cli.py meeting.mp4

# With SRT subtitles
python cli.py meeting.mp4 --srt

# Transcribe English video
python cli.py interview.mp4 --language en --srt

# Faster with smaller model (less accurate)
python cli.py meeting.mp4 --model base

# Custom output path
python cli.py meeting.mp4 --output ~/transcripts/meeting.txt --srt
```

## Output Formats

### TXT (always generated)

```
[00:00:00 --> 00:00:04] Hyvää päivää, tervetuloa tähän kokoukseen.
[00:00:04 --> 00:00:09] Käydään ensin läpi viime viikon asiat.
```

### SRT (with --srt flag)

```
1
00:00:00,000 --> 00:00:04,320
Hyvää päivää, tervetuloa tähän kokoukseen.

2
00:00:04,320 --> 00:00:09,150
Käydään ensin läpi viime viikon asiat.
```

## Models

First run downloads the model automatically. Sizes:

| Model | Download | Speed | Accuracy | Finnish Quality |
|-------|----------|-------|----------|-----------------|
| base | ~150MB | Fastest | Lower | Poor |
| medium | ~1.5GB | Medium | Good | Acceptable |
| large-v3 | ~3GB | Slowest | Best | Recommended |

GPU (CUDA) is auto-detected. Falls back to CPU automatically.

> **Note:** `tiny` and `small` models produce very poor results for Finnish and are not recommended.

## Tests

```bash
python -m pytest tests/ -v
python -m mypy cli.py transcriber.py --strict
```

## Troubleshooting

- **"ffmpeg not found"**: Install ffmpeg and ensure it's on your PATH
- **First run is slow**: Model downloads on first use (large-v3 is ~3GB)
- **CUDA errors**: Ensure CUDA 12 + cuDNN 9 are installed, or it will fall back to CPU
- **Poor Finnish accuracy**: Use `large-v3` model. Smaller models struggle with Finnish.
