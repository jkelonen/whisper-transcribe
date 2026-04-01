# Video Transcriber (faster-whisper)

Transcribe video files to timestamped text and SRT subtitles using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Supports 99+ languages (default: Finnish).

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

**Supported formats:** `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`

**Default output:** Saved next to the video as `<name>_transcript.txt` (and `<name>_transcript.srt` with `--srt`).

### Examples

```bash
# Finnish (default)
python cli.py meeting.mp4

# English
python cli.py interview.mp4 --language en

# With SRT subtitles
python cli.py meeting.mp4 --srt

# Faster with smaller model (less accurate)
python cli.py meeting.mp4 --model base

# More accurate with larger model (but slower)
python cli.py meeting.mp4 --model large-v3

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

| Model | Download | Speed | Accuracy |
|-------|----------|-------|----------|
| tiny | ~75MB | Fastest | Lowest |
| base | ~150MB | Fast | Lower |
| small | ~500MB | Moderate | Moderate |
| medium | ~1.5GB | Slow | Good |
| large-v3 | ~3GB | Slowest | Best |

GPU (CUDA) is auto-detected. Falls back to CPU automatically.

> **Note:** For non-English languages (especially Finnish), `large-v3` is strongly recommended. Smaller models struggle with less common languages.

## Tests

```bash
python -m pytest tests/ -v
python -m mypy cli.py transcriber.py --strict
```

## Troubleshooting

- **"ffmpeg not found"**: Install ffmpeg and ensure it's on your PATH
- **First run is slow**: Model downloads on first use (large-v3 is ~3GB)
- **CUDA errors**: Ensure CUDA 12 + cuDNN 9 are installed, or it will fall back to CPU
- **Poor accuracy for non-English languages**: Use `large-v3` model. Smaller models struggle with less common languages.
