import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess
from transcriber import (
    Segment, format_timestamp_srt, format_timestamp_txt, validate_video_file,
    format_txt, format_srt, extract_audio, load_model, transcribe_audio, write_output,
)


class TestSegment:
    def test_creation(self):
        seg = Segment(start=0.0, end=4.0, text="Hei maailma.")
        assert seg.start == 0.0
        assert seg.end == 4.0
        assert seg.text == "Hei maailma."

    def test_equality(self):
        a = Segment(start=0.0, end=4.0, text="Hei.")
        b = Segment(start=0.0, end=4.0, text="Hei.")
        assert a == b


class TestFormatTimestampTxt:
    def test_zero(self):
        assert format_timestamp_txt(0.0) == "00:00:00"

    def test_seconds_only(self):
        assert format_timestamp_txt(45.0) == "00:00:45"

    def test_minutes_and_seconds(self):
        assert format_timestamp_txt(125.0) == "00:02:05"

    def test_hours(self):
        assert format_timestamp_txt(3661.0) == "01:01:01"

    def test_truncates_subsecond(self):
        assert format_timestamp_txt(4.9) == "00:00:04"


class TestFormatTimestampSrt:
    def test_zero(self):
        assert format_timestamp_srt(0.0) == "00:00:00,000"

    def test_with_milliseconds(self):
        assert format_timestamp_srt(4.32) == "00:00:04,320"

    def test_hours_with_millis(self):
        assert format_timestamp_srt(3661.5) == "01:01:01,500"


class TestValidateVideoFile:
    def test_valid_mp4(self, tmp_path):
        video = tmp_path / "test.mp4"
        video.touch()
        result = validate_video_file(str(video))
        assert result == video

    def test_valid_mkv(self, tmp_path):
        video = tmp_path / "test.mkv"
        video.touch()
        result = validate_video_file(str(video))
        assert result == video

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            validate_video_file("/nonexistent/video.mp4")

    def test_unsupported_format(self, tmp_path):
        video = tmp_path / "test.pdf"
        video.touch()
        with pytest.raises(ValueError, match="Unsupported format"):
            validate_video_file(str(video))


class TestFormatTxt:
    def test_single_segment(self):
        segments = [Segment(start=0.0, end=4.0, text="Hei maailma.")]
        result = format_txt(segments)
        assert result == "[00:00:00 --> 00:00:04] Hei maailma."

    def test_multiple_segments(self):
        segments = [
            Segment(start=0.0, end=4.0, text="Hei maailma."),
            Segment(start=4.0, end=9.5, text="Mitä kuuluu?"),
        ]
        result = format_txt(segments)
        expected = "[00:00:00 --> 00:00:04] Hei maailma.\n[00:00:04 --> 00:00:09] Mitä kuuluu?"
        assert result == expected

    def test_empty_segments(self):
        assert format_txt([]) == ""


class TestFormatSrt:
    def test_single_segment(self):
        segments = [Segment(start=0.0, end=4.32, text="Hei maailma.")]
        result = format_srt(segments)
        expected = "1\n00:00:00,000 --> 00:00:04,320\nHei maailma.\n"
        assert result == expected

    def test_multiple_segments(self):
        segments = [
            Segment(start=0.0, end=4.32, text="Hei maailma."),
            Segment(start=4.32, end=9.15, text="Mitä kuuluu?"),
        ]
        result = format_srt(segments)
        expected = (
            "1\n00:00:00,000 --> 00:00:04,320\nHei maailma.\n\n"
            "2\n00:00:04,320 --> 00:00:09,150\nMitä kuuluu?\n"
        )
        assert result == expected

    def test_empty_segments(self):
        assert format_srt([]) == ""


class TestExtractAudio:
    @patch("transcriber.subprocess.run")
    def test_calls_ffmpeg_correctly(self, mock_run, tmp_path):
        video = tmp_path / "test.mp4"
        video.touch()
        output = tmp_path / "output.wav"

        result = extract_audio(video, output)

        assert result == output
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd
        assert str(video) in cmd
        assert str(output) in cmd

    @patch("transcriber.subprocess.run")
    def test_creates_temp_file_when_no_output(self, mock_run, tmp_path):
        video = tmp_path / "test.mp4"
        video.touch()

        result = extract_audio(video)

        assert result.suffix == ".wav"

    @patch("transcriber.subprocess.run", side_effect=FileNotFoundError)
    def test_missing_ffmpeg(self, mock_run, tmp_path):
        video = tmp_path / "test.mp4"
        video.touch()

        with pytest.raises(RuntimeError, match="ffmpeg not found"):
            extract_audio(video)

    @patch("transcriber.subprocess.run")
    def test_ffmpeg_failure_includes_stderr(self, mock_run, tmp_path):
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg", stderr=b"No audio stream found"
        )
        video = tmp_path / "test.mp4"
        video.touch()

        with pytest.raises(RuntimeError, match="No audio stream found"):
            extract_audio(video)


class TestLoadModel:
    @patch("transcriber.WhisperModel")
    def test_loads_with_defaults(self, MockModel):
        load_model()
        MockModel.assert_called_once_with("large-v3", device="auto", compute_type="auto")

    @patch("transcriber.WhisperModel")
    def test_loads_custom_model(self, MockModel):
        load_model("base")
        MockModel.assert_called_once_with("base", device="auto", compute_type="auto")


class TestTranscribeAudio:
    @patch("transcriber.tqdm")
    @patch("transcriber.WhisperModel")
    def test_returns_segments(self, MockModel, mock_tqdm):
        mock_tqdm.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_tqdm.return_value.__exit__ = MagicMock(return_value=False)
        mock_segment_1 = MagicMock(start=0.0, end=4.0, text=" Hei maailma. ")
        mock_segment_2 = MagicMock(start=4.0, end=9.0, text=" Mitä kuuluu? ")
        mock_info = MagicMock(duration=9.0)
        mock_instance = MockModel.return_value
        mock_instance.transcribe.return_value = (
            iter([mock_segment_1, mock_segment_2]),
            mock_info,
        )

        model = load_model("base")
        result = transcribe_audio(model, Path("audio.wav"), language="fi")

        assert result == [
            Segment(start=0.0, end=4.0, text="Hei maailma."),
            Segment(start=4.0, end=9.0, text="Mitä kuuluu?"),
        ]
        mock_instance.transcribe.assert_called_once_with("audio.wav", language="fi")

    @patch("transcriber.tqdm")
    @patch("transcriber.WhisperModel")
    def test_default_language_fi(self, MockModel, mock_tqdm):
        mock_tqdm.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_tqdm.return_value.__exit__ = MagicMock(return_value=False)
        mock_instance = MockModel.return_value
        mock_instance.transcribe.return_value = (iter([]), MagicMock(duration=0.0))

        model = load_model()
        transcribe_audio(model, Path("audio.wav"))

        mock_instance.transcribe.assert_called_once_with("audio.wav", language="fi")

    @patch("transcriber.tqdm")
    @patch("transcriber.WhisperModel")
    def test_custom_language(self, MockModel, mock_tqdm):
        mock_tqdm.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_tqdm.return_value.__exit__ = MagicMock(return_value=False)
        mock_instance = MockModel.return_value
        mock_instance.transcribe.return_value = (iter([]), MagicMock(duration=0.0))

        model = load_model()
        transcribe_audio(model, Path("audio.wav"), language="en")

        mock_instance.transcribe.assert_called_once_with("audio.wav", language="en")


class TestWriteOutput:
    def test_writes_content_to_file(self, tmp_path):
        output = tmp_path / "output.txt"
        write_output("hello world", output)
        assert output.read_text(encoding="utf-8") == "hello world"

    def test_handles_finnish_characters(self, tmp_path):
        output = tmp_path / "output.txt"
        content = "[00:00:00 --> 00:00:04] Hyvää päivää."
        write_output(content, output)
        assert output.read_text(encoding="utf-8") == content
