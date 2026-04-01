import pytest
from unittest.mock import patch, MagicMock
from cli import parse_args, main
from transcriber import Segment


class TestParseArgs:
    def test_minimal_args(self):
        args = parse_args(["video.mp4"])
        assert args.video_path == "video.mp4"
        assert args.model == "large-v3"
        assert args.language == "fi"
        assert args.output is None
        assert args.srt is False

    def test_all_options(self):
        args = parse_args([
            "video.mp4", "--model", "base", "--language", "en",
            "--output", "out.txt", "--srt"
        ])
        assert args.video_path == "video.mp4"
        assert args.model == "base"
        assert args.language == "en"
        assert args.output == "out.txt"
        assert args.srt is True

    def test_missing_video_path(self):
        with pytest.raises(SystemExit):
            parse_args([])


class TestMain:
    @patch("cli.write_output")
    @patch("cli.format_srt")
    @patch("cli.format_txt", return_value="[00:00:00 --> 00:00:04] Hei.")
    @patch("cli.transcribe_audio", return_value=[Segment(0, 4, "Hei.")])
    @patch("cli.load_model")
    @patch("cli.extract_audio")
    @patch("cli.validate_video_file")
    def test_success_txt_only(
        self, mock_validate, mock_extract, mock_load,
        mock_transcribe, mock_fmt_txt, mock_fmt_srt, mock_write,
        tmp_path,
    ):
        video = tmp_path / "test.mp4"
        video.touch()
        mock_validate.return_value = video
        mock_extract.return_value = tmp_path / "temp.wav"
        (tmp_path / "temp.wav").touch()

        result = main([str(video)])

        assert result == 0
        mock_validate.assert_called_once()
        mock_load.assert_called_once()
        mock_extract.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_fmt_txt.assert_called_once()
        mock_fmt_srt.assert_not_called()

    @patch("cli.write_output")
    @patch("cli.format_srt", return_value="1\n00:00:00,000 --> 00:00:04,000\nHei.\n")
    @patch("cli.format_txt", return_value="[00:00:00 --> 00:00:04] Hei.")
    @patch("cli.transcribe_audio", return_value=[Segment(0, 4, "Hei.")])
    @patch("cli.load_model")
    @patch("cli.extract_audio")
    @patch("cli.validate_video_file")
    def test_success_with_srt(
        self, mock_validate, mock_extract, mock_load,
        mock_transcribe, mock_fmt_txt, mock_fmt_srt, mock_write,
        tmp_path,
    ):
        video = tmp_path / "test.mp4"
        video.touch()
        mock_validate.return_value = video
        mock_extract.return_value = tmp_path / "temp.wav"
        (tmp_path / "temp.wav").touch()

        result = main([str(video), "--srt"])

        assert result == 0
        mock_fmt_srt.assert_called_once()
        assert mock_write.call_count == 2

    def test_file_not_found(self):
        result = main(["/nonexistent/video.mp4"])
        assert result == 1

    @patch("cli.load_model")
    @patch("cli.extract_audio", side_effect=RuntimeError("ffmpeg not found"))
    @patch("cli.validate_video_file")
    def test_ffmpeg_error(self, mock_validate, mock_extract, mock_load, tmp_path):
        video = tmp_path / "test.mp4"
        video.touch()
        mock_validate.return_value = video

        result = main([str(video)])

        assert result == 1
