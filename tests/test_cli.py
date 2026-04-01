import pytest
from cli import parse_args


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
