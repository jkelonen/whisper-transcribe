from transcriber import Segment, format_timestamp_srt, format_timestamp_txt


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
