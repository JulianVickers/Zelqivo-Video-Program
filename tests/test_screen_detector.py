"""Screen-marker detection: flatness + colour match, run-length debounce, backdating."""
import shutil
import statistics
import subprocess

import numpy as np
import pytest

from multicam_editor.logic.screen_detector import (
    DEFAULT_MAX_FLAT_STD,
    FrameStats,
    MarkerColor,
    ScreenInterval,
    ScreenScanCancelled,
    ScreenScanError,
    frame_stats,
    intervals_from_flags,
    is_marker,
    sample_frame_stats,
    scan_screen_intervals,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)

_M = MarkerColor(y=100.0, u=200.0, v=150.0, tolerance=10.0)


class TestFrameStats:
    def test_flat_frame_reports_means_and_zero_spread(self):
        frame = np.empty((3, 36, 64), dtype=np.uint8)
        frame[0], frame[1], frame[2] = 100, 50, 200
        stats = frame_stats(frame)
        assert stats.y == pytest.approx(100.0)
        assert stats.u == pytest.approx(50.0)
        assert stats.v == pytest.approx(200.0)
        assert stats.flat_std == pytest.approx(0.0)

    def test_noisy_frame_reports_large_spread(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, size=(3, 36, 64), dtype=np.uint8)
        assert frame_stats(frame).flat_std > 50.0

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            frame_stats(np.zeros((4, 36, 64), dtype=np.uint8))


class TestIsMarker:
    def test_exact_match_flat_frame(self):
        assert is_marker(FrameStats(100.0, 200.0, 150.0, 0.0), _M)

    def test_tolerance_and_flatness_boundaries_inclusive(self):
        assert is_marker(FrameStats(110.0, 190.0, 160.0, DEFAULT_MAX_FLAT_STD), _M)

    @pytest.mark.parametrize("channel", ["y", "u", "v"])
    def test_outside_tolerance_on_any_channel_rejected(self, channel):
        values = {"y": 100.0, "u": 200.0, "v": 150.0}
        values[channel] += 11.0
        assert not is_marker(FrameStats(flat_std=0.0, **values), _M)

    def test_marker_coloured_but_noisy_frame_rejected(self):
        assert not is_marker(FrameStats(100.0, 200.0, 150.0, 20.0), _M)

    def test_negative_tolerance_rejected(self):
        with pytest.raises(ValueError):
            MarkerColor(1.0, 2.0, 3.0, tolerance=-1.0)


class TestIntervalsFromFlags:
    # 10 samples/s: enter = 5 samples (0.5 s), exit = 20 samples (2.0 s)
    def _run(self, flags, fps=10.0, enter_s=0.5, exit_s=2.0):
        return intervals_from_flags(flags, fps, enter_s, exit_s)

    def test_empty(self):
        assert self._run([]) == []

    def test_start_is_backdated_to_onset(self):
        flags = [False] * 10 + [True] * 30 + [False] * 30
        assert self._run(flags) == [ScreenInterval(1000, 4000)]

    def test_blip_shorter_than_enter_rejected(self):
        assert self._run([True] * 4 + [False] * 20) == []

    def test_exactly_enter_samples_accepted(self):
        assert self._run([True] * 5 + [False] * 25) == [ScreenInterval(0, 500)]

    def test_short_gap_bridged(self):
        flags = [True] * 10 + [False] * 19 + [True] * 10
        assert self._run(flags) == [ScreenInterval(0, 3900)]

    def test_long_gap_splits(self):
        flags = [True] * 10 + [False] * 20 + [True] * 10
        assert self._run(flags) == [ScreenInterval(0, 1000), ScreenInterval(3000, 4000)]

    def test_open_run_closes_at_end_of_stream(self):
        assert self._run([False] * 5 + [True] * 20) == [ScreenInterval(500, 2500)]

    def test_unconfirmed_run_at_end_dropped(self):
        assert self._run([False] * 5 + [True] * 3) == []

    def test_other_sample_rate_scales_debounce(self):
        # 5 samples/s: enter = ceil(2.5) = 3 samples, exit = 10 samples
        flags = [False] * 5 + [True] * 3 + [False] * 10
        assert self._run(flags, fps=5.0) == [ScreenInterval(1000, 1600)]

    def test_invalid_parameters_rejected(self):
        with pytest.raises(ValueError):
            intervals_from_flags([True], 0.0, 0.5, 2.0)
        with pytest.raises(ValueError):
            intervals_from_flags([True], 10.0, -0.5, 2.0)
        with pytest.raises(ValueError):
            intervals_from_flags([True], 10.0, 0.5, -2.0)


# ---------------------------------------------------------------- ffmpeg-backed

_CLIP_RATE = 10
_CLIP_SIZE = "160x90"


def _source(kind, seconds):
    base = f"s={_CLIP_SIZE}:r={_CLIP_RATE}:d={seconds}"
    specs = {
        "marker": f"color=c=magenta:{base}",
        "wrong": f"color=c=green:{base}",
        "noise": f"color=c=0x808080:{base},noise=alls=60:allf=t+u",
        "marker_noise": f"color=c=magenta:{base},noise=alls=60:allf=t+u",
        "testsrc": f"testsrc={base}",
    }
    return specs[kind] + ",format=yuv420p,setsar=1"


def _make_clip(tmp_path, name, segments):
    out = tmp_path / name
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for kind, seconds in segments:
        cmd += ["-f", "lavfi", "-i", _source(kind, seconds)]
    n = len(segments)
    inputs = "".join(f"[{i}:v]" for i in range(n))
    cmd += ["-filter_complex", f"{inputs}concat=n={n}:v=1:a=0[v]", "-map", "[v]",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True)
    return str(out)


def _reference_marker(tmp_path, tolerance=12.0):
    """Marker colour measured through the scanner's own conversion path."""
    stats = sample_frame_stats(_make_clip(tmp_path, "ref.mp4", [("marker", 2)]))
    mid = stats[len(stats) // 2]
    return MarkerColor(mid.y, mid.u, mid.v, tolerance=tolerance)


def _assert_near(intervals, expected, slack_ms=200):
    assert len(intervals) == len(expected), f"got {intervals}, expected ~{expected}"
    for got, (start, end) in zip(intervals, expected):
        assert abs(got.start_ms - start) <= slack_ms, f"start {got} vs {start}"
        assert abs(got.end_ms - end) <= slack_ms, f"end {got} vs {end}"


@needs_ffmpeg
class TestScanWithFfmpeg:
    def test_sample_rate_gives_expected_frame_count(self, tmp_path):
        clip = _make_clip(tmp_path, "a.mp4", [("marker", 3)])
        assert 28 <= len(sample_frame_stats(clip)) <= 31

    def test_marker_is_flat_and_noise_is_not(self, tmp_path):
        marker = sample_frame_stats(_make_clip(tmp_path, "m.mp4", [("marker", 2)]))
        noise = sample_frame_stats(_make_clip(tmp_path, "n.mp4", [("noise", 2)]))
        assert max(s.flat_std for s in marker) <= DEFAULT_MAX_FLAT_STD
        assert statistics.median(s.flat_std for s in noise) > 2 * DEFAULT_MAX_FLAT_STD

    def test_marker_between_noise_found_and_backdated(self, tmp_path):
        clip = _make_clip(tmp_path, "c.mp4", [("noise", 2), ("marker", 3), ("noise", 2)])
        found = scan_screen_intervals(clip, _reference_marker(tmp_path))
        _assert_near(found, [(2000, 5000)])

    def test_blip_shorter_than_debounce_ignored(self, tmp_path):
        clip = _make_clip(tmp_path, "d.mp4", [("noise", 2), ("marker", 0.3), ("noise", 2)])
        assert scan_screen_intervals(clip, _reference_marker(tmp_path)) == []

    def test_short_dropout_bridged(self, tmp_path):
        clip = _make_clip(tmp_path, "e.mp4", [("marker", 2), ("testsrc", 1), ("marker", 2)])
        found = scan_screen_intervals(clip, _reference_marker(tmp_path))
        _assert_near(found, [(0, 5000)])

    def test_long_dropout_splits(self, tmp_path):
        clip = _make_clip(tmp_path, "f.mp4", [("marker", 2), ("testsrc", 3), ("marker", 2)])
        found = scan_screen_intervals(clip, _reference_marker(tmp_path))
        _assert_near(found, [(0, 2000), (5000, 7000)])

    def test_wrong_colour_flat_clip_ignored(self, tmp_path):
        clip = _make_clip(tmp_path, "g.mp4", [("wrong", 3)])
        assert scan_screen_intervals(clip, _reference_marker(tmp_path)) == []

    def test_marker_coloured_noise_rejected_by_flatness(self, tmp_path):
        clip = _make_clip(tmp_path, "h.mp4", [("marker_noise", 3)])
        stats = sample_frame_stats(clip)
        assert statistics.median(s.flat_std for s in stats) > 2 * DEFAULT_MAX_FLAT_STD
        assert scan_screen_intervals(clip, _reference_marker(tmp_path)) == []

    def test_tolerance_controls_match(self, tmp_path):
        clip = _make_clip(tmp_path, "i.mp4", [("marker", 3)])
        ref = _reference_marker(tmp_path)
        tight = MarkerColor(ref.y + 30.0, ref.u, ref.v, tolerance=12.0)
        loose = MarkerColor(ref.y + 30.0, ref.u, ref.v, tolerance=40.0)
        assert scan_screen_intervals(clip, tight) == []
        _assert_near(scan_screen_intervals(clip, loose), [(0, 3000)])

    def test_missing_file_raises_typed_error(self, tmp_path):
        with pytest.raises(ScreenScanError):
            scan_screen_intervals(str(tmp_path / "nope.mp4"), MarkerColor(100.0, 128.0, 128.0))

    def test_cancel_callback_stops_scan(self, tmp_path):
        clip = _make_clip(tmp_path, "k.mp4", [("marker", 2)])
        with pytest.raises(ScreenScanCancelled):
            sample_frame_stats(clip, cancel_callback=lambda: True)
