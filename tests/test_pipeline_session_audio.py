"""2c: the concat stage muxes the session mix when audio sources are configured."""
from __future__ import annotations

import json
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from multicam_editor.logic.audio_sources import AudioSource
from multicam_editor.logic.processing_pipeline import ProcessingPipeline
from multicam_editor.utils.signals import ProcessingSignals

_FF = "multicam_editor.utils.ffmpeg"

TRAVIS = AudioSource(source_id="travis", path="/rec/travis.wav")
DAKOTA = AudioSource(source_id="dakota", path="/rec/dakota.wav")
_LEGACY_HELPERS = (
    "_add_audio_from_video",
    "_replace_audio_track",
    "_add_external_audio_with_offset",
)


def _make(sources):
    signals = ProcessingSignals()
    errors: list[str] = []
    signals.error.connect(lambda message: errors.append(message))
    pipeline = ProcessingPipeline(["/v/0.mp4", "/v/1.mp4"], signals)
    pipeline._audio_sources = list(sources)
    pipeline._cleanup = MagicMock()
    return pipeline, errors


def _boom(name):
    def boom(*args, **kwargs):
        raise AssertionError(f"{name} must not be called here")

    return boom


def _rendered(tmp_path):
    path = tmp_path / "rendered.mp4"
    path.write_bytes(b"x")
    return str(path)


def _filter(args):
    return args[args.index("-filter_complex") + 1]


def _values(args, flag):
    return [args[i + 1] for i, a in enumerate(args) if a == flag]


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    calls: list[list[str]] = []
    state = {"success": True, "error": None, "available": True}

    class Proc:
        def __init__(self, args, output_path, *a, **k):
            calls.append(list(args))

        def run(self):
            return SimpleNamespace(success=state["success"], error=state["error"])

    monkeypatch.setattr(f"{_FF}.FFmpegProcess", Proc)
    monkeypatch.setattr(f"{_FF}.is_ffmpeg_available", lambda: state["available"])
    return SimpleNamespace(calls=calls, state=state)


# ---- command construction --------------------------------------------------

def test_two_sources_are_mixed_at_offset_zero_without_shortest():
    pipeline, _ = _make([TRAVIS, DAKOTA])
    args = pipeline._build_session_mix_args("/r/video.mp4", "/o/out.mp4")
    assert args[:2] == ["ffmpeg", "-y"]
    assert _values(args, "-i") == ["/r/video.mp4", "/rec/travis.wav", "/rec/dakota.wav"]
    assert "-itsoffset" not in args
    assert "-shortest" not in args
    assert _values(args, "-c:v") == ["copy"]
    assert _values(args, "-map") == ["0:v:0", "[aout]"]
    flt = _filter(args)
    assert "amix=inputs=2:duration=longest:normalize=0" in flt
    assert "[1:a:0]" in flt and "[2:a:0]" in flt
    assert args[-1] == "/o/out.mp4"


def test_single_source_skips_amix():
    pipeline, _ = _make([TRAVIS])
    flt = _filter(pipeline._build_session_mix_args("/r/video.mp4", "/o/out.mp4"))
    assert "amix" not in flt
    assert "[1:a:0]" in flt and "[aout]" in flt


def test_sources_not_included_in_output_are_left_out():
    quiet = AudioSource(source_id="room", path="/rec/room.wav", include_in_output=False)
    pipeline, _ = _make([quiet, DAKOTA])
    args = pipeline._build_session_mix_args("/r/video.mp4", "/o/out.mp4")
    assert _values(args, "-i") == ["/r/video.mp4", "/rec/dakota.wav"]
    assert "amix" not in _filter(args)


def test_no_included_source_is_an_error():
    quiet = AudioSource(source_id="room", path="/rec/room.wav", include_in_output=False)
    pipeline, _ = _make([quiet])
    with pytest.raises(ValueError, match="include_in_output"):
        pipeline._build_session_mix_args("/r/video.mp4", "/o/out.mp4")


def test_channel_and_gain_are_applied_per_source():
    src = AudioSource(source_id="x", path="/rec/x.wav", channel_index=2, gain_db=-6.0)
    pipeline, _ = _make([src, DAKOTA])
    flt = _filter(pipeline._build_session_mix_args("/r/video.mp4", "/o/out.mp4"))
    assert "pan=mono|c0=c2" in flt
    assert "volume=-6dB" in flt
    assert flt.index("pan=mono") < flt.index("volume=")
    assert flt.count("volume=") == 1  # dakota has default gain: no filter


# ---- concat stage ----------------------------------------------------------

def test_concat_muxes_session_mix_not_primary_camera_audio(tmp_path, monkeypatch, fake_ffmpeg):
    pipeline, errors = _make([TRAVIS, DAKOTA])
    for name in _LEGACY_HELPERS:
        monkeypatch.setattr(pipeline, name, _boom(name))
    out = str(tmp_path / "final.mp4")
    result = pipeline._stage_concat([_rendered(tmp_path)], out)
    assert result == out and not errors
    assert len(fake_ffmpeg.calls) == 1
    assert "amix" in _filter(fake_ffmpeg.calls[0])
    pipeline._cleanup.assert_called_once()


def test_concat_failure_stops_without_falling_back(tmp_path, monkeypatch, fake_ffmpeg):
    fake_ffmpeg.state.update(success=False, error="boom")
    pipeline, errors = _make([TRAVIS, DAKOTA])
    for name in _LEGACY_HELPERS:
        monkeypatch.setattr(pipeline, name, _boom(name))
    result = pipeline._stage_concat([_rendered(tmp_path)], str(tmp_path / "final.mp4"))
    assert result is None
    assert errors and "boom" in errors[0]
    pipeline._cleanup.assert_called_once()


def test_concat_reports_missing_ffmpeg(tmp_path, monkeypatch, fake_ffmpeg):
    fake_ffmpeg.state["available"] = False
    pipeline, errors = _make([TRAVIS])
    for name in _LEGACY_HELPERS:
        monkeypatch.setattr(pipeline, name, _boom(name))
    result = pipeline._stage_concat([_rendered(tmp_path)], str(tmp_path / "final.mp4"))
    assert result is None
    assert errors and "ffmpeg" in errors[0].lower()


def test_concat_reports_no_output_sources(tmp_path, monkeypatch, fake_ffmpeg):
    quiet = AudioSource(source_id="room", path="/rec/room.wav", include_in_output=False)
    pipeline, errors = _make([quiet])
    for name in _LEGACY_HELPERS:
        monkeypatch.setattr(pipeline, name, _boom(name))
    result = pipeline._stage_concat([_rendered(tmp_path)], str(tmp_path / "final.mp4"))
    assert result is None
    assert errors and "include_in_output" in errors[0]
    assert fake_ffmpeg.calls == []


def test_concat_legacy_path_still_uses_primary_camera_audio(tmp_path, monkeypatch):
    pipeline, errors = _make([])
    seen = {}

    def fake(video, primary, out):
        seen["primary"] = primary
        return out

    monkeypatch.setattr(pipeline, "_add_audio_from_video", fake)
    out = str(tmp_path / "final.mp4")
    assert pipeline._stage_concat([_rendered(tmp_path)], out) == out
    assert seen["primary"] == "/v/0.mp4"


# ---- real ffmpeg -----------------------------------------------------------

@pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg and ffprobe"
)
def test_real_mux_keeps_full_video_and_longest_audio(tmp_path):
    def ff(*a):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], check=True)

    video, a1, a2 = tmp_path / "v.mp4", tmp_path / "a1.wav", tmp_path / "a2.wav"
    ff("-f", "lavfi", "-i", "color=c=blue:s=64x64:r=10:d=4", "-c:v", "mpeg4", str(video))
    ff("-f", "lavfi", "-i", "sine=frequency=440:d=3", "-ar", "16000", str(a1))
    ff("-f", "lavfi", "-i", "sine=frequency=660:d=4", "-ar", "16000", str(a2))

    pipeline, errors = _make(
        [AudioSource(source_id="a", path=str(a1)), AudioSource(source_id="b", path=str(a2))]
    )
    out = tmp_path / "out.mp4"
    assert pipeline._add_session_audio(str(video), str(out)) == str(out), errors

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration",
         "-of", "json", str(out)],
        capture_output=True, text=True, check=True,
    )
    dur = {s["codec_type"]: float(s["duration"]) for s in json.loads(probe.stdout)["streams"]}
    assert dur["video"] == pytest.approx(4.0, abs=0.25)
    assert dur["audio"] == pytest.approx(4.0, abs=0.25)
