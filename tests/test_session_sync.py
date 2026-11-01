"""Tests for aligning cameras to a mixer session by loudness envelope."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from multicam_editor.logic.envelope_sync import ENVELOPE_FPS
from multicam_editor.logic.session_sync import (
    AudioDecodeError,
    align_cameras_to_session,
    decode_mono_16k,
)

SR = 16000
HOP = SR // ENVELOPE_FPS


def _bursts(rng, seconds):
    frames = int(seconds * ENVELOPE_FPS)
    activity = np.zeros(frames)
    for _ in range(int(seconds)):
        start = int(rng.integers(0, frames - 150))
        activity[start:start + int(rng.integers(20, 100))] = 1.0
    activity = np.convolve(activity, np.ones(8) / 8, mode="same")
    return rng.standard_normal(frames * HOP) * np.repeat(activity, HOP)


def _noise(rng, n_samples, level=0.05):
    return level * rng.standard_normal(n_samples)


def _camera(rng, sources, lead_s):
    mix = 0.4 * sum(sources)
    return np.concatenate([
        _noise(rng, int(round(lead_s * SR))),
        mix + _noise(rng, len(mix)),
        _noise(rng, 5 * SR),
    ])


def _decoder(audio, calls=None):
    def decode(path):
        if calls is not None:
            calls.append(path)
        value = audio[path]
        if isinstance(value, Exception):
            raise value
        return value
    return decode


def _run_ffmpeg_result(stdout=b"", stderr=b"", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_decode_returns_scaled_float_samples():
    pcm = np.array([0, 16384, -32768], dtype="<i2").tobytes()
    with patch("multicam_editor.logic.session_sync.subprocess.run",
               return_value=_run_ffmpeg_result(stdout=pcm)) as run:
        samples = decode_mono_16k("clip.mp4")

    np.testing.assert_allclose(samples, [0.0, 0.5, -1.0])
    command = run.call_args[0][0]
    assert command[0] == "ffmpeg" and "clip.mp4" in command
    assert "16000" in command and "-vn" in command


def test_decode_reports_ffmpeg_failure():
    with patch("multicam_editor.logic.session_sync.subprocess.run",
               return_value=_run_ffmpeg_result(stderr=b"boom", returncode=1)):
        with pytest.raises(AudioDecodeError, match="boom"):
            decode_mono_16k("clip.mp4")


def test_decode_rejects_empty_output():
    with patch("multicam_editor.logic.session_sync.subprocess.run",
               return_value=_run_ffmpeg_result(stdout=b"")):
        with pytest.raises(AudioDecodeError, match="no audio"):
            decode_mono_16k("silent.mp4")


def test_decode_reports_missing_ffmpeg():
    with patch("multicam_editor.logic.session_sync.subprocess.run",
               side_effect=FileNotFoundError()):
        with pytest.raises(AudioDecodeError, match="ffmpeg"):
            decode_mono_16k("clip.mp4")


def test_results_follow_camera_order_with_offsets():
    rng = np.random.default_rng(10)
    a, b = _bursts(rng, 120), _bursts(rng, 120)
    audio = {
        "s_a.wav": a,
        "s_b.wav": b,
        "cam0.mp4": _camera(rng, [a, b], lead_s=35.11),
        "cam1.mp4": _camera(rng, [a, b], lead_s=12.0),
    }

    results = align_cameras_to_session(
        ["cam0.mp4", "cam1.mp4"], ["s_a.wav", "s_b.wav"], decode=_decoder(audio)
    )

    assert [r.camera_index for r in results] == [0, 1]
    assert [r.path for r in results] == ["cam0.mp4", "cam1.mp4"]
    assert results[0].offset_ms == pytest.approx(-35110.0, abs=20.0)
    assert results[1].offset_ms == pytest.approx(-12000.0, abs=20.0)
    assert all(r.status == "ok" and r.reliable for r in results)


def test_unreliable_match_is_flagged_but_keeps_its_offset():
    rng = np.random.default_rng(11)
    a = _bursts(rng, 120)
    audio = {"s.wav": a, "cam.mp4": _noise(rng, 120 * SR, level=0.1)}

    (result,) = align_cameras_to_session(
        ["cam.mp4"], ["s.wav"], decode=_decoder(audio)
    )

    assert result.status == "unreliable"
    assert not result.reliable
    assert isinstance(result.offset_ms, float)
    assert "not trustworthy" in result.message


def test_one_camera_failure_does_not_stop_the_others():
    rng = np.random.default_rng(12)
    a = _bursts(rng, 120)
    audio = {
        "s.wav": a,
        "cam0.mp4": _camera(rng, [a], lead_s=20.0),
        "cam1.mp4": AudioDecodeError("cannot read cam1.mp4"),
        "cam2.mp4": _camera(rng, [a], lead_s=8.0),
    }

    results = align_cameras_to_session(
        ["cam0.mp4", "cam1.mp4", "cam2.mp4"], ["s.wav"], decode=_decoder(audio)
    )

    assert [r.status for r in results] == ["ok", "failed", "ok"]
    assert "cannot read cam1.mp4" in results[1].message
    assert results[1].offset_ms == 0.0 and not results[1].reliable
    assert results[2].offset_ms == pytest.approx(-8000.0, abs=20.0)


def test_session_decode_failure_fails_every_camera():
    rng = np.random.default_rng(13)
    audio = {
        "s0.wav": AudioDecodeError("cannot read s0.wav"),
        "cam0.mp4": _noise(rng, 10 * SR),
        "cam1.mp4": _noise(rng, 10 * SR),
    }

    results = align_cameras_to_session(
        ["cam0.mp4", "cam1.mp4"], ["s0.wav"], decode=_decoder(audio)
    )

    assert [r.status for r in results] == ["failed", "failed"]
    assert all("s0.wav" in r.message for r in results)


def test_camera_too_short_to_match_is_failed_not_raised():
    rng = np.random.default_rng(14)
    a = _bursts(rng, 120)
    audio = {"s.wav": a, "tiny.mp4": _noise(rng, SR // 2)}

    (result,) = align_cameras_to_session(
        ["tiny.mp4"], ["s.wav"], decode=_decoder(audio)
    )

    assert result.status == "failed"
    assert "one second" in result.message


def test_session_files_are_decoded_once_each():
    rng = np.random.default_rng(15)
    a, b = _bursts(rng, 60), _bursts(rng, 60)
    audio = {
        "s_a.wav": a,
        "s_b.wav": b,
        "cam0.mp4": _camera(rng, [a, b], lead_s=3.0),
        "cam1.mp4": _camera(rng, [a, b], lead_s=6.0),
    }
    calls = []

    align_cameras_to_session(
        ["cam0.mp4", "cam1.mp4"], ["s_a.wav", "s_b.wav"],
        decode=_decoder(audio, calls),
    )

    assert calls.count("s_a.wav") == 1
    assert calls.count("s_b.wav") == 1
    assert calls.count("cam0.mp4") == 1


def test_requires_at_least_one_session_file():
    with pytest.raises(ValueError):
        align_cameras_to_session(["cam.mp4"], [], decode=_decoder({}))
