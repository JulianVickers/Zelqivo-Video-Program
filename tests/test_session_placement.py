"""Tests for placing cameras on the mixer-session timeline."""
from __future__ import annotations

import numpy as np
import pytest

from multicam_editor.logic import session_placement as sp
from multicam_editor.logic.session_placement import (
    SessionPlacementError,
    place_cameras_on_session,
)
from multicam_editor.logic.session_sync import AudioDecodeError
from multicam_editor.logic.waveform_refine import RefinedOffset

SR = 16000


def speechlike(n_samples: int, seed: int) -> np.ndarray:
    """Noise with a random loudness contour, so envelopes are distinctive."""
    rng = np.random.default_rng(seed)
    block = int(0.4 * SR)
    levels = np.exp(rng.uniform(np.log(0.01), np.log(1.0), size=n_samples // block + 1))
    contour = np.repeat(levels, block)[:n_samples]
    return (rng.standard_normal(n_samples) * contour).astype(np.float32)


def camera_recording(session: np.ndarray, preroll_samples: int, seed: int) -> np.ndarray:
    """Camera that started ``preroll_samples`` before the session.

    camera[n + k] == session[n] (scaled), k = preroll_samples,
    so the expected session offset is -k / SR * 1000 ms.
    """
    rng = np.random.default_rng(seed)
    pre = speechlike(preroll_samples, seed=seed + 1000)
    body = 0.5 * session + 0.01 * rng.standard_normal(len(session)).astype(np.float32)
    return np.concatenate([pre, body]).astype(np.float32)


class FakeAudio:
    """Decode stand-in: path -> array, or an exception to raise."""

    def __init__(self, files):
        self.files = files
        self.calls: list[str] = []

    def __call__(self, path):
        self.calls.append(path)
        value = self.files[path]
        if isinstance(value, Exception):
            raise value
        return value


def test_places_camera_and_refines_to_sample_accuracy():
    session = speechlike(60 * SR, seed=1)
    preroll = 112_048  # 7.003 s, deliberately off the 10 ms envelope grid
    audio = FakeAudio({
        "/x/cam.mp4": camera_recording(session, preroll, seed=2),
        "/x/s.wav": session,
    })

    placement = place_cameras_on_session(["/x/cam.mp4"], ["/x/s.wav"], decode=audio)

    cam = placement.cameras[0]
    assert cam.camera_index == 0
    assert cam.status == "ok"
    assert cam.refined is True
    assert cam.session_offset_ms == pytest.approx(-7003.0, abs=1.0)
    assert cam.slot_offset_ms == pytest.approx(7003.0, abs=1.0)
    assert placement.session_duration_ms == 60_000
    assert placement.warnings == ()
    assert placement.failures == ()


def test_two_sources_early_camera_and_late_camera():
    a = speechlike(60 * SR, seed=11)
    b = speechlike(60 * SR, seed=12)
    mix = a + b
    early = camera_recording(mix, 48_000, seed=3)  # starts 3 s before the show
    rng = np.random.default_rng(4)
    late = (0.5 * mix[80_000:] + 0.01 * rng.standard_normal(len(mix) - 80_000)).astype(np.float32)
    audio = FakeAudio({
        "/x/1.mp4": early,
        "/x/2.mp4": late,  # starts 5 s into the show
        "/x/a.wav": a,
        "/x/b.wav": b,
    })

    placement = place_cameras_on_session(
        ["/x/1.mp4", "/x/2.mp4"], ["/x/a.wav", "/x/b.wav"], decode=audio
    )

    c0, c1 = placement.cameras
    assert (c0.camera_index, c1.camera_index) == (0, 1)
    assert c0.status == c1.status == "ok"
    assert c0.session_offset_ms == pytest.approx(-3000.0, abs=1.0)
    assert c0.slot_offset_ms == pytest.approx(3000.0, abs=1.0)
    assert c1.session_offset_ms == pytest.approx(5000.0, abs=1.0)
    assert c1.slot_offset_ms == pytest.approx(-5000.0, abs=1.0)


def test_unrelated_camera_is_not_placed_and_error_names_it():
    session = speechlike(60 * SR, seed=21)
    audio = FakeAudio({
        "/shoot/unrelated.mp4": speechlike(70 * SR, seed=22),
        "/shoot/s.wav": session,
    })

    placement = place_cameras_on_session(
        ["/shoot/unrelated.mp4"], ["/shoot/s.wav"], decode=audio
    )

    cam = placement.cameras[0]
    assert cam.status in ("unreliable", "failed")
    assert placement.failures == (cam,)
    with pytest.raises(SessionPlacementError) as excinfo:
        placement.require_all_placed()
    text = str(excinfo.value)
    assert "unrelated.mp4" in text
    assert cam.status in text
    assert "peak" in text.lower()


def test_one_bad_camera_does_not_block_the_others():
    session = speechlike(60 * SR, seed=31)
    audio = FakeAudio({
        "/x/1.mp4": camera_recording(session, 64_000, seed=32),
        "/x/2.mp4": speechlike(65 * SR, seed=33),
        "/x/3.mp4": AudioDecodeError("no audio stream"),
        "/x/4.mp4": camera_recording(session, 32_000, seed=34),
        "/x/s.wav": session,
    })

    placement = place_cameras_on_session(
        ["/x/1.mp4", "/x/2.mp4", "/x/3.mp4", "/x/4.mp4"], ["/x/s.wav"], decode=audio
    )

    c1, c2, c3, c4 = placement.cameras
    assert c1.status == "ok"
    assert c1.session_offset_ms == pytest.approx(-4000.0, abs=1.0)
    assert c2.status != "ok"
    assert c3.status == "failed"
    assert "no audio stream" in c3.message
    assert c3.session_offset_ms == 0.0
    assert c3.slot_offset_ms == 0.0
    assert c4.status == "ok"
    assert c4.session_offset_ms == pytest.approx(-2000.0, abs=1.0)
    assert [c.camera_index for c in placement.failures] == [1, 2]


def test_undecodable_source_raises():
    audio = FakeAudio({
        "/x/cam.mp4": speechlike(10 * SR, seed=41),
        "/x/s.wav": AudioDecodeError("corrupt wav"),
    })
    with pytest.raises(SessionPlacementError, match="s.wav"):
        place_cameras_on_session(["/x/cam.mp4"], ["/x/s.wav"], decode=audio)


def test_no_sources_is_a_programming_error():
    audio = FakeAudio({"/x/cam.mp4": speechlike(10 * SR, seed=42)})
    with pytest.raises(ValueError):
        place_cameras_on_session(["/x/cam.mp4"], [], decode=audio)


@pytest.mark.parametrize("mode", ["rejected", "raises"])
def test_refine_failure_keeps_coarse_offset_and_warns(monkeypatch, mode):
    session = speechlike(60 * SR, seed=51)
    audio = FakeAudio({
        "/x/cam.mp4": camera_recording(session, 64_000, seed=52),
        "/x/s.wav": session,
    })

    def fake_refine(camera, session_audio, coarse_offset_ms, **kwargs):
        if mode == "raises":
            raise RuntimeError("refine exploded")
        return RefinedOffset(
            coarse_offset_ms=coarse_offset_ms,
            offset_ms=coarse_offset_ms,
            shift_ms=0.0,
            peak=0.1,
            windows_used=1,
            spread_ms=40.0,
            window_starts_s=(1.0,),
            refined=False,
            message="window spread too large",
        )

    monkeypatch.setattr(sp, "refine_offset", fake_refine)

    placement = place_cameras_on_session(["/x/cam.mp4"], ["/x/s.wav"], decode=audio)

    cam = placement.cameras[0]
    assert cam.status == "ok"
    assert cam.refined is False
    assert cam.session_offset_ms == pytest.approx(-4000.0, abs=10.0)
    assert cam.slot_offset_ms == pytest.approx(-cam.session_offset_ms)
    assert placement.failures == ()
    assert len(placement.warnings) == 1
    assert "cam.mp4" in placement.warnings[0]


def test_each_file_is_decoded_once():
    session = speechlike(60 * SR, seed=61)
    audio = FakeAudio({
        "/x/1.mp4": camera_recording(session, 64_000, seed=62),
        "/x/2.mp4": camera_recording(session, 32_000, seed=63),
        "/x/s.wav": session,
    })

    place_cameras_on_session(
        ["/x/1.mp4", "/x/2.mp4"], ["/x/s.wav", "/x/s.wav"], decode=audio
    )

    assert audio.calls.count("/x/s.wav") == 1
    assert audio.calls.count("/x/1.mp4") == 1
    assert audio.calls.count("/x/2.mp4") == 1


def test_source_length_mismatch_warns_and_uses_longest():
    a = speechlike(60 * SR, seed=71)
    b = speechlike(50 * SR, seed=72)
    mix = a.copy()
    mix[: len(b)] += b
    audio = FakeAudio({
        "/x/cam.mp4": camera_recording(mix, 64_000, seed=73),
        "/x/a.wav": a,
        "/x/b.wav": b,
    })

    placement = place_cameras_on_session(
        ["/x/cam.mp4"], ["/x/a.wav", "/x/b.wav"], decode=audio
    )

    assert placement.session_duration_ms == 60_000
    assert any("duration" in w.lower() for w in placement.warnings)
