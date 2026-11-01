"""Tests for coarse whole-file synchronization by loudness-envelope correlation."""

import numpy as np
import pytest

from multicam_editor.logic.envelope_sync import (
    ENVELOPE_FPS,
    MIN_PEAK,
    combine_envelopes,
    loudness_envelope,
    match_envelopes,
)

SR = 16000
HOP = SR // ENVELOPE_FPS


def _bursts(rng, seconds):
    """Irregular noise bursts: speech-like loudness changes, no periodicity."""
    frames = int(seconds * ENVELOPE_FPS)
    activity = np.zeros(frames)
    for _ in range(int(seconds)):
        start = int(rng.integers(0, frames - 150))
        activity[start:start + int(rng.integers(20, 100))] = 1.0
    activity = np.convolve(activity, np.ones(8) / 8, mode="same")
    return rng.standard_normal(frames * HOP) * np.repeat(activity, HOP)


def _noise(rng, n_samples, level=0.05):
    return level * rng.standard_normal(n_samples)


def _camera_hearing(rng, session, lead_s=0.0, tail_s=5.0):
    """A camera that hears the session quietly, with room noise around it."""
    return np.concatenate([
        _noise(rng, int(round(lead_s * SR))),
        0.4 * session + _noise(rng, len(session)),
        _noise(rng, int(round(tail_s * SR))),
    ])


def test_envelope_has_100_frames_per_second_and_unit_norm():
    rng = np.random.default_rng(0)
    env = loudness_envelope(0.1 * rng.standard_normal(SR * 3), SR)
    assert len(env) == 3 * ENVELOPE_FPS
    assert np.linalg.norm(env) == pytest.approx(1.0)


def test_envelope_rejects_too_short_audio_and_odd_sample_rates():
    with pytest.raises(ValueError):
        loudness_envelope(np.zeros(100), SR)
    with pytest.raises(ValueError):
        loudness_envelope(np.zeros(SR), 22050)


def test_match_uses_the_audio_sync_sign_convention():
    rng = np.random.default_rng(1)
    reference = rng.standard_normal(6000)
    other = reference[300:5000]  # other's t=0 is 3 s into the reference

    forward = match_envelopes(reference, other)
    assert forward.offset_ms == pytest.approx(-3000.0)
    assert forward.reliable

    reverse = match_envelopes(other, reference)
    assert reverse.offset_ms == pytest.approx(3000.0)


def test_unrelated_envelopes_are_not_reliable():
    rng = np.random.default_rng(2)
    match = match_envelopes(rng.standard_normal(20000), rng.standard_normal(20000))
    assert not match.reliable


def test_match_requires_at_least_one_second_of_envelope():
    with pytest.raises(ValueError):
        match_envelopes(np.ones(50), np.ones(5000))
    with pytest.raises(ValueError):
        match_envelopes(np.ones(5000), np.ones(50))


def test_combine_truncates_to_shortest_and_normalizes():
    rng = np.random.default_rng(3)
    combined = combine_envelopes(
        [rng.standard_normal(1000), rng.standard_normal(800)]
    )
    assert len(combined) == 800
    assert np.linalg.norm(combined) == pytest.approx(1.0)


def test_combine_requires_input():
    with pytest.raises(ValueError):
        combine_envelopes([])


def test_camera_started_before_session_gives_negative_offset():
    """Real case: camera 2 was rolling 35.11 s before the session began."""
    rng = np.random.default_rng(42)
    session = _bursts(rng, 120)
    camera = _camera_hearing(rng, session, lead_s=35.11)

    match = match_envelopes(loudness_envelope(camera), loudness_envelope(session))

    assert match.offset_ms == pytest.approx(-35110.0, abs=20.0)
    assert match.peak > 0.5


def test_camera_started_after_session_gives_positive_offset():
    rng = np.random.default_rng(43)
    session = _bursts(rng, 120)
    camera = _camera_hearing(rng, session[int(7.5 * SR):])

    match = match_envelopes(loudness_envelope(camera), loudness_envelope(session))

    assert match.offset_ms == pytest.approx(7500.0, abs=20.0)
    assert match.peak > 0.5


def test_camera_covering_only_part_of_the_session_still_aligns():
    """Wide camera 4 recorded only the first few minutes of the show."""
    rng = np.random.default_rng(44)
    session = _bursts(rng, 180)
    camera = _camera_hearing(rng, session[: 60 * SR], lead_s=5.0, tail_s=0.0)

    match = match_envelopes(loudness_envelope(camera), loudness_envelope(session))

    assert match.offset_ms == pytest.approx(-5000.0, abs=20.0)
    assert match.peak > MIN_PEAK
