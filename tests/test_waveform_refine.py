"""Tests for sample-level refinement of a coarse session offset."""

import numpy as np
import pytest

from multicam_editor.logic.waveform_refine import mix_sources, refine_offset

SR = 16000


def _camera(rng, session, k, tail_s=3.0, level=0.5, noise=0.05):
    """A camera that hears the session ``k`` samples into its own file."""
    cam = noise * rng.standard_normal(k + len(session) + int(tail_s * SR))
    cam[k:k + len(session)] += level * session
    return cam


def test_refines_to_sample_accuracy():
    rng = np.random.default_rng(20)
    session = 0.3 * rng.standard_normal(60 * SR)
    k = 5 * SR + 37  # true position; the coarse estimate is 2.3 ms off

    result = refine_offset(_camera(rng, session, k), session, coarse_offset_ms=-5000.0)

    assert result.refined
    assert result.offset_ms == pytest.approx(-k * 1000.0 / SR, abs=0.01)
    assert result.shift_ms == pytest.approx(-2.3125, abs=0.01)
    assert result.peak > 0.9
    assert result.windows_used == 3


def test_camera_that_started_after_the_session():
    rng = np.random.default_rng(21)
    session = 0.3 * rng.standard_normal(60 * SR)
    m = 7 * SR + 21  # camera t=0 is m samples into the session
    camera = 0.5 * session[m:] + 0.05 * rng.standard_normal(len(session) - m)

    result = refine_offset(camera, session, coarse_offset_ms=7000.0)

    assert result.refined
    assert result.offset_ms == pytest.approx(m * 1000.0 / SR, abs=0.01)


def test_unrelated_audio_keeps_the_coarse_offset():
    rng = np.random.default_rng(22)
    session = 0.3 * rng.standard_normal(60 * SR)
    camera = 0.3 * rng.standard_normal(70 * SR)

    result = refine_offset(camera, session, coarse_offset_ms=-5000.0)

    assert not result.refined
    assert result.offset_ms == -5000.0
    assert result.shift_ms == 0.0
    assert "correlation" in result.message


def test_no_usable_window_when_camera_cannot_cover_one():
    rng = np.random.default_rng(23)
    session = 0.3 * rng.standard_normal(60 * SR)
    camera = 0.3 * rng.standard_normal(5 * SR)

    result = refine_offset(camera, session, coarse_offset_ms=0.0)

    assert not result.refined
    assert result.windows_used == 0
    assert "window" in result.message


def test_coarse_error_beyond_search_range_is_not_refined():
    rng = np.random.default_rng(24)
    session = 0.3 * rng.standard_normal(60 * SR)
    k = 5 * SR + int(0.3 * SR)  # 300 ms away from the coarse estimate

    result = refine_offset(
        _camera(rng, session, k), session, coarse_offset_ms=-5000.0, search_ms=100.0
    )

    assert not result.refined
    assert result.offset_ms == -5000.0


def test_prefers_loud_session_windows():
    rng = np.random.default_rng(25)
    session = np.concatenate([
        0.001 * rng.standard_normal(30 * SR),  # near-silent first half
        1.0 * rng.standard_normal(30 * SR),
    ])
    k = 3 * SR + 11

    result = refine_offset(
        _camera(rng, session, k), session, coarse_offset_ms=-3000.0, max_windows=1
    )

    assert result.refined
    assert result.offset_ms == pytest.approx(-k * 1000.0 / SR, abs=0.01)
    assert 30.0 <= result.window_starts_s[0] <= 50.0


def test_disagreeing_windows_are_not_refined():
    """Two loud passages that imply different offsets cannot both be right."""
    rng = np.random.default_rng(26)
    session = np.zeros(60 * SR)
    session[5 * SR:15 * SR] = 0.3 * rng.standard_normal(10 * SR)
    session[35 * SR:45 * SR] = 0.3 * rng.standard_normal(10 * SR)

    k = 5 * SR
    camera = 0.05 * rng.standard_normal(k + len(session) + 800 + 3 * SR)
    camera[k:k + 25 * SR] += 0.5 * session[:25 * SR]
    camera[k + 800 + 25 * SR:k + 800 + len(session)] += 0.5 * session[25 * SR:]

    result = refine_offset(camera, session, coarse_offset_ms=-5000.0, max_windows=2)

    assert not result.refined
    assert result.offset_ms == -5000.0
    assert result.spread_ms == pytest.approx(50.0, abs=0.5)
    assert "disagree" in result.message


def test_mix_sources_truncates_and_peak_normalizes():
    rng = np.random.default_rng(27)
    mixed = mix_sources([rng.standard_normal(100), 3.0 * rng.standard_normal(80)])

    assert len(mixed) == 80
    assert np.max(np.abs(mixed)) == pytest.approx(1.0)


def test_mix_sources_requires_input():
    with pytest.raises(ValueError):
        mix_sources([])


def test_invalid_parameters_are_rejected():
    audio = np.zeros(20 * SR)
    with pytest.raises(ValueError):
        refine_offset(audio, audio, 0.0, window_s=0.0)
    with pytest.raises(ValueError):
        refine_offset(audio, audio, 0.0, search_ms=-1.0)
    with pytest.raises(ValueError):
        refine_offset(audio, audio, 0.0, max_windows=0)
