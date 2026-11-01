"""Coarse whole-file synchronization by loudness-envelope correlation.

Waveform cross-correlation over a short opening window (see ``audio_sync``)
only works when two recordings start within a few seconds of each other.
Hand-started cameras and a separately recorded mixer session can be minutes
apart, so no common moment exists inside the window and the match is noise.

This module compares slow loudness curves over the *whole* recording instead.
Speech makes both a close mic and a room mic get louder and quieter together,
whatever the waveforms look like. Resolution is one envelope frame (10 ms);
a later waveform refinement can narrow that.

Sign convention (same as ``audio_sync``): ``offset_ms`` positive means the
second recording's content appears LATER in its own file than in the
reference; negative means EARLIER. Real example: a camera was already
recording when a mixer session began, and the session's time zero falls
35.11 s into the camera file. With the camera as reference and the session as
the other recording, ``offset_ms`` is -35110.

Inputs are mono float samples in [-1, 1]. Pure computation: no files, no
ffmpeg.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

ENVELOPE_FPS = 100

# Provisional decision thresholds. Real recordings with a confirmed match
# scored peak 0.26-0.86 and peak_ratio 4.8-17; unrelated envelopes score a
# peak near 0.03 and a ratio near 1.
MIN_PEAK = 0.15
MIN_PEAK_RATIO = 3.0

_DRIFT_WINDOW_FRAMES = 200  # 2 s moving average removed from the log envelope
_EXCLUSION_FRAMES = 100     # runner-up peaks must be > 1 s from the winner
_MIN_FRAMES = 100           # need at least 1 s of envelope to match


@dataclass(frozen=True)
class EnvelopeMatch:
    """Result of matching two loudness envelopes.

    ``peak`` is the normalized correlation at the best lag (1.0 is perfect).
    ``peak_ratio`` is that peak divided by the best competing peak more than
    one second away, so a sharp, unique match scores high.
    """

    offset_ms: float
    peak: float
    peak_ratio: float
    reliable: bool


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / (np.linalg.norm(values) + 1e-12)


def loudness_envelope(samples: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Return a unit-norm log-loudness curve with ``ENVELOPE_FPS`` frames/s."""
    if sample_rate % ENVELOPE_FPS != 0:
        raise ValueError(
            f"sample_rate must be a multiple of {ENVELOPE_FPS}, got {sample_rate}"
        )
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("samples must be mono (one-dimensional)")

    hop = sample_rate // ENVELOPE_FPS
    frames = len(x) // hop
    if frames < 1:
        raise ValueError("audio is shorter than one envelope frame")

    rms = np.sqrt((x[: frames * hop].reshape(frames, hop) ** 2).mean(axis=1))
    env = np.log(rms + 1e-4)
    window = min(_DRIFT_WINDOW_FRAMES, frames)
    env = env - np.convolve(env, np.ones(window) / window, mode="same")
    return _unit(env)


def combine_envelopes(envelopes: Sequence[np.ndarray]) -> np.ndarray:
    """Sum several envelopes that share a start time into one unit-norm curve.

    Sources exported from one mixer session start together, so they are
    truncated to the shortest and summed frame by frame. The mix of all
    sources matches a room microphone better than any single source does.
    """
    if not len(envelopes):
        raise ValueError("at least one envelope is required")
    length = min(len(e) for e in envelopes)
    total = sum(_unit(np.asarray(e)[:length]) for e in envelopes)
    return _unit(total)


def match_envelopes(
    reference: np.ndarray,
    other: np.ndarray,
    min_peak: float = MIN_PEAK,
    min_peak_ratio: float = MIN_PEAK_RATIO,
) -> EnvelopeMatch:
    """Find where ``other`` sits relative to ``reference`` (see module docs)."""
    ref = _unit(reference)
    ext = _unit(other)
    if len(ref) < _MIN_FRAMES or len(ext) < _MIN_FRAMES:
        raise ValueError("each envelope needs at least one second of audio")

    size = 1 << int(np.ceil(np.log2(len(ref) + len(ext))))
    corr = np.fft.irfft(
        np.fft.rfft(ref, size) * np.conj(np.fft.rfft(ext, size)), size
    )
    # corr at lag k is sum(ref[n + k] * ext[n]); lags run -(len(ext)-1)..len(ref)-1
    lags = np.arange(-(len(ext) - 1), len(ref))
    values = corr[lags % size]

    best = int(np.argmax(values))
    peak = float(values[best])
    far = np.abs(lags - lags[best]) > _EXCLUSION_FRAMES
    runner_up = float(values[far].max()) if far.any() else 0.0
    ratio = peak / max(runner_up, 1e-9) if peak > 0 else 0.0

    return EnvelopeMatch(
        offset_ms=0.0 - float(lags[best]) * 1000.0 / ENVELOPE_FPS,
        peak=peak,
        peak_ratio=ratio,
        reliable=peak >= min_peak and ratio >= min_peak_ratio,
    )
