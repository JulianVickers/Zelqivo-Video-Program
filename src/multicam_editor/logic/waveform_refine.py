"""Sample-level refinement of a coarse session-to-camera offset.

``envelope_sync`` finds where a session sits in a camera file to within about
10 ms by comparing loudness curves. This module narrows that to a single
sample by correlating raw waveforms over a few short, loud windows, searching
only a small range around the coarse answer.

Sign convention (same as ``audio_sync`` and ``envelope_sync``): ``offset_ms``
is -(k / sample_rate) in ms, where ``camera[n + k]`` lines up with
``session[n]``. An offset of -35110 means the session's time zero falls
35.11 s into the camera file.

Correlation is energy-normalized (1.0 is a perfect match), unlike the app's
older score. A refinement is accepted only when every window that clears
``min_peak`` agrees within ``max_spread_ms``; otherwise the coarse offset is
returned unchanged. The room mic and a close mic hear different waveforms,
so a weak result is expected on some material and is reported, not hidden.

Pure computation on mono float arrays: no files, no ffmpeg.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Provisional. Not yet tuned against real room-mic recordings.
MIN_REFINE_PEAK = 0.3
# Windows dominated by different speakers sit at different distances from a
# room mic, so a few ms of real disagreement is physical. Lip sync only
# becomes noticeable at tens of ms.
MAX_SPREAD_MS = 25.0


@dataclass(frozen=True)
class RefinedOffset:
    """Outcome of refining a coarse offset.

    When ``refined`` is False, ``offset_ms`` equals ``coarse_offset_ms`` and
    ``shift_ms`` is 0. ``windows_used`` counts windows that cleared the
    correlation floor; ``window_starts_s`` lists every window tried.
    """

    coarse_offset_ms: float
    offset_ms: float
    shift_ms: float
    peak: float
    windows_used: int
    spread_ms: float
    window_starts_s: tuple[float, ...]
    refined: bool
    message: str


def mix_sources(sources: Sequence[np.ndarray]) -> np.ndarray:
    """Sum sources that share a start time, truncated to the shortest."""
    if not len(sources):
        raise ValueError("at least one source is required")
    length = min(len(s) for s in sources)
    total = sum(np.asarray(s, dtype=np.float64)[:length] for s in sources)
    return total / (np.max(np.abs(total)) + 1e-12)


def _normalized_xcorr(segment: np.ndarray, template: np.ndarray) -> np.ndarray:
    """NCC of ``template`` against every position inside ``segment``."""
    n_s, n_t = len(segment), len(template)
    size = 1 << int(np.ceil(np.log2(n_s + n_t)))
    corr = np.fft.irfft(
        np.fft.rfft(segment, size) * np.conj(np.fft.rfft(template, size)), size
    )
    raw = corr[: n_s - n_t + 1]
    cumulative = np.concatenate([[0.0], np.cumsum(segment ** 2)])
    energy = cumulative[n_t:] - cumulative[:-n_t]
    denominator = np.sqrt(np.maximum(energy, 0.0)) * np.linalg.norm(template) + 1e-12
    return raw / denominator


def _loud_window_starts(
    session: np.ndarray,
    camera_len: int,
    k0: int,
    window: int,
    search: int,
    step: int,
    max_windows: int,
) -> list[int]:
    """Start samples of the loudest non-overlapping windows both files cover."""
    lo = max(0, search - k0)
    hi = min(len(session) - window, camera_len - k0 - window - search)
    if hi < lo:
        return []

    starts = np.arange(lo, hi + 1, step)
    cumulative = np.concatenate([[0.0], np.cumsum(session ** 2)])
    energy = cumulative[starts + window] - cumulative[starts]

    chosen: list[int] = []
    for i in np.argsort(-energy, kind="stable"):
        if energy[i] <= 0:
            break
        start = int(starts[i])
        if all(abs(start - other) >= window for other in chosen):
            chosen.append(start)
            if len(chosen) == max_windows:
                break
    return sorted(chosen)


def refine_offset(
    camera: np.ndarray,
    session: np.ndarray,
    coarse_offset_ms: float,
    sample_rate: int = 16000,
    window_s: float = 10.0,
    search_ms: float = 100.0,
    max_windows: int = 3,
    min_peak: float = MIN_REFINE_PEAK,
    max_spread_ms: float = MAX_SPREAD_MS,
) -> RefinedOffset:
    """Refine ``coarse_offset_ms`` using waveform correlation (see module docs)."""
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    if search_ms < 0:
        raise ValueError("search_ms must not be negative")
    if max_windows < 1:
        raise ValueError("max_windows must be at least 1")

    camera = np.asarray(camera, dtype=np.float64)
    session = np.asarray(session, dtype=np.float64)

    window = int(round(window_s * sample_rate))
    search = int(round(search_ms * sample_rate / 1000.0))
    k0 = int(round(-coarse_offset_ms * sample_rate / 1000.0))

    def unrefined(message, peak=0.0, used=0, spread=0.0, starts=()):
        return RefinedOffset(
            coarse_offset_ms=coarse_offset_ms,
            offset_ms=coarse_offset_ms,
            shift_ms=0.0,
            peak=peak,
            windows_used=used,
            spread_ms=spread,
            window_starts_s=tuple(starts),
            refined=False,
            message=message,
        )

    starts = _loud_window_starts(
        session, len(camera), k0, window, search, sample_rate, max_windows
    )
    if not starts:
        return unrefined("no loud window is covered by both recordings")
    starts_s = [s / sample_rate for s in starts]

    shifts, peaks, best_overall = [], [], 0.0
    for start in starts:
        template = session[start:start + window]
        begin = start + k0 - search
        segment = camera[begin:begin + window + 2 * search]
        ncc = _normalized_xcorr(segment, template)
        best = int(np.argmax(ncc))
        best_overall = max(best_overall, float(ncc[best]))
        if ncc[best] >= min_peak:
            shifts.append(best - search)
            peaks.append(float(ncc[best]))

    if not shifts:
        return unrefined(
            f"no window reached correlation {min_peak:.2f} "
            f"(best {best_overall:.2f}); keeping coarse offset",
            peak=best_overall,
            starts=starts_s,
        )

    spread_ms = (max(shifts) - min(shifts)) * 1000.0 / sample_rate
    if spread_ms > max_spread_ms:
        return unrefined(
            f"windows disagree by {spread_ms:.1f} ms; keeping coarse offset",
            peak=max(peaks),
            used=len(shifts),
            spread=spread_ms,
            starts=starts_s,
        )

    k = k0 + int(round(float(np.median(shifts))))
    offset_ms = -k * 1000.0 / sample_rate
    return RefinedOffset(
        coarse_offset_ms=coarse_offset_ms,
        offset_ms=offset_ms,
        shift_ms=offset_ms - coarse_offset_ms,
        peak=max(peaks),
        windows_used=len(shifts),
        spread_ms=spread_ms,
        window_starts_s=tuple(starts_s),
        refined=True,
        message=(
            f"refined by {offset_ms - coarse_offset_ms:+.2f} ms "
            f"from {len(shifts)} window(s)"
        ),
    )
