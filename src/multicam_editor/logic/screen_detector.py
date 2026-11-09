"""Find the stretches of a screen-capture file that show the marker.

A frame counts as marker when it is flat (near-zero spread) and its mean Y/U/V
sits within a tolerance of the configured colour. Flatness is what separates a
deliberate marker from a dark, noisy camera picture or a busy screen.

The file is scanned once, sequentially, at a low sample rate on downscaled
frames. Per-sample flags are then debounced by run length: a run must last
``enter_s`` before it counts (noise rejection only, not a minimum dwell), and
it ends only after ``exit_s`` of non-marker samples. Interval starts are
backdated to the first marker sample of the run; ends are the end of the last
marker sample. Times are relative to the first sampled frame (t = 0); placing
them on the session timeline is the caller's job.
"""
from __future__ import annotations

import logging
import math
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

SCAN_FPS = 10.0
SCAN_WIDTH = 64
SCAN_HEIGHT = 36
ENTER_S = 0.5
EXIT_S = 2.0
DEFAULT_TOLERANCE = 12.0
DEFAULT_MAX_FLAT_STD = 3.0

_FRAME_BYTES = 3 * SCAN_WIDTH * SCAN_HEIGHT


class ScreenScanError(RuntimeError):
    """The screen-capture file could not be scanned."""


class ScreenScanCancelled(Exception):
    """The scan was stopped by the caller's cancel callback."""


@dataclass(frozen=True)
class MarkerColor:
    """Marker colour as mean Y/U/V (0-255) with one per-channel tolerance."""

    y: float
    u: float
    v: float
    tolerance: float = DEFAULT_TOLERANCE

    def __post_init__(self):
        if self.tolerance < 0:
            raise ValueError("tolerance must be >= 0")


@dataclass(frozen=True)
class FrameStats:
    """Mean Y/U/V of one sampled frame and its flatness (max per-plane std)."""

    y: float
    u: float
    v: float
    flat_std: float


@dataclass(frozen=True)
class ScreenInterval:
    """A stretch of the file showing the marker, in ms from the first sample."""

    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def frame_stats(frame: np.ndarray) -> FrameStats:
    """Stats for one planar YUV 4:4:4 frame shaped (3, height, width)."""
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"expected a (3, height, width) frame, got shape {arr.shape}")
    planes = arr.astype(np.float32).reshape(3, -1)
    means = planes.mean(axis=1)
    return FrameStats(
        float(means[0]), float(means[1]), float(means[2]), float(planes.std(axis=1).max())
    )


def is_marker(
    stats: FrameStats, marker: MarkerColor, max_flat_std: float = DEFAULT_MAX_FLAT_STD
) -> bool:
    """True when the frame is flat and within tolerance of the marker colour."""
    return (
        stats.flat_std <= max_flat_std
        and abs(stats.y - marker.y) <= marker.tolerance
        and abs(stats.u - marker.u) <= marker.tolerance
        and abs(stats.v - marker.v) <= marker.tolerance
    )


def _samples_for(seconds: float, fps: float) -> int:
    return max(1, math.ceil(seconds * fps - 1e-9))


def intervals_from_flags(
    flags: Sequence[bool], fps: float, enter_s: float = ENTER_S, exit_s: float = EXIT_S
) -> list[ScreenInterval]:
    """Debounce per-sample marker flags into backdated intervals."""
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if enter_s < 0 or exit_s < 0:
        raise ValueError("enter_s and exit_s must be >= 0")
    enter_n = _samples_for(enter_s, fps)
    exit_n = _samples_for(exit_s, fps)

    def ms(index: int) -> int:
        return int(round(index * 1000.0 / fps))

    intervals: list[ScreenInterval] = []
    active = False
    run_start: Optional[int] = None
    last_marker = 0
    misses = 0
    for i, flag in enumerate(flags):
        if not active:
            if flag:
                if run_start is None:
                    run_start = i
                if i - run_start + 1 >= enter_n:
                    active = True
                    last_marker = i
                    misses = 0
            else:
                run_start = None
        elif flag:
            last_marker = i
            misses = 0
        else:
            misses += 1
            if misses >= exit_n:
                intervals.append(ScreenInterval(ms(run_start), ms(last_marker + 1)))
                active = False
                run_start = None
    if active:
        intervals.append(ScreenInterval(ms(run_start), ms(last_marker + 1)))
    return intervals


def sample_frame_stats(
    path: str,
    fps: float = SCAN_FPS,
    cancel_callback: Optional[Callable[[], bool]] = None,
    timeout: float = 900.0,
) -> list[FrameStats]:
    """Stream downscaled frames from ffmpeg and return per-sample stats.

    Frames are read incrementally (not buffered whole) so long recordings stay
    cheap and ``cancel_callback`` is honoured between frames.
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")
    command = [
        "ffmpeg", "-loglevel", "error", "-i", path,
        "-map", "0:v:0", "-an",
        "-vf", f"fps={fps:g},scale={SCAN_WIDTH}:{SCAN_HEIGHT}:flags=area",
        "-pix_fmt", "yuv444p", "-f", "rawvideo", "-",
    ]
    stats: list[FrameStats] = []
    with tempfile.TemporaryFile() as err:
        try:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=err)
        except FileNotFoundError as exc:
            raise ScreenScanError("ffmpeg was not found on this system") from exc
        deadline = time.monotonic() + timeout
        completed = False
        try:
            while True:
                if cancel_callback is not None and cancel_callback():
                    raise ScreenScanCancelled(f"scan of {path} cancelled")
                if time.monotonic() > deadline:
                    raise ScreenScanError(f"scanning {path} timed out after {timeout:.0f}s")
                raw = proc.stdout.read(_FRAME_BYTES)
                if len(raw) < _FRAME_BYTES:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(3, SCAN_HEIGHT, SCAN_WIDTH)
                stats.append(frame_stats(frame))
            returncode = proc.wait()
            completed = True
        finally:
            if not completed:
                proc.kill()
                proc.wait()
            proc.stdout.close()
        err.seek(0)
        detail = err.read().decode("utf-8", "replace").strip()

    if returncode != 0:
        raise ScreenScanError(
            f"ffmpeg could not scan {path}: {detail or f'exit code {returncode}'}"
        )
    if not stats:
        raise ScreenScanError(f"no video frames decoded from {path}")
    return stats


def scan_screen_intervals(
    path: str,
    marker: MarkerColor,
    *,
    fps: float = SCAN_FPS,
    enter_s: float = ENTER_S,
    exit_s: float = EXIT_S,
    max_flat_std: float = DEFAULT_MAX_FLAT_STD,
    cancel_callback: Optional[Callable[[], bool]] = None,
    timeout: float = 900.0,
) -> list[ScreenInterval]:
    """Scan ``path`` and return the debounced marker intervals."""
    stats = sample_frame_stats(path, fps, cancel_callback, timeout)
    flags = [is_marker(s, marker, max_flat_std) for s in stats]
    intervals = intervals_from_flags(flags, fps, enter_s, exit_s)
    logger.debug("screen scan %s: %d samples, %d intervals", path, len(stats), len(intervals))
    return intervals
