"""Place cameras on the mixer-session timeline.

With configured audio sources, the sources form one session whose start is
show time zero. Each camera is matched coarsely to the session by loudness
envelope (session_sync) and then refined to sample level by waveform
correlation (waveform_refine).

Failure policy:
  * A failed or unreliable *coarse* match is a failure; the caller decides
    what to do (the pipeline stops, naming the camera and the numbers).
  * A rejected or crashed *refinement* keeps the coarse offset and adds a
    warning. The coarse match is already trustworthy.
  * An undecodable *source* raises SessionPlacementError: nothing can be
    placed without the session.

Sign conventions (see session_sync): ``session_offset_ms`` of -35110 means
the session's time zero is 35.11 s into the camera file. ``slot_offset_ms``
is the negation, in the convention of the pipeline's ``_camera_offsets``
(camera_time = timeline_time + slot_offset).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .envelope_sync import MIN_PEAK, MIN_PEAK_RATIO
from .session_sync import SAMPLE_RATE, align_cameras_to_session, decode_mono_16k
from .timeline_ruler import session_offset_to_seek_offset
from .waveform_refine import mix_sources, refine_offset

logger = logging.getLogger(__name__)

# Sources that differ in length by more than this probably do not share a start.
SOURCE_LENGTH_TOLERANCE_MS = 1000


class SessionPlacementError(RuntimeError):
    """A camera or audio source could not be placed on the session timeline."""


@dataclass(frozen=True)
class CameraPlacement:
    camera_index: int
    path: str
    status: str  # "ok" | "unreliable" | "failed"
    session_offset_ms: float
    slot_offset_ms: float
    peak: float
    peak_ratio: float
    refined: bool
    refine_shift_ms: float
    message: str


@dataclass(frozen=True)
class SessionPlacement:
    cameras: tuple[CameraPlacement, ...]
    session_duration_ms: int
    warnings: tuple[str, ...] = ()

    @property
    def failures(self) -> tuple[CameraPlacement, ...]:
        return tuple(c for c in self.cameras if c.status != "ok")

    def require_all_placed(self) -> None:
        """Raise SessionPlacementError naming every camera that is not placed."""
        failed = self.failures
        if not failed:
            return
        lines = [
            f"{len(failed)} camera(s) could not be placed on the session timeline:"
        ]
        for cam in failed:
            line = (
                f"  {os.path.basename(cam.path)}: {cam.status} "
                f"(offset {cam.session_offset_ms:.0f} ms, "
                f"peak {cam.peak:.2f}, peak ratio {cam.peak_ratio:.1f})"
            )
            if cam.message:
                line += f" - {cam.message}"
            lines.append(line)
        lines.append(
            f"A match needs peak >= {MIN_PEAK} and peak ratio >= {MIN_PEAK_RATIO}. "
            "Check that the camera recorded the show's audio and overlaps the "
            "session, or remove it from the inputs."
        )
        raise SessionPlacementError("\n".join(lines))


def place_cameras_on_session(
    camera_paths: Sequence[str],
    source_paths: Sequence[str],
    *,
    decode: Callable[[str], np.ndarray] = decode_mono_16k,
) -> SessionPlacement:
    """Place every camera on the session timeline defined by the sources."""
    sources = list(dict.fromkeys(source_paths))  # one decode per distinct file
    if not sources:
        raise ValueError("At least one audio source is required to build a session timeline")

    cache: dict[str, np.ndarray] = {}

    def cached_decode(path: str) -> np.ndarray:
        if path not in cache:
            cache[path] = decode(path)
        return cache[path]

    source_audio: list[np.ndarray] = []
    for path in sources:
        try:
            source_audio.append(cached_decode(path))
        except Exception as exc:
            raise SessionPlacementError(
                f"Could not decode audio source {os.path.basename(path)}: {exc}"
            ) from exc

    warnings: list[str] = []
    lengths = [len(a) for a in source_audio]
    session_samples = max(lengths)
    spread_ms = (session_samples - min(lengths)) / SAMPLE_RATE * 1000.0
    if spread_ms > SOURCE_LENGTH_TOLERANCE_MS:
        warnings.append(
            "Audio sources differ in duration "
            f"({min(lengths) / SAMPLE_RATE:.1f} s to {session_samples / SAMPLE_RATE:.1f} s); "
            "they may not share a start time"
        )

    alignments = align_cameras_to_session(
        list(camera_paths), sources, decode=cached_decode
    )

    session_mix: np.ndarray | None = None
    cameras: list[CameraPlacement] = []
    for al in alignments:
        offset = al.offset_ms
        refined = False
        shift = 0.0
        camera_audio = cache.pop(al.path, None)  # free memory as we go

        if al.status == "ok":
            name = os.path.basename(al.path)
            try:
                if camera_audio is None:
                    raise RuntimeError("camera audio is not available")
                if session_mix is None:
                    session_mix = mix_sources([
                        np.pad(a, (0, session_samples - len(a))) for a in source_audio
                    ])
                result = refine_offset(camera_audio, session_mix, al.offset_ms)
            except Exception as exc:
                logger.warning("Refinement failed for %s: %s", name, exc)
                warnings.append(
                    f"{name}: refinement failed ({exc}); keeping coarse offset"
                )
            else:
                if result.refined:
                    offset, refined, shift = result.offset_ms, True, result.shift_ms
                else:
                    warnings.append(
                        f"{name}: refinement rejected ({result.message}); "
                        "keeping coarse offset"
                    )

        cameras.append(CameraPlacement(
            camera_index=al.camera_index,
            path=al.path,
            status=al.status,
            session_offset_ms=offset,
            # "+ 0.0" turns a -0.0 into 0.0 so QA output never shows "-0.0".
            slot_offset_ms=session_offset_to_seek_offset(offset) + 0.0,
            peak=al.peak,
            peak_ratio=al.peak_ratio,
            refined=refined,
            refine_shift_ms=shift,
            message=al.message,
        ))

    return SessionPlacement(
        cameras=tuple(cameras),
        session_duration_ms=int(round(session_samples / SAMPLE_RATE * 1000.0)),
        warnings=tuple(warnings),
    )
