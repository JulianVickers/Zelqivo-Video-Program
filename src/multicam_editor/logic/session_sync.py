"""Align cameras to a mixer session using whole-file loudness envelopes.

Wraps the pure matching in ``envelope_sync`` with audio decoding. The session
is one or more files that share a start time (for example per-channel exports
from one DAW session). They are decoded once and combined into a single
envelope, and each camera is matched against it.

Sign convention (same as ``audio_sync``): ``offset_ms`` is where the session's
time zero falls relative to the camera file. A negative offset of -35110 means
the session began 35.11 s into the camera recording, i.e. to find a session
moment ``t`` in the camera file, look at ``t + 35.11 s``.

A weak match is returned with ``status="unreliable"`` and its offset intact.
Deciding what to do about it (stop, fall back) belongs to the caller. A camera
that cannot be decoded or matched is ``"failed"`` with offset 0, and never
prevents other cameras from being aligned.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .envelope_sync import combine_envelopes, loudness_envelope, match_envelopes

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class AudioDecodeError(RuntimeError):
    """Audio could not be decoded from a media file."""


def decode_mono_16k(path: str, timeout: float = 900.0) -> np.ndarray:
    """Decode a file's audio to mono 16 kHz float samples in [-1, 1]."""
    command = [
        "ffmpeg", "-loglevel", "error", "-i", path,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise AudioDecodeError("ffmpeg was not found on this system") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError(f"decoding {path} timed out after {timeout:.0f}s") from exc

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise AudioDecodeError(
            f"ffmpeg could not decode {path}: {detail or f'exit code {result.returncode}'}"
        )

    raw = result.stdout[: len(result.stdout) // 2 * 2]
    if not raw:
        raise AudioDecodeError(f"no audio decoded from {path}")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


@dataclass(frozen=True)
class CameraSessionAlignment:
    """Where the session sits relative to one camera recording.

    ``status`` is ``"ok"`` (reliable match), ``"unreliable"`` (a match was
    computed but is too weak to trust) or ``"failed"`` (offset is 0.0 and
    ``message`` explains why).
    """

    camera_index: int
    path: str
    offset_ms: float
    peak: float
    peak_ratio: float
    reliable: bool
    status: str
    message: str


def _failed(index: int, path: str, message: str) -> CameraSessionAlignment:
    return CameraSessionAlignment(index, path, 0.0, 0.0, 0.0, False, "failed", message)


def align_cameras_to_session(
    camera_paths: Sequence[str],
    session_paths: Sequence[str],
    decode: Callable[[str], np.ndarray] = decode_mono_16k,
) -> tuple[CameraSessionAlignment, ...]:
    """Return one alignment per camera, in the order given."""
    if not session_paths:
        raise ValueError("at least one session audio file is required")

    try:
        session = combine_envelopes(
            [loudness_envelope(decode(path), SAMPLE_RATE) for path in session_paths]
        )
    except Exception as exc:
        logger.error("Session audio unusable: %s", exc)
        message = f"session audio unusable: {exc}"
        return tuple(_failed(i, p, message) for i, p in enumerate(camera_paths))

    results = []
    for index, path in enumerate(camera_paths):
        try:
            camera = loudness_envelope(decode(path), SAMPLE_RATE)
            match = match_envelopes(camera, session)
        except Exception as exc:
            logger.warning("Camera %d (%s) could not be aligned: %s", index, path, exc)
            results.append(_failed(index, path, str(exc)))
            continue

        if match.reliable:
            status = "ok"
            message = (
                f"session offset {match.offset_ms:.0f} ms "
                f"(peak {match.peak:.2f}, ratio {match.peak_ratio:.1f})"
            )
        else:
            status = "unreliable"
            message = (
                f"weak match (peak {match.peak:.2f}, ratio {match.peak_ratio:.1f}); "
                "offset not trustworthy"
            )
            logger.warning("Camera %d (%s): %s", index, path, message)

        results.append(CameraSessionAlignment(
            camera_index=index,
            path=path,
            offset_ms=match.offset_ms,
            peak=match.peak,
            peak_ratio=match.peak_ratio,
            reliable=match.reliable,
            status=status,
            message=message,
        ))
    return tuple(results)
