"""Independent synchronization of configured external audio sources."""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable

from .audio_sources import AudioSource
from .audio_sync import align_audio_offset
from ..utils.ffmpeg import extract_audio_to_wav

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioSourceAlignment:
    """Synchronization result for one configured external audio source."""

    source_id: str
    path: str
    channel_index: int | None
    offset_ms: float
    status: str
    message: str


def align_audio_sources(
    sources: Iterable[AudioSource],
    reference_audio_path: str,
) -> Dict[str, AudioSourceAlignment]:
    """Align each configured source independently to a reference WAV.

    A failure for one source does not prevent other sources from being
    processed. Results are keyed by stable ``source_id`` values.

    The temporary extracted WAV is always removed after each attempt.
    """
    results: Dict[str, AudioSourceAlignment] = {}

    for source in sources:
        temporary_wav = None

        try:
            fd, temporary_wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

            extraction = extract_audio_to_wav(
                source.path,
                output_path=temporary_wav,
                sample_rate=16000,
                mono=True,
                channel_index=source.channel_index,
            )

            if not extraction.success:
                message = extraction.error or "Audio extraction failed"
                results[source.source_id] = AudioSourceAlignment(
                    source_id=source.source_id,
                    path=source.path,
                    channel_index=source.channel_index,
                    offset_ms=0.0,
                    status="failed",
                    message=message,
                )
                logger.warning(
                    "Audio source %s extraction failed: %s",
                    source.source_id,
                    message,
                )
                continue

            offset_ms, message = align_audio_offset(
                temporary_wav,
                reference_audio_path,
            )
            status = "ok" if message == "ok" else "failed"

            results[source.source_id] = AudioSourceAlignment(
                source_id=source.source_id,
                path=source.path,
                channel_index=source.channel_index,
                offset_ms=offset_ms if status == "ok" else 0.0,
                status=status,
                message=message,
            )

            if status == "ok":
                logger.info(
                    "Audio source %s aligned: offset=%.1f ms",
                    source.source_id,
                    offset_ms,
                )
            else:
                logger.warning(
                    "Audio source %s alignment failed: %s",
                    source.source_id,
                    message,
                )

        except Exception as exc:
            logger.exception(
                "Unexpected synchronization error for audio source %s",
                source.source_id,
            )
            results[source.source_id] = AudioSourceAlignment(
                source_id=source.source_id,
                path=source.path,
                channel_index=source.channel_index,
                offset_ms=0.0,
                status="failed",
                message=str(exc),
            )

        finally:
            if temporary_wav and os.path.isfile(temporary_wav):
                try:
                    os.remove(temporary_wav)
                except OSError:
                    logger.warning(
                        "Could not remove temporary audio WAV: %s",
                        temporary_wav,
                    )

    return results
