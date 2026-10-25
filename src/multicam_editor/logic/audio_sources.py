"""Configuration models for independent external audio sources.

These models deliberately keep audio-source identity, speaker identity, and
camera identity separate. They are configuration primitives; pipeline and UI
integration can be added without changing their meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Optional


class AudioOutputMode(str, Enum):
    """How selected external audio sources are delivered in the output."""

    MIXDOWN = "mixdown"
    PRESERVE_TRACKS = "preserve_tracks"


@dataclass(frozen=True)
class AudioSource:
    """One independently addressable external audio source.

    ``channel_index`` is zero-based. ``None`` means use the input's default
    audio stream/channel layout; an integer selects one channel as a mono
    source. ``speaker_id`` identifies the person associated with this source,
    not a camera index. Detection and final-output inclusion are independent.
    """

    source_id: str
    path: str
    label: Optional[str] = None
    channel_index: Optional[int] = None
    speaker_id: Optional[str] = None
    use_for_detection: bool = True
    include_in_output: bool = True
    gain_db: float = 0.0

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if not self.path.strip():
            raise ValueError("path must not be empty")
        if self.channel_index is not None and self.channel_index < 0:
            raise ValueError("channel_index must be zero-based and non-negative")
        if not math.isfinite(self.gain_db):
            raise ValueError("gain_db must be a finite number")
        if self.speaker_id is not None and not self.speaker_id.strip():
            raise ValueError("speaker_id must be None or a non-empty string")
        if self.label is not None and not self.label.strip():
            raise ValueError("label must be None or a non-empty string")

    @property
    def display_name(self) -> str:
        """Human-readable source name, preferring an explicit label."""
        if self.label:
            return self.label
        filename = Path(self.path).name
        if self.channel_index is not None:
            return f"{filename} — Channel {self.channel_index + 1}"
        return filename


@dataclass(frozen=True)
class CameraSpeakerAssignment:
    """Assign a camera to a speaker, or explicitly leave it unmapped.

    ``camera_index`` is zero-based to match the existing pipeline. A ``None``
    speaker is a valid intentional state (for example, a wide-shot camera).
    """

    camera_index: int
    speaker_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.camera_index < 0:
            raise ValueError("camera_index must be zero-based and non-negative")
        if self.speaker_id is not None and not self.speaker_id.strip():
            raise ValueError("speaker_id must be None or a non-empty string")


@dataclass(frozen=True)
class AudioOutputConfig:
    """Output policy for external audio sources."""

    mode: AudioOutputMode = AudioOutputMode.MIXDOWN

    def __post_init__(self) -> None:
        if not isinstance(self.mode, AudioOutputMode):
            try:
                object.__setattr__(self, "mode", AudioOutputMode(self.mode))
            except (TypeError, ValueError) as exc:
                allowed = ", ".join(mode.value for mode in AudioOutputMode)
                raise ValueError(f"mode must be one of: {allowed}") from exc


def validate_audio_sources(sources: list[AudioSource]) -> None:
    """Validate a source collection, including unique stable source IDs."""
    ids = [source.source_id for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("audio source IDs must be unique")


def validate_camera_assignments(
    assignments: list[CameraSpeakerAssignment],
) -> None:
    """Require at most one assignment entry per camera index."""
    indices = [assignment.camera_index for assignment in assignments]
    if len(indices) != len(set(indices)):
        raise ValueError("camera assignments must have unique camera indices")
