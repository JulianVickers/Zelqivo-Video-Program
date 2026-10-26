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



class TimelineBoundaryPolicy(str, Enum):
    """Whether output boundaries follow legacy behavior or preserve all media."""

    STANDARD = "standard"
    PRESERVE_ALL_MEDIA = "preserve_all_media"


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



@dataclass(frozen=True)
class TimelineBoundaryConfig:
    """Controls whether rendering may discard media at timeline boundaries.

    STANDARD retains the application's existing behavior.
    PRESERVE_ALL_MEDIA requires rendering to retain the complete duration of
    every selected source, extending the output timeline when necessary.
    """

    policy: TimelineBoundaryPolicy = TimelineBoundaryPolicy.STANDARD

    def __post_init__(self) -> None:
        if not isinstance(self.policy, TimelineBoundaryPolicy):
            try:
                object.__setattr__(
                    self, "policy", TimelineBoundaryPolicy(self.policy)
                )
            except (TypeError, ValueError) as exc:
                allowed = ", ".join(policy.value for policy in TimelineBoundaryPolicy)
                raise ValueError(f"policy must be one of: {allowed}") from exc



class TimelinePlacementBasis(str, Enum):
    """How a source's position on the shared timeline was established."""

    RECORDED_TIMESTAMP = "recorded_timestamp"
    USER_DEFINED = "user_defined"
    SYNC_ESTIMATE = "sync_estimate"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TimelineSourceSpan:
    """A source's placement and complete duration on a shared timeline.

    ``start_ms`` is relative to a common reference and may be negative.
    It is not inferred from duration or assumed to be zero.

    A synchronization estimate describes estimated placement, not a
    verified recording-start timestamp.
    """

    source_id: str
    start_ms: Optional[int]
    duration_ms: int
    placement_basis: TimelinePlacementBasis = TimelinePlacementBasis.UNKNOWN

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if isinstance(self.duration_ms, bool) or not isinstance(self.duration_ms, int):
            raise ValueError("duration_ms must be an integer")
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        if self.start_ms is not None and (
            isinstance(self.start_ms, bool) or not isinstance(self.start_ms, int)
        ):
            raise ValueError("start_ms must be an integer or None")

        if not isinstance(self.placement_basis, TimelinePlacementBasis):
            try:
                object.__setattr__(
                    self, "placement_basis",
                    TimelinePlacementBasis(self.placement_basis),
                )
            except (TypeError, ValueError) as exc:
                allowed = ", ".join(item.value for item in TimelinePlacementBasis)
                raise ValueError(f"placement_basis must be one of: {allowed}") from exc

        if self.start_ms is None and self.placement_basis is not TimelinePlacementBasis.UNKNOWN:
            raise ValueError("unknown start_ms requires UNKNOWN placement_basis")
        if self.start_ms is not None and self.placement_basis is TimelinePlacementBasis.UNKNOWN:
            raise ValueError("known start_ms requires an explicit placement basis")

    @property
    def end_ms(self) -> Optional[int]:
        """End position on the shared timeline, or None if placement is unknown."""
        if self.start_ms is None:
            return None
        return self.start_ms + self.duration_ms


@dataclass(frozen=True)
class PreservationTimeline:
    """The complete shared timeline required to retain all supplied sources."""

    start_ms: int
    end_ms: int
    source_spans: tuple[TimelineSourceSpan, ...]

    @property
    def duration_ms(self) -> int:
        """Duration after shifting the earliest source start to output time zero."""
        return self.end_ms - self.start_ms

    @property
    def has_estimated_placements(self) -> bool:
        """Whether any source position depends on synchronization estimation."""
        return any(
            span.placement_basis is TimelinePlacementBasis.SYNC_ESTIMATE
            for span in self.source_spans
        )


@dataclass(frozen=True)
class OutputTimelinePlacement:
    """A source's position relative to the beginning of the output timeline."""

    source_id: str
    output_start_ms: int
    duration_ms: int
    placement_basis: TimelinePlacementBasis

    @property
    def output_end_ms(self) -> int:
        """Exclusive end position relative to output time zero."""
        return self.output_start_ms + self.duration_ms


def build_preservation_timeline(
    source_spans: list[TimelineSourceSpan],
) -> PreservationTimeline:
    """Calculate bounds covering every source without trimming source content.

    All source starts must be explicitly placed. Unknown positions are rejected
    because a complete shared timeline cannot be calculated honestly without
    them. Estimated positions are retained but remain identifiable as estimates.
    """
    if not source_spans:
        raise ValueError("at least one source span is required")

    source_ids = [span.source_id for span in source_spans]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("timeline source IDs must be unique")

    unknown = [span.source_id for span in source_spans if span.start_ms is None]
    if unknown:
        raise ValueError(
            "cannot build preservation timeline with unknown source placement: "
            + ", ".join(unknown)
        )

    starts = [span.start_ms for span in source_spans]
    ends = [span.end_ms for span in source_spans]
    # Unknown values have already been rejected above.
    return PreservationTimeline(
        start_ms=min(starts),
        end_ms=max(ends),
        source_spans=tuple(source_spans),
    )


def map_sources_to_output_timeline(
    timeline: PreservationTimeline,
) -> tuple[OutputTimelinePlacement, ...]:
    """Map every placed source to output time without trimming its duration.

    The earliest source starts at output time zero. All other source positions
    are shifted by the same amount, preserving their relative timing.
    """
    placements = []

    for span in timeline.source_spans:
        if span.start_ms is None:
            raise ValueError(
                f"cannot map source with unknown placement: {span.source_id}"
            )

        placements.append(
            OutputTimelinePlacement(
                source_id=span.source_id,
                output_start_ms=span.start_ms - timeline.start_ms,
                duration_ms=span.duration_ms,
                placement_basis=span.placement_basis,
            )
        )

    return tuple(placements)


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
