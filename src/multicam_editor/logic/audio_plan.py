"""Resolve which audio source each camera uses for speaker detection.

Pure configuration logic: no audio is read and no files are touched. The
resolver joins three independent pieces of configuration:

* audio sources, each optionally tied to a speaker,
* camera assignments (camera -> speaker, or explicitly no speaker),
* per-source alignment results from synchronization.

It never guesses. A speaker with no source, or with several competing
detection sources and no explicit choice, is an error. A failed alignment is
reported in the plan rather than hidden, and ``require_usable_alignment``
turns that into an error when the caller needs trustworthy offsets.

Offset convention (matches ``audio_sync``): a positive ``offset_ms`` means the
source is late relative to the reference, so aligning it means trimming its
start; a negative offset means padding silence at its start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from .audio_sources import (
    AudioSource,
    CameraSpeakerAssignment,
    validate_audio_sources,
    validate_camera_assignments,
)

if TYPE_CHECKING:
    from .audio_source_sync import AudioSourceAlignment


class AudioPlanError(ValueError):
    """The audio-source configuration cannot be resolved."""


class AmbiguousSpeakerSourceError(AudioPlanError):
    """A speaker has several detection sources and no explicit choice."""


@dataclass(frozen=True)
class CameraAudioPlan:
    """Detection audio for one camera.

    ``alignment_status`` is one of:
    ``"not_applicable"`` (no detection source), ``"ok"``, ``"failed"``
    (synchronization could not find a trustworthy offset), or ``"missing"``
    (no synchronization result exists for the source).
    """

    camera_index: int
    speaker_id: Optional[str]
    source: Optional[AudioSource]
    offset_ms: float
    alignment_status: str

    @property
    def is_mapped(self) -> bool:
        """The camera is assigned to a speaker."""
        return self.speaker_id is not None

    @property
    def has_detection_audio(self) -> bool:
        """A dedicated source supplies this camera's detection audio."""
        return self.source is not None


def _pick_detection_source(
    speaker_id: str,
    sources: Sequence[AudioSource],
    source_by_id: Mapping[str, AudioSource],
    preferred_sources: Mapping[str, str],
) -> Optional[AudioSource]:
    candidates = [s for s in sources if s.speaker_id == speaker_id]
    if not candidates:
        raise AudioPlanError(
            f"no audio source is configured for speaker {speaker_id!r}"
        )

    preferred_id = preferred_sources.get(speaker_id)
    if preferred_id is not None:
        chosen = source_by_id[preferred_id]
        if not chosen.use_for_detection:
            raise AudioPlanError(
                f"preferred source {preferred_id!r} for speaker {speaker_id!r} "
                "is not enabled for detection"
            )
        return chosen

    detection = [s for s in candidates if s.use_for_detection]
    if not detection:
        return None
    if len(detection) > 1:
        names = ", ".join(s.source_id for s in detection)
        raise AmbiguousSpeakerSourceError(
            f"speaker {speaker_id!r} has several detection sources ({names}); "
            "choose one explicitly"
        )
    return detection[0]


def resolve_camera_audio_plan(
    num_cameras: int,
    sources: Sequence[AudioSource],
    assignments: Sequence[CameraSpeakerAssignment],
    alignments: Optional[Mapping[str, "AudioSourceAlignment"]] = None,
    preferred_sources: Optional[Mapping[str, str]] = None,
) -> tuple[CameraAudioPlan, ...]:
    """Return one ``CameraAudioPlan`` per camera, in camera order.

    A camera with no assignment, or an explicit ``None`` speaker (for example
    a wide shot), has no detection source. ``preferred_sources`` maps a
    speaker ID to the ``source_id`` that should win when several sources
    belong to that speaker.
    """
    if num_cameras < 1:
        raise AudioPlanError("num_cameras must be at least 1")

    sources = list(sources)
    assignments = list(assignments)
    validate_audio_sources(sources)
    validate_camera_assignments(assignments)

    by_camera: dict[int, CameraSpeakerAssignment] = {}
    for assignment in assignments:
        if assignment.camera_index >= num_cameras:
            raise AudioPlanError(
                f"camera assignment {assignment.camera_index} is out of range "
                f"for {num_cameras} camera(s)"
            )
        by_camera[assignment.camera_index] = assignment

    source_by_id = {s.source_id: s for s in sources}
    preferred = dict(preferred_sources or {})
    for speaker_id, source_id in preferred.items():
        chosen = source_by_id.get(source_id)
        if chosen is None:
            raise AudioPlanError(f"preferred source {source_id!r} does not exist")
        if chosen.speaker_id != speaker_id:
            raise AudioPlanError(
                f"preferred source {source_id!r} does not belong to "
                f"speaker {speaker_id!r}"
            )

    alignment_results = alignments or {}
    plan = []
    for camera_index in range(num_cameras):
        assignment = by_camera.get(camera_index)
        speaker_id = assignment.speaker_id if assignment else None

        if speaker_id is None:
            plan.append(CameraAudioPlan(camera_index, None, None, 0.0, "not_applicable"))
            continue

        source = _pick_detection_source(speaker_id, sources, source_by_id, preferred)
        if source is None:
            plan.append(CameraAudioPlan(camera_index, speaker_id, None, 0.0, "not_applicable"))
            continue

        result = alignment_results.get(source.source_id)
        if result is None:
            status, offset_ms = "missing", 0.0
        elif result.status == "ok":
            status, offset_ms = "ok", float(result.offset_ms)
        else:
            status, offset_ms = "failed", 0.0
        plan.append(CameraAudioPlan(camera_index, speaker_id, source, offset_ms, status))

    return tuple(plan)


def require_usable_alignment(plan: Sequence[CameraAudioPlan]) -> None:
    """Raise if any camera's dedicated source lacks a trustworthy offset.

    Cameras without a detection source (such as a wide shot) are ignored.
    A mapped source is never silently dropped or used at a guessed offset.
    """
    problems = []
    seen = set()
    for camera in plan:
        if camera.source is None or camera.alignment_status == "ok":
            continue
        if camera.source.source_id in seen:
            continue
        seen.add(camera.source.source_id)
        problems.append(
            f"{camera.source.display_name} (camera {camera.camera_index + 1}): "
            f"alignment {camera.alignment_status}"
        )
    if problems:
        raise AudioPlanError(
            "cannot use mapped audio sources without a trustworthy sync offset: "
            + "; ".join(problems)
        )
