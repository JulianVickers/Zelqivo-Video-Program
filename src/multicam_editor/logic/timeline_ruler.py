"""Express cameras on a shared "show" timeline and repair cuts to fit.

With a mixer session as the ruler, show time zero is the session start. Each
camera has a *seek offset*: ``camera_time = show_time + seek_offset_ms``. This
is the slot ``_stage_render`` already uses for ``_camera_offsets``. It is the
negation of the session offset reported by ``session_sync``: a session offset
of -35110 (the session begins 35.11 s into the camera file) means the camera
seeks +35110 ms to reach show time zero.

A camera only has footage for show times in ``[start_ms, end_ms]``. Start may
be negative (the camera was rolling before the show) and end may fall short
of the show (a wide shot that stopped early). A cut to a camera with no
footage cannot be rendered, since single-pass rendering trims and
concatenates and cannot pad. ``repair_cut_plan`` splits such cuts and moves
the uncovered stretch to the most recently used camera that does have
footage, falling back to the lowest-numbered covering camera.

Pure computation: no files, no pipeline state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .decision_engine import CutSegment


def session_offset_to_seek_offset(session_offset_ms: float) -> float:
    """Convert a session offset into the render slot's seek offset."""
    return 0.0 - session_offset_ms


@dataclass(frozen=True)
class CameraCoverage:
    """Show-time span for which one camera has footage (milliseconds).

    Bounds are rounded inward, so a cut inside them never reads past the
    camera's real footage.
    """

    camera_index: int
    seek_offset_ms: float
    start_ms: int
    end_ms: int

    def covers(self, start_ms: int, end_ms: int) -> bool:
        return self.start_ms <= start_ms and end_ms <= self.end_ms


def build_camera_coverage(
    session_offsets_ms: Sequence[float],
    durations_ms: Sequence[int],
    *,
    excluded: frozenset[int] | set[int] = frozenset(),
) -> tuple[CameraCoverage, ...]:
    """Build one ``CameraCoverage`` per camera, in camera order.

    Cameras in ``excluded`` get empty coverage (start == end == 0), so the
    repair never uses them; their durations are not checked.
    """
    if len(session_offsets_ms) != len(durations_ms):
        raise ValueError("need one duration per session offset")
    excluded = frozenset(excluded)
    for index in sorted(excluded):
        if not 0 <= index < len(durations_ms):
            raise ValueError(f"excluded camera {index} does not exist")

    coverage = []
    for index, (offset, duration) in enumerate(zip(session_offsets_ms, durations_ms)):
        if index in excluded:
            coverage.append(CameraCoverage(
                camera_index=index, seek_offset_ms=0.0, start_ms=0, end_ms=0,
            ))
            continue
        if duration <= 0:
            raise ValueError(f"camera {index} has a non-positive duration")
        seek = session_offset_to_seek_offset(offset)
        coverage.append(CameraCoverage(
            camera_index=index,
            seek_offset_ms=seek,
            start_ms=math.ceil(-seek),
            end_ms=math.floor(duration - seek),
        ))
    return tuple(coverage)


@dataclass(frozen=True)
class Substitution:
    """A stretch of show time moved from the wanted camera to another."""

    start_ms: int
    end_ms: int
    wanted_camera: int
    used_camera: int


@dataclass(frozen=True)
class RepairResult:
    """A cut plan that fits camera coverage, plus what had to change.

    ``uncovered`` lists stretches no camera has footage for. Those cuts keep
    their wanted camera and must be handled by the caller.
    """

    cuts: tuple[CutSegment, ...]
    substitutions: tuple[Substitution, ...]
    uncovered: tuple[tuple[int, int], ...]


def repair_cut_plan(
    cuts: Sequence[CutSegment],
    coverage: Sequence[CameraCoverage],
) -> RepairResult:
    """Split cuts at coverage edges and replace uncovered stretches."""
    if not coverage:
        raise ValueError("camera coverage is required")

    last_camera = len(coverage) - 1
    pieces: list[tuple[int, int, int]] = []
    raw_subs: list[Substitution] = []
    raw_uncovered: list[tuple[int, int]] = []
    history: list[int] = []

    for cut in cuts:
        wanted = min(max(int(cut.camera_id), 0), last_camera)
        start, end = int(cut.start_ms), int(cut.end_ms)
        if end <= start:
            continue

        bounds = {start, end}
        for cam in coverage:
            for edge in (cam.start_ms, cam.end_ms):
                if start < edge < end:
                    bounds.add(edge)
        edges = sorted(bounds)

        for x, y in zip(edges, edges[1:]):
            if coverage[wanted].covers(x, y):
                used = wanted
            else:
                used = next(
                    (c for c in reversed(history) if coverage[c].covers(x, y)), None
                )
                if used is None:
                    used = next(
                        (cam.camera_index for cam in coverage if cam.covers(x, y)), None
                    )
                if used is None:
                    raw_uncovered.append((x, y))
                    used = wanted
                else:
                    raw_subs.append(Substitution(x, y, wanted, used))
            pieces.append((x, y, used))
            if not history or history[-1] != used:
                history.append(used)

    merged: list[tuple[int, int, int]] = []
    for x, y, camera in pieces:
        if merged and merged[-1][2] == camera and merged[-1][1] == x:
            merged[-1] = (merged[-1][0], y, camera)
        else:
            merged.append((x, y, camera))

    subs: list[Substitution] = []
    for sub in raw_subs:
        prev = subs[-1] if subs else None
        if (prev and prev.end_ms == sub.start_ms
                and prev.wanted_camera == sub.wanted_camera
                and prev.used_camera == sub.used_camera):
            subs[-1] = Substitution(prev.start_ms, sub.end_ms, prev.wanted_camera, prev.used_camera)
        else:
            subs.append(sub)

    gaps: list[tuple[int, int]] = []
    for x, y in raw_uncovered:
        if gaps and gaps[-1][1] == x:
            gaps[-1] = (gaps[-1][0], y)
        else:
            gaps.append((x, y))

    return RepairResult(
        cuts=tuple(CutSegment(x, y, camera) for x, y, camera in merged),
        substitutions=tuple(subs),
        uncovered=tuple(gaps),
    )


@dataclass(frozen=True)
class TimelineSegment:
    """One stretch of show time to render; ``camera_id`` None means black."""

    start_ms: int
    end_ms: int
    camera_id: int | None


def build_timeline_segments(
    cuts: Sequence[CutSegment],
    uncovered: Sequence[tuple[int, int]],
) -> tuple[TimelineSegment, ...]:
    """Subtract no-footage ranges from repaired cuts.

    ``repair_cut_plan`` keeps an uncovered stretch on its wanted camera and
    merges it into a same-camera neighbour, so the black stretches are only
    visible in ``uncovered``. This splits the cuts around them again.
    Contiguous segments that are identical (same camera, or both black) are
    merged, and the total span is preserved.
    """
    gaps: list[tuple[int, int]] = []
    for lo, hi in sorted((int(a), int(b)) for a, b in uncovered):
        if hi <= lo:
            continue
        if gaps and lo <= gaps[-1][1]:
            gaps[-1] = (gaps[-1][0], max(gaps[-1][1], hi))
        else:
            gaps.append((lo, hi))

    pieces: list[tuple[int, int, int | None]] = []
    for cut in cuts:
        start, end = int(cut.start_ms), int(cut.end_ms)
        if end <= start:
            continue
        cursor = start
        for lo, hi in gaps:
            if hi <= cursor:
                continue
            if lo >= end:
                break
            if lo > cursor:
                pieces.append((cursor, lo, cut.camera_id))
            black_end = min(hi, end)
            pieces.append((max(lo, cursor), black_end, None))
            cursor = black_end
        if cursor < end:
            pieces.append((cursor, end, cut.camera_id))

    merged: list[tuple[int, int, int | None]] = []
    for x, y, camera in pieces:
        if merged and merged[-1][2] == camera and merged[-1][1] == x:
            merged[-1] = (merged[-1][0], y, camera)
        else:
            merged.append((x, y, camera))

    return tuple(TimelineSegment(x, y, camera) for x, y, camera in merged)
