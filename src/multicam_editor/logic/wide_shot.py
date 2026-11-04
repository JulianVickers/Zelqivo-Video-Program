"""Wide-shot post-processing of a finished cut plan.

Pure logic, no pipeline state. The wide shot is a camera that no speaker is
mapped to. Two rules put it into the plan:

  * Nobody speaking: a stretch of at least ``nobody_min_ms`` in which no
    source is active shows the wide shot (clipped to what the wide camera
    recorded; a clipped piece under ``min_clip_ms`` is dropped).
  * Cutaways: after ``hold_ms`` on one camera, a ``cutaway_ms`` wide shot,
    preferably at a pause of that camera's speaker. A long hold gets a
    cutaway after every ``hold_ms``. A cutaway must lie fully inside the wide
    camera's coverage and leave ``min_clip_ms`` of the held camera.

The decision engine and the speaker detection are not involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .decision_engine import CutSegment

_RETRY_STEP_MS = 1000


@dataclass(frozen=True)
class WideShotSettings:
    nobody_min_ms: int = 2000
    hold_ms: int = 20000
    cutaway_ms: int = 3000
    min_clip_ms: int = 1000
    pause_min_ms: int = 600
    pause_search_ms: int = 5000


def apply_wide_shot(
    cuts: Sequence[CutSegment],
    *,
    wide_camera: int | None,
    wide_coverage: Sequence[tuple[int, int]],
    activity: Mapping[int, Sequence[bool]],
    window_ms: int = 200,
    settings: WideShotSettings = WideShotSettings(),
) -> list[CutSegment]:
    """Return the cut plan with the wide shot worked in.

    ``wide_coverage`` is the wide camera's footage as (start, end) ranges on
    the session timeline. ``activity`` maps a camera index to one bool per
    ``window_ms`` window (True: its speaker is active); an empty mapping skips
    the silence rule and the pause placement.
    """
    result = _merge(cuts)
    if not result or wide_camera is None:
        return result

    if activity:
        total = max(c.end_ms for c in result)
        intervals: list[tuple[int, int]] = []
        for start, end in _silent_runs(activity, window_ms, total):
            if end - start < settings.nobody_min_ms:
                continue
            for lo, hi in wide_coverage:
                a, b = max(start, lo), min(end, hi)
                if b - a >= settings.min_clip_ms:
                    intervals.append((a, b))
        result = _merge(_overlay(result, intervals, wide_camera))

    cutaways = _cutaway_intervals(
        result, wide_camera, wide_coverage, activity, window_ms, settings
    )
    return _merge(_overlay(result, cutaways, wide_camera))


def _active(track: Sequence[bool], window_ms: int, ms: int) -> bool:
    w = ms // window_ms
    return 0 <= w < len(track) and bool(track[w])


def _silent_runs(
    activity: Mapping[int, Sequence[bool]], window_ms: int, total_ms: int
) -> list[tuple[int, int]]:
    count = -(-total_ms // window_ms)
    runs: list[tuple[int, int]] = []
    start = None
    for w in range(count):
        silent = not any(w < len(t) and t[w] for t in activity.values())
        if silent and start is None:
            start = w
        elif not silent and start is not None:
            runs.append((start * window_ms, min(w * window_ms, total_ms)))
            start = None
    if start is not None:
        runs.append((start * window_ms, total_ms))
    return runs


def _inside(start: int, end: int, coverage: Sequence[tuple[int, int]]) -> bool:
    return any(lo <= start and end <= hi for lo, hi in coverage)


def _cutaway_intervals(
    cuts: Sequence[CutSegment],
    wide_camera: int,
    wide_coverage: Sequence[tuple[int, int]],
    activity: Mapping[int, Sequence[bool]],
    window_ms: int,
    s: WideShotSettings,
) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    pause_windows = max(1, -(-s.pause_min_ms // window_ms))
    for cut in cuts:
        if cut.camera_id == wide_camera:
            continue
        track = activity.get(cut.camera_id)
        t = cut.start_ms
        while t + s.hold_ms + s.cutaway_ms + s.min_clip_ms <= cut.end_ms:
            earliest = t + s.hold_ms
            candidates: list[int] = []
            if track is not None:
                first = -(-earliest // window_ms) * window_ms
                for start in range(first, earliest + s.pause_search_ms + 1, window_ms):
                    if all(
                        not _active(track, window_ms, start + k * window_ms)
                        for k in range(pause_windows)
                    ):
                        candidates.append(start)
            candidates.append(earliest)
            placed = next(
                (
                    (a, a + s.cutaway_ms)
                    for a in candidates
                    if a + s.cutaway_ms + s.min_clip_ms <= cut.end_ms
                    and _inside(a, a + s.cutaway_ms, wide_coverage)
                ),
                None,
            )
            if placed is not None:
                found.append(placed)
                t = placed[1]
            else:
                t += _RETRY_STEP_MS
    return found


def _overlay(
    cuts: Sequence[CutSegment],
    intervals: Sequence[tuple[int, int]],
    wide_camera: int,
) -> list[CutSegment]:
    if not intervals:
        return list(cuts)
    ordered = sorted(cuts, key=lambda c: c.start_ms)
    bounds = sorted(
        {c.start_ms for c in ordered}
        | {c.end_ms for c in ordered}
        | {edge for interval in intervals for edge in interval}
    )
    out: list[CutSegment] = []
    i = 0
    for a, b in zip(bounds, bounds[1:]):
        while i < len(ordered) and ordered[i].end_ms <= a:
            i += 1
        if i == len(ordered) or ordered[i].start_ms > a:
            continue  # gap in the plan: leave it alone
        wide = any(lo <= a and b <= hi for lo, hi in intervals)
        out.append(CutSegment(a, b, wide_camera if wide else ordered[i].camera_id))
    return out


def _merge(cuts: Sequence[CutSegment]) -> list[CutSegment]:
    merged: list[CutSegment] = []
    for cut in cuts:
        last = merged[-1] if merged else None
        if last and last.camera_id == cut.camera_id and last.end_ms == cut.start_ms:
            merged[-1] = CutSegment(last.start_ms, cut.end_ms, cut.camera_id)
        else:
            merged.append(cut)
    return merged
