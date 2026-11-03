"""Turning repaired cuts plus uncovered ranges into render segments (None = black)."""
from __future__ import annotations

import pytest

from multicam_editor.logic.decision_engine import CutSegment
from multicam_editor.logic.timeline_ruler import (
    TimelineSegment,
    build_camera_coverage,
    build_timeline_segments,
    repair_cut_plan,
)


def seg(start, end, camera):
    return TimelineSegment(start, end, camera)


def test_no_uncovered_ranges_passes_cuts_through():
    cuts = [CutSegment(0, 4000, 0), CutSegment(4000, 9000, 2)]
    assert build_timeline_segments(cuts, []) == (seg(0, 4000, 0), seg(4000, 9000, 2))


def test_gap_at_the_end_of_a_cut_becomes_black():
    cuts = [CutSegment(0, 6000, 1)]
    assert build_timeline_segments(cuts, [(5000, 6000)]) == (
        seg(0, 5000, 1),
        seg(5000, 6000, None),
    )


def test_leading_gap_becomes_black():
    cuts = [CutSegment(0, 10000, 1)]
    assert build_timeline_segments(cuts, [(0, 3000)]) == (
        seg(0, 3000, None),
        seg(3000, 10000, 1),
    )


def test_gap_spanning_two_cuts_is_one_black_segment():
    cuts = [CutSegment(0, 4000, 0), CutSegment(4000, 8000, 2)]
    assert build_timeline_segments(cuts, [(3000, 5000)]) == (
        seg(0, 3000, 0),
        seg(3000, 5000, None),
        seg(5000, 8000, 2),
    )


def test_cut_entirely_inside_a_gap_is_black_and_merges_with_neighbours():
    cuts = [CutSegment(0, 2000, 0), CutSegment(2000, 4000, 1), CutSegment(4000, 6000, 0)]
    assert build_timeline_segments(cuts, [(0, 6000)]) == (seg(0, 6000, None),)


def test_several_gaps_inside_one_cut():
    cuts = [CutSegment(0, 10000, 3)]
    assert build_timeline_segments(cuts, [(1000, 2000), (6000, 7000)]) == (
        seg(0, 1000, 3),
        seg(1000, 2000, None),
        seg(2000, 6000, 3),
        seg(6000, 7000, None),
        seg(7000, 10000, 3),
    )


def test_contiguous_cuts_on_the_same_camera_merge():
    cuts = [CutSegment(0, 1000, 1), CutSegment(1000, 2000, 1)]
    assert build_timeline_segments(cuts, []) == (seg(0, 2000, 1),)


def test_uncovered_range_outside_every_cut_is_ignored():
    cuts = [CutSegment(0, 5000, 0)]
    assert build_timeline_segments(cuts, [(7000, 9000)]) == (seg(0, 5000, 0),)


def test_total_duration_and_contiguity_are_preserved():
    cuts = [CutSegment(0, 3000, 0), CutSegment(3000, 7000, 1), CutSegment(7000, 12000, 0)]
    result = build_timeline_segments(cuts, [(2500, 3500), (6000, 8000), (11000, 12000)])
    assert result[0].start_ms == 0
    assert result[-1].end_ms == 12000
    assert all(a.end_ms == b.start_ms for a, b in zip(result, result[1:]))
    assert all(s.end_ms > s.start_ms for s in result)


def test_repair_then_segments_for_a_camera_that_starts_late_and_stops_early():
    # Camera starts 20 s into the show and has 100 s of footage: covers 20-120 s.
    coverage = build_camera_coverage([20_000.0], [100_000])
    repaired = repair_cut_plan([CutSegment(0, 150_000, 0)], coverage)

    # repair keeps the uncovered stretches on the wanted camera and merges them.
    assert repaired.uncovered == ((0, 20_000), (120_000, 150_000))
    assert repaired.cuts == (CutSegment(0, 150_000, 0),)

    assert build_timeline_segments(repaired.cuts, repaired.uncovered) == (
        seg(0, 20_000, None),
        seg(20_000, 120_000, 0),
        seg(120_000, 150_000, None),
    )
