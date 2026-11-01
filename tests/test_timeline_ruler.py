"""Tests for expressing cameras on a mixer-session timeline."""

import pytest

from multicam_editor.logic.decision_engine import CutSegment
from multicam_editor.logic.timeline_ruler import (
    Substitution,
    build_camera_coverage,
    repair_cut_plan,
    session_offset_to_seek_offset,
)

# Real PageCast recordings: session offsets (ms) and camera durations (ms).
REAL_OFFSETS = [-809270.0, -35110.0, -39250.0, -20530.0]
REAL_DURATIONS = [2177000, 1395000, 1403000, 424000]


def _triples(result):
    return [(c.start_ms, c.end_ms, c.camera_id) for c in result.cuts]


def _real_coverage():
    return build_camera_coverage(REAL_OFFSETS, REAL_DURATIONS)


def test_seek_offset_negates_session_offset():
    assert session_offset_to_seek_offset(-35110.0) == 35110.0
    assert session_offset_to_seek_offset(5000.0) == -5000.0
    assert session_offset_to_seek_offset(0.0) == 0.0


def test_coverage_from_real_recordings():
    coverage = _real_coverage()

    assert [c.camera_index for c in coverage] == [0, 1, 2, 3]
    assert [c.seek_offset_ms for c in coverage] == [809270.0, 35110.0, 39250.0, 20530.0]
    assert [c.start_ms for c in coverage] == [-809270, -35110, -39250, -20530]
    assert [c.end_ms for c in coverage] == [1367730, 1359890, 1363750, 403470]
    # Every closeup covers the whole 1349 s show; the wide shot stops early.
    assert all(c.covers(0, 1349000) for c in coverage[:3])
    assert not coverage[3].covers(0, 1349000)


def test_coverage_rounds_conservatively():
    (coverage,) = build_camera_coverage([-20522.375], [424000])
    assert coverage.start_ms == -20522
    assert coverage.end_ms == 403477


def test_coverage_rejects_bad_input():
    with pytest.raises(ValueError):
        build_camera_coverage([0.0], [1000, 2000])
    with pytest.raises(ValueError):
        build_camera_coverage([0.0], [0])


def test_repair_leaves_covered_plan_unchanged():
    plan = [CutSegment(0, 10000, 0), CutSegment(10000, 20000, 1), CutSegment(20000, 30000, 2)]

    result = repair_cut_plan(plan, _real_coverage())

    assert _triples(result) == [(0, 10000, 0), (10000, 20000, 1), (20000, 30000, 2)]
    assert result.substitutions == ()
    assert result.uncovered == ()


def test_repair_substitutes_most_recent_covering_camera():
    """The wide shot (camera 4) has no footage after 403.47 s of the show."""
    plan = [CutSegment(0, 400000, 1), CutSegment(400000, 410000, 3)]

    result = repair_cut_plan(plan, _real_coverage())

    assert _triples(result) == [
        (0, 400000, 1),
        (400000, 403470, 3),
        (403470, 410000, 1),
    ]
    assert result.substitutions == (Substitution(403470, 410000, 3, 1),)
    assert result.uncovered == ()


def test_repair_when_camera_starts_after_show_start():
    coverage = build_camera_coverage([0.0, 5000.0], [100000, 100000])

    result = repair_cut_plan([CutSegment(0, 10000, 1)], coverage)

    assert _triples(result) == [(0, 5000, 0), (5000, 10000, 1)]
    assert result.substitutions == (Substitution(0, 5000, 1, 0),)


def test_repair_reports_footage_nobody_covers():
    coverage = build_camera_coverage([0.0], [1000])

    result = repair_cut_plan([CutSegment(0, 2000, 0)], coverage)

    assert _triples(result) == [(0, 2000, 0)]
    assert result.substitutions == ()
    assert result.uncovered == ((1000, 2000),)


def test_repair_merges_adjacent_pieces_of_the_same_camera():
    coverage = build_camera_coverage([0.0, -5000.0], [100000, 100000])

    result = repair_cut_plan([CutSegment(0, 100000, 0)], coverage)

    assert _triples(result) == [(0, 100000, 0)]
    assert result.substitutions == ()


def test_repair_clamps_out_of_range_camera():
    coverage = build_camera_coverage([0.0, 0.0], [100000, 100000])

    result = repair_cut_plan([CutSegment(0, 1000, 9)], coverage)

    assert _triples(result) == [(0, 1000, 1)]


def test_repair_requires_coverage():
    with pytest.raises(ValueError):
        repair_cut_plan([CutSegment(0, 1000, 0)], ())
