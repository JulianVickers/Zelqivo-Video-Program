from __future__ import annotations

from multicam_editor.logic.decision_engine import CutSegment
from multicam_editor.logic.wide_shot import WideShotSettings, apply_wide_shot

W = 200
WIDE = 2
FULL = [(0, 60000)]
NO_HOLD = WideShotSettings(hold_ms=10**9)  # isolates the silence pass


def _cuts(*rows):
    return [CutSegment(s, e, c) for s, e, c in rows]


def _rows(cuts):
    return [(c.start_ms, c.end_ms, c.camera_id) for c in cuts]


def _track(total_ms, active):
    return [
        any(a <= w * W < b for a, b in active) for w in range(total_ms // W)
    ]


def _run(cuts, activity=None, coverage=FULL, settings=WideShotSettings(), wide=WIDE):
    return apply_wide_shot(
        cuts,
        wide_camera=wide,
        wide_coverage=coverage,
        activity=activity or {},
        window_ms=W,
        settings=settings,
    )


def test_no_wide_camera_returns_cuts_unchanged():
    cuts = _cuts((0, 30000, 0), (30000, 60000, 1))
    assert _rows(_run(cuts, wide=None)) == _rows(cuts)


def test_cuts_on_the_wide_camera_are_never_cut_away():
    cuts = _cuts((0, 60000, WIDE))
    assert _rows(_run(cuts)) == [(0, 60000, WIDE)]


def test_silence_of_two_seconds_cuts_to_wide():
    cuts = _cuts((0, 30000, 0), (30000, 60000, 1))
    activity = {
        0: _track(60000, [(0, 10000), (14000, 30000)]),
        1: _track(60000, [(30000, 60000)]),
    }
    assert _rows(_run(cuts, activity, settings=NO_HOLD)) == [
        (0, 10000, 0), (10000, 14000, WIDE), (14000, 30000, 0), (30000, 60000, 1),
    ]


def test_shorter_silence_is_ignored():
    cuts = _cuts((0, 30000, 0), (30000, 60000, 1))
    activity = {
        0: _track(60000, [(0, 10000), (11800, 30000)]),
        1: _track(60000, [(30000, 60000)]),
    }
    assert _rows(_run(cuts, activity, settings=NO_HOLD)) == _rows(cuts)


def test_silence_is_clipped_to_wide_coverage():
    cuts = _cuts((0, 30000, 0), (30000, 60000, 1))
    activity = {
        0: _track(60000, [(0, 10000), (14000, 30000)]),
        1: _track(60000, [(30000, 60000)]),
    }
    assert _rows(_run(cuts, activity, coverage=[(0, 12000)], settings=NO_HOLD)) == [
        (0, 10000, 0), (10000, 12000, WIDE), (12000, 30000, 0), (30000, 60000, 1),
    ]


def test_silence_clip_shorter_than_min_clip_is_dropped():
    cuts = _cuts((0, 30000, 0), (30000, 60000, 1))
    activity = {
        0: _track(60000, [(0, 10000), (14000, 30000)]),
        1: _track(60000, [(30000, 60000)]),
    }
    assert _rows(_run(cuts, activity, coverage=[(0, 10500)], settings=NO_HOLD)) == _rows(cuts)


def test_no_activity_skips_the_silence_pass():
    cuts = _cuts((0, 10000, 0))
    assert _rows(_run(cuts, {})) == [(0, 10000, 0)]


def test_cutaway_after_a_20_second_hold():
    cuts = _cuts((0, 30000, 0))
    assert _rows(_run(cuts)) == [
        (0, 20000, 0), (20000, 23000, WIDE), (23000, 30000, 0),
    ]


def test_cutaway_prefers_a_pause():
    cuts = _cuts((0, 30000, 0))
    activity = {0: _track(30000, [(0, 24000), (25000, 30000)])}
    assert _rows(_run(cuts, activity)) == [
        (0, 24000, 0), (24000, 27000, WIDE), (27000, 30000, 0),
    ]


def test_long_hold_gets_repeated_cutaways():
    cuts = _cuts((0, 60000, 0))
    assert _rows(_run(cuts)) == [
        (0, 20000, 0), (20000, 23000, WIDE), (23000, 43000, 0),
        (43000, 46000, WIDE), (46000, 60000, 0),
    ]


def test_hold_shorter_than_limit_is_untouched():
    cuts = _cuts((0, 15000, 0), (15000, 25000, 1))
    assert _rows(_run(cuts)) == _rows(cuts)


def test_cutaway_must_leave_a_usable_remainder():
    cuts = _cuts((0, 22500, 0))
    assert _rows(_run(cuts)) == [(0, 22500, 0)]


def test_cutaway_outside_wide_coverage_is_skipped():
    cuts = _cuts((0, 30000, 0))
    assert _rows(_run(cuts, coverage=[(0, 15000)])) == [(0, 30000, 0)]


def test_result_is_contiguous_and_merged():
    cuts = _cuts((0, 30000, 0), (30000, 60000, 1))
    activity = {
        0: _track(60000, [(0, 10000), (14000, 30000)]),
        1: _track(60000, [(30000, 60000)]),
    }
    result = _run(cuts, activity)
    rows = _rows(result)
    assert rows[0][0] == 0 and rows[-1][1] == 60000
    for (_, end, cam), (start, _, next_cam) in zip(rows, rows[1:]):
        assert end == start
        assert cam != next_cam
    assert (10000, 14000, WIDE) in rows
    assert (50000, 53000, WIDE) in rows
