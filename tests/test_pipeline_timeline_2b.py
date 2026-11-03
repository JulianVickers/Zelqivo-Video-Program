"""2b: session-length timeline, cut repair in decision, black segments in render."""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from multicam_editor.logic.processing_pipeline import ProcessingPipeline
from multicam_editor.logic.timeline_ruler import TimelineSegment
from multicam_editor.utils.signals import ProcessingSignals

_PIPE = "multicam_editor.logic.processing_pipeline"


class _Settings:
    def __init__(self, *args, **kwargs):
        pass

    def value(self, key, default=None, type=None):
        return default


def _camera(index, session_offset_ms):
    from multicam_editor.logic.session_placement import CameraPlacement

    return CameraPlacement(
        camera_index=index,
        path=f"/v/{index}.mp4",
        status="ok",
        session_offset_ms=session_offset_ms,
        slot_offset_ms=-session_offset_ms + 0.0,
        peak=0.6,
        peak_ratio=12.0,
        refined=False,
        refine_shift_ms=0.0,
        message="",
    )


def _placement(*cameras, duration_ms=60_000):
    from multicam_editor.logic.session_placement import SessionPlacement

    return SessionPlacement(cameras=tuple(cameras), session_duration_ms=duration_ms, warnings=())


def _make(durations, session_offsets=None, session_ms=60_000):
    signals = ProcessingSignals()
    errors: list[str] = []
    signals.error.connect(lambda message: errors.append(message))
    files = [f"/v/{i}.mp4" for i in range(len(durations))]
    pipeline = ProcessingPipeline(files, signals)
    pipeline._probe_results = [SimpleNamespace(duration_ms=d) for d in durations]
    pipeline.speaker_switching_enabled = False
    pipeline._speaker_segments = []
    if session_offsets is not None:
        cams = [_camera(i, off) for i, off in enumerate(session_offsets)]
        pipeline._session_placement = _placement(*cams, duration_ms=session_ms)
        pipeline._camera_offsets = {c.camera_index: c.slot_offset_ms for c in cams}
    return pipeline, errors


def _decide(pipeline):
    pipeline._qa_exporter = MagicMock()
    with patch(f"{_PIPE}.QSettings", _Settings):
        ok = pipeline._stage_decision()
    return ok, pipeline._qa_exporter


def _render(pipeline):
    captured = {}

    def fake_render(cuts, output_path, resolution, fps):
        captured["cuts"] = list(cuts)
        return SimpleNamespace(cancelled=False, success=True, error=None)

    with patch(f"{_PIPE}.QSettings", _Settings), \
            patch(f"{_PIPE}.render_single_pass", fake_render), \
            patch.object(ProcessingPipeline, "_check_disk_space", lambda self, **kw: None):
        result = pipeline._stage_render()
    return result, captured


# ---- _timeline_duration_ms -------------------------------------------------

def test_duration_uses_session_length_with_placement():
    pipeline, _ = _make([30_000, 30_000], [0, 20_000], session_ms=60_000)
    assert pipeline._timeline_duration_ms() == 60_000


def test_duration_is_longest_probe_without_placement():
    pipeline, _ = _make([30_000, 45_000])
    assert pipeline._timeline_duration_ms() == 45_000


# ---- decision --------------------------------------------------------------

def test_decision_repairs_cuts_and_records_black():
    pipeline, errors = _make([30_000, 30_000], [0, 20_000])
    ok, qa = _decide(pipeline)
    assert ok and not errors
    calls = [c.kwargs for c in qa.add_cut.call_args_list]
    spans = [(c["start_ms"], c["end_ms"], c["camera_index"]) for c in calls]
    assert spans == [(0, 30_000, 0), (30_000, 50_000, 1), (50_000, 60_000, -1)]
    assert calls[2]["reason"] == "no_footage"
    assert calls[2]["speaker_id"] == -1
    qa.set_total_duration.assert_called_once_with(60_000)


def test_decision_keeps_timeline_segments_for_render():
    pipeline, _ = _make([30_000, 30_000], [0, 20_000])
    _decide(pipeline)
    assert pipeline._timeline_segments == (
        TimelineSegment(0, 30_000, 0),
        TimelineSegment(30_000, 50_000, 1),
        TimelineSegment(50_000, 60_000, None),
    )
    plan = pipeline._cut_plan
    assert plan[0].start_ms == 0 and plan[-1].end_ms == 60_000
    assert all(a.end_ms == b.start_ms for a, b in zip(plan, plan[1:]))


def test_decision_warns_for_substitution_and_uncovered(caplog):
    pipeline, _ = _make([30_000, 30_000], [0, 20_000])
    with caplog.at_level(logging.WARNING):
        _decide(pipeline)
    text = " ".join(r.getMessage().lower() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "substitut" in text
    assert "no footage" in text


def test_decision_fails_when_no_camera_has_footage():
    pipeline, errors = _make([30_000, 30_000], [70_000, 80_000])
    ok, _ = _decide(pipeline)
    assert ok is False
    assert errors and "footage" in errors[0].lower()


def test_decision_legacy_path_has_no_segments():
    pipeline, _ = _make([30_000, 45_000])
    ok, qa = _decide(pipeline)
    assert ok
    assert pipeline._timeline_segments is None
    assert qa.set_total_duration.call_args.args == (45_000,)


# ---- render ----------------------------------------------------------------

def test_render_builds_black_and_offset_cuts():
    pipeline, errors = _make([30_000, 30_000], [0, 20_000])
    _decide(pipeline)
    result, got = _render(pipeline)
    assert result and not errors
    cuts = got["cuts"]
    assert [c.black for c in cuts] == [False, False, True]
    assert (cuts[0].source_path, cuts[0].start_ms, cuts[0].end_ms) == ("/v/0.mp4", 0, 30_000)
    # camera 1 starts at show time 20 s: show 30-50 s is camera 10-30 s
    assert (cuts[1].source_path, cuts[1].start_ms, cuts[1].end_ms) == ("/v/1.mp4", 10_000, 30_000)
    assert (cuts[2].start_ms, cuts[2].end_ms) == (50_000, 60_000)
    assert [c.cut_index for c in cuts] == [0, 1, 2]


def test_render_errors_instead_of_clamping_to_a_sliver():
    pipeline, errors = _make([30_000, 30_000], [0, 20_000])
    pipeline._timeline_segments = (TimelineSegment(0, 40_000, 0),)  # camera 0 ends at 30 s
    result, got = _render(pipeline)
    assert result is None
    assert "cuts" not in got
    assert errors and "camera 0" in errors[0].lower()


def test_render_tolerates_millisecond_truncation():
    pipeline, errors = _make([30_000, 30_000], [0, 20_000])
    pipeline._timeline_segments = (TimelineSegment(0, 30_002, 0),)
    result, got = _render(pipeline)
    assert result and not errors
    assert got["cuts"][0].end_ms == 30_000


def test_render_legacy_path_still_clamps():
    pipeline, _ = _make([30_000, 30_000])
    pipeline._camera_offsets = {0: 0.0, 1: 5_000.0}
    from multicam_editor.logic.decision_engine import CutSegment

    pipeline._cut_plan = [CutSegment(0, 10_000, 0), CutSegment(10_000, 40_000, 1)]
    result, got = _render(pipeline)
    assert result
    assert (got["cuts"][1].start_ms, got["cuts"][1].end_ms) == (15_000, 30_000)
    assert not any(c.black for c in got["cuts"])
