"""Pipeline wiring for session-based camera placement (configured audio sources)."""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from multicam_editor.logic.audio_source_sync import AudioSourceAlignment
from multicam_editor.logic.audio_sources import AudioSource
from multicam_editor.logic.processing_pipeline import ProcessingPipeline
from multicam_editor.utils.signals import ProcessingSignals

_PLACEMENT = "multicam_editor.logic.session_placement.place_cameras_on_session"
_FORBIDDEN_IN_SESSION_MODE = (
    "multicam_editor.logic.audio_sync.align_cameras",
    "multicam_editor.logic.audio_sync.align_audio_offset",
    "multicam_editor.logic.audio_source_sync.align_audio_sources",
    "multicam_editor.utils.ffmpeg.extract_audio_to_wav",
)

TRAVIS = AudioSource(source_id="travis", path="/rec/travis.wav")
DAKOTA = AudioSource(source_id="dakota", path="/rec/dakota.wav")


def _camera(index, path, session_offset_ms, status="ok", peak=0.6, ratio=12.0):
    from multicam_editor.logic.session_placement import CameraPlacement

    return CameraPlacement(
        camera_index=index,
        path=path,
        status=status,
        session_offset_ms=session_offset_ms,
        slot_offset_ms=-session_offset_ms,
        peak=peak,
        peak_ratio=ratio,
        refined=True,
        refine_shift_ms=1.5,
        message="",
    )


def _placement(*cameras, duration_ms=1_349_000, warnings=()):
    from multicam_editor.logic.session_placement import SessionPlacement

    return SessionPlacement(
        cameras=tuple(cameras),
        session_duration_ms=duration_ms,
        warnings=tuple(warnings),
    )


def _make(files, sources=()):
    signals = ProcessingSignals()
    errors: list[str] = []
    signals.error.connect(lambda message: errors.append(message))
    pipeline = ProcessingPipeline(list(files), signals)
    pipeline._audio_sources = list(sources)
    return pipeline, errors


def _boom(name):
    def boom(*args, **kwargs):
        raise AssertionError(f"{name} must not be called here")

    return boom


def _forbid_legacy(monkeypatch):
    for target in _FORBIDDEN_IN_SESSION_MODE:
        monkeypatch.setattr(target, _boom(target))


def _two_camera_placement():
    return _placement(
        _camera(0, "/shoot/1.mp4", -809270.0),
        _camera(1, "/shoot/2.mp4", -35110.0),
    )


def test_initial_state_has_no_session_placement():
    pipeline, _ = _make(["1.mp4", "2.mp4"])
    assert pipeline._session_placement is None


def test_session_mode_skips_legacy_alignment_and_places_all_cameras(monkeypatch):
    pipeline, errors = _make(["/shoot/1.mp4", "/shoot/2.mp4"], [TRAVIS, DAKOTA])
    _forbid_legacy(monkeypatch)
    placement = _two_camera_placement()
    calls = []

    def fake_place(camera_paths, source_paths, **kwargs):
        calls.append((list(camera_paths), list(source_paths)))
        return placement

    monkeypatch.setattr(_PLACEMENT, fake_place)

    assert pipeline._stage_align() is True

    assert calls == [(["/shoot/1.mp4", "/shoot/2.mp4"], ["/rec/travis.wav", "/rec/dakota.wav"])]
    assert errors == []
    assert pipeline._session_placement is placement


def test_slot_offsets_are_negated_session_offsets_for_every_camera(monkeypatch):
    pipeline, _ = _make(["/shoot/1.mp4", "/shoot/2.mp4"], [TRAVIS])
    monkeypatch.setattr(_PLACEMENT, lambda *a, **k: _two_camera_placement())

    assert pipeline._stage_align() is True

    # The primary camera is NOT pinned to 0 on a session ruler.
    assert pipeline._camera_offsets == {0: 809270.0, 1: 35110.0}


def test_sources_get_session_defined_alignments(monkeypatch):
    left = AudioSource(source_id="left", path="/rec/multi.wav", channel_index=0)
    right = AudioSource(source_id="right", path="/rec/multi.wav", channel_index=1)
    pipeline, _ = _make(["/shoot/1.mp4", "/shoot/2.mp4"], [left, right])
    seen = []

    def fake_place(camera_paths, source_paths, **kwargs):
        seen.append(set(source_paths))
        return _placement(
            _camera(0, "/shoot/1.mp4", -1000.0),
            _camera(1, "/shoot/2.mp4", -1000.0),
        )

    monkeypatch.setattr(_PLACEMENT, fake_place)

    assert pipeline._stage_align() is True

    assert seen == [{"/rec/multi.wav"}]
    assert pipeline._audio_source_alignments == {
        "left": AudioSourceAlignment(
            source_id="left", path="/rec/multi.wav", channel_index=0,
            offset_ms=0.0, status="ok",
            message=pipeline._audio_source_alignments["left"].message,
        ),
        "right": AudioSourceAlignment(
            source_id="right", path="/rec/multi.wav", channel_index=1,
            offset_ms=0.0, status="ok",
            message=pipeline._audio_source_alignments["right"].message,
        ),
    }
    assert "session" in pipeline._audio_source_alignments["left"].message.lower()


def test_unplaceable_camera_stops_the_stage_with_numbers(monkeypatch):
    pipeline, errors = _make(["/shoot/1.mp4", "/shoot/2.mp4"], [TRAVIS])
    pipeline._qa_exporter = MagicMock()
    placement = _placement(
        _camera(0, "/shoot/1.mp4", -809270.0),
        _camera(1, "/shoot/2.mp4", -35110.0, status="unreliable", peak=0.12, ratio=2.1),
    )
    monkeypatch.setattr(_PLACEMENT, lambda *a, **k: placement)

    assert pipeline._stage_align() is False

    assert len(errors) == 1
    message = errors[0]
    assert "2.mp4" in message
    assert "1.mp4" not in message
    assert "unreliable" in message
    assert "35110" in message
    assert "0.12" in message
    assert "2.1" in message
    pipeline._qa_exporter.set_camera_alignments.assert_called_once()


def test_placement_error_stops_the_stage_with_its_message(monkeypatch):
    from multicam_editor.logic.session_placement import SessionPlacementError

    pipeline, errors = _make(["/shoot/1.mp4", "/shoot/2.mp4"], [TRAVIS])

    def failing_place(*args, **kwargs):
        raise SessionPlacementError("Could not decode audio source travis.wav: corrupt")

    monkeypatch.setattr(_PLACEMENT, failing_place)

    assert pipeline._stage_align() is False
    assert errors == ["Could not decode audio source travis.wav: corrupt"]


def test_qa_camera_alignments_include_session_numbers(monkeypatch):
    pipeline, _ = _make(["/shoot/1.mp4", "/shoot/2.mp4"], [TRAVIS])
    pipeline._qa_exporter = MagicMock()
    monkeypatch.setattr(_PLACEMENT, lambda *a, **k: _two_camera_placement())

    assert pipeline._stage_align() is True

    (alignments,), _kwargs = pipeline._qa_exporter.set_camera_alignments.call_args
    second = alignments[1]
    assert second["camera_index"] == 1
    assert second["offset_ms"] == 35110.0           # same convention as _camera_offsets
    assert second["session_offset_ms"] == -35110.0  # raw session offset
    for key in ("status", "peak", "peak_ratio", "refined", "refine_shift_ms", "message"):
        assert key in second


def test_placement_warnings_are_logged(monkeypatch, caplog):
    pipeline, _ = _make(["/shoot/1.mp4", "/shoot/2.mp4"], [TRAVIS])
    placement = _placement(
        _camera(0, "/shoot/1.mp4", -1000.0),
        _camera(1, "/shoot/2.mp4", -1000.0),
        warnings=["1.mp4: refinement rejected (window spread too large); keeping coarse offset"],
    )
    monkeypatch.setattr(_PLACEMENT, lambda *a, **k: placement)

    with caplog.at_level(logging.WARNING):
        assert pipeline._stage_align() is True

    assert any("refinement rejected" in r.getMessage() for r in caplog.records)


def test_without_sources_legacy_alignment_is_untouched(monkeypatch):
    pipeline, errors = _make(["1.mp4", "2.mp4"])
    monkeypatch.setattr("multicam_editor.logic.audio_sync.align_cameras", lambda *a, **k: [])
    monkeypatch.setattr(_PLACEMENT, _boom("place_cameras_on_session"))

    assert pipeline._stage_align() is True

    assert pipeline._camera_offsets == {0: 0.0}
    assert pipeline._audio_source_alignments == {}
    assert pipeline._session_placement is None
    assert errors == []


def test_diarize_refuses_configured_sources_until_detection_supports_them():
    pipeline, errors = _make(["1.mp4", "2.mp4"], [TRAVIS])
    pipeline.speaker_switching_enabled = True

    assert pipeline._stage_diarize() is False

    assert len(errors) == 1
    assert "audio source" in errors[0].lower()


def test_diarize_with_switching_off_still_works_with_sources():
    pipeline, errors = _make(["1.mp4", "2.mp4"], [TRAVIS])
    pipeline.speaker_switching_enabled = False

    assert pipeline._stage_diarize() is True

    assert pipeline._speaker_segments == []
    assert errors == []


_PREFLIGHT_PATCH = "multicam_editor.logic.preflight.PreflightManager"


def _mock_preflight_pass(mock_cls):
    result = MagicMock()
    result.ok = True
    result.critical_errors = []
    result.warnings = []
    mock_cls.return_value.run_full_check.return_value = result


def test_run_stops_when_alignment_stage_fails():
    signals = ProcessingSignals()
    pipeline = ProcessingPipeline(["1.mp4", "2.mp4"], signals)

    with patch(_PREFLIGHT_PATCH) as preflight, \
         patch("multicam_editor.logic.processing_pipeline.probe") as probe, \
         patch.object(ProcessingPipeline, "_stage_align", return_value=False) as align, \
         patch.object(ProcessingPipeline, "_stage_diarize") as diarize, \
         patch.object(ProcessingPipeline, "_stage_decision", return_value=False), \
         patch.object(ProcessingPipeline, "_save_checkpoint") as save:
        _mock_preflight_pass(preflight)
        probe.return_value = SimpleNamespace(duration_ms=5000, error=None)
        result = pipeline.run()

    assert result.success is False
    assert "Alignment stage failed" in result.error
    align.assert_called_once()
    diarize.assert_not_called()
    assert [c.args[0] for c in save.call_args_list] == ["PROBE"]
