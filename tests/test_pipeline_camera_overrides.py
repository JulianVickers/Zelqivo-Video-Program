"""Plumbing for per-camera overrides: excluded coverage, run() validation, hand-off."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from multicam_editor.logic.processing_pipeline import ProcessingPipeline
from multicam_editor.logic.processing_worker import ProcessingThread, ProcessingWorker
from multicam_editor.logic.session_placement import CameraOverride
from multicam_editor.logic.timeline_ruler import build_camera_coverage
from multicam_editor.utils.signals import ProcessingSignals

_PIPE = "multicam_editor.logic.processing_pipeline"
_WORKER = "multicam_editor.logic.processing_worker"
_FILES = ["/v/0.mp4", "/v/1.mp4"]


class _Settings:
    def __init__(self, *args, **kwargs):
        pass

    def value(self, key, default=None, type=None):
        return default


def _camera(index, session_offset_ms, status="ok"):
    from multicam_editor.logic.session_placement import CameraPlacement

    return CameraPlacement(
        camera_index=index,
        path=f"/v/{index}.mp4",
        status=status,
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


def _make(durations):
    signals = ProcessingSignals()
    errors: list[str] = []
    signals.error.connect(lambda message: errors.append(message))
    files = [f"/v/{i}.mp4" for i in range(len(durations))]
    pipeline = ProcessingPipeline(files, signals)
    pipeline._probe_results = [SimpleNamespace(duration_ms=d) for d in durations]
    pipeline.speaker_switching_enabled = False
    pipeline._speaker_segments = []
    return pipeline, errors


def _session_pipeline(durations, statuses):
    pipeline, errors = _make(durations)
    cams = [_camera(i, 0.0, status) for i, status in enumerate(statuses)]
    pipeline._session_placement = _placement(*cams)
    pipeline._camera_offsets = {c.camera_index: c.slot_offset_ms for c in cams}
    return pipeline, errors


def _decide(pipeline):
    pipeline._qa_exporter = MagicMock()
    with patch(f"{_PIPE}.QSettings", _Settings):
        return pipeline._stage_decision()


def _run(**kwargs):
    signals = ProcessingSignals()
    errors: list[str] = []
    signals.error.connect(lambda message: errors.append(message))
    pipeline = ProcessingPipeline(list(_FILES), signals)
    return pipeline.run(**kwargs), errors


class TestExcludedCoverage:
    def test_excluded_camera_has_empty_coverage(self):
        cov = build_camera_coverage([0, 0, 0], [1000, 2000, 3000], excluded=frozenset({1}))
        assert [c.camera_index for c in cov] == [0, 1, 2]
        assert (cov[1].start_ms, cov[1].end_ms) == (0, 0)
        assert not cov[1].covers(0, 500)
        assert cov[0].covers(0, 1000)
        assert cov[2].covers(0, 3000)

    def test_excluded_camera_skips_the_duration_check(self):
        cov = build_camera_coverage([0, 0], [0, 3000], excluded=frozenset({0}))
        assert cov[1].covers(0, 3000)

    def test_excluded_index_out_of_range_raises(self):
        with pytest.raises(ValueError):
            build_camera_coverage([0, 0], [1000, 1000], excluded=frozenset({2}))

    def test_default_matches_the_old_positional_call(self):
        assert build_camera_coverage([0, 500], [1000, 2000]) == build_camera_coverage(
            [0, 500], [1000, 2000], excluded=frozenset()
        )


class TestExcludedCamerasInDecision:
    @pytest.mark.parametrize("excluded_index", [0, 1])
    def test_excluded_camera_is_never_used(self, excluded_index):
        statuses = ["ok", "ok"]
        statuses[excluded_index] = "excluded"
        pipeline, errors = _session_pipeline([60_000, 60_000], statuses)
        ok = _decide(pipeline)
        assert ok, errors
        used = {seg.camera_id for seg in pipeline._timeline_segments}
        assert used == {1 - excluded_index}

    def test_excluded_camera_with_unusable_duration_does_not_break_the_run(self):
        pipeline, errors = _session_pipeline([0, 60_000], ["excluded", "ok"])
        ok = _decide(pipeline)
        assert ok, errors
        assert errors == []

    def test_every_camera_excluded_is_a_loud_error(self):
        pipeline, errors = _session_pipeline([60_000, 60_000], ["excluded", "excluded"])
        assert _decide(pipeline) is False
        assert errors and "No camera has any footage" in errors[-1]


class TestRunValidation:
    def test_overrides_without_audio_sources_fail(self):
        result, errors = _run(camera_overrides={0: CameraOverride(exclude=True)})
        assert result.success is False
        assert errors and "camera overrides" in errors[0]

    def test_overrides_with_legacy_external_audio_fail(self):
        result, errors = _run(
            external_audio="/a.wav", camera_overrides={0: CameraOverride(exclude=True)}
        )
        assert result.success is False
        assert errors and "camera overrides" in errors[0]

    def test_override_for_a_nonexistent_camera_fails(self):
        with patch(f"{_PIPE}.validate_audio_sources"):
            result, errors = _run(
                audio_sources=[SimpleNamespace(path="/a.wav")],
                camera_overrides={5: CameraOverride(exclude=True)},
            )
        assert result.success is False
        assert errors and "camera 5" in errors[0]

    def test_override_value_must_be_a_camera_override(self):
        with patch(f"{_PIPE}.validate_audio_sources"):
            result, errors = _run(
                audio_sources=[SimpleNamespace(path="/a.wav")],
                camera_overrides={0: "exclude"},
            )
        assert result.success is False
        assert errors and "CameraOverride" in errors[0]

    def test_valid_overrides_pass_validation(self):
        pipeline = ProcessingPipeline(list(_FILES), ProcessingSignals())
        pipeline._audio_sources = [SimpleNamespace(path="/a.wav")]
        pipeline._validate_camera_overrides(None)
        pipeline._validate_camera_overrides({})
        pipeline._validate_camera_overrides({
            0: CameraOverride(manual_offset_ms=-35110.0),
            1: CameraOverride(exclude=True),
        })


def _align(overrides):
    pipeline, errors = _make([60_000, 60_000])
    pipeline._audio_sources = [SimpleNamespace(source_id="s0", path="/a.wav", channel_index=0)]
    pipeline._camera_overrides = overrides
    pipeline._qa_exporter = MagicMock()
    placement = _placement(_camera(0, 0.0), _camera(1, 0.0, "excluded"))
    target = "multicam_editor.logic.session_placement.place_cameras_on_session"
    with patch(target, return_value=placement) as place:
        ok = pipeline._align_to_session()
    return ok, errors, place


class TestAlignPassesOverrides:
    def test_overrides_reach_placement(self):
        overrides = {1: CameraOverride(exclude=True)}
        ok, errors, place = _align(overrides)
        assert ok, errors
        assert place.call_args.kwargs["overrides"] == overrides

    def test_no_overrides_means_none_reach_placement(self):
        ok, errors, place = _align({})
        assert ok, errors
        assert not place.call_args.kwargs.get("overrides")


class TestWorkerAndThreadHandOff:
    def test_worker_keeps_camera_overrides(self):
        overrides = {1: CameraOverride(exclude=True)}
        worker = ProcessingWorker(list(_FILES), camera_overrides=overrides)
        assert worker.camera_overrides == overrides

    def test_thread_keeps_camera_overrides(self):
        overrides = {1: CameraOverride(exclude=True)}
        thread = ProcessingThread(list(_FILES), camera_overrides=overrides)
        assert thread.camera_overrides == overrides

    def test_worker_passes_overrides_to_pipeline_run(self):
        overrides = {1: CameraOverride(exclude=True)}
        worker = ProcessingWorker(list(_FILES), camera_overrides=overrides)
        result = MagicMock(
            cancelled=True, success=False, error=None, speaker_segments=[],
            detection_time_s=0.0, detection_model="",
        )
        with patch(f"{_WORKER}.ProcessingPipeline") as pipeline_cls:
            pipeline_cls.return_value.run.return_value = result
            worker.run()
        assert pipeline_cls.return_value.run.call_args.kwargs["camera_overrides"] == overrides

    def test_thread_passes_overrides_to_worker(self):
        overrides = {1: CameraOverride(exclude=True)}
        thread = ProcessingThread(list(_FILES), camera_overrides=overrides)
        with patch(f"{_WORKER}.ProcessingWorker") as worker_cls:
            thread.run()
        assert worker_cls.call_args.kwargs["camera_overrides"] == overrides
