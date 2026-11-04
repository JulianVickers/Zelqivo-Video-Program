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


def test_diarize_with_sources_and_nothing_mapped_fails(detection):
    pipeline, errors = _make(["1.mp4", "2.mp4"], [TRAVIS])
    pipeline.speaker_switching_enabled = True

    assert pipeline._stage_diarize() is False

    assert len(errors) == 1
    assert "mapped" in errors[0]


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


# --- camera -> speaker mapping becomes per-camera refinement sources ------

from multicam_editor.logic.pipeline_config import PipelineConfig


def _speaker(source_id, path, speaker_id):
    return AudioSource(source_id=source_id, path=path, speaker_id=speaker_id)


def _align_with_map(speaker_map, sources, files=("a.mp4", "b.mp4", "c.mp4")):
    pipeline, errors = _make(files, sources)
    pipeline._config = PipelineConfig(speaker_to_cameras_map=speaker_map)
    cameras = [_camera(i, f, -1000.0 * (i + 1)) for i, f in enumerate(files)]
    with patch(_PLACEMENT, return_value=_placement(*cameras)) as placed:
        ok = pipeline._align_to_session()
    return ok, errors, placed



def _mapping(speaker_map, sources, files=("a.mp4", "b.mp4", "c.mp4")):
    pipeline, _ = _make(files, sources)
    pipeline._config = PipelineConfig(speaker_to_cameras_map=speaker_map)
    return pipeline._camera_sources_from_mapping()


def test_no_mapping_has_no_camera_sources():
    assert _mapping({}, [_speaker("t", "/rec/t.wav", "travis")]) is None


def test_placement_is_given_no_camera_sources():
    """Every camera refines against the full session mix."""
    ok, errors, placed = _align_with_map(
        {0: [0]}, [_speaker("t", "/rec/t.wav", "travis")]
    )
    assert ok and not errors
    assert "camera_sources" not in placed.call_args.kwargs



def test_mapping_becomes_per_camera_source_paths():
    sources = [
        _speaker("t", "/rec/t.wav", "travis"),
        _speaker("d", "/rec/d.wav", "dakota"),
    ]
    assert _mapping({0: [0], 1: [1, 2]}, sources) == {
        0: ["/rec/t.wav"],
        1: ["/rec/d.wav"],
        2: ["/rec/d.wav"],
    }



def test_speaker_with_several_sources_uses_all_of_them():
    sources = [
        _speaker("t1", "/rec/t1.wav", "travis"),
        _speaker("t2", "/rec/t2.wav", "travis"),
        _speaker("d", "/rec/d.wav", "dakota"),
    ]
    assert _mapping({0: [0]}, sources) == {0: ["/rec/t1.wav", "/rec/t2.wav"]}



def test_speaker_numbers_follow_first_appearance_in_sources():
    sources = [
        _speaker("d", "/rec/d.wav", "dakota"),
        _speaker("t", "/rec/t.wav", "travis"),
    ]
    assert _mapping({0: [1]}, sources) == {1: ["/rec/d.wav"]}


def test_speaker_beyond_available_sources_fails_before_placement():
    sources = [
        _speaker("t", "/rec/t.wav", "travis"),
        _speaker("d", "/rec/d.wav", "dakota"),
    ]
    ok, errors, placed = _align_with_map({5: [0]}, sources)
    assert ok is False
    assert len(errors) == 1 and "Speaker 6" in errors[0]
    placed.assert_not_called()


def test_mapping_without_any_source_speakers_fails():
    ok, errors, placed = _align_with_map(
        {0: [0]}, [AudioSource(source_id="m", path="/rec/m.wav")]
    )
    assert ok is False
    assert len(errors) == 1 and "Speaker 1" in errors[0]
    placed.assert_not_called()


def test_camera_mapped_to_two_speakers_fails():
    sources = [
        _speaker("t", "/rec/t.wav", "travis"),
        _speaker("d", "/rec/d.wav", "dakota"),
    ]
    ok, errors, placed = _align_with_map({0: [0], 1: [0]}, sources)
    assert ok is False
    assert len(errors) == 1 and "camera 0" in errors[0]
    placed.assert_not_called()


# --- detection on configured audio sources (FAST_RULES) -------------------

import os

import pytest

from multicam_editor.logic.active_speaker import SpeakerSegment
from multicam_editor.logic.switching_strategy import SwitchingStrategy

_EXTRACT = "multicam_editor.utils.ffmpeg.extract_audio_to_wav"
_BACKEND = "multicam_editor.logic.processing_pipeline.RealEnergyVADBackend"


@pytest.fixture
def detection(monkeypatch):
    """Fake WAV extraction and energy backend; records what detection was fed."""
    state = SimpleNamespace(
        extract_calls=[], backends=[], segments=[], extract_error=None
    )

    def fake_extract(path, sample_rate=16000, mono=True, channel_index=None):
        state.extract_calls.append((path, channel_index))
        if state.extract_error:
            return SimpleNamespace(
                success=False, output_path=None, error=state.extract_error
            )
        tag = "all" if channel_index is None else f"ch{channel_index}"
        return SimpleNamespace(
            success=True,
            output_path=f"/tmp/{os.path.basename(path)}.{tag}.wav",
            error=None,
        )

    class FakeBackend:
        def __init__(self, **kwargs):
            self.paths = None
            self.num_channels = None
            state.backends.append(self)

        def set_camera_audio_paths(self, paths):
            self.paths = list(paths)

        def diarize(self, audio_path, num_channels=2):
            self.num_channels = num_channels
            return list(state.segments)

    monkeypatch.setattr(_EXTRACT, fake_extract)
    monkeypatch.setattr(_BACKEND, FakeBackend)
    monkeypatch.setattr(
        ProcessingPipeline,
        "_load_switching_strategy",
        lambda self: SwitchingStrategy.FAST_RULES,
    )
    return state


def _seg(start, end, speaker):
    return SpeakerSegment(start_ms=start, end_ms=end, speaker_id=speaker)


def _rows(pipeline):
    return [(s.start_ms, s.end_ms, s.speaker_id) for s in pipeline._speaker_segments]


def _diarize(speaker_map, sources, files=("a.mp4", "b.mp4", "c.mp4"), placement=None):
    pipeline, errors = _make(files, sources)
    pipeline._config = PipelineConfig(speaker_to_cameras_map=speaker_map)
    pipeline._session_placement = placement
    ok = pipeline._stage_diarize()
    return pipeline, ok, errors


def _two_speakers():
    return [
        AudioSource(source_id="t", path="/rec/t.wav", speaker_id="travis"),
        AudioSource(source_id="d", path="/rec/d.wav", speaker_id="dakota"),
    ]


def test_detection_runs_on_each_mapped_speakers_source(detection):
    detection.segments = [_seg(0, 4000, 0), _seg(4000, 9000, 1)]
    pipeline, ok, errors = _diarize({0: [0], 1: [1]}, _two_speakers())
    assert ok and not errors
    assert detection.extract_calls == [("/rec/t.wav", None), ("/rec/d.wav", None)]
    backend = detection.backends[0]
    assert backend.paths == ["/tmp/t.wav.all.wav", "/tmp/d.wav.all.wav"]
    assert backend.num_channels == 2
    assert _rows(pipeline) == [(0, 4000, 0), (4000, 9000, 1)]


def test_returned_speaker_indices_become_real_camera_indices(detection):
    detection.segments = [_seg(0, 4000, 0), _seg(4000, 9000, 1)]
    pipeline, ok, _ = _diarize({0: [2], 1: [0]}, _two_speakers())
    assert ok
    assert _rows(pipeline) == [(0, 4000, 2), (4000, 9000, 0)]


def test_speaker_with_several_cameras_uses_the_lowest_numbered(detection):
    detection.segments = [_seg(0, 5000, 0)]
    pipeline, ok, _ = _diarize({0: [1, 2]}, _two_speakers())
    assert ok
    assert _rows(pipeline) == [(0, 5000, 1)]
    # the unmapped speaker is never extracted or analysed
    assert detection.extract_calls == [("/rec/t.wav", None)]
    assert detection.backends[0].num_channels == 1


def test_excluded_camera_is_not_chosen_for_a_speaker(detection):
    detection.segments = [_seg(0, 5000, 0)]
    placement = _placement(
        _camera(0, "a.mp4", 0.0),
        _camera(1, "b.mp4", 0.0, status="excluded"),
        _camera(2, "c.mp4", 0.0),
    )
    pipeline, ok, _ = _diarize({0: [1, 2]}, _two_speakers(), placement=placement)
    assert ok
    assert _rows(pipeline) == [(0, 5000, 2)]


def test_channel_sources_are_extracted_by_channel(detection):
    detection.segments = [_seg(0, 3000, 0), _seg(3000, 6000, 1)]
    sources = [
        AudioSource(source_id="c0", path="/rec/mixer.wav", channel_index=0, speaker_id="travis"),
        AudioSource(source_id="c1", path="/rec/mixer.wav", channel_index=1, speaker_id="dakota"),
    ]
    pipeline, ok, _ = _diarize({0: [0], 1: [1]}, sources)
    assert ok
    assert detection.extract_calls == [("/rec/mixer.wav", 0), ("/rec/mixer.wav", 1)]
    assert detection.backends[0].paths == [
        "/tmp/mixer.wav.ch0.wav",
        "/tmp/mixer.wav.ch1.wav",
    ]


def test_several_detection_sources_for_one_speaker_is_an_error(detection):
    sources = [
        AudioSource(source_id="t1", path="/rec/t1.wav", speaker_id="travis"),
        AudioSource(source_id="t2", path="/rec/t2.wav", speaker_id="travis"),
    ]
    _, ok, errors = _diarize({0: [0]}, sources)
    assert ok is False
    assert len(errors) == 1 and "not implemented" not in errors[0]
    assert detection.extract_calls == [] and detection.backends == []


def test_use_for_detection_false_picks_the_other_source(detection):
    detection.segments = [_seg(0, 5000, 0)]
    sources = [
        AudioSource(
            source_id="t1", path="/rec/t1.wav", speaker_id="travis",
            use_for_detection=False,
        ),
        AudioSource(source_id="t2", path="/rec/t2.wav", speaker_id="travis"),
    ]
    _, ok, _ = _diarize({0: [0]}, sources)
    assert ok
    assert detection.extract_calls == [("/rec/t2.wav", None)]


def test_nothing_mapped_is_an_error(detection):
    _, ok, errors = _diarize({}, _two_speakers())
    assert ok is False
    assert len(errors) == 1 and "mapped" in errors[0]
    assert detection.extract_calls == []


def test_other_strategies_are_refused_with_sources(detection, monkeypatch):
    monkeypatch.setattr(
        ProcessingPipeline,
        "_load_switching_strategy",
        lambda self: SwitchingStrategy.BEST_LIPS,
    )
    _, ok, errors = _diarize({0: [0]}, _two_speakers())
    assert ok is False
    assert len(errors) == 1 and "Fast" in errors[0]
    assert detection.extract_calls == []


def test_extraction_failure_stops_the_stage(detection):
    detection.extract_error = "boom"
    _, ok, errors = _diarize({0: [0]}, _two_speakers())
    assert ok is False
    assert len(errors) == 1 and "boom" in errors[0]
    assert detection.backends == []


def test_speaker_with_all_sources_disabled_for_detection_is_an_error(detection):
    sources = [
        AudioSource(
            source_id="t1", path="/rec/t1.wav", speaker_id="travis",
            use_for_detection=False,
        ),
    ]
    _, ok, errors = _diarize({0: [0]}, sources)
    assert ok is False
    assert len(errors) == 1 and "enabled for detection" in errors[0]
    assert detection.extract_calls == [] and detection.backends == []


import wave

import numpy as np


@pytest.fixture
def real_energy_detection(monkeypatch, tmp_path):
    """Fake only WAV extraction; the real energy backend analyses the files."""
    wavs = {}

    def fake_extract(path, sample_rate=16000, mono=True, channel_index=None):
        return SimpleNamespace(success=True, output_path=wavs[path], error=None)

    monkeypatch.setattr(_EXTRACT, fake_extract)
    monkeypatch.setattr(
        ProcessingPipeline,
        "_load_switching_strategy",
        lambda self: SwitchingStrategy.FAST_RULES,
    )

    def write(source_path, samples):
        out = str(tmp_path / (os.path.basename(source_path) + ".wav"))
        with wave.open(out, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
        wavs[source_path] = out

    return SimpleNamespace(write=write)


def test_quiet_mixer_exports_are_not_gated_away(real_energy_detection):
    """Real PageBrahs levels: speech RMS 0.009/0.024/0.013 over a 0.0002 floor.

    The legacy 0.01 silence floor put every gate above these levels, so
    detection collapsed to a single segment.
    """
    rate, turn_s = 16000, 6
    levels = [0.009, 0.024, 0.013]  # travis, dakota, julian
    turns = [0, 1, 2] * 5
    total = len(turns) * turn_s * rate
    rng = np.random.default_rng(7)
    signals = [rng.normal(0, 0.0002, total) for _ in levels]
    for t, speaker in enumerate(turns):
        a, b = t * turn_s * rate, (t + 1) * turn_s * rate
        signals[speaker][a:b] = rng.normal(0, levels[speaker], b - a)

    sources = [
        AudioSource(source_id="t", path="/rec/t.wav", speaker_id="travis"),
        AudioSource(source_id="d", path="/rec/d.wav", speaker_id="dakota"),
        AudioSource(source_id="j", path="/rec/j.wav", speaker_id="julian"),
    ]
    for source, samples in zip(sources, signals):
        real_energy_detection.write(source.path, samples)

    camera_of = [2, 0, 1]  # speaker index -> camera index
    pipeline, ok, errors = _diarize({0: [2], 1: [0], 2: [1]}, sources)
    assert ok and not errors

    rows = _rows(pipeline)
    assert len(rows) >= 12, rows

    def camera_at(ms):
        return next((c for s, e, c in rows if s <= ms < e), None)

    for t, speaker in enumerate(turns):
        middle = int((t * turn_s + turn_s / 2) * 1000)
        assert camera_at(middle) == camera_of[speaker], (t, middle, rows)


from multicam_editor.logic.decision_engine import CutSegment


def _write_wav(path, samples):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def test_activity_from_wav_marks_speech_windows(tmp_path):
    from multicam_editor.logic.wide_shot import activity_from_wav

    rng = np.random.default_rng(3)
    samples = rng.normal(0, 0.0002, 12 * 16000)
    samples[6 * 16000:] = rng.normal(0, 0.02, 6 * 16000)
    path = tmp_path / "s.wav"
    _write_wav(path, samples)
    track = activity_from_wav(str(path))
    assert len(track) == 60
    assert not any(track[:30]) and all(track[30:])


def test_activity_from_wav_all_silence_is_inactive(tmp_path):
    from multicam_editor.logic.wide_shot import activity_from_wav

    path = tmp_path / "z.wav"
    _write_wav(path, np.zeros(5 * 16000))
    track = activity_from_wav(str(path))
    assert len(track) == 25 and not any(track)


def test_detection_keeps_activity_by_camera(real_energy_detection):
    rate, turn_s = 16000, 6
    turns = [0, 1, 0, 1]
    total = len(turns) * turn_s * rate
    rng = np.random.default_rng(11)
    signals = [rng.normal(0, 0.0002, total) for _ in range(2)]
    for t, speaker in enumerate(turns):
        a, b = t * turn_s * rate, (t + 1) * turn_s * rate
        signals[speaker][a:b] = rng.normal(0, 0.02, b - a)
    sources = _two_speakers()
    for source, samples in zip(sources, signals):
        real_energy_detection.write(source.path, samples)

    pipeline, ok, errors = _diarize({0: [1], 1: [0]}, sources)
    assert ok and not errors
    act = pipeline._source_activity
    assert sorted(act) == [0, 1]
    assert len(act[1]) == 120
    assert act[1][15] and not act[0][15]  # 3 s: speaker 0 (camera 1) talks
    assert act[0][45] and not act[1][45]  # 9 s: speaker 1 (camera 0) talks


def _wide_pipeline(speaker_map, durations=(60000,) * 4, excluded=(), activity=None):
    files = ("a.mp4", "b.mp4", "c.mp4", "d.mp4")[: len(durations)]
    pipeline, _ = _make(files, [_speaker("t", "/rec/t.wav", "travis")])
    pipeline._config = PipelineConfig(speaker_to_cameras_map=speaker_map)
    pipeline._session_placement = SimpleNamespace(cameras=[
        SimpleNamespace(
            camera_index=i,
            session_offset_ms=0.0,
            status="excluded" if i in excluded else "ok",
        )
        for i in range(len(files))
    ])
    pipeline._probe_results = [SimpleNamespace(duration_ms=d) for d in durations]
    pipeline._cut_plan = [CutSegment(0, 60000, 0)]
    pipeline._source_activity = activity or {}
    return pipeline


def _plan(pipeline):
    return [(c.start_ms, c.end_ms, c.camera_id) for c in pipeline._cut_plan]


def test_wide_shot_cutaways_use_the_unmapped_camera():
    pipeline = _wide_pipeline({0: [0], 1: [1], 2: [2]})
    pipeline._apply_wide_shot()
    assert _plan(pipeline) == [
        (0, 20000, 0), (20000, 23000, 3), (23000, 43000, 0),
        (43000, 46000, 3), (46000, 60000, 0),
    ]


def test_wide_shot_skipped_when_every_camera_is_mapped():
    pipeline = _wide_pipeline({0: [0], 1: [1], 2: [2], 3: [3]})
    pipeline._apply_wide_shot()
    assert _plan(pipeline) == [(0, 60000, 0)]


def test_wide_shot_skipped_when_nothing_is_mapped():
    pipeline = _wide_pipeline({})
    pipeline._apply_wide_shot()
    assert _plan(pipeline) == [(0, 60000, 0)]


def test_wide_shot_skipped_when_the_wide_camera_is_excluded():
    pipeline = _wide_pipeline({0: [0], 1: [1], 2: [2]}, excluded=(3,))
    pipeline._apply_wide_shot()
    assert _plan(pipeline) == [(0, 60000, 0)]


def test_wide_shot_stays_inside_the_wide_cameras_footage():
    pipeline = _wide_pipeline(
        {0: [0], 1: [1], 2: [2]}, durations=(60000, 60000, 60000, 15000)
    )
    pipeline._apply_wide_shot()
    assert _plan(pipeline) == [(0, 60000, 0)]
