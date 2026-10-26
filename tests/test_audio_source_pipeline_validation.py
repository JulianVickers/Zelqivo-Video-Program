"""Regression tests for external audio-source pipeline configuration."""
from unittest.mock import patch

from multicam_editor.logic.audio_sources import AudioSource
from multicam_editor.utils.signals import ProcessingSignals
from multicam_editor.logic.processing_pipeline import ProcessingPipeline


def _source(source_id: str = "mixer-channel-1") -> AudioSource:
    return AudioSource(
        source_id=source_id,
        path="/recordings/mixer.wav",
        channel_index=0,
        speaker_id="travis",
    )


def test_duplicate_audio_source_ids_fail_before_qa_run_starts():
    pipeline = ProcessingPipeline(input_files=["file1.mp4", "file2.mp4"], signals=ProcessingSignals())
    with patch.object(pipeline._qa_exporter, "start_run") as start_run:
        result = pipeline.run(audio_sources=[_source(), _source()])

    assert not result.success
    assert "Duplicate" in (result.error or "") or "unique" in (result.error or "").lower()
    start_run.assert_not_called()


def test_legacy_external_audio_and_audio_sources_conflict_before_qa_run_starts():
    pipeline = ProcessingPipeline(input_files=["file1.mp4", "file2.mp4"], signals=ProcessingSignals())
    with patch.object(pipeline._qa_exporter, "start_run") as start_run:
        result = pipeline.run(
            external_audio="/recordings/legacy.wav",
            audio_sources=[_source()],
        )

    assert not result.success
    assert "Pass either legacy external_audio or audio_sources, not both" in (result.error or "")
    start_run.assert_not_called()
