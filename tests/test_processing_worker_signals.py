"""Tests for ProcessingWorker signals."""

from unittest.mock import MagicMock, patch

from multicam_editor.logic.audio_sources import (
    AudioSource,
    TimelineBoundaryConfig,
    TimelineBoundaryPolicy,
)
from multicam_editor.logic.processing_pipeline import PipelineResult
from multicam_editor.logic.processing_worker import ProcessingWorker


class TestProcessingWorkerSignals:
    """Test signal emissions from ProcessingWorker."""

    def test_detection_stats_signal_emitted(self):
        """Worker should emit detection_stats when result contains them."""
        worker = ProcessingWorker(
            input_files=["file1.mp4", "file2.mp4"],
            speaker_switching_enabled=True,
        )

        mock_stats_slot = MagicMock()
        worker.detection_stats.connect(mock_stats_slot)

        with patch(
            "multicam_editor.logic.processing_worker.ProcessingPipeline"
        ) as mock_pipeline:
            mock_pipeline.return_value.run.return_value = PipelineResult(
                success=True,
                output_path="out.mp4",
                detection_time_s=42.5,
                detection_model="Test Model",
            )

            worker.run()

        mock_stats_slot.assert_called_once_with(42.5, "Test Model")

    def test_detection_stats_signal_not_emitted_if_zero(self):
        """Worker should not emit detection_stats if time is 0."""
        worker = ProcessingWorker(
            input_files=["file1.mp4", "file2.mp4"],
            speaker_switching_enabled=True,
        )

        mock_stats_slot = MagicMock()
        worker.detection_stats.connect(mock_stats_slot)

        with patch(
            "multicam_editor.logic.processing_worker.ProcessingPipeline"
        ) as mock_pipeline:
            mock_pipeline.return_value.run.return_value = PipelineResult(
                success=True,
                output_path="out.mp4",
                detection_time_s=0.0,
                detection_model="",
            )

            worker.run()

        mock_stats_slot.assert_not_called()

    def test_audio_sources_are_passed_to_pipeline(self):
        """Worker should forward configured audio sources to pipeline.run()."""
        source = AudioSource(
            source_id="mixer-channel-1",
            path="/recordings/mixer.wav",
            channel_index=0,
            speaker_id="travis",
        )
        worker = ProcessingWorker(
            input_files=["camera-a.mp4", "camera-b.mp4"],
            audio_sources=[source],
        )

        with patch(
            "multicam_editor.logic.processing_worker.ProcessingPipeline"
        ) as mock_pipeline:
            mock_pipeline.return_value.run.return_value = PipelineResult(
                success=True,
                output_path="out.mp4",
            )
            worker.run()

        mock_pipeline.return_value.run.assert_called_once_with(
            external_audio=None,
            resolution="1080p",
            output_path=None,
            audio_sources=[source],
            camera_overrides={},
        )

    def test_timeline_boundary_config_is_passed_to_pipeline(self):
        """Worker should forward an explicit timeline policy to the pipeline."""
        config = TimelineBoundaryConfig(
            policy=TimelineBoundaryPolicy.PRESERVE_ALL_MEDIA
        )
        worker = ProcessingWorker(
            input_files=["camera-a.mp4", "camera-b.mp4"],
            timeline_boundary_config=config,
        )

        with patch(
            "multicam_editor.logic.processing_worker.ProcessingPipeline"
        ) as mock_pipeline:
            mock_pipeline.return_value.run.return_value = PipelineResult(
                success=True,
                output_path="out.mp4",
            )
            worker.run()

        assert mock_pipeline.call_args.kwargs["timeline_boundary_config"] == config
