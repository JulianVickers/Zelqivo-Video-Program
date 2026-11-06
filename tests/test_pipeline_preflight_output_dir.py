"""The pipeline must give preflight an output *directory*, never the file path."""
from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from multicam_editor.logic.preflight import PreflightManager
from multicam_editor.logic.processing_pipeline import ProcessingPipeline
from multicam_editor.utils.signals import ProcessingSignals

_PREFLIGHT_PATCH = "multicam_editor.logic.preflight.PreflightManager"


def _passing_result():
    result = MagicMock()
    result.ok = True
    result.critical_errors = []
    result.warnings = []
    return result


def _run_until_align(output_path):
    """Run the pipeline; it stops right after preflight and probe."""
    pipeline = ProcessingPipeline(["1.mp4", "2.mp4"], ProcessingSignals())
    with patch("multicam_editor.logic.processing_pipeline.probe") as probe, \
         patch.object(ProcessingPipeline, "_stage_align", return_value=False), \
         patch.object(ProcessingPipeline, "_save_checkpoint"):
        probe.return_value = SimpleNamespace(duration_ms=5000, error=None)
        return pipeline.run(output_path=output_path)


def _second_arg(preflight_cls):
    call = preflight_cls.return_value.run_full_check.call_args
    return call.args[1] if len(call.args) > 1 else call.kwargs["output_dir"]


def test_preflight_receives_parent_directory_of_output_file(tmp_path):
    out = str(tmp_path / "multicam_2026-11-05_20-17.mp4")
    with patch(_PREFLIGHT_PATCH) as preflight:
        preflight.return_value.run_full_check.return_value = _passing_result()
        _run_until_align(out)
    assert _second_arg(preflight) == str(tmp_path)


def test_preflight_receives_temp_dir_without_output_path():
    with patch(_PREFLIGHT_PATCH) as preflight:
        preflight.return_value.run_full_check.return_value = _passing_result()
        _run_until_align(None)
    assert _second_arg(preflight) == tempfile.gettempdir()


def test_real_write_check_does_not_create_directory_at_output_file_path(tmp_path):
    """Regression: the write check used to makedirs() the output file's own path."""
    out = tmp_path / "multicam_2026-11-05_20-17.mp4"

    def run_real_write_check(self_, input_files, output_dir=None):
        self_._check_write_permissions(output_dir)
        return _passing_result()

    with patch.object(
        PreflightManager, "run_full_check", autospec=True, side_effect=run_real_write_check
    ):
        _run_until_align(str(out))
    assert not out.is_dir()
    assert not os.path.exists(out)
