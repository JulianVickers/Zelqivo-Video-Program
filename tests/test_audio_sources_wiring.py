"""Main-window wiring for the multi-source audio dialog (commit 3)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtWidgets import QMessageBox

from multicam_editor.logic.audio_source_config import SourceRow
from multicam_editor.logic.switching_strategy import SwitchingStrategy

MW = "multicam_editor.ui.main_window"
AUTO = "Auto (best effort)"
SETTING_KEYS = ["magic/director/model_index"] + [
    f"processing/camera_{i}_speaker" for i in range(4)
]


@pytest.fixture
def main_window(qtbot, qapp):
    """Create MainWindow for testing (same as the fixture in test_ui.py)."""
    from multicam_editor.ui.main_window import MainWindow
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    yield window
    window.close()


@pytest.fixture(autouse=True)
def _restore_settings(main_window):
    saved = {k: main_window.settings.value(k) for k in SETTING_KEYS}
    yield
    for k, v in saved.items():
        if v is None:
            main_window.settings.remove(k)
        else:
            main_window.settings.setValue(k, v)


def _rows(tmp_path, names=("Travis", "Dakota", "Julian")):
    rows = []
    for n in names:
        p = tmp_path / f"{n}.wav"
        p.write_bytes(b"")
        rows.append(SourceRow(path=str(p), speaker_name=n))
    return rows


def _set_clips(window, monkeypatch, tmp_path, stems):
    clips = [
        SimpleNamespace(path=str(tmp_path / f"{s}.mp4"), duration_ms=60000)
        for s in stems
    ]
    monkeypatch.setattr(window.project, "clips", lambda: clips)
    window._refresh_camera_mapping_ui()
    return clips


def _accept(window, rows):
    with patch(f"{MW}.show_audio_sources_dialog", return_value=rows):
        window.btn_audio_sources.click()


class TestAudioSourcesButton:
    def test_starts_empty(self, main_window):
        assert main_window._audio_source_rows == []
        assert main_window.lbl_audio_sources.text() == "None"

    def test_accept_stores_rows_and_summary(self, main_window, tmp_path):
        rows = _rows(tmp_path)
        _accept(main_window, rows)
        assert main_window._audio_source_rows == rows
        assert main_window.lbl_audio_sources.text() == "3 sources, 3 speakers"

    def test_singular_summary(self, main_window, tmp_path):
        _accept(main_window, _rows(tmp_path, ("Julian",)))
        assert main_window.lbl_audio_sources.text() == "1 source, 1 speaker"

    def test_cancel_changes_nothing(self, main_window, tmp_path):
        rows = _rows(tmp_path)
        _accept(main_window, rows)
        _accept(main_window, None)
        assert main_window._audio_source_rows == rows

    def test_dialog_receives_current_rows(self, main_window, tmp_path):
        rows = _rows(tmp_path)
        _accept(main_window, rows)
        with patch(f"{MW}.show_audio_sources_dialog", return_value=None) as dlg:
            main_window.btn_audio_sources.click()
        assert list(dlg.call_args.args[0]) == rows


class TestMutualExclusion:
    def test_sources_uncheck_and_disable_legacy(self, main_window, tmp_path):
        main_window.chk_external_audio.setChecked(True)
        _accept(main_window, _rows(tmp_path))
        assert not main_window.chk_external_audio.isChecked()
        assert not main_window.chk_external_audio.isEnabled()

    def test_clearing_sources_reenables_legacy(self, main_window, tmp_path):
        _accept(main_window, _rows(tmp_path))
        _accept(main_window, [])
        assert main_window.chk_external_audio.isEnabled()
        assert main_window.lbl_audio_sources.text() == "None"


class TestCameraCombos:
    def test_combo_labels_and_values(self, main_window, monkeypatch, tmp_path):
        _set_clips(main_window, monkeypatch, tmp_path, ["Travis", "Dakota", "Julian"])
        _accept(main_window, _rows(tmp_path))
        combo = main_window._camera_combos[0]
        assert combo.itemText(0) == AUTO
        assert combo.itemText(1) == "Speaker 1 - Travis"
        assert combo.itemData(1) == "Speaker 1"
        assert combo.itemData(3) == "Speaker 3"

    def test_prefilled_by_file_stem(self, main_window, monkeypatch, tmp_path):
        _set_clips(main_window, monkeypatch, tmp_path, ["Julian", "Travis", "Dakota"])
        _accept(main_window, _rows(tmp_path))
        assert main_window.get_camera_speaker_mapping() == {
            0: "Speaker 3", 1: "Speaker 1", 2: "Speaker 2",
        }

    def test_mapping_returns_value_not_label(self, main_window, monkeypatch, tmp_path):
        _set_clips(main_window, monkeypatch, tmp_path, ["Travis", "Dakota"])
        _accept(main_window, _rows(tmp_path))
        for v in main_window.get_camera_speaker_mapping().values():
            assert " - " not in v

    def test_unmatched_camera_stays_auto(self, main_window, monkeypatch, tmp_path):
        _set_clips(main_window, monkeypatch, tmp_path, ["Travis", "Wide"])
        _accept(main_window, _rows(tmp_path))
        mapping = main_window.get_camera_speaker_mapping()
        assert mapping[0] == "Speaker 1"
        assert mapping[1].startswith("Auto")

    def test_saved_values_ignored_when_sources_present(
        self, main_window, monkeypatch, tmp_path
    ):
        main_window.settings.setValue("processing/camera_0_speaker", "Speaker 2")
        _set_clips(main_window, monkeypatch, tmp_path, ["Travis", "Dakota"])
        _accept(main_window, _rows(tmp_path))
        assert main_window.get_camera_speaker_mapping()[0] == "Speaker 1"

    def test_hand_edited_camera_survives_reaccept(
        self, main_window, monkeypatch, tmp_path
    ):
        rows = _rows(tmp_path)
        _set_clips(main_window, monkeypatch, tmp_path, ["Travis", "Dakota"])
        _accept(main_window, rows)
        combo = main_window._camera_combos[0]
        combo.setCurrentIndex(combo.findData("Speaker 2"))
        _accept(main_window, rows)
        assert main_window.get_camera_speaker_mapping()[0] == "Speaker 2"
        assert main_window.get_camera_speaker_mapping()[1] == "Speaker 2"

    def test_legacy_mode_unchanged(self, main_window, monkeypatch, tmp_path):
        _set_clips(main_window, monkeypatch, tmp_path, ["Travis", "Dakota"])
        mapping = main_window.get_camera_speaker_mapping()
        assert all(v == AUTO for v in mapping.values())
        assert main_window._camera_combos[0].findText("Speaker 1") >= 0


@pytest.fixture
def armed(main_window, monkeypatch, tmp_path):
    """Patch everything on_process_videos touches beyond the guards."""
    seen = SimpleNamespace(strategies=[], thread=MagicMock(), question=None, warning=None)

    def fake_preflight(strategy, show_dialog_callback=None):
        seen.strategies.append(strategy)
        return SimpleNamespace(final_strategy=strategy, gpu_available=True)

    stack = [
        patch(f"{MW}.run_gpu_preflight_check", side_effect=fake_preflight),
        patch(f"{MW}.should_warn_long_project", return_value=False),
        patch(f"{MW}.check_preflight_warnings", return_value=[]),
        patch(f"{MW}.ProcessingProgressDialog", MagicMock()),
        patch(f"{MW}.ProcessingThread", seen.thread),
        patch.object(main_window, "_generate_output_path", return_value=str(tmp_path / "out.mp4")),
    ]
    seen.question = patch.object(
        QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
    )
    seen.warning = patch.object(
        QMessageBox, "warning", return_value=QMessageBox.StandardButton.Ok
    )
    started = [p.start() for p in stack]
    seen.question_mock = seen.question.start()
    seen.warning_mock = seen.warning.start()
    yield seen
    seen.question.stop()
    seen.warning.stop()
    for p in stack:
        p.stop()


class TestProcessGuards:
    def _configure(self, window, monkeypatch, tmp_path, stems=("Travis", "Dakota")):
        _set_clips(window, monkeypatch, tmp_path, list(stems))
        _accept(window, _rows(tmp_path))

    def test_sources_passed_to_thread(self, main_window, monkeypatch, tmp_path, armed):
        main_window.settings.setValue("magic/director/model_index", 0)
        self._configure(main_window, monkeypatch, tmp_path)
        main_window.on_process_videos()
        kwargs = armed.thread.call_args.kwargs
        assert kwargs["external_audio"] is None
        assert len(kwargs["audio_sources"]) == 3

    def test_fast_strategy_no_prompt(self, main_window, monkeypatch, tmp_path, armed):
        main_window.settings.setValue("magic/director/model_index", 0)
        self._configure(main_window, monkeypatch, tmp_path)
        main_window.on_process_videos()
        armed.question_mock.assert_not_called()
        assert armed.strategies == [SwitchingStrategy.FAST_RULES]

    def test_non_fast_yes_overrides_locally(self, main_window, monkeypatch, tmp_path, armed):
        main_window.settings.setValue("magic/director/model_index", 1)
        self._configure(main_window, monkeypatch, tmp_path)
        main_window.on_process_videos()
        armed.question_mock.assert_called_once()
        assert armed.strategies == [SwitchingStrategy.FAST_RULES]
        assert main_window.settings.value("magic/director/model_index", type=int) == 1
        armed.thread.assert_called_once()

    def test_non_fast_no_aborts(self, main_window, monkeypatch, tmp_path, armed):
        main_window.settings.setValue("magic/director/model_index", 2)
        self._configure(main_window, monkeypatch, tmp_path)
        armed.question_mock.return_value = QMessageBox.StandardButton.No
        main_window.on_process_videos()
        armed.thread.assert_not_called()
        assert armed.strategies == []

    def test_all_auto_stops_with_message(self, main_window, monkeypatch, tmp_path, armed):
        main_window.settings.setValue("magic/director/model_index", 0)
        self._configure(main_window, monkeypatch, tmp_path, stems=("Wide1", "Wide2"))
        main_window.on_process_videos()
        armed.warning_mock.assert_called()
        armed.thread.assert_not_called()

    def test_no_sources_keeps_legacy_path(self, main_window, monkeypatch, tmp_path, armed):
        main_window.settings.setValue("magic/director/model_index", 1)
        _set_clips(main_window, monkeypatch, tmp_path, ["a", "b"])
        main_window.on_process_videos()
        armed.question_mock.assert_not_called()
        assert armed.strategies == [SwitchingStrategy.BALANCED_LIPS_ENERGY]
        assert armed.thread.call_args.kwargs.get("audio_sources") in (None, [])
