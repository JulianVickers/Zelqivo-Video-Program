"""Offscreen tests for AudioSourcesDialog."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QFileDialog

from multicam_editor.logic.audio_source_config import SourceRow
from multicam_editor.ui.audio_sources_dialog import (
    COL_DETECT,
    COL_GAIN,
    COL_NAME,
    COL_NUMBER,
    AudioSourcesDialog,
    show_audio_sources_dialog,
)


@pytest.fixture
def files(tmp_path):
    paths = []
    for name in ("session_Travis.wav", "session_Dakota.wav", "session_Julian.wav"):
        p = tmp_path / name
        p.write_bytes(b"")
        paths.append(str(p))
    return paths


def make(qtbot, rows=()):
    dlg = AudioSourcesDialog(rows)
    qtbot.addWidget(dlg)
    return dlg


def names(dlg):
    return [r.speaker_name for r in dlg.rows()]


def test_empty_dialog_is_valid(qtbot):
    dlg = make(qtbot)
    assert dlg.rows() == []
    assert dlg.btn_ok.isEnabled()
    assert dlg.lbl_problems.text() == ""


def test_add_files_to_empty_table_uses_default_names(qtbot, files):
    dlg = make(qtbot)
    with patch.object(QFileDialog, "getOpenFileNames", return_value=(files, "")):
        dlg.btn_add.click()
    assert names(dlg) == ["Travis", "Dakota", "Julian"]
    assert [dlg.table.item(r, COL_NUMBER).text() for r in range(3)] == [
        "Speaker 1", "Speaker 2", "Speaker 3",
    ]
    assert dlg.btn_ok.isEnabled()


def test_cancelled_file_dialog_adds_nothing(qtbot):
    dlg = make(qtbot)
    with patch.object(QFileDialog, "getOpenFileNames", return_value=([], "")):
        dlg.btn_add.click()
    assert dlg.rows() == []


def test_adding_later_keeps_edited_names_and_uses_stem(qtbot, files, tmp_path):
    dlg = make(qtbot)
    dlg.add_paths(files[:2])
    dlg.table.item(0, COL_NAME).setText("Trav")
    extra = tmp_path / "wide_mic.wav"
    extra.write_bytes(b"")
    dlg.add_paths([str(extra)])
    assert names(dlg) == ["Trav", "Dakota", "wide_mic"]


def test_duplicate_paths_skipped(qtbot, files):
    dlg = make(qtbot)
    dlg.add_paths(files)
    dlg.add_paths([files[0]])
    assert len(dlg.rows()) == 3


def test_prefilled_rows_round_trip(qtbot, files):
    rows = [
        SourceRow(files[0], "Travis", True, False, -3.5),
        SourceRow(files[1], "Dakota", False, True, 0.0),
    ]
    dlg = make(qtbot, rows)
    assert dlg.rows() == rows


def test_move_down_and_up_renumbers(qtbot, files):
    dlg = make(qtbot)
    dlg.add_paths(files)
    dlg.table.selectRow(0)
    dlg.btn_down.click()
    assert names(dlg) == ["Dakota", "Travis", "Julian"]
    assert dlg.table.currentRow() == 1
    assert dlg.table.item(1, COL_NUMBER).text() == "Speaker 2"
    dlg.btn_up.click()
    dlg.btn_up.click()  # already at the top: no-op
    assert names(dlg) == ["Travis", "Dakota", "Julian"]


def test_remove_row(qtbot, files):
    dlg = make(qtbot)
    dlg.add_paths(files)
    dlg.table.selectRow(1)
    dlg.btn_remove.click()
    assert names(dlg) == ["Travis", "Julian"]


def test_duplicate_detection_source_for_one_speaker_blocks_ok(qtbot, files):
    dlg = make(qtbot)
    dlg.add_paths(files[:2])
    dlg.table.item(1, COL_NAME).setText("Travis")
    assert not dlg.btn_ok.isEnabled()
    assert "2 sources used for detection" in dlg.lbl_problems.text()
    # a room mic that is output-only is fine
    dlg.table.item(1, COL_DETECT).setCheckState(Qt.CheckState.Unchecked)
    assert dlg.btn_ok.isEnabled()
    assert dlg.table.item(0, COL_NUMBER).text() == dlg.table.item(1, COL_NUMBER).text() == "Speaker 1"


def test_blank_name_blocks_ok_and_shows_dash(qtbot, files):
    dlg = make(qtbot)
    dlg.add_paths(files[:1])
    dlg.table.item(0, COL_NAME).setText("  ")
    assert not dlg.btn_ok.isEnabled()
    assert dlg.table.item(0, COL_NUMBER).text() == "—"


def test_bad_gain_blocks_ok_and_good_gain_parses(qtbot, files):
    dlg = make(qtbot)
    dlg.add_paths(files[:1])
    dlg.table.item(0, COL_GAIN).setText("loud")
    assert not dlg.btn_ok.isEnabled()
    dlg.table.item(0, COL_GAIN).setText("-6,5")
    assert dlg.btn_ok.isEnabled()
    assert dlg.rows()[0].gain_db == -6.5


def test_missing_file_blocks_ok(qtbot, tmp_path):
    dlg = make(qtbot, [SourceRow(str(tmp_path / "gone.wav"), "A")])
    assert not dlg.btn_ok.isEnabled()
    assert "file not found" in dlg.lbl_problems.text()


def test_show_helper_returns_rows_on_accept_and_none_on_cancel(qtbot, files):
    rows = [SourceRow(files[0], "Travis")]
    with patch.object(AudioSourcesDialog, "exec", return_value=QDialog.DialogCode.Accepted):
        assert show_audio_sources_dialog(rows) == rows
    with patch.object(AudioSourcesDialog, "exec", return_value=QDialog.DialogCode.Rejected):
        assert show_audio_sources_dialog(rows) is None
