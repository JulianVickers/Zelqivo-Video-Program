"""Dialog for configuring independent external audio sources (one file per speaker).

Thin Qt layer over ``logic.audio_source_config``: all validation, naming and
numbering rules live there. Row order defines the "Speaker N" numbering.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..logic.audio_source_config import (
    SourceRow,
    default_speaker_names,
    speaker_order,
    validate_rows,
)

logger = logging.getLogger(__name__)

COL_NUMBER, COL_FILE, COL_NAME, COL_DETECT, COL_OUTPUT, COL_GAIN = range(6)
_HEADERS = ["Speaker #", "File", "Speaker name", "Use for detection", "Include in output", "Gain (dB)"]
_PATH_ROLE = Qt.ItemDataRole.UserRole
_AUDIO_FILTER = "Audio Files (*.wav *.mp3 *.aac *.m4a *.flac);;All Files (*.*)"


class AudioSourcesDialog(QDialog):
    """Edit the list of audio sources; ``rows()`` returns the result."""

    def __init__(
        self,
        rows: Sequence[SourceRow] = (),
        parent=None,
        start_dir: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Audio Sources")
        self.setModal(True)
        self.setMinimumSize(760, 340)
        self._start_dir = start_dir or os.path.expanduser("~")
        self._busy = False
        self._init_ui()
        self._populate(list(rows))

    # ----- UI -----

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        hint = QLabel(
            "One audio file per speaker. Speaker numbers follow row order; "
            "cameras are mapped to these speakers afterwards."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 12px;")
        layout.addWidget(hint)

        self.table = QTableWidget(0, len(_HEADERS))
        self.table.setObjectName("tblAudioSources")
        self.table.setHorizontalHeaderLabels(_HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(COL_NUMBER, 80)
        self.table.setColumnWidth(COL_FILE, 220)
        self.table.setColumnWidth(COL_NAME, 140)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table, 1)

        row_buttons = QHBoxLayout()
        self.btn_add = QPushButton("Add files…")
        self.btn_add.setObjectName("btnAddSources")
        self.btn_add.clicked.connect(self._on_add)
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.setObjectName("btnRemoveSource")
        self.btn_remove.clicked.connect(self._on_remove)
        self.btn_up = QPushButton("Move up")
        self.btn_up.setObjectName("btnMoveUp")
        self.btn_up.clicked.connect(lambda: self._move(-1))
        self.btn_down = QPushButton("Move down")
        self.btn_down.setObjectName("btnMoveDown")
        self.btn_down.clicked.connect(lambda: self._move(1))
        for b in (self.btn_add, self.btn_remove, self.btn_up, self.btn_down):
            row_buttons.addWidget(b)
        row_buttons.addStretch(1)
        layout.addLayout(row_buttons)

        self.lbl_problems = QLabel("")
        self.lbl_problems.setObjectName("lblSourceProblems")
        self.lbl_problems.setWordWrap(True)
        self.lbl_problems.setStyleSheet("color: #c0392b; font-size: 12px;")
        layout.addWidget(self.lbl_problems)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.btn_ok = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    # ----- model <-> table -----

    def rows(self) -> list[SourceRow]:
        """Current table contents as SourceRow objects."""
        result: list[SourceRow] = []
        for r in range(self.table.rowCount()):
            file_item = self.table.item(r, COL_FILE)
            gain_text = self.table.item(r, COL_GAIN).text().strip().replace(",", ".")
            try:
                gain = float(gain_text)
            except ValueError:
                gain = float("nan")  # reported by validate_rows
            result.append(
                SourceRow(
                    path=file_item.data(_PATH_ROLE) or "",
                    speaker_name=self.table.item(r, COL_NAME).text(),
                    use_for_detection=self.table.item(r, COL_DETECT).checkState() == Qt.CheckState.Checked,
                    include_in_output=self.table.item(r, COL_OUTPUT).checkState() == Qt.CheckState.Checked,
                    gain_db=gain,
                )
            )
        return result

    def problems(self) -> list[str]:
        return validate_rows(self.rows(), check_files=True)

    def _populate(self, rows: list[SourceRow], select: Optional[int] = None) -> None:
        self._busy = True
        try:
            self.table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                self.table.setItem(r, COL_NUMBER, self._readonly_item(""))
                file_item = self._readonly_item(os.path.basename(row.path) or row.path)
                file_item.setData(_PATH_ROLE, row.path)
                file_item.setToolTip(row.path)
                self.table.setItem(r, COL_FILE, file_item)
                self.table.setItem(r, COL_NAME, QTableWidgetItem(row.speaker_name))
                self.table.setItem(r, COL_DETECT, self._check_item(row.use_for_detection))
                self.table.setItem(r, COL_OUTPUT, self._check_item(row.include_in_output))
                self.table.setItem(r, COL_GAIN, QTableWidgetItem(f"{row.gain_db:g}"))
        finally:
            self._busy = False
        if select is not None and 0 <= select < len(rows):
            self.table.selectRow(select)
        self._refresh()

    @staticmethod
    def _readonly_item(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        return item

    @staticmethod
    def _check_item(checked: bool) -> QTableWidgetItem:
        item = QTableWidgetItem()
        item.setFlags(
            Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable
        )
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        return item

    def _refresh(self) -> None:
        """Update the speaker-number column, problem text and OK state."""
        rows = self.rows()
        numbers = {name: n for n, name in enumerate(speaker_order(rows), start=1)}
        self._busy = True
        try:
            for r, row in enumerate(rows):
                n = numbers.get(row.speaker)
                self.table.item(r, COL_NUMBER).setText(f"Speaker {n}" if n else "—")
        finally:
            self._busy = False
        problems = validate_rows(rows, check_files=True)
        self.lbl_problems.setText("\n".join(problems))
        self.btn_ok.setEnabled(not problems)
        has_selection = self.table.currentRow() >= 0
        self.btn_remove.setEnabled(has_selection)
        self.btn_up.setEnabled(has_selection)
        self.btn_down.setEnabled(has_selection)

    def _on_item_changed(self, _item) -> None:
        if not self._busy:
            self._refresh()

    # ----- actions -----

    def _on_add(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Sources", self._start_dir, _AUDIO_FILTER
        )
        if paths:
            self.add_paths(list(paths))

    def add_paths(self, paths: Sequence[str]) -> None:
        """Append files as new rows, skipping paths already listed."""
        rows = self.rows()
        existing = {os.path.normcase(os.path.abspath(r.path)) for r in rows if r.path}
        new_paths: list[str] = []
        for p in paths:
            key = os.path.normcase(os.path.abspath(p))
            if key not in existing:
                existing.add(key)
                new_paths.append(p)
        if not new_paths:
            return
        if rows:
            names = [default_speaker_names([p])[0] for p in new_paths]
        else:
            names = default_speaker_names(new_paths)
        rows.extend(SourceRow(p, n) for p, n in zip(new_paths, names))
        self._start_dir = os.path.dirname(new_paths[-1]) or self._start_dir
        self._populate(rows, select=len(rows) - 1)

    def _on_remove(self) -> None:
        r = self.table.currentRow()
        if r < 0:
            return
        rows = self.rows()
        del rows[r]
        self._populate(rows, select=min(r, len(rows) - 1))

    def _move(self, delta: int) -> None:
        r = self.table.currentRow()
        target = r + delta
        rows = self.rows()
        if r < 0 or not 0 <= target < len(rows):
            return
        rows[r], rows[target] = rows[target], rows[r]
        self._populate(rows, select=target)


def show_audio_sources_dialog(
    rows: Sequence[SourceRow] = (),
    parent=None,
    start_dir: Optional[str] = None,
) -> Optional[list[SourceRow]]:
    """Show the dialog; return the edited rows on OK, or None if cancelled."""
    dialog = AudioSourcesDialog(rows, parent, start_dir)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.rows()
    logger.info("Audio sources dialog cancelled")
    return None
