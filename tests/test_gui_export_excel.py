from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

from veldra.gui import services


class _DummySheet:
    def __init__(self) -> None:
        self.rows = []
        self.title = ""

    def append(self, row):
        self.rows.append(row)

    def cell(self, row: int, column: int, value: str):
        self.rows.append((row, column, value))


class _DummyWorkbook:
    def __init__(self) -> None:
        self.active = _DummySheet()
        self._sheets = [self.active]

    def create_sheet(self, name: str):
        sheet = _DummySheet()
        sheet.title = name
        self._sheets.append(sheet)
        return sheet

    def save(self, path):
        Path(path).write_bytes(b"xlsx")


def test_export_excel_report(monkeypatch, tmp_path) -> None:
    fake_art = SimpleNamespace(metrics={"auc": 0.8}, config={"task": {"type": "binary"}})
    monkeypatch.setattr(services, "Artifact", SimpleNamespace(load=lambda _p: fake_art))

    dummy_mod = types.SimpleNamespace(Workbook=_DummyWorkbook)
    monkeypatch.setitem(sys.modules, "openpyxl", dummy_mod)

    artifact_path = tmp_path / "art"
    artifact_path.mkdir(parents=True, exist_ok=True)

    out = services.export_excel_report(str(artifact_path))
    assert out.endswith(".xlsx")
    assert Path(out).exists()
