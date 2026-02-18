from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from veldra.api.exceptions import VeldraValidationError
from veldra.gui import services


def test_export_pdf_report_missing_dependency(monkeypatch, tmp_path) -> None:
    html_path = tmp_path / "report.html"
    html_path.write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(services, "export_html_report", lambda _p: str(html_path))

    sys.modules.pop("weasyprint", None)
    with pytest.raises(VeldraValidationError):
        services.export_pdf_report(str(tmp_path / "artifact"))


def test_export_pdf_report_success(monkeypatch, tmp_path) -> None:
    html_path = tmp_path / "report.html"
    html_path.write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(services, "export_html_report", lambda _p: str(html_path))

    class _HTML:
        def __init__(self, filename: str) -> None:
            self.filename = filename

        def write_pdf(self, out: str) -> None:
            Path(out).write_text("pdf", encoding="utf-8")

    monkeypatch.setitem(sys.modules, "weasyprint", SimpleNamespace(HTML=_HTML))
    out = services.export_pdf_report(str(tmp_path / "artifact"))
    assert out.endswith(".pdf")
    assert Path(out).exists()
