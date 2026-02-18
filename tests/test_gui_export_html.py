from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from veldra.gui import services


def test_export_html_report(monkeypatch, tmp_path) -> None:
    fake_art = SimpleNamespace(
        metrics={"mean": {"auc": 0.8}},
        config={"task": {"type": "binary"}},
        task_type="binary",
        run_id="run-1",
    )
    monkeypatch.setattr(services, "Artifact", SimpleNamespace(load=lambda _p: fake_art))

    artifact_path = tmp_path / "art"
    artifact_path.mkdir(parents=True, exist_ok=True)

    out = services.export_html_report(str(artifact_path))
    assert out.endswith(".html")
    text = Path(out).read_text(encoding="utf-8")
    assert "Veldra Report" in text
    assert "auc" in text
