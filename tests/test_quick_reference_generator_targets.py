from __future__ import annotations

import json
import types

import scripts.generate_quick_reference_notebook_specs as quickref_specs
import scripts.generate_quick_reference_notebooks as generator


def test_preview_alias_routes_to_canonical_and_warns(monkeypatch, capsys) -> None:
    quickref_calls: list[dict[str, object]] = []

    module = types.SimpleNamespace(
        generate_quick_reference_notebooks=lambda **kwargs: quickref_calls.append(kwargs)
    )
    monkeypatch.setattr(generator.importlib, "import_module", lambda _name: module)

    generator.main(["--target", "preview_alias", "--no-execute"])

    captured = capsys.readouterr()
    assert "deprecated" in captured.err.lower()
    assert quickref_calls == [{"execute": False, "notebook_subdir": "quick_reference"}]


def test_target_all_generates_canonical_only(monkeypatch) -> None:
    quickref_calls: list[dict[str, object]] = []

    module = types.SimpleNamespace(
        generate_quick_reference_notebooks=lambda **kwargs: quickref_calls.append(kwargs)
    )
    monkeypatch.setattr(generator.importlib, "import_module", lambda _name: module)

    generator.main(["--target", "all", "--no-execute"])

    assert quickref_calls == [{"execute": False, "notebook_subdir": "quick_reference"}]


def test_quick_reference_notebook_content_is_identical_across_output_dirs(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(quickref_specs, "NB_DIR", tmp_path / "notebooks")

    canonical_paths = quickref_specs.generate_quick_reference_notebooks(
        execute=False,
        notebook_subdir="quick_reference",
    )
    preview_paths = quickref_specs.generate_quick_reference_notebooks(
        execute=False,
        notebook_subdir="quick_reference_preview",
    )

    assert len(canonical_paths) == len(preview_paths) == 13
    for canonical_path, preview_path in zip(canonical_paths, preview_paths):
        assert canonical_path.name == preview_path.name
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
        preview = json.loads(preview_path.read_text(encoding="utf-8"))
        for notebook in (canonical, preview):
            for cell in notebook.get("cells", []):
                cell.pop("id", None)
        assert canonical == preview
