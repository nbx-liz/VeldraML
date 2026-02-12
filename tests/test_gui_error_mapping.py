from __future__ import annotations

from veldra.api.exceptions import (
    VeldraArtifactError,
    VeldraNotImplementedError,
    VeldraValidationError,
)
from veldra.gui.services import normalize_gui_error


def test_normalize_gui_error_for_known_error_types() -> None:
    assert normalize_gui_error(VeldraValidationError("bad input")).startswith("Validation error:")
    assert normalize_gui_error(VeldraArtifactError("bad artifact")).startswith("Artifact error:")
    assert normalize_gui_error(VeldraNotImplementedError("pending")).startswith("Not implemented:")


def test_normalize_gui_error_for_generic_exception() -> None:
    assert normalize_gui_error(RuntimeError("oops")) == "RuntimeError: oops"
