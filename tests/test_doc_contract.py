from __future__ import annotations

import inspect
from importlib import import_module

import veldra.api as api

PUBLIC_API_FUNCTIONS = [
    "fit",
    "tune",
    "estimate_dr",
    "evaluate",
    "predict",
    "simulate",
    "export",
]

ALGORITHM_FUNCTIONS = [
    ("veldra.modeling.regression", "train_regression_with_cv"),
    ("veldra.modeling.binary", "train_binary_with_cv"),
    ("veldra.modeling.multiclass", "train_multiclass_with_cv"),
    ("veldra.modeling.frontier", "train_frontier_with_cv"),
    ("veldra.modeling.tuning", "run_tuning"),
    ("veldra.causal.dr", "run_dr_estimation"),
    ("veldra.causal.dr_did", "run_dr_did_estimation"),
    ("veldra.simulate.engine", "normalize_scenarios"),
    ("veldra.simulate.engine", "apply_scenario"),
    ("veldra.simulate.engine", "build_simulation_frame"),
]


def _doc(obj: object) -> str:
    return inspect.getdoc(obj) or ""


def test_public_api_exports_have_docstrings() -> None:
    missing: list[str] = []
    for name in api.__all__:
        obj = getattr(api, name)
        if not _doc(obj).strip():
            missing.append(name)
    assert not missing, f"Missing docstrings for public API exports: {missing}"


def test_public_api_functions_have_numpy_sections() -> None:
    missing_sections: dict[str, list[str]] = {}
    for name in PUBLIC_API_FUNCTIONS:
        doc = _doc(getattr(api, name))
        missing: list[str] = []
        for section in ("Parameters", "Returns", "Raises"):
            if section not in doc:
                missing.append(section)
        if missing:
            missing_sections[name] = missing
    assert not missing_sections, f"Missing API doc sections: {missing_sections}"


def test_major_algorithm_functions_have_notes() -> None:
    missing: list[str] = []
    for module_name, func_name in ALGORITHM_FUNCTIONS:
        module = import_module(module_name)
        func = getattr(module, func_name)
        doc = _doc(func)
        if "Notes" not in doc:
            missing.append(f"{module_name}.{func_name}")
    assert not missing, f"Missing Notes section for algorithm functions: {missing}"
