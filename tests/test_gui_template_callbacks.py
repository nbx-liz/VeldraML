"""GUI callback compatibility checks for template application."""

from __future__ import annotations

from types import SimpleNamespace

from veldra.gui import app as app_module


def test_library_actions_apply_template_for_both_routes(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "train-template-apply-btn.n_clicks"}]),
    )
    out_train = app_module._cb_phase30_library_actions(
        1,
        0,
        0,
        0,
        "regression_baseline",
        None,
        "cfg",
        "template",
        "",
        [],
        {},
    )
    assert "config_version: 1" in out_train[0]

    monkeypatch.setattr(
        app_module,
        "callback_context",
        SimpleNamespace(triggered=[{"prop_id": "config-template-apply-btn.n_clicks"}]),
    )
    out_config = app_module._cb_phase30_library_actions(
        1,
        0,
        0,
        0,
        "regression_baseline",
        None,
        "cfg",
        "template",
        "",
        [],
        {},
    )
    assert out_config[0] == out_train[0]


def test_run_validation_gate_blocks_invalid(monkeypatch) -> None:
    monkeypatch.setattr(
        app_module,
        "validate_config_with_guidance",
        lambda _yaml: {
            "ok": False,
            "errors": [{"path": "task", "message": "invalid", "suggestions": []}],
        },
    )
    monkeypatch.setattr(
        app_module,
        "submit_run_job",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("must not call")),
    )

    msg = app_module._cb_enqueue_run_job(
        1,
        "fit",
        "task:\n  type: regression\n",
        "configs/gui_run.yaml",
        "examples/data/causal_dr_tune_demo.csv",
        "",
        "",
        "python",
        "normal",
    )
    assert "Validation blocked run" in msg
