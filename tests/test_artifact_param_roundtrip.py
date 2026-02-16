from __future__ import annotations

from pathlib import Path

from veldra.api import Artifact, evaluate, fit


def test_binary_artifact_roundtrip_preserves_phase258_train_params(tmp_path, binary_frame) -> None:
    frame = binary_frame(rows=90, seed=64, coef1=1.2, coef2=-0.8, noise=0.35)
    data_path = tmp_path / "binary_train.csv"
    frame.to_csv(data_path, index=False)

    run = fit(
        {
            "config_version": 1,
            "task": {"type": "binary"},
            "data": {"path": str(data_path), "target": "target"},
            "split": {"type": "stratified", "n_splits": 3, "seed": 21},
            "train": {
                "auto_num_leaves": True,
                "num_leaves_ratio": 0.5,
                "lgb_params": {"max_depth": 5},
                "feature_weights": {"x1": 2.0},
                "top_k": 10,
            },
            "postprocess": {"calibration": "platt"},
            "export": {"artifact_dir": str(tmp_path / "artifacts")},
        }
    )

    artifact_path = Path(run.artifact_path)
    config_text = (artifact_path / "run_config.yaml").read_text(encoding="utf-8")
    assert "auto_num_leaves: true" in config_text
    assert "num_leaves_ratio: 0.5" in config_text
    assert "feature_weights:" in config_text
    assert "top_k: 10" in config_text

    loaded = Artifact.load(run.artifact_path)
    assert loaded.run_config.train.auto_num_leaves is True
    assert loaded.run_config.train.num_leaves_ratio == 0.5
    assert loaded.run_config.train.feature_weights == {"x1": 2.0}
    assert loaded.run_config.train.top_k == 10

    pred = loaded.predict(frame[["x1", "x2"]])
    assert list(pred.columns) == ["p_cal", "p_raw", "label_pred"]

    eval_result = evaluate(loaded, frame)
    assert "precision_at_10" in eval_result.metrics
