from veldra.api.artifact import Artifact
from veldra.config.models import RunConfig


def test_artifact_save_load_roundtrip(tmp_path) -> None:
    config = RunConfig.model_validate(
        {
            "config_version": 1,
            "task": {"type": "regression"},
            "data": {"target": "y"},
            "export": {"artifact_dir": str(tmp_path)},
        }
    )
    artifact = Artifact.from_config(
        config,
        run_id="run-0001",
        feature_schema={"feature_a": "float"},
    )
    artifact_path = tmp_path / "artifact"
    artifact.save(artifact_path)

    loaded = Artifact.load(artifact_path)
    assert loaded.manifest.run_id == "run-0001"
    assert loaded.manifest.python_version
    assert loaded.manifest.dependencies
    assert loaded.feature_schema == {"feature_a": "float"}
    assert loaded.run_config.model_dump(mode="json") == config.model_dump(mode="json")
