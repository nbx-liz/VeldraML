import logging

from veldra.api.logging import log_event


def test_structured_logging_contract(caplog) -> None:
    logger = logging.getLogger("veldra.test")

    with caplog.at_level(logging.INFO):
        payload = log_event(
            logger=logger,
            level=logging.INFO,
            message="contract check",
            run_id="run-1",
            artifact_path="artifacts/run-1",
            task_type="binary",
            stage="fit",
        )

    assert payload["run_id"] == "run-1"
    assert payload["artifact_path"] == "artifacts/run-1"
    assert payload["task_type"] == "binary"
    assert payload["stage"] == "fit"

    record = caplog.records[-1]
    assert record.run_id == "run-1"
    assert record.artifact_path == "artifacts/run-1"
    assert record.task_type == "binary"
