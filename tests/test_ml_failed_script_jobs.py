"""Integration tests asserting that failing shell scripts produce failed jobs with diagnostic artifacts."""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo, ModelMetadata, PeriodType
from chapkit.artifact import ArtifactHierarchy
from chapkit.ml import ShellModelRunner

HIERARCHY = ArtifactHierarchy(
    name="failing_shell_pipeline",
    level_labels={0: "ml_training_workspace", 1: "ml_prediction"},
)


class FailingConfig(BaseConfig):
    """Minimal config for the failing-script fixtures."""

    prediction_periods: int = 3


def build_client(train_command: str, predict_command: str, project_root: Path) -> TestClient:
    """Build a TestClient for an ML service rooted at an empty project directory."""
    original_cwd = Path.cwd()
    os.chdir(project_root)
    try:
        runner: ShellModelRunner[FailingConfig] = ShellModelRunner(
            train_command=train_command,
            predict_command=predict_command,
        )
    finally:
        os.chdir(original_cwd)
    app = MLServiceBuilder(
        info=MLServiceInfo(
            id="failing-shell-service",
            display_name="Failing Shell Service",
            model_metadata=ModelMetadata(
                author="Test",
                author_assessed_status=AssessedStatus.gray,
            ),
            period_type=PeriodType.monthly,
        ),
        config_schema=FailingConfig,
        hierarchy=HIERARCHY,
        runner=runner,
    ).build()
    return TestClient(app)


@pytest.fixture
def failing_train_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    """Client whose training script writes to stderr and exits non-zero."""
    with build_client(
        train_command="echo 'boom: training blew up' >&2; exit 3",
        predict_command="echo 'value' > {output_file}",
        project_root=tmp_path,
    ) as client:
        yield client


@pytest.fixture
def failing_predict_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    """Client whose prediction script writes to stderr and exits non-zero."""
    with build_client(
        train_command="echo trained",
        predict_command="echo 'boom: prediction blew up' >&2; exit 4",
        project_root=tmp_path,
    ) as client:
        yield client


def wait_for_job(client: TestClient, job_id: str, timeout: float = 20.0) -> dict[str, Any]:
    """Poll a job until it reaches a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        job = cast(dict[str, Any], response.json())
        if job["status"] in ("completed", "failed", "canceled"):
            return job
        time.sleep(0.1)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")


def create_config(client: TestClient) -> str:
    """Create a config and return its id."""
    response = client.post("/api/v1/configs", json={"name": "failing-config", "data": {}})
    assert response.status_code == 201
    return cast(str, response.json()["id"])


TRAIN_DATA = {"columns": ["feature", "target"], "data": [[1.0, 2.0], [2.0, 4.0]]}


def test_failing_train_script_fails_the_job(failing_train_client: TestClient) -> None:
    """A non-zero training exit code fails the job and keeps the diagnostic artifact."""
    config_id = create_config(failing_train_client)

    train_response = failing_train_client.post(
        "/api/v1/ml/$train",
        json={"config_id": config_id, "data": TRAIN_DATA},
    )
    assert train_response.status_code == 202
    submission = train_response.json()

    job = wait_for_job(failing_train_client, submission["job_id"])

    assert job["status"] == "failed"
    assert job["error"] is not None
    assert "exit code 3" in job["error"]
    assert submission["artifact_id"] in job["error"]
    assert "boom: training blew up" in job["error"]
    assert job["error_traceback"]

    artifact_response = failing_train_client.get(f"/api/v1/artifacts/{submission['artifact_id']}")
    assert artifact_response.status_code == 200
    metadata = artifact_response.json()["data"]["metadata"]
    assert metadata["status"] == "failed"
    assert metadata["exit_code"] == 3
    assert "boom: training blew up" in metadata["stderr"]


def test_failing_predict_script_fails_the_job(failing_predict_client: TestClient) -> None:
    """A non-zero prediction exit code fails the job and keeps the diagnostic workspace artifact."""
    config_id = create_config(failing_predict_client)

    train_response = failing_predict_client.post(
        "/api/v1/ml/$train",
        json={"config_id": config_id, "data": TRAIN_DATA},
    )
    train_submission = train_response.json()
    train_job = wait_for_job(failing_predict_client, train_submission["job_id"])
    assert train_job["status"] == "completed"

    predict_response = failing_predict_client.post(
        "/api/v1/ml/$predict",
        json={
            "artifact_id": train_submission["artifact_id"],
            "historic": {"columns": ["feature", "target"], "data": [[1.0, 2.0]]},
            "future": {"columns": ["feature"], "data": [[3.0]]},
        },
    )
    assert predict_response.status_code == 202
    predict_submission = predict_response.json()

    predict_job = wait_for_job(failing_predict_client, predict_submission["job_id"])

    assert predict_job["status"] == "failed"
    assert "exit code 4" in predict_job["error"]
    assert predict_submission["artifact_id"] in predict_job["error"]
    assert "boom: prediction blew up" in predict_job["error"]

    artifact_response = failing_predict_client.get(f"/api/v1/artifacts/{predict_submission['artifact_id']}")
    assert artifact_response.status_code == 200
    artifact_data = artifact_response.json()["data"]
    assert artifact_data["type"] == "ml_prediction_workspace"
    assert artifact_data["metadata"]["status"] == "failed"
    assert artifact_data["metadata"]["exit_code"] == 4


def test_format_stderr_tail_keeps_last_lines() -> None:
    """The stderr tail keeps only the last non-empty lines, joined on one line."""
    from chapkit.ml.runner import format_stderr_tail

    assert format_stderr_tail("a\n\nb\nc\n", tail_lines=2) == "b | c"
    assert format_stderr_tail("   \n\n") == "<no stderr output>"


def test_model_run_failed_error_message_carries_context() -> None:
    """The failure message names the phase, exit code, artifact id and stderr tail."""
    from chapkit.ml import ModelRunFailedError

    error = ModelRunFailedError("train", 7, "01ARZ3NDEKTSV4RRFFQ69G5FAV", "line one\nline two")

    message = str(error)
    assert "train script failed with exit code 7" in message
    assert "01ARZ3NDEKTSV4RRFFQ69G5FAV" in message
    assert "line one | line two" in message
    assert error.phase == "train"
    assert error.exit_code == 7


def test_runner_result_helpers_treat_non_dict_results_as_success() -> None:
    """Runner results that are not workspace dicts report exit code 0 and no stderr."""
    from chapkit.ml.manager import _exit_code_of, _stderr_of

    assert _exit_code_of("not-a-dict") == 0
    assert _stderr_of("not-a-dict") == ""
    assert _exit_code_of({"exit_code": 5}) == 5
    assert _stderr_of({"stderr": "boom"}) == "boom"
