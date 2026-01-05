"""CLI scaffolding integration tests.

These tests verify that scaffolded projects work end-to-end:
- `chapkit init` creates valid projects
- Projects can be patched to use local chapkit
- Train/predict workflows complete successfully
- Artifact hierarchy is correct

Run with: pytest -m slow tests/test_cli_scaffolding.py -v
"""

from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.slow


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port: int = s.getsockname()[1]
        return port


@pytest.fixture
def chapkit_root() -> Path:
    """Get the root directory of the chapkit project."""
    return Path(__file__).parent.parent


@pytest.fixture
def scaffold_project(tmp_path: Path, chapkit_root: Path) -> Callable[[str, str], Path]:
    """Scaffold a project and patch to use local chapkit."""

    def _scaffold(name: str, template: str = "ml") -> Path:
        # Run chapkit init using uv run from the chapkit project root
        result = subprocess.run(
            [
                "uv",
                "run",
                "chapkit",
                "init",
                name,
                "--template",
                template,
                "--path",
                str(tmp_path),
            ],
            cwd=chapkit_root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"chapkit init failed: {result.stderr}"

        project_dir = tmp_path / name

        # Patch pyproject.toml to use local chapkit
        pyproject = project_dir / "pyproject.toml"
        content = pyproject.read_text()

        # Replace the chapkit dependency with a path-based dependency
        local_path = chapkit_root.absolute()
        content = re.sub(
            r'"chapkit>=[\d.]+"',
            f'"chapkit @ file://{local_path}"',
            content,
        )
        pyproject.write_text(content)

        # Install dependencies
        result = subprocess.run(
            ["uv", "sync"],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"uv sync failed: {result.stderr}"

        return project_dir

    return _scaffold


@contextmanager
def run_service(project_dir: Path, port: int) -> Generator[str, None, None]:
    """Start FastAPI service and yield base URL."""
    env = os.environ.copy()
    env["PORT"] = str(port)

    process = subprocess.Popen(
        ["uv", "run", "fastapi", "run", "main.py", "--port", str(port)],
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    base_url = f"http://127.0.0.1:{port}"

    # Wait for service to start
    start_time = time.time()
    timeout = 30
    while time.time() - start_time < timeout:
        try:
            with httpx.Client(timeout=2) as client:
                resp = client.get(f"{base_url}/health")
                if resp.status_code == 200:
                    break
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        time.sleep(0.5)
    else:
        # Cleanup and fail
        process.terminate()
        process.wait(timeout=5)
        stdout, stderr = process.communicate()
        raise TimeoutError(
            f"Service did not start within {timeout}s. stdout: {stdout.decode()}, stderr: {stderr.decode()}"
        )

    try:
        yield base_url
    finally:
        # Graceful shutdown
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)


def wait_for_job_completion(client: httpx.Client, base_url: str, job_id: str, timeout: float = 60.0) -> dict[str, Any]:
    """Poll job status until completion or timeout."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        resp = client.get(f"{base_url}/api/v1/jobs/{job_id}")
        assert resp.status_code == 200, f"Failed to get job: {resp.text}"
        job: dict[str, Any] = resp.json()

        if job["status"] in ("completed", "failed", "canceled"):
            return job

        time.sleep(0.5)

    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


@pytest.mark.slow
def test_scaffold_functional_project_structure(
    scaffold_project: Callable[[str, str], Path],
) -> None:
    """Test that scaffolded functional ML project has correct structure."""
    project_dir = scaffold_project("test-functional", "ml")

    # Verify project structure
    assert (project_dir / "main.py").exists()
    assert (project_dir / "pyproject.toml").exists()
    assert (project_dir / "Dockerfile").exists()
    assert (project_dir / "README.md").exists()
    assert (project_dir / "compose.yml").exists()
    assert (project_dir / "postman_collection.json").exists()
    assert (project_dir / ".gitignore").exists()

    # Verify main.py contains functional runner
    main_content = (project_dir / "main.py").read_text()
    assert "FunctionalModelRunner" in main_content
    assert "on_train" in main_content
    assert "on_predict" in main_content


@pytest.mark.slow
def test_scaffold_shell_project_structure(
    scaffold_project: Callable[[str, str], Path],
) -> None:
    """Test that scaffolded shell ML project has correct structure."""
    project_dir = scaffold_project("test-shell", "ml-shell")

    # Verify project structure
    assert (project_dir / "main.py").exists()
    assert (project_dir / "pyproject.toml").exists()
    assert (project_dir / "scripts").is_dir()
    assert (project_dir / "scripts" / "train_model.py").exists()
    assert (project_dir / "scripts" / "predict_model.py").exists()

    # Verify main.py contains shell runner
    main_content = (project_dir / "main.py").read_text()
    assert "ShellModelRunner" in main_content
    assert "train_command" in main_content
    assert "predict_command" in main_content


@pytest.mark.slow
def test_scaffold_with_monitoring_structure(tmp_path: Path, chapkit_root: Path) -> None:
    """Test scaffolding with monitoring stack generates all files."""
    result = subprocess.run(
        [
            "uv",
            "run",
            "chapkit",
            "init",
            "test-monitor",
            "--with-monitoring",
            "--path",
            str(tmp_path),
        ],
        cwd=chapkit_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"chapkit init failed: {result.stderr}"

    project_dir = tmp_path / "test-monitor"

    # Verify monitoring files exist
    assert (project_dir / "compose.yml").exists()
    assert (project_dir / "monitoring").is_dir()
    assert (project_dir / "monitoring" / "prometheus").is_dir()
    assert (project_dir / "monitoring" / "prometheus" / "prometheus.yml").exists()
    assert (project_dir / "monitoring" / "grafana").is_dir()
    assert (project_dir / "monitoring" / "grafana" / "provisioning").is_dir()
    assert (project_dir / "monitoring" / "grafana" / "provisioning" / "datasources").is_dir()
    assert (project_dir / "monitoring" / "grafana" / "provisioning" / "dashboards").is_dir()

    # Verify compose.yml has monitoring services
    compose_content = (project_dir / "compose.yml").read_text()
    assert "prometheus" in compose_content
    assert "grafana" in compose_content


@pytest.mark.slow
def test_scaffold_functional_train_predict(
    scaffold_project: Callable[[str, str], Path],
) -> None:
    """Test scaffolded functional ML project with train and predict workflow."""
    project_dir = scaffold_project("test-ml-workflow", "ml")
    port = find_free_port()

    with run_service(project_dir, port) as base_url:
        with httpx.Client(timeout=30) as client:
            # 1. Create config
            config_resp = client.post(
                f"{base_url}/api/v1/configs",
                json={"name": "test-config", "data": {}},
            )
            assert config_resp.status_code == 201, f"Config creation failed: {config_resp.text}"
            config_id = config_resp.json()["id"]

            # 2. Train
            train_resp = client.post(
                f"{base_url}/api/v1/ml/$train",
                json={
                    "config_id": config_id,
                    "data": {
                        "columns": ["x", "y"],
                        "data": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                    },
                },
            )
            assert train_resp.status_code == 202, f"Train submission failed: {train_resp.text}"
            train_data = train_resp.json()
            artifact_id = train_data["artifact_id"]
            job_id = train_data["job_id"]

            # 3. Wait for training
            job = wait_for_job_completion(client, base_url, job_id)
            assert job["status"] == "completed", f"Training failed: {job.get('error')}"

            # 4. Verify training artifact
            artifact_resp = client.get(f"{base_url}/api/v1/artifacts/{artifact_id}")
            assert artifact_resp.status_code == 200
            artifact = artifact_resp.json()

            assert artifact["level"] == 0
            assert artifact["data"]["type"] == "ml_training_workspace"
            assert artifact["data"]["content_type"] == "application/zip"
            assert artifact["data"]["metadata"]["status"] == "success"

            # 5. Predict
            predict_resp = client.post(
                f"{base_url}/api/v1/ml/$predict",
                json={
                    "artifact_id": artifact_id,
                    "historic": {"columns": ["x"], "data": []},
                    "future": {"columns": ["x"], "data": [[7.0], [8.0]]},
                },
            )
            assert predict_resp.status_code == 202, f"Predict submission failed: {predict_resp.text}"
            pred_data = predict_resp.json()
            pred_artifact_id = pred_data["artifact_id"]
            pred_job_id = pred_data["job_id"]

            # 6. Wait for prediction
            pred_job = wait_for_job_completion(client, base_url, pred_job_id)
            assert pred_job["status"] == "completed", f"Prediction failed: {pred_job.get('error')}"

            # 7. Verify prediction artifact linkage
            pred_artifact_resp = client.get(f"{base_url}/api/v1/artifacts/{pred_artifact_id}")
            assert pred_artifact_resp.status_code == 200
            pred_artifact = pred_artifact_resp.json()

            assert pred_artifact["level"] == 1
            assert pred_artifact["parent_id"] == artifact_id
            assert pred_artifact["data"]["type"] == "ml_prediction"
            assert pred_artifact["data"]["metadata"]["status"] == "success"

            # 8. Verify prediction DataFrame content
            predictions = pred_artifact["data"]["content"]
            assert "columns" in predictions
            assert "data" in predictions
            assert len(predictions["data"]) > 0, "Predictions should have data rows"
            assert "sample_0" in predictions["columns"], "Predictions should have sample_0 column"

            # 9. Verify level 2 prediction workspace artifact exists
            tree_resp = client.get(f"{base_url}/api/v1/artifacts/{pred_artifact_id}/$tree")
            assert tree_resp.status_code == 200
            tree = tree_resp.json()

            # Tree root is the prediction artifact
            assert tree["id"] == pred_artifact_id
            assert tree["level"] == 1

            # Should have children (the prediction workspace)
            assert "children" in tree
            assert len(tree["children"]) >= 1

            # Verify workspace artifact structure
            workspace_artifact = tree["children"][0]
            assert workspace_artifact["level"] == 2
            assert workspace_artifact["parent_id"] == pred_artifact_id
            assert workspace_artifact["data"]["type"] == "ml_prediction_workspace"
            assert workspace_artifact["data"]["content_type"] == "application/zip"


@pytest.mark.slow
def test_scaffold_shell_train_predict(
    scaffold_project: Callable[[str, str], Path],
) -> None:
    """Test scaffolded shell ML project with train and predict workflow."""
    project_dir = scaffold_project("test-shell-workflow", "ml-shell")
    port = find_free_port()

    with run_service(project_dir, port) as base_url:
        with httpx.Client(timeout=60) as client:
            # 1. Create config
            config_resp = client.post(
                f"{base_url}/api/v1/configs",
                json={"name": "shell-test-config", "data": {}},
            )
            assert config_resp.status_code == 201, f"Config creation failed: {config_resp.text}"
            config_id = config_resp.json()["id"]

            # 2. Train
            train_resp = client.post(
                f"{base_url}/api/v1/ml/$train",
                json={
                    "config_id": config_id,
                    "data": {
                        "columns": ["x", "y", "target"],
                        "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                    },
                },
            )
            assert train_resp.status_code == 202, f"Train submission failed: {train_resp.text}"
            train_data = train_resp.json()
            artifact_id = train_data["artifact_id"]
            job_id = train_data["job_id"]

            # 3. Wait for training (shell scripts may take longer)
            job = wait_for_job_completion(client, base_url, job_id, timeout=120)
            assert job["status"] == "completed", f"Training failed: {job.get('error')}"

            # 4. Verify training artifact
            artifact_resp = client.get(f"{base_url}/api/v1/artifacts/{artifact_id}")
            assert artifact_resp.status_code == 200
            artifact = artifact_resp.json()

            assert artifact["level"] == 0
            assert artifact["data"]["type"] == "ml_training_workspace"
            assert artifact["data"]["content_type"] == "application/zip"

            # 5. Predict
            predict_resp = client.post(
                f"{base_url}/api/v1/ml/$predict",
                json={
                    "artifact_id": artifact_id,
                    "historic": {"columns": ["x", "y"], "data": []},
                    "future": {"columns": ["x", "y"], "data": [[10.0, 11.0], [12.0, 13.0]]},
                },
            )
            assert predict_resp.status_code == 202, f"Predict submission failed: {predict_resp.text}"
            pred_data = predict_resp.json()
            pred_artifact_id = pred_data["artifact_id"]
            pred_job_id = pred_data["job_id"]

            # 6. Wait for prediction
            pred_job = wait_for_job_completion(client, base_url, pred_job_id, timeout=120)
            assert pred_job["status"] == "completed", f"Prediction failed: {pred_job.get('error')}"

            # 7. Verify prediction artifact linkage
            pred_artifact_resp = client.get(f"{base_url}/api/v1/artifacts/{pred_artifact_id}")
            assert pred_artifact_resp.status_code == 200
            pred_artifact = pred_artifact_resp.json()

            assert pred_artifact["level"] == 1
            assert pred_artifact["parent_id"] == artifact_id
            assert pred_artifact["data"]["type"] == "ml_prediction"

            # 8. Verify prediction DataFrame content
            predictions = pred_artifact["data"]["content"]
            assert "columns" in predictions
            assert "data" in predictions
            assert len(predictions["data"]) > 0, "Predictions should have data rows"
            assert "sample_0" in predictions["columns"], "Predictions should have sample_0 column"

            # 9. Verify level 2 prediction workspace artifact exists
            tree_resp = client.get(f"{base_url}/api/v1/artifacts/{pred_artifact_id}/$tree")
            assert tree_resp.status_code == 200
            tree = tree_resp.json()

            # Tree root is the prediction artifact
            assert tree["id"] == pred_artifact_id
            assert tree["level"] == 1

            # Should have children (the prediction workspace)
            assert "children" in tree
            assert len(tree["children"]) >= 1

            # Verify workspace artifact structure
            workspace_artifact = tree["children"][0]
            assert workspace_artifact["level"] == 2
            assert workspace_artifact["parent_id"] == pred_artifact_id
            assert workspace_artifact["data"]["type"] == "ml_prediction_workspace"
            assert workspace_artifact["data"]["content_type"] == "application/zip"


@pytest.mark.slow
def test_scaffold_multiple_predictions_from_same_model(
    scaffold_project: Callable[[str, str], Path],
) -> None:
    """Test making multiple predictions from the same trained model."""
    project_dir = scaffold_project("test-multi-predict", "ml")
    port = find_free_port()

    with run_service(project_dir, port) as base_url:
        with httpx.Client(timeout=30) as client:
            # Create config and train model
            config_resp = client.post(
                f"{base_url}/api/v1/configs",
                json={"name": "multi-predict-config", "data": {}},
            )
            config_id = config_resp.json()["id"]

            train_resp = client.post(
                f"{base_url}/api/v1/ml/$train",
                json={
                    "config_id": config_id,
                    "data": {
                        "columns": ["x", "y"],
                        "data": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                    },
                },
            )
            artifact_id = train_resp.json()["artifact_id"]
            job = wait_for_job_completion(client, base_url, train_resp.json()["job_id"])
            assert job["status"] == "completed"

            # Make multiple predictions
            prediction_artifact_ids = []
            for i in range(3):
                predict_resp = client.post(
                    f"{base_url}/api/v1/ml/$predict",
                    json={
                        "artifact_id": artifact_id,
                        "historic": {"columns": ["x"], "data": []},
                        "future": {"columns": ["x"], "data": [[float(10 + i)]]},
                    },
                )
                assert predict_resp.status_code == 202

                pred_job = wait_for_job_completion(client, base_url, predict_resp.json()["job_id"])
                assert pred_job["status"] == "completed"
                prediction_artifact_ids.append(predict_resp.json()["artifact_id"])

            # Verify all prediction artifacts exist and link to the same model
            assert len(set(prediction_artifact_ids)) == 3  # All unique

            for pred_id in prediction_artifact_ids:
                artifact_resp = client.get(f"{base_url}/api/v1/artifacts/{pred_id}")
                artifact = artifact_resp.json()
                assert artifact["parent_id"] == artifact_id
                assert artifact["level"] == 1


@pytest.mark.slow
def test_scaffold_config_artifact_linkage(
    scaffold_project: Callable[[str, str], Path],
) -> None:
    """Test that config is linked to artifacts via artifact operations."""
    project_dir = scaffold_project("test-config-link", "ml")
    port = find_free_port()

    with run_service(project_dir, port) as base_url:
        with httpx.Client(timeout=30) as client:
            # Create config
            config_resp = client.post(
                f"{base_url}/api/v1/configs",
                json={"name": "link-test-config", "data": {}},
            )
            config_id = config_resp.json()["id"]

            # Train model
            train_resp = client.post(
                f"{base_url}/api/v1/ml/$train",
                json={
                    "config_id": config_id,
                    "data": {
                        "columns": ["x", "y"],
                        "data": [[1.0, 2.0]],
                    },
                },
            )
            artifact_id = train_resp.json()["artifact_id"]
            job = wait_for_job_completion(client, base_url, train_resp.json()["job_id"])
            assert job["status"] == "completed"

            # Get config's artifacts
            artifacts_resp = client.get(f"{base_url}/api/v1/configs/{config_id}/$artifacts")
            assert artifacts_resp.status_code == 200
            artifacts = artifacts_resp.json()

            # Should have at least one artifact linked
            assert len(artifacts) >= 1
            artifact_ids = [a["id"] for a in artifacts]
            assert artifact_id in artifact_ids
