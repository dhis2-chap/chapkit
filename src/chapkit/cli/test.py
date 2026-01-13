"""End-to-end test command for chapkit ML services."""

import random
import time
from typing import Annotated, Any

import httpx
import typer
from ulid import ULID


class TestRunner:
    """Orchestrates end-to-end ML service testing."""

    def __init__(self, base_url: str, timeout: float = 60.0, verbose: bool = False) -> None:
        """Initialize TestRunner with service URL and options."""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verbose = verbose
        self.client = httpx.Client(timeout=30.0)

        # Track created resources for optional cleanup
        self.created_config_ids: list[str] = []
        self.created_artifact_ids: list[str] = []

        # Service info (populated by fetch_service_info)
        self.required_covariates: list[str] = []
        self.requires_geo: bool = False
        self.allow_free_additional_continuous_covariates: bool = False

    def check_service_health(self) -> tuple[bool, str]:
        """Verify service is running and healthy."""
        try:
            response = self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    return True, "Service is healthy"
            return False, f"Health check returned: {response.text}"
        except httpx.ConnectError:
            return False, f"Cannot connect to {self.base_url}"
        except httpx.TimeoutException:
            return False, f"Connection to {self.base_url} timed out"

    def fetch_service_info(self) -> tuple[bool, str]:
        """Fetch service info to discover required covariates and geo requirements."""
        try:
            response = self.client.get(f"{self.base_url}/api/v1/info")
            if response.status_code == 200:
                data = response.json()
                self.required_covariates = data.get("required_covariates", [])
                self.requires_geo = data.get("requires_geo", False)
                self.allow_free_additional_continuous_covariates = data.get(
                    "allow_free_additional_continuous_covariates", False
                )
                return True, "Service info fetched"
            return False, f"Failed to fetch service info: {response.text}"
        except Exception as e:
            return False, f"Error fetching service info: {e}"

    def create_config(self, name: str, data: dict[str, Any]) -> tuple[bool, str, str | None]:
        """Create a config and return (success, message, config_id)."""
        try:
            response = self.client.post(f"{self.base_url}/api/v1/configs", json={"name": name, "data": data})
            if response.status_code in (200, 201):
                config = response.json()
                config_id = config["id"]
                self.created_config_ids.append(config_id)
                return True, f"Created config: {config_id}", config_id
            return False, f"Failed to create config: {response.text}", None
        except Exception as e:
            return False, f"Error creating config: {e}", None

    def submit_training(
        self, config_id: str, data: dict[str, Any], geo: dict[str, Any] | None = None
    ) -> tuple[bool, str, str | None, str | None]:
        """Submit training job and return (success, message, job_id, artifact_id)."""
        try:
            request_body: dict[str, Any] = {"config_id": config_id, "data": data}
            if geo:
                request_body["geo"] = geo

            response = self.client.post(f"{self.base_url}/api/v1/ml/$train", json=request_body)
            if response.status_code == 202:
                result = response.json()
                return (True, result["message"], result["job_id"], result["artifact_id"])
            return False, f"Failed to submit training: {response.text}", None, None
        except Exception as e:
            return False, f"Error submitting training: {e}", None, None

    def submit_prediction(
        self,
        artifact_id: str,
        historic: dict[str, Any],
        future: dict[str, Any],
        geo: dict[str, Any] | None = None,
    ) -> tuple[bool, str, str | None, str | None]:
        """Submit prediction job and return (success, message, job_id, artifact_id)."""
        try:
            request_body: dict[str, Any] = {"artifact_id": artifact_id, "historic": historic, "future": future}
            if geo:
                request_body["geo"] = geo

            response = self.client.post(f"{self.base_url}/api/v1/ml/$predict", json=request_body)
            if response.status_code == 202:
                result = response.json()
                return (True, result["message"], result["job_id"], result["artifact_id"])
            return False, f"Failed to submit prediction: {response.text}", None, None
        except Exception as e:
            return False, f"Error submitting prediction: {e}", None, None

    def wait_for_job(self, job_id: str) -> tuple[bool, str, dict[str, Any] | None]:
        """Poll job until completion, return (success, message, job_record)."""
        start_time = time.time()
        poll_interval = 0.2  # Start with 200ms
        max_poll_interval = 2.0  # Max 2 seconds between polls

        while time.time() - start_time < self.timeout:
            try:
                response = self.client.get(f"{self.base_url}/api/v1/jobs/{job_id}")
                if response.status_code != 200:
                    return False, f"Failed to get job status: {response.text}", None

                job = response.json()
                status = job.get("status")

                if status == "completed":
                    return True, "Job completed successfully", job
                elif status == "failed":
                    error = job.get("error", "Unknown error")
                    return False, f"Job failed: {error}", job
                elif status == "canceled":
                    return False, "Job was canceled", job

                # Exponential backoff
                time.sleep(poll_interval)
                poll_interval = min(poll_interval * 1.5, max_poll_interval)

            except Exception as e:
                return False, f"Error polling job: {e}", None

        return False, f"Job did not complete within {self.timeout}s", None

    def verify_artifact(self, artifact_id: str) -> tuple[bool, str, dict[str, Any] | None]:
        """Verify artifact exists and has valid structure."""
        try:
            response = self.client.get(f"{self.base_url}/api/v1/artifacts/{artifact_id}")
            if response.status_code != 200:
                return False, f"Artifact not found: {response.text}", None

            artifact = response.json()

            # Basic validation
            if "id" not in artifact or "data" not in artifact:
                return False, "Artifact missing required fields", artifact

            self.created_artifact_ids.append(artifact_id)
            return True, "Artifact verified", artifact
        except Exception as e:
            return False, f"Error verifying artifact: {e}", None

    def cleanup_resources(self) -> list[str]:
        """Delete created configs and artifacts. Returns list of errors."""
        errors: list[str] = []

        # Delete artifacts first (due to hierarchy)
        for artifact_id in reversed(self.created_artifact_ids):
            try:
                response = self.client.delete(f"{self.base_url}/api/v1/artifacts/{artifact_id}")
                if response.status_code not in (200, 204, 404):
                    errors.append(f"Failed to delete artifact {artifact_id}")
            except Exception as e:
                errors.append(f"Error deleting artifact {artifact_id}: {e}")

        # Then delete configs
        for config_id in self.created_config_ids:
            try:
                response = self.client.delete(f"{self.base_url}/api/v1/configs/{config_id}")
                if response.status_code not in (200, 204, 404):
                    errors.append(f"Failed to delete config {config_id}")
            except Exception as e:
                errors.append(f"Error deleting config {config_id}: {e}")

        return errors

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()


class TestDataGenerator:
    """Generate synthetic test data for ML workflows."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize generator with optional random seed."""
        if seed is not None:
            random.seed(seed)

    def generate_training_data(
        self,
        num_rows: int = 100,
        num_dimensions: int = 2,
        num_features: int = 3,
        required_covariates: list[str] | None = None,
        extra_covariates: int = 0,
    ) -> dict[str, Any]:
        """Generate training DataFrame with dimension and numeric columns."""
        required_covariates = required_covariates or []

        # Build columns list
        columns: list[str] = []

        # Dimension columns
        for i in range(num_dimensions):
            columns.append(f"dim_{i}")

        # Feature columns
        for i in range(num_features):
            columns.append(f"feature_{i}")

        # Required covariates from service info
        for cov in required_covariates:
            if cov not in columns:
                columns.append(cov)

        # Extra continuous covariates if allowed
        for i in range(extra_covariates):
            columns.append(f"extra_covariate_{i}")

        # Generate data
        data: list[list[Any]] = []
        for _ in range(num_rows):
            row: list[Any] = []

            # Dimension values (categorical)
            for i in range(num_dimensions):
                row.append(f"cat_{i}_{random.randint(0, 4)}")

            # Feature values (floats)
            for _ in range(num_features):
                row.append(random.uniform(0, 100))

            # Required covariate values (floats)
            for _ in required_covariates:
                row.append(random.uniform(0, 100))

            # Extra covariate values (floats)
            for _ in range(extra_covariates):
                row.append(random.uniform(0, 100))

            data.append(row)

        return {"columns": columns, "data": data}

    def generate_prediction_data(
        self,
        num_rows: int = 10,
        num_dimensions: int = 2,
        num_features: int = 3,
        required_covariates: list[str] | None = None,
        extra_covariates: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate historic and future DataFrames for prediction."""
        required_covariates = required_covariates or []

        # Build columns list (same as training but without target)
        columns: list[str] = []

        # Dimension columns
        for i in range(num_dimensions):
            columns.append(f"dim_{i}")

        # Feature columns
        for i in range(num_features):
            columns.append(f"feature_{i}")

        # Required covariates
        for cov in required_covariates:
            if cov not in columns:
                columns.append(cov)

        # Extra covariates
        for i in range(extra_covariates):
            columns.append(f"extra_covariate_{i}")

        # Empty historic (matching scaffolded template pattern)
        historic: dict[str, Any] = {"columns": columns, "data": []}

        # Future data to predict
        future = self.generate_training_data(
            num_rows=num_rows,
            num_dimensions=num_dimensions,
            num_features=num_features,
            required_covariates=required_covariates,
            extra_covariates=extra_covariates,
        )

        return historic, future

    def generate_config_data(self, variation: int = 0) -> dict[str, Any]:
        """Generate config data with variations for different hyperparameters."""
        return {
            "test_param_1": variation * 0.1,
            "test_param_2": f"variation_{variation}",
            "test_seed": variation,
        }

    def generate_geo_data(self, num_features: int = 5) -> dict[str, Any]:
        """Generate simple GeoJSON FeatureCollection with Point geometries."""
        features: list[dict[str, Any]] = []

        for i in range(num_features):
            # Random coordinates (longitude, latitude)
            lon = random.uniform(-180, 180)
            lat = random.uniform(-90, 90)

            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {"id": f"location_{i}"},
                }
            )

        return {"type": "FeatureCollection", "features": features}


def test_command(
    url: Annotated[
        str,
        typer.Option("--url", "-u", help="Base URL of running chapkit service"),
    ] = "http://localhost:8000",
    num_configs: Annotated[
        int,
        typer.Option("--configs", "-c", help="Number of configs to create"),
    ] = 1,
    num_trainings: Annotated[
        int,
        typer.Option("--trainings", "-t", help="Number of training jobs per config"),
    ] = 1,
    num_predictions: Annotated[
        int,
        typer.Option("--predictions", "-p", help="Number of predictions per trained model"),
    ] = 1,
    num_rows: Annotated[
        int,
        typer.Option("--rows", "-r", help="Number of rows in generated training data"),
    ] = 100,
    timeout: Annotated[
        float,
        typer.Option("--timeout", help="Job completion timeout in seconds"),
    ] = 60.0,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed output including job statuses"),
    ] = False,
    cleanup: Annotated[
        bool,
        typer.Option("--cleanup/--no-cleanup", help="Delete created test artifacts and configs after test"),
    ] = False,
) -> None:
    """Run end-to-end test of the ML service workflow."""
    runner = TestRunner(url, timeout=timeout, verbose=verbose)
    generator = TestDataGenerator(seed=42)  # Reproducible data

    # Statistics tracking
    stats: dict[str, Any] = {
        "configs_created": 0,
        "trainings_completed": 0,
        "trainings_failed": 0,
        "predictions_completed": 0,
        "predictions_failed": 0,
        "start_time": time.time(),
    }

    try:
        # 1. Health check
        typer.echo(f"Connecting to service at {url}...")
        success, message = runner.check_service_health()
        if not success:
            typer.echo(f"  [FAILED] {message}", err=True)
            typer.echo("Make sure the service is running with: uv run python main.py")
            raise typer.Exit(code=1)
        typer.echo(f"  [OK] {message}")
        typer.echo()

        # 2. Fetch service info
        success, message = runner.fetch_service_info()
        if not success:
            typer.echo(f"  [WARNING] {message}")
        elif verbose:
            typer.echo(
                f"  Service info: required_covariates={runner.required_covariates}, "
                f"requires_geo={runner.requires_geo}, "
                f"allow_free_additional={runner.allow_free_additional_continuous_covariates}"
            )
            typer.echo()

        # Determine extra covariates to add
        extra_covariates = 2 if runner.allow_free_additional_continuous_covariates else 0

        # Generate geo if required
        geo_data = generator.generate_geo_data() if runner.requires_geo else None

        # 3. Create configs
        typer.echo(f"Creating {num_configs} config(s)...")
        config_ids: list[str] = []
        for i in range(num_configs):
            config_name = f"test_config_{ULID()}"
            config_data = generator.generate_config_data(variation=i)

            success, message, config_id = runner.create_config(config_name, config_data)
            if success and config_id:
                config_ids.append(config_id)
                stats["configs_created"] += 1
                if verbose:
                    typer.echo(f"  [OK] {message}")
            else:
                typer.echo(f"  [FAILED] {message}", err=True)
                raise typer.Exit(code=1)

        typer.echo(f"  Created {len(config_ids)} config(s)")
        typer.echo()

        # 4. Run trainings
        total_trainings = num_configs * num_trainings
        typer.echo(f"Running {total_trainings} training job(s)...")

        model_artifacts: list[tuple[str, str]] = []  # List of (config_id, artifact_id) tuples

        for config_id in config_ids:
            for _ in range(num_trainings):
                training_data = generator.generate_training_data(
                    num_rows=num_rows,
                    required_covariates=runner.required_covariates,
                    extra_covariates=extra_covariates,
                )

                success, msg, job_id, artifact_id = runner.submit_training(config_id, training_data, geo_data)
                if not success:
                    typer.echo(f"  [FAILED] Submit: {msg}", err=True)
                    stats["trainings_failed"] += 1
                    continue

                if verbose:
                    typer.echo(f"  Submitted training job {job_id}")

                # Wait for completion
                success, msg, _ = runner.wait_for_job(job_id)  # type: ignore[arg-type]
                if success:
                    stats["trainings_completed"] += 1
                    model_artifacts.append((config_id, artifact_id))  # type: ignore[arg-type]
                    if verbose:
                        typer.echo(f"  [OK] Training {artifact_id}: {msg}")
                else:
                    stats["trainings_failed"] += 1
                    typer.echo(f"  [FAILED] Training job {job_id}: {msg}", err=True)

        typer.echo(f"  Completed: {stats['trainings_completed']}, Failed: {stats['trainings_failed']}")
        typer.echo()

        # 5. Run predictions
        if model_artifacts:
            total_predictions = len(model_artifacts) * num_predictions
            typer.echo(f"Running {total_predictions} prediction job(s)...")

            for _, model_artifact_id in model_artifacts:
                for _ in range(num_predictions):
                    historic, future = generator.generate_prediction_data(
                        num_rows=10,
                        required_covariates=runner.required_covariates,
                        extra_covariates=extra_covariates,
                    )

                    success, msg, job_id, pred_artifact_id = runner.submit_prediction(
                        model_artifact_id, historic, future, geo_data
                    )
                    if not success:
                        typer.echo(f"  [FAILED] Submit: {msg}", err=True)
                        stats["predictions_failed"] += 1
                        continue

                    if verbose:
                        typer.echo(f"  Submitted prediction job {job_id}")

                    # Wait for completion
                    success, msg, _ = runner.wait_for_job(job_id)  # type: ignore[arg-type]
                    if success:
                        stats["predictions_completed"] += 1

                        # Verify artifact exists
                        v_success, v_msg, _ = runner.verify_artifact(pred_artifact_id)  # type: ignore[arg-type]
                        if v_success and verbose:
                            typer.echo(f"  [OK] Prediction {pred_artifact_id}: verified")
                    else:
                        stats["predictions_failed"] += 1
                        typer.echo(f"  [FAILED] Prediction job {job_id}: {msg}", err=True)

            typer.echo(f"  Completed: {stats['predictions_completed']}, Failed: {stats['predictions_failed']}")
            typer.echo()

        # 6. Summary
        elapsed = time.time() - stats["start_time"]

        typer.echo("=" * 50)
        typer.echo("TEST SUMMARY")
        typer.echo("=" * 50)
        typer.echo(f"Service URL:           {url}")
        typer.echo(f"Elapsed time:          {elapsed:.2f}s")
        typer.echo(f"Configs created:       {stats['configs_created']}")
        typer.echo(f"Trainings completed:   {stats['trainings_completed']}")
        typer.echo(f"Trainings failed:      {stats['trainings_failed']}")
        typer.echo(f"Predictions completed: {stats['predictions_completed']}")
        typer.echo(f"Predictions failed:    {stats['predictions_failed']}")
        typer.echo()

        # Determine overall status
        total_failures = stats["trainings_failed"] + stats["predictions_failed"]
        if total_failures == 0 and stats["trainings_completed"] > 0:
            typer.echo("Result: ALL TESTS PASSED")
        elif total_failures > 0:
            typer.echo(f"Result: {total_failures} FAILURE(S)", err=True)
        else:
            typer.echo("Result: NO TESTS RUN", err=True)

        # 7. Optional cleanup
        if cleanup:
            typer.echo()
            typer.echo("Cleaning up test resources...")
            errors = runner.cleanup_resources()
            if errors:
                for error in errors:
                    typer.echo(f"  Warning: {error}", err=True)
            else:
                typer.echo("  Cleanup complete")

        # Exit with appropriate code
        if total_failures > 0:
            raise typer.Exit(code=1)

    finally:
        runner.close()
