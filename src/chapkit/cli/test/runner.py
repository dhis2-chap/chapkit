"""Test runner for orchestrating ML service testing."""

import time
from typing import Any

import httpx


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

    def fetch_config_schema(self) -> tuple[bool, str, dict[str, Any] | None]:
        """Fetch config JSON schema from service."""
        try:
            response = self.client.get(f"{self.base_url}/api/v1/configs/$schema")
            if response.status_code == 200:
                return True, "Schema fetched", response.json()
            return False, f"Failed to fetch schema: {response.text}", None
        except Exception as e:
            return False, f"Error fetching schema: {e}", None

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

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()
