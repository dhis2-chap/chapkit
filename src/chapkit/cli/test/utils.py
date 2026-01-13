"""Utility functions for test command."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx


def save_test_data(directory: Path, filename: str, data: dict[str, Any]) -> None:
    """Save test data to JSON file."""
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def start_service_subprocess(project_root: Path, port: int = 8000) -> subprocess.Popen[bytes]:
    """Start the service as a subprocess with in-memory database."""
    env = os.environ.copy()
    env["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

    process = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=project_root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process


def wait_for_service_ready(url: str, timeout: float = 30.0) -> tuple[bool, str]:
    """Poll health endpoint until service is ready."""
    start_time = time.time()
    poll_interval = 0.5

    while time.time() - start_time < timeout:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{url}/health")
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        return True, "Service is ready"
        except (httpx.ConnectError, httpx.TimeoutException):
            pass

        time.sleep(poll_interval)

    return False, f"Service did not become ready within {timeout}s"


def find_project_main(start_path: Path) -> Path | None:
    """Find main.py in the project root."""
    for parent in [start_path, *start_path.parents]:
        main_py = parent / "main.py"
        if main_py.exists():
            return main_py
    return None
