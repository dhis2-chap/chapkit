"""Smoke tests ensuring every example app imports and starts; the only tests allowed to touch examples/."""

from __future__ import annotations

import importlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Each entry is (module_name, allows_unhealthy). The artifact example wires a
# randomized health check (check_flaky_service) that can report 503, so only it
# may answer non-200; every other example must be healthy on startup.
EXAMPLE_MODULES = [
    ("examples.artifact.main", True),
    ("examples.config.main", False),
    ("examples.library_usage.main", False),
    ("examples.ml_class.main", False),
]


@pytest.mark.parametrize(("module_name", "allows_unhealthy"), EXAMPLE_MODULES)
def test_example_app_starts(module_name: str, allows_unhealthy: bool) -> None:
    """Import the example app, run its lifespan, and hit the health endpoint."""
    if module_name == "examples.ml_class.main":
        pytest.importorskip("sklearn")
        pytest.importorskip("pandas")

    module = importlib.import_module(module_name)
    app = module.app
    assert isinstance(app, FastAPI)

    with TestClient(app) as client:
        response = client.get("/health")
        expected = (200, 503) if allows_unhealthy else (200,)
        assert response.status_code in expected
