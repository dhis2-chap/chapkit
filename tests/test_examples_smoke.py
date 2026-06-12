"""Smoke tests ensuring every example app imports and starts; the only tests allowed to touch examples/."""

from __future__ import annotations

import importlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

EXAMPLE_MODULES = [
    "examples.artifact.main",
    "examples.config.main",
    "examples.library_usage.main",
    "examples.ml_class.main",
]


@pytest.mark.parametrize("module_name", EXAMPLE_MODULES)
def test_example_app_starts(module_name: str) -> None:
    """Import the example app, run its lifespan, and hit the health endpoint."""
    if module_name == "examples.ml_class.main":
        pytest.importorskip("sklearn")
        pytest.importorskip("pandas")

    module = importlib.import_module(module_name)
    app = module.app
    assert isinstance(app, FastAPI)

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code in (200, 503)
