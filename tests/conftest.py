"""Test configuration and shared fixtures for chapkit."""

from pydantic import Field

from chapkit import BaseConfig


class DemoConfig(BaseConfig):
    """Concrete config schema for testing."""

    x: int
    y: int
    z: int
    tags: list[str] = Field(default_factory=list)
