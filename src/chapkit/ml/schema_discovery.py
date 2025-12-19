"""Schema discovery for external model scripts.

This module provides utilities to discover configuration schemas and service info
from external model scripts (R, Python, etc.) that implement an `info` command.

The expected JSON output format from `info --format json`:
{
    "service_info": {
        "period_type": "month",
        "required_covariates": ["rainfall", "temperature"],
        "allows_additional_continuous_covariates": true
    },
    "config_schema": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Model Configuration",
        "type": "object",
        "properties": {
            "smoothing": {"type": "number", "default": 0.5}
        }
    }
}
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

from pydantic import Field, create_model
from servicekit.logging import get_logger

from chapkit.config.schemas import BaseConfig

logger = get_logger(__name__)


class ModelInfo:
    """Discovered model information from an external script's info command."""

    def __init__(
        self,
        service_info: dict[str, Any],
        config_schema: dict[str, Any] | None,
        config_class: type[BaseConfig],
    ) -> None:
        """Initialize with discovered info.

        Args:
            service_info: Service metadata (period_type, required_covariates, etc.)
            config_schema: JSON Schema for configuration (may be None)
            config_class: Dynamically generated Pydantic config class
        """
        self.service_info = service_info
        self.config_schema = config_schema
        self.config_class = config_class

    @property
    def period_type(self) -> str:
        """Get the model's supported period type."""
        return self.service_info.get("period_type", "any")

    @property
    def required_covariates(self) -> list[str]:
        """Get the model's required covariates."""
        return self.service_info.get("required_covariates", [])

    @property
    def allows_additional_continuous_covariates(self) -> bool:
        """Whether the model accepts additional continuous covariates."""
        return self.service_info.get("allows_additional_continuous_covariates", False)


def discover_model_info(
    info_command: str,
    *,
    cwd: str | Path | None = None,
    timeout: float = 30.0,
    model_name: str = "ExternalModel",
) -> ModelInfo:
    """Discover model info by executing an info command synchronously.

    This is typically called at service startup to discover the schema
    from an external model script.

    Args:
        info_command: Command to execute (e.g., "Rscript model.R info --format json")
        cwd: Working directory for command execution
        timeout: Command timeout in seconds
        model_name: Name for the generated config class

    Returns:
        ModelInfo with service_info, config_schema, and generated config_class

    Raises:
        RuntimeError: If command fails or output is invalid
    """
    logger.info("discovering_model_info", command=info_command, cwd=str(cwd) if cwd else None)

    try:
        result = subprocess.run(
            info_command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Info command timed out after {timeout}s: {info_command}") from e

    if result.returncode != 0:
        raise RuntimeError(
            f"Info command failed with exit code {result.returncode}:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    # Parse JSON output
    try:
        # Find the JSON object in stdout (skip any non-JSON output like R package loading messages)
        stdout = result.stdout.strip()
        json_start = stdout.find("{")
        if json_start == -1:
            raise RuntimeError(
                f"No JSON object found in info command output:\n"
                f"stdout: {result.stdout}"
            )
        json_str = stdout[json_start:]
        info_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse info command output as JSON:\n"
            f"stdout: {result.stdout}\n"
            f"error: {e}"
        ) from e

    # Extract components
    service_info = info_data.get("service_info", {})
    config_schema = info_data.get("config_schema")

    # Generate Pydantic config class from schema
    config_class = create_config_from_schema(config_schema, model_name=model_name)

    logger.info(
        "model_info_discovered",
        period_type=service_info.get("period_type"),
        required_covariates=service_info.get("required_covariates"),
        has_config_schema=config_schema is not None,
    )

    return ModelInfo(
        service_info=service_info,
        config_schema=config_schema,
        config_class=config_class,
    )


async def discover_model_info_async(
    info_command: str,
    *,
    cwd: str | Path | None = None,
    timeout: float = 30.0,
    model_name: str = "ExternalModel",
) -> ModelInfo:
    """Async version of discover_model_info.

    Useful when called from async context (e.g., FastAPI lifespan).
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: discover_model_info(info_command, cwd=cwd, timeout=timeout, model_name=model_name),
    )


def create_config_from_schema(
    schema: dict[str, Any] | None,
    *,
    model_name: str = "DynamicConfig",
) -> type[BaseConfig]:
    """Create a Pydantic BaseConfig subclass from a JSON Schema.

    Supports basic JSON Schema types: string, number, integer, boolean, array.
    For complex schemas, consider defining the Pydantic model manually.

    Args:
        schema: JSON Schema dict, or None for empty config
        model_name: Name for the generated class

    Returns:
        A new BaseConfig subclass with fields matching the schema
    """
    if schema is None or not schema.get("properties"):
        # Empty config - create a class that accepts any extra fields
        return create_model(model_name, __base__=BaseConfig)  # type: ignore[call-overload]

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    # Build field definitions
    field_definitions: dict[str, Any] = {}

    for prop_name, prop_schema in properties.items():
        python_type = _json_schema_to_python_type(prop_schema)
        default = prop_schema.get("default")
        description = prop_schema.get("description")

        # Create field with metadata
        field_kwargs: dict[str, Any] = {}
        if description:
            field_kwargs["description"] = description

        # Handle constraints
        if "minimum" in prop_schema:
            field_kwargs["ge"] = prop_schema["minimum"]
        if "maximum" in prop_schema:
            field_kwargs["le"] = prop_schema["maximum"]
        if "exclusiveMinimum" in prop_schema:
            field_kwargs["gt"] = prop_schema["exclusiveMinimum"]
        if "exclusiveMaximum" in prop_schema:
            field_kwargs["lt"] = prop_schema["exclusiveMaximum"]
        if "minLength" in prop_schema:
            field_kwargs["min_length"] = prop_schema["minLength"]
        if "maxLength" in prop_schema:
            field_kwargs["max_length"] = prop_schema["maxLength"]

        # Determine if field is required
        is_required = prop_name in required

        if is_required and default is None:
            # Required field without default
            field_definitions[prop_name] = (python_type, Field(**field_kwargs))
        else:
            # Optional field or has default
            field_definitions[prop_name] = (python_type, Field(default=default, **field_kwargs))

    return create_model(model_name, __base__=BaseConfig, **field_definitions)  # type: ignore[call-overload]


def _json_schema_to_python_type(prop_schema: dict[str, Any]) -> type:
    """Convert JSON Schema type to Python type."""
    json_type = prop_schema.get("type", "string")

    type_map = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    # Handle nullable types
    if isinstance(json_type, list):
        # e.g., ["string", "null"]
        non_null_types = [t for t in json_type if t != "null"]
        if non_null_types:
            base_type = type_map.get(non_null_types[0], str)
            return base_type | None  # type: ignore[return-value]
        return str | None  # type: ignore[return-value]

    return type_map.get(json_type, str)
