"""Tests for artifact data serialization with non-JSON-serializable types."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime

import pytest
from ulid import ULID

from chapkit import ArtifactOut


class CustomNonSerializable:
    """A custom class that is not JSON-serializable."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"CustomNonSerializable(value={self.value})"


class TestArtifactSerialization:
    """Test artifact data field serialization with various data types."""

    def test_json_serializable_data_passes_through(self) -> None:
        """JSON-serializable data should be returned unchanged."""
        # Test various JSON-serializable types
        test_cases = [
            {"key": "value", "number": 42},
            [1, 2, 3, 4, 5],
            "simple string",
            42,
            3.14,
            True,
            None,
            {"nested": {"data": [1, 2, {"deep": "value"}]}},
        ]

        for data in test_cases:
            artifact = ArtifactOut(
                id=ULID(),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                data=data,
                parent_id=None,
                level=0,
            )

            # Serialize to dict (triggers field_serializer)
            serialized = artifact.model_dump()

            # Data should be unchanged
            assert serialized["data"] == data

            # The full artifact should be JSON-serializable via Pydantic
            json_str = artifact.model_dump_json()
            assert json_str is not None

            # Can parse back
            parsed = json.loads(json_str)
            assert parsed["data"] == data

    def test_custom_object_returns_metadata(self) -> None:
        """Non-serializable custom objects should return metadata."""
        custom_obj = CustomNonSerializable(value=42)

        artifact = ArtifactOut(
            id=ULID(),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            data=custom_obj,
            parent_id=None,
            level=0,
        )

        serialized = artifact.model_dump()

        # Should return metadata dict
        assert isinstance(serialized["data"], dict)
        assert serialized["data"]["_type"] == "CustomNonSerializable"
        assert serialized["data"]["_module"] == __name__
        assert "CustomNonSerializable(value=42)" in serialized["data"]["_repr"]
        assert "not JSON-serializable" in serialized["data"]["_serialization_error"]

        # The whole artifact should now be JSON-serializable via Pydantic
        json_str = artifact.model_dump_json()
        assert json_str is not None

        # Can parse back
        parsed = json.loads(json_str)
        assert parsed["data"]["_type"] == "CustomNonSerializable"

    def test_function_returns_metadata(self) -> None:
        """Functions should return metadata instead of crashing."""

        def my_function(x: int) -> int:
            return x * 2

        artifact = ArtifactOut(
            id=ULID(),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            data=my_function,
            parent_id=None,
            level=0,
        )

        serialized = artifact.model_dump()

        assert isinstance(serialized["data"], dict)
        assert serialized["data"]["_type"] == "function"
        assert "my_function" in serialized["data"]["_repr"]
        assert "_serialization_error" in serialized["data"]

    def test_bytes_returns_metadata(self) -> None:
        """Bytes objects should return metadata."""
        binary_data = b"\x00\x01\x02\x03\xff"

        artifact = ArtifactOut(
            id=ULID(),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            data=binary_data,
            parent_id=None,
            level=0,
        )

        serialized = artifact.model_dump()

        assert isinstance(serialized["data"], dict)
        assert serialized["data"]["_type"] == "bytes"
        assert "_serialization_error" in serialized["data"]

    def test_very_long_repr_is_truncated(self) -> None:
        """Very long repr strings should be truncated."""

        class LongRepr:
            def __repr__(self) -> str:
                return "x" * 1000  # Very long repr

        obj = LongRepr()

        artifact = ArtifactOut(
            id=ULID(),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            data=obj,
            parent_id=None,
            level=0,
        )

        serialized = artifact.model_dump()

        # Repr should be truncated to 200 chars + "..."
        assert len(serialized["data"]["_repr"]) <= 203  # 200 + "..."
        assert serialized["data"]["_repr"].endswith("...")

    def test_complex_nested_structure_with_non_serializable_parts(self) -> None:
        """Nested structures with both serializable and non-serializable parts."""
        # This should pass - the outer structure is serializable
        data = {
            "name": "experiment",
            "params": {"lr": 0.001, "epochs": 100},
            "results": [1, 2, 3],
        }

        artifact = ArtifactOut(
            id=ULID(),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            data=data,
            parent_id=None,
            level=0,
        )

        serialized = artifact.model_dump()
        assert serialized["data"] == data

        # Should be JSON-serializable via Pydantic
        json_str = artifact.model_dump_json()
        assert json_str is not None

    def test_set_returns_metadata(self) -> None:
        """Sets are not JSON-serializable and should return metadata."""
        data = {1, 2, 3, 4, 5}

        artifact = ArtifactOut(
            id=ULID(),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            data=data,
            parent_id=None,
            level=0,
        )

        serialized = artifact.model_dump()

        assert isinstance(serialized["data"], dict)
        assert serialized["data"]["_type"] == "set"
        assert "_serialization_error" in serialized["data"]

    @pytest.mark.skipif(
        importlib.util.find_spec("torch") is None,
        reason="Requires torch to be installed",
    )
    def test_torch_tensor_returns_metadata(self) -> None:
        """PyTorch tensors should return metadata (optional test)."""
        try:
            import torch  # type: ignore[import-not-found]

            tensor = torch.randn(3, 3)

            artifact = ArtifactOut(
                id=ULID(),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                data=tensor,
                parent_id=None,
                level=0,
            )

            serialized = artifact.model_dump()

            assert isinstance(serialized["data"], dict)
            assert serialized["data"]["_type"] == "Tensor"
            assert serialized["data"]["_module"] == "torch"
            assert "_serialization_error" in serialized["data"]
        except ImportError:
            pytest.skip("torch not installed")

    @pytest.mark.skipif(
        importlib.util.find_spec("sklearn") is None,
        reason="Requires scikit-learn to be installed",
    )
    def test_sklearn_model_returns_metadata(self) -> None:
        """Scikit-learn models should return metadata (optional test)."""
        try:
            import numpy as np
            from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]

            # Train a simple linear regression model
            X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
            y = np.dot(X, np.array([1, 2])) + 3
            model = LinearRegression().fit(X, y)

            artifact = ArtifactOut(
                id=ULID(),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                data=model,
                parent_id=None,
                level=0,
            )

            serialized = artifact.model_dump()

            assert isinstance(serialized["data"], dict)
            assert serialized["data"]["_type"] == "LinearRegression"
            assert "sklearn" in serialized["data"]["_module"]
            assert "_serialization_error" in serialized["data"]

            # Should be JSON-serializable via Pydantic
            json_str = artifact.model_dump_json()
            assert json_str is not None

            # Can parse back
            parsed = json.loads(json_str)
            assert parsed["data"]["_type"] == "LinearRegression"
        except ImportError:
            pytest.skip("scikit-learn not installed")

    def test_model_dump_json_works_with_non_serializable(self) -> None:
        """The model_dump_json() method should work with non-serializable data."""
        custom_obj = CustomNonSerializable(value=123)

        artifact = ArtifactOut(
            id=ULID(),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            data=custom_obj,
            parent_id=None,
            level=0,
        )

        # Should not raise an exception
        json_str = artifact.model_dump_json()
        assert json_str is not None

        # Should be parseable
        parsed = json.loads(json_str)
        assert parsed["data"]["_type"] == "CustomNonSerializable"
