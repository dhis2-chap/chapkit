"""Tests for typed artifact data schemas."""

import pytest
from pydantic import ValidationError

from chapkit.artifact.schemas import (
    GenericArtifactData,
    GenericMetadata,
    MLMetadata,
    MLPredictionArtifactData,
    MLTrainingWorkspaceArtifactData,
    validate_artifact_data,
)

# Tests for MLMetadata


def test_ml_metadata_valid_success():
    """Test MLMetadata with success status."""
    metadata = MLMetadata(
        status="success",
        config_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        started_at="2025-01-10T10:00:00Z",
        completed_at="2025-01-10T10:00:42Z",
        duration_seconds=42.5,
    )

    assert metadata.status == "success"
    assert metadata.config_id == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    assert metadata.duration_seconds == 42.5


def test_ml_metadata_valid_failed():
    """Test MLMetadata with failed status."""
    metadata = MLMetadata(
        status="failed",
        config_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        started_at="2025-01-10T10:00:00Z",
        completed_at="2025-01-10T10:00:15Z",
        duration_seconds=15.2,
    )

    assert metadata.status == "failed"


def test_ml_metadata_invalid_status():
    """Test MLMetadata rejects invalid status."""
    with pytest.raises(ValidationError):
        MLMetadata(
            status="pending",  # type: ignore[arg-type]  # Testing invalid status
            config_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
            started_at="2025-01-10T10:00:00Z",
            completed_at="2025-01-10T10:00:15Z",
            duration_seconds=15.2,
        )


# Tests for GenericMetadata


def test_generic_metadata_allows_extra_fields():
    """Test GenericMetadata allows arbitrary fields."""
    metadata = GenericMetadata(
        experiment_name="test",  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]  # Testing extra fields
        custom_field="value",  # pyright: ignore[reportCallIssue]  # Testing extra fields
        nested={"data": 123},  # pyright: ignore[reportCallIssue]  # Testing extra fields
    )

    assert metadata.model_extra["experiment_name"] == "test"  # type: ignore[index]  # model_extra is dict when extra="allow"
    assert metadata.model_extra["custom_field"] == "value"  # type: ignore[index]  # model_extra is dict when extra="allow"
    assert metadata.model_extra["nested"] == {"data": 123}  # type: ignore[index]  # model_extra is dict when extra="allow"


def test_generic_metadata_empty():
    """Test GenericMetadata can be empty."""
    metadata = GenericMetadata()
    assert metadata.model_dump() == {}


# Tests for MLTrainingWorkspaceArtifactData


def test_ml_training_workspace_artifact_data_valid():
    """Test MLTrainingWorkspaceArtifactData with valid data."""
    metadata = MLMetadata(
        status="success",
        config_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        started_at="2025-01-10T10:00:00Z",
        completed_at="2025-01-10T10:00:42Z",
        duration_seconds=42.5,
    )

    artifact_data = MLTrainingWorkspaceArtifactData(
        type="ml_training_workspace",
        metadata=metadata,
        content=b"model bytes",
        content_type="application/zip",
        content_size=1024,
    )

    assert artifact_data.type == "ml_training_workspace"
    assert artifact_data.metadata.status == "success"
    assert artifact_data.content == b"model bytes"
    assert artifact_data.content_type == "application/zip"
    assert artifact_data.content_size == 1024


def test_ml_training_workspace_artifact_data_wrong_type():
    """Test MLTrainingWorkspaceArtifactData rejects wrong type value."""
    metadata = MLMetadata(
        status="success",
        config_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        started_at="2025-01-10T10:00:00Z",
        completed_at="2025-01-10T10:00:42Z",
        duration_seconds=42.5,
    )

    with pytest.raises(ValidationError):
        MLTrainingWorkspaceArtifactData(
            type="ml_prediction",  # type: ignore[arg-type]  # Testing wrong type
            metadata=metadata,
            content=b"model bytes",
        )


def test_ml_training_workspace_artifact_data_optional_fields():
    """Test MLTrainingWorkspaceArtifactData with optional fields as None."""
    metadata = MLMetadata(
        status="success",
        config_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        started_at="2025-01-10T10:00:00Z",
        completed_at="2025-01-10T10:00:42Z",
        duration_seconds=42.5,
    )

    artifact_data = MLTrainingWorkspaceArtifactData(
        metadata=metadata,
        content=b"model bytes",
        content_type=None,
        content_size=None,
    )

    assert artifact_data.content_type is None
    assert artifact_data.content_size is None


# Tests for MLPredictionArtifactData


def test_ml_prediction_artifact_data_valid():
    """Test MLPredictionArtifactData with valid data."""
    metadata = MLMetadata(
        status="success",
        config_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        started_at="2025-01-10T11:00:00Z",
        completed_at="2025-01-10T11:00:05Z",
        duration_seconds=5.2,
    )

    artifact_data = MLPredictionArtifactData(
        type="ml_prediction",
        metadata=metadata,
        content={"predictions": [1, 2, 3]},
        content_type="text/csv",
        content_size=500,
    )

    assert artifact_data.type == "ml_prediction"
    assert artifact_data.metadata.status == "success"
    assert artifact_data.content_type == "text/csv"


# Tests for GenericArtifactData


def test_generic_artifact_data_valid():
    """Test GenericArtifactData with custom metadata."""
    metadata = GenericMetadata(
        experiment_name="rainfall-prediction",  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]  # Testing extra fields
        dataset_version="2024-11-10",  # pyright: ignore[reportCallIssue]  # Testing extra fields
    )

    artifact_data = GenericArtifactData(
        type="generic",
        metadata=metadata,
        content=b"zip bytes",
        content_type="application/zip",
        content_size=2048,
    )

    assert artifact_data.type == "generic"
    assert artifact_data.metadata.model_extra["experiment_name"] == "rainfall-prediction"  # type: ignore[index]  # model_extra is dict when extra="allow"
    assert artifact_data.content_type == "application/zip"


# Tests for Discriminated Union


def test_discriminated_union_ml_training_workspace():
    """Test discriminated union routes to MLTrainingWorkspaceArtifactData."""
    data_dict = {
        "type": "ml_training_workspace",
        "metadata": {
            "status": "success",
            "config_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "started_at": "2025-01-10T10:00:00Z",
            "completed_at": "2025-01-10T10:00:42Z",
            "duration_seconds": 42.5,
        },
        "content": b"model",
        "content_type": "application/zip",
        "content_size": 1024,
    }

    # Pydantic's discriminated union should route to correct type
    artifact_data = MLTrainingWorkspaceArtifactData.model_validate(data_dict)
    assert isinstance(artifact_data, MLTrainingWorkspaceArtifactData)
    assert artifact_data.type == "ml_training_workspace"


def test_discriminated_union_ml_prediction():
    """Test discriminated union routes to MLPredictionArtifactData."""
    data_dict = {
        "type": "ml_prediction",
        "metadata": {
            "status": "success",
            "config_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "started_at": "2025-01-10T11:00:00Z",
            "completed_at": "2025-01-10T11:00:05Z",
            "duration_seconds": 5.2,
        },
        "content": {"predictions": []},
        "content_type": "text/csv",
    }

    artifact_data = MLPredictionArtifactData.model_validate(data_dict)
    assert isinstance(artifact_data, MLPredictionArtifactData)
    assert artifact_data.type == "ml_prediction"


def test_discriminated_union_generic():
    """Test discriminated union routes to GenericArtifactData."""
    data_dict = {
        "type": "generic",
        "metadata": {"custom": "value"},
        "content": b"data",
    }

    artifact_data = GenericArtifactData.model_validate(data_dict)
    assert isinstance(artifact_data, GenericArtifactData)
    assert artifact_data.type == "generic"


# Tests for validate_artifact_data helper


def test_validate_artifact_data_ml_training_workspace():
    """Test validate_artifact_data with ml_training_workspace type."""
    data_dict = {
        "type": "ml_training_workspace",
        "metadata": {
            "status": "success",
            "config_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "started_at": "2025-01-10T10:00:00Z",
            "completed_at": "2025-01-10T10:00:42Z",
            "duration_seconds": 42.5,
        },
        "content": b"model",
    }

    result = validate_artifact_data(data_dict)
    assert isinstance(result, MLTrainingWorkspaceArtifactData)
    assert result.metadata.status == "success"


def test_validate_artifact_data_ml_prediction():
    """Test validate_artifact_data with ml_prediction type."""
    data_dict = {
        "type": "ml_prediction",
        "metadata": {
            "status": "success",
            "config_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "started_at": "2025-01-10T11:00:00Z",
            "completed_at": "2025-01-10T11:00:05Z",
            "duration_seconds": 5.2,
        },
        "content": {},
    }

    result = validate_artifact_data(data_dict)
    assert isinstance(result, MLPredictionArtifactData)


def test_validate_artifact_data_generic():
    """Test validate_artifact_data with generic type."""
    data_dict = {
        "type": "generic",
        "metadata": {"custom": "value"},
        "content": "data",
    }

    result = validate_artifact_data(data_dict)
    assert isinstance(result, GenericArtifactData)


def test_validate_artifact_data_no_type_defaults_to_generic():
    """Test validate_artifact_data defaults to generic when no type specified."""
    data_dict = {
        "metadata": {},
        "content": "data",
    }

    result = validate_artifact_data(data_dict)
    assert isinstance(result, GenericArtifactData)


def test_validate_artifact_data_unknown_type_raises_error():
    """Test validate_artifact_data raises error for unknown type."""
    data_dict = {
        "type": "unknown_type",
        "metadata": {},
        "content": "data",
    }

    # Unknown types should raise ValidationError since type is a Literal
    with pytest.raises(ValidationError):
        validate_artifact_data(data_dict)


def test_validate_artifact_data_invalid_raises_error():
    """Test validate_artifact_data raises ValidationError for invalid data."""
    data_dict = {
        "type": "ml_training_workspace",
        "metadata": {
            "status": "invalid_status",  # Invalid
            "config_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
            "started_at": "2025-01-10T10:00:00Z",
            "completed_at": "2025-01-10T10:00:42Z",
            "duration_seconds": 42.5,
        },
        "content": b"model",
    }

    with pytest.raises(ValidationError):
        validate_artifact_data(data_dict)


# Tests for extra field forbidding


def test_base_artifact_data_forbids_extra_fields():
    """Test BaseArtifactData forbids extra fields."""
    metadata = MLMetadata(
        status="success",
        config_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        started_at="2025-01-10T10:00:00Z",
        completed_at="2025-01-10T10:00:42Z",
        duration_seconds=42.5,
    )

    with pytest.raises(ValidationError):
        MLTrainingWorkspaceArtifactData(
            type="ml_training_workspace",
            metadata=metadata,
            content=b"model",
            extra_field="not allowed",  # type: ignore[call-arg]  # Testing forbidden field
        )


# Tests for serialization


def test_ml_training_workspace_artifact_data_serialization():
    """Test MLTrainingWorkspaceArtifactData serialization to dict."""
    metadata = MLMetadata(
        status="success",
        config_id="01ARZ3NDEKTSV4RRFFQ69G5FAV",
        started_at="2025-01-10T10:00:00Z",
        completed_at="2025-01-10T10:00:42Z",
        duration_seconds=42.5,
    )

    artifact_data = MLTrainingWorkspaceArtifactData(
        metadata=metadata,
        content="simple content",
        content_type="text/plain",
        content_size=100,
    )

    data_dict = artifact_data.model_dump()

    assert data_dict["type"] == "ml_training_workspace"
    assert data_dict["metadata"]["status"] == "success"
    assert data_dict["metadata"]["config_id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    assert data_dict["content"] == "simple content"
    assert data_dict["content_type"] == "text/plain"
    assert data_dict["content_size"] == 100


def test_generic_artifact_data_serialization_with_extra():
    """Test GenericArtifactData serialization preserves extra metadata fields."""
    metadata = GenericMetadata(
        experiment="test",  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]  # Testing extra fields
        version="1.0",  # pyright: ignore[reportCallIssue]  # Testing extra fields
    )

    artifact_data = GenericArtifactData(
        metadata=metadata,
        content="data",
    )

    data_dict = artifact_data.model_dump()

    assert data_dict["type"] == "generic"
    assert data_dict["metadata"]["experiment"] == "test"
    assert data_dict["metadata"]["version"] == "1.0"
