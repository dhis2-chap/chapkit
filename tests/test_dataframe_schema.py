"""Tests for the self-describing schema on chapkit.data.DataFrame."""

from __future__ import annotations

import json

from chapkit.data import DataFrame, DataFrameSchema


class TestBackwardCompatibility:
    """A DataFrame without a schema must serialize exactly as before."""

    def test_plain_dump_has_no_schema_key(self) -> None:
        """model_dump omits the schema key entirely when no schema is set."""
        frame = DataFrame(columns=["a", "b"], data=[[1, 2.5]])
        assert frame.model_dump() == {"columns": ["a", "b"], "data": [[1, 2.5]]}

    def test_plain_json_has_no_schema_key(self) -> None:
        """JSON serialization omits the schema key when no schema is set."""
        frame = DataFrame(columns=["a"], data=[[1]])
        assert "schema" not in json.loads(frame.model_dump_json())

    def test_validates_legacy_payload_without_schema(self) -> None:
        """A columns/data payload (no schema) still validates and has schema None."""
        frame = DataFrame.model_validate({"columns": ["a"], "data": [[1]]})
        assert frame.table_schema is None

    def test_by_alias_dump_omits_schema_when_absent(self) -> None:
        """by_alias serialization (used by FastAPI responses) also omits a null schema."""
        frame = DataFrame(columns=["a", "b"], data=[[1, 2.5]])
        assert frame.model_dump(by_alias=True) == {"columns": ["a", "b"], "data": [[1, 2.5]]}
        assert "schema" not in json.loads(frame.model_dump_json(by_alias=True))


class TestWithSchema:
    """with_schema attaches a contract-derived self-describing schema."""

    def test_canonical_required_and_inferred_types(self) -> None:
        """Types come from the canonical set, required covariates, then inference."""
        frame = DataFrame(
            columns=["time_period", "location", "population", "rainfall", "feature_0", "note"],
            data=[["2020-01", "loc_0", 1000, 3.4, 5.1, "hi"]],
        )
        described = frame.with_schema(required_covariates=["feature_0"])
        assert described.table_schema is not None
        types = {field.name: field.type for field in described.table_schema.fields}
        assert types["time_period"] == "string"
        assert types["location"] == "string"
        assert types["population"] == "integer"
        assert types["rainfall"] == "number"
        assert types["feature_0"] == "number"  # declared required covariate
        assert types["note"] == "string"  # inferred from values

    def test_config_schema_types_applied(self) -> None:
        """A config JSON Schema supplies types for matching, non-canonical columns."""
        frame = DataFrame(columns=["flag"], data=[[True]])
        described = frame.with_schema(config_schema={"properties": {"flag": {"type": "boolean"}}})
        assert described.table_schema is not None
        assert described.table_schema.fields[0].type == "boolean"

    def test_schema_field_order_matches_columns(self) -> None:
        """The schema lists one field per column, in column order."""
        frame = DataFrame(columns=["x", "y", "z"], data=[[1, 2, 3]])
        described = frame.with_schema()
        assert described.table_schema is not None
        assert [field.name for field in described.table_schema.fields] == ["x", "y", "z"]

    def test_described_dump_exposes_schema_under_schema_key(self) -> None:
        """The attached schema serializes under the JSON key "schema"."""
        described = DataFrame(columns=["x"], data=[[1]]).with_schema()
        dumped = described.model_dump()
        assert "schema" in dumped
        assert dumped["schema"]["fields"][0] == {
            "name": "x",
            "type": "integer",
            "title": None,
            "description": None,
        }

    def test_described_dump_by_alias_includes_schema(self) -> None:
        """A described frame still emits the schema under by_alias serialization."""
        described = DataFrame(columns=["x"], data=[[1]]).with_schema()
        assert "schema" in described.model_dump(by_alias=True)

    def test_with_schema_round_trips_through_validation(self) -> None:
        """A described DataFrame can be re-parsed from its own dump via the alias."""
        described = DataFrame(columns=["x"], data=[[1]]).with_schema()
        reparsed = DataFrame.model_validate(described.model_dump())
        assert isinstance(reparsed.table_schema, DataFrameSchema)
        assert reparsed.table_schema.fields[0].name == "x"
