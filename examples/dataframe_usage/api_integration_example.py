"""Example showing DataFrame integration with FastAPI."""

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from chapkit.data import DataFrame

app = FastAPI(title="DataFrame API Example")


class DataUploadResponse(BaseModel):
    """Response for data upload."""

    rows: int
    columns: list[str]
    types: dict[str, str]
    sample: list[dict]


class DataStats(BaseModel):
    """Data statistics."""

    total_rows: int
    total_columns: int
    column_types: dict[str, str]
    null_counts: dict[str, int]


@app.post("/data/$upload", response_model=DataUploadResponse)
async def upload_csv(file: UploadFile) -> DataUploadResponse:
    """Upload CSV data and return metadata."""
    # Read uploaded file
    content = await file.read()
    df = DataFrame.from_csv(csv_string=content.decode())

    # Validate structure
    try:
        df.validate_structure()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV structure: {e}")

    # Return metadata
    return DataUploadResponse(
        rows=df.shape[0], columns=df.columns, types=df.infer_types(), sample=df.head(5).to_dict(orient="records")
    )


@app.post("/data/$validate")
async def validate_data(file: UploadFile) -> dict:
    """Validate uploaded CSV data quality."""
    content = await file.read()
    df = DataFrame.from_csv(csv_string=content.decode())

    # Check required columns
    required_columns = ["id", "name", "value"]
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing)}")

    # Check for nulls in required columns
    nulls = df.has_nulls()
    null_required = [col for col in required_columns if nulls.get(col, False)]
    if null_required:
        raise HTTPException(status_code=400, detail=f"Required columns contain nulls: {', '.join(null_required)}")

    return {"status": "valid", "message": "Data validation passed"}


@app.get("/data/$download")
async def download_csv() -> Response:
    """Download data as CSV."""
    # Generate sample data
    df = DataFrame.from_dict({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "score": [95.5, 87.0, 92.3]})

    csv_data = df.to_csv()

    return Response(
        content=csv_data, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=data.csv"}
    )


@app.get("/data/$stats", response_model=DataStats)
async def get_stats() -> DataStats:
    """Get data statistics."""
    # Get data (from database, cache, etc.)
    df = DataFrame.from_dict(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", None, "Eve"],
            "age": [25, 30, None, 28, 42],
            "score": [95.5, 87.0, 92.3, 88.5, 91.0],
        }
    )

    # Calculate null counts
    null_counts = {}
    for col in df.columns:
        col_idx = df.columns.index(col)
        null_count = sum(1 for row in df.data if row[col_idx] is None)
        null_counts[col] = null_count

    return DataStats(
        total_rows=df.shape[0], total_columns=df.shape[1], column_types=df.infer_types(), null_counts=null_counts
    )


@app.post("/data/$transform")
async def transform_data(file: UploadFile) -> Response:
    """Transform uploaded CSV data."""
    content = await file.read()
    df = DataFrame.from_csv(csv_string=content.decode())

    # Apply transformations
    df = (
        df.select([col for col in df.columns if not col.startswith("_")])  # Remove private columns
        .rename({col: col.lower() for col in df.columns})  # Lowercase column names
        .head(1000)  # Limit to 1000 rows
    )

    # Return transformed data
    csv_data = df.to_csv()
    return Response(content=csv_data, media_type="text/csv")


@app.get("/data/$sample")
async def get_sample(n: int = 10, random_state: int = 42) -> list[dict]:
    """Get random sample of data."""
    # Get full dataset
    df = DataFrame.from_dict({"id": list(range(1, 101)), "value": [i * 2.5 for i in range(1, 101)]})

    # Sample
    sample = df.sample(n=min(n, df.shape[0]), random_state=random_state)

    result: list[dict] = sample.to_dict(orient="records")
    return result


if __name__ == "__main__":
    import uvicorn

    print("Starting DataFrame API example...")
    print("Available endpoints:")
    print("  POST /data/$upload - Upload CSV and get metadata")
    print("  POST /data/$validate - Validate CSV data quality")
    print("  GET  /data/$download - Download sample CSV")
    print("  GET  /data/$stats - Get data statistics")
    print("  POST /data/$transform - Transform CSV data")
    print("  GET  /data/$sample - Get random sample")
    print()
    print("Visit http://localhost:8000/docs for interactive documentation")

    uvicorn.run(app, host="0.0.0.0", port=8000)
