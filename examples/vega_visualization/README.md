# Vega Visualization Service

A proof-of-concept visualization service that transforms DataFrame data to Vega-Lite specifications. This demonstrates the **Transformation Service Pattern** (Phase 5.1 from the enhancement roadmap).

## Features

- **Data Transformation**: Universal DataFrame → Vega-Lite grammar
- **Multiple Chart Types**: Line, bar, scatter, heatmap, boxplot, histogram
- **Multiple Output Formats**: JSON spec, PNG, SVG, or standalone HTML
- **Multi-Format Input**: Works with pandas, polars, xarray, dicts, records
- **Data Processing**: Aggregation, grouping, filtering
- **Built with Altair**: Clean, Pythonic chart generation
- **REST API**: Clean `/$operation` endpoints following servicekit conventions
- **Type Safety**: Full Pydantic validation for requests and responses
- **Extensible**: Easy to add new chart types and transformations

## Architecture

This example follows the **Transformation Router Pattern**:

```
Input (DataFrame) → Router → Processing → Output (Vega-Lite Spec)
```

Key components:
- **VegaRouter**: Custom router extending `Router` base class
- **VegaGenerateRequest**: Input schema with data and visualization parameters
- **VegaResponse**: Output schema with Vega-Lite spec and metadata
- **Data Processing**: Pandas operations for aggregation and transformation

## Quick Start

### Local Development

```bash
# Install dependencies
uv sync

# Run the service
uv run uvicorn main:app --reload

# Or use fastapi CLI
uv run fastapi dev main.py
```

The API will be available at http://localhost:8000

### Docker Deployment

```bash
# Build and run
docker compose up

# Detached mode
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

## API Endpoints

### Core Endpoints

- `GET /health` - Health check
- `GET /api/v1/system` - Service information
- `GET /metrics` - Prometheus metrics
- `GET /` - Landing page with API docs

### Visualization Operations

- `POST /api/v1/visualizations/$generate` - Generate Vega-Lite spec from data
- `POST /api/v1/visualizations/$aggregate` - Aggregate data and visualize

## Output Formats

The service supports multiple output formats via the `format` parameter:

| Format | Content-Type | Description | Use Case |
|--------|-------------|-------------|----------|
| `json` | `application/json` | Vega-Lite JSON spec | For Vega renderers, embedding in web apps |
| `png` | `image/png` | Rendered PNG image (2x scale) | Static images, reports, emails |
| `svg` | `image/svg+xml` | Rendered SVG vector graphic | High-quality prints, scaling |
| `html` | `text/html` | Standalone HTML page | Direct browser viewing, sharing |

**Default:** `json` (Vega-Lite specification)

## Usage Examples

### Example 1: Simple Line Chart (JSON)

```bash
curl -X POST http://localhost:8000/api/v1/visualizations/\$generate \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "columns": ["month", "sales", "region"],
      "data": [
        ["Jan", 120, "North"],
        ["Feb", 150, "North"],
        ["Mar", 180, "North"],
        ["Jan", 100, "South"],
        ["Feb", 130, "South"],
        ["Mar", 160, "South"]
      ]
    },
    "chart_type": "line",
    "x_field": "month",
    "y_field": "sales",
    "color_field": "region",
    "title": "Monthly Sales by Region",
    "format": "json"
  }'
```

**Response:**
```json
{
  "spec": {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "data": {
      "values": [
        {"month": "Jan", "sales": 120, "region": "North"},
        {"month": "Feb", "sales": 150, "region": "North"},
        ...
      ]
    },
    "mark": {"type": "line", "point": true, "tooltip": true},
    "encoding": {
      "x": {"field": "month", "type": "nominal"},
      "y": {"field": "sales", "type": "quantitative"},
      "color": {"field": "region", "type": "nominal"}
    },
    "title": "Monthly Sales by Region",
    "width": 600,
    "height": 400
  },
  "row_count": 6,
  "columns": ["month", "sales", "region"]
}
```

**Note**: To use with the Vega online renderer, copy only the `spec` object contents (not the entire response).

### Example 2: Bar Chart with Aggregation

```bash
curl -X POST http://localhost:8000/api/v1/visualizations/\$generate \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "columns": ["category", "value"],
      "data": [
        ["A", 10],
        ["A", 15],
        ["A", 12],
        ["B", 20],
        ["B", 25],
        ["C", 8]
      ]
    },
    "chart_type": "bar",
    "x_field": "category",
    "y_field": "value",
    "aggregate": "mean",
    "title": "Average Value by Category"
  }'
```

### Example 3: Generate PNG Image

```bash
curl -X POST http://localhost:8000/api/v1/visualizations/\$generate \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "columns": ["x", "y"],
      "data": [[1, 10], [2, 25], [3, 15], [4, 30], [5, 20]]
    },
    "chart_type": "line",
    "x_field": "x",
    "y_field": "y",
    "title": "Trend Analysis",
    "format": "png"
  }' \
  --output chart.png
```

Saves a high-resolution PNG image (2x scale for retina displays).

### Example 4: Generate SVG Vector Graphic

```bash
curl -X POST http://localhost:8000/api/v1/visualizations/\$generate \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "columns": ["category", "value"],
      "data": [["A", 10], ["B", 20], ["C", 15]]
    },
    "chart_type": "bar",
    "x_field": "category",
    "y_field": "value",
    "title": "Category Breakdown",
    "format": "svg"
  }' \
  --output chart.svg
```

Perfect for high-quality prints and infinite scaling.

### Example 5: Generate Standalone HTML

```bash
curl -X POST http://localhost:8000/api/v1/visualizations/\$generate \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "columns": ["date", "temperature"],
      "data": [["Mon", 72], ["Tue", 68], ["Wed", 75], ["Thu", 70], ["Fri", 73]]
    },
    "chart_type": "line",
    "x_field": "date",
    "y_field": "temperature",
    "title": "Weekly Temperature",
    "format": "html"
  }' \
  --output chart.html
```

Open `chart.html` in any browser for an interactive visualization.

### Example 6: Data Aggregation

Use the `/$aggregate` endpoint to group and aggregate data before visualization:

```bash
curl -X POST http://localhost:8000/api/v1/visualizations/\$aggregate \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "columns": ["date", "product", "quantity", "revenue"],
      "data": [
        ["2025-01-01", "Widget", 5, 100],
        ["2025-01-01", "Gadget", 3, 75],
        ["2025-01-02", "Widget", 7, 140],
        ["2025-01-02", "Gadget", 4, 100],
        ["2025-01-03", "Widget", 6, 120],
        ["2025-01-03", "Gadget", 5, 125]
      ]
    },
    "group_by": ["date", "product"],
    "agg_field": "revenue",
    "agg_func": "sum",
    "chart_type": "bar",
    "title": "Total Revenue by Date and Product"
  }'
```

### Example 4: Scatter Plot

```bash
curl -X POST http://localhost:8000/api/v1/visualizations/\$generate \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "columns": ["temperature", "ice_cream_sales", "season"],
      "data": [
        [65, 100, "Spring"],
        [70, 150, "Spring"],
        [75, 200, "Summer"],
        [80, 280, "Summer"],
        [85, 320, "Summer"],
        [55, 50, "Fall"]
      ]
    },
    "chart_type": "scatter",
    "x_field": "temperature",
    "y_field": "ice_cream_sales",
    "color_field": "season",
    "title": "Ice Cream Sales vs Temperature"
  }'
```

### Example 5: Histogram

```bash
curl -X POST http://localhost:8000/api/v1/visualizations/\$generate \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "columns": ["age"],
      "data": [
        [25], [28], [32], [35], [38], [42], [45], [48], [52], [55],
        [22], [26], [29], [33], [36], [40], [44], [47], [50], [54]
      ]
    },
    "chart_type": "histogram",
    "x_field": "age",
    "title": "Age Distribution"
  }'
```

### Example 6: Heatmap

```bash
curl -X POST http://localhost:8000/api/v1/visualizations/\$generate \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "columns": ["hour", "day", "traffic"],
      "data": [
        [8, "Mon", 120],
        [9, "Mon", 180],
        [10, "Mon", 150],
        [8, "Tue", 110],
        [9, "Tue", 190],
        [10, "Tue", 160]
      ]
    },
    "chart_type": "heatmap",
    "x_field": "hour",
    "y_field": "day",
    "aggregate": "sum",
    "title": "Traffic Heatmap"
  }'
```

## Python Client Example

```python
import requests
import pandas as pd
from chapkit.data import DataFrame

# Create sample data
df = pd.DataFrame({
    "month": ["Jan", "Feb", "Mar", "Apr"],
    "revenue": [10000, 12000, 15000, 18000],
    "expenses": [7000, 8000, 9000, 10000],
})

# Convert to DataFrame schema
data_schema = DataFrame.from_pandas(df)

# Generate line chart
response = requests.post(
    "http://localhost:8000/api/v1/visualizations/$generate",
    json={
        "data": data_schema.model_dump(),
        "chart_type": "line",
        "x_field": "month",
        "y_field": "revenue",
        "title": "Monthly Revenue",
    },
)

vega_spec = response.json()["spec"]
print(vega_spec)

# You can now render this spec using Vega-Lite in your frontend
# or use altair/vega libraries to display it
```

## Supported Chart Types

| Chart Type | Use Case | Required Fields |
|------------|----------|-----------------|
| `line` | Time series, trends | `x_field`, `y_field` |
| `bar` | Comparisons, categorical data | `x_field`, `y_field` |
| `scatter` | Correlations, relationships | `x_field`, `y_field` |
| `heatmap` | Density, patterns | `x_field`, `y_field` |
| `boxplot` | Distribution, outliers | `x_field`, `y_field` |
| `histogram` | Frequency distribution | `x_field` or `y_field` |

## Aggregation Functions

Supported aggregation functions for the `aggregate` parameter:
- `mean` - Average value
- `sum` - Total sum
- `count` - Count of records
- `median` - Median value
- `min` - Minimum value
- `max` - Maximum value

## Architecture Details

### Router Pattern

The `VegaRouter` extends servicekit's `Router` base class:

```python
class VegaRouter(Router):
    def _register_routes(self) -> None:
        # Register endpoints using @self.router decorators
        # Endpoints use $ prefix for operations (/$generate, /$aggregate)
```

This follows servicekit conventions:
- **Collection operations**: Use `/$operation` prefix
- **Type safety**: Pydantic models for request/response
- **Standard patterns**: Consistent with CRUD routers

### Data Flow

```
1. Client sends DataFrame + parameters
   ↓
2. Pydantic validates request schema
   ↓
3. Router converts DataFrame → pandas.DataFrame
   ↓
4. Data processing (optional aggregation)
   ↓
5. Vega-Lite spec generation
   ↓
6. Response with spec + metadata
```

### Extension Points

Easy to extend with new features:

**Add new chart type:**
```python
elif chart_type == "area":
    spec["mark"] = {"type": "area", "tooltip": True}
    spec["encoding"] = self._build_encoding(...)
```

**Add data transformations:**
```python
@self.router.post("/$filter")
async def filter_and_visualize(request: FilterRequest):
    df = request.data.to_dataframe()
    filtered_df = df[df[request.filter_field] > request.threshold]
    # Generate viz from filtered data
```

**Add caching (Phase 2.2):**
```python
from servicekit import InMemoryCache

cache = InMemoryCache(maxsize=1000, ttl=3600)

app = (
    BaseServiceBuilder(...)
    .with_cache(cache)  # Enable caching
    .build()
)
```

## Roadmap Alignment

This example demonstrates **Phase 5.1: Transformation Service Pattern** from the enhancement roadmap:

- ✅ **5.1.1**: TransformationRouter pattern (stateless data-in → data-out)
- ✅ Input validation with Pydantic schemas
- ✅ RESTful API with `/$operation` endpoints
- 🔄 **5.1.2**: Cache integration (ready for Phase 2)
- 🔄 **5.4.3**: Complete Vega service example
- 🔄 **6.1**: Store Vega specs as artifacts (chapkit integration)

## Integration with Chap-Core

This service can be used as a **visualization extension** for chap-core:

1. **ML Service generates predictions** (DataFrame)
2. **Sends data to Vega service** (POST /$generate)
3. **Receives Vega-Lite spec** (ready for frontend rendering)
4. **Optionally stores spec as artifact** (for reuse)

## Performance Considerations

Current implementation:
- ✅ Lightweight data transformation
- ✅ Minimal dependencies (pandas + pydantic)
- ⚠️ No caching (every request regenerates spec)
- ⚠️ In-memory processing (limited by available RAM)

**Future optimizations** (from roadmap):
- Add caching layer (Phase 2.2) - 10x faster for repeated requests
- Add batch processing (Phase 2.3) - handle multiple visualizations
- Add storage backend (Phase 3.1) - persist generated specs

## Troubleshooting

### Port Already in Use

```bash
# Change port in compose.yml or:
uv run uvicorn main:app --port 8080
```

### Invalid Field Names

Ensure field names in `x_field`, `y_field`, `color_field` exist in your data columns.

### Unsupported Chart Type

Check the supported chart types table above. Add custom types by extending `_build_spec()`.

## Next Steps

- **Add caching** - Implement Phase 2.2 caching for repeated requests
- **Store artifacts** - Save Vega specs to chapkit artifacts
- **Add filters** - Pre-process data with filtering operations
- **Custom themes** - Add Vega theme customization
- **Export formats** - Generate PNG/SVG from specs

## Related Examples

- `core_api/` - Basic CRUD service patterns
- `monitoring/` - Prometheus metrics and monitoring
- `../chapkit/examples/ml_functional/` - ML service with artifacts

## References

- [Vega-Lite Documentation](https://vega.github.io/vega-lite/)
- [Servicekit Enhancement Roadmap](../../designs/enhancement-roadmap.md) (Phase 5.1)
- [DataFrame Schema](../../src/servicekit/data/dataframe.py)
