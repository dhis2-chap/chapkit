# Testing ML Services

This guide covers how to test chapkit ML services during development.

## Using the `chapkit test` Command

The `chapkit test` command runs end-to-end tests against your ML service, verifying the complete workflow from config creation through training and prediction.

**Note:** This command only appears when running `chapkit` from inside a chapkit project directory (a directory containing `main.py` with chapkit imports).

### Basic Usage

First, start your service:

```bash
uv run python main.py
```

Then in another terminal, run the test:

```bash
chapkit test
```

### Auto-Starting the Service

Use `--start-service` to automatically start the service with an in-memory database:

```bash
chapkit test --start-service
```

This is the easiest way to test your service - it handles starting and stopping the service automatically.

### Command Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--url` | `-u` | `http://localhost:8000` | Service URL |
| `--configs` | `-c` | `1` | Number of configs to create |
| `--trainings` | `-t` | `1` | Training jobs per config |
| `--predictions` | `-p` | `1` | Predictions per trained model |
| `--rows` | `-r` | `100` | Target rows in training data (locations x periods) |
| `--timeout` | | `60.0` | Job completion timeout (seconds) |
| `--delay` | `-d` | `1.0` | Delay between job submissions (seconds) |
| `--verbose` | `-v` | `false` | Show detailed output |
| `--start-service` | | `false` | Auto-start service with in-memory DB |
| `--save-data` | | `false` | Save generated test data files |
| `--save-data-dir` | | `target` | Directory for saved test data |
| `--parallel` | | `1` | Number of jobs to run in parallel (experimental) |
| `--debug` | | `false` | Show full stack traces on errors |
| `--period-type` | | `monthly` | Period format: `monthly` (YYYY-mm) or `weekly` (YYYY-Wxx) |

### Examples

Run a quick test with auto-start:

```bash
chapkit test --start-service
```

Run multiple configs, trainings, and predictions:

```bash
chapkit test --start-service -c 2 -t 2 -p 5 -v
```

Test against a remote service:

```bash
chapkit test --url http://my-service:8000
```

Save generated test data for inspection:

```bash
chapkit test --start-service --save-data
ls target/  # Contains JSON and CSV files for training/prediction data
```

The `--save-data` option creates:
- `config_*.json` - Configuration data
- `training_*.json` / `training_*.csv` - Training panel data
- `prediction_*_historic.json` / `.csv` - Historic data for prediction
- `prediction_*_future.json` / `.csv` - Future data for prediction
- `geo.json` - GeoJSON with polygon geometries (if service requires geo)

Run jobs in parallel (experimental):

```bash
chapkit test --start-service -c 2 -t 4 -p 4 --parallel 4
```

Use weekly periods instead of monthly:

```bash
chapkit test --start-service --period-type weekly --save-data
# Generates periods like 2020-W01, 2020-W02, etc.
```

### Generated Data Structure

The test data generator creates panel data for climate-health correlation analysis:

```
time_period, location, disease_cases, feature_0, feature_1, feature_2
2020-01,     location_0, 42.0,        23.1,      45.2,      67.3
2020-01,     location_1, 38.0,        25.3,      41.8,      62.1
2020-01,     location_2, 51.0,        18.7,      52.1,      71.4
2020-02,     location_0, 35.0,        21.4,      48.9,      65.8
...
```

- **time_period**: Monthly (YYYY-mm) or weekly (YYYY-Wxx) format
- **location**: Matches GeoJSON `properties.id` values
- **disease_cases**: Health outcome (positive integer as float)
- **feature_N**: Climate/covariate data

Training data uses periods starting from 2020, prediction future data uses 2025.

## Manual Service Startup

For more control, you can start the service manually with specific configurations.

### Using In-Memory Database

For faster testing without persistent data:

```bash
DATABASE_URL="sqlite+aiosqlite:///:memory:" uv run python main.py
```

### Using a Test Database File

To persist test data for debugging:

```bash
DATABASE_URL="sqlite+aiosqlite:///test_data/test.db" uv run python main.py
```

## Testing with Docker

### Build and Run

```bash
docker build -t my-ml-service .
docker run -p 8000:8000 -e DATABASE_URL="sqlite+aiosqlite:///:memory:" my-ml-service
```

Then test from the host:

```bash
chapkit test --url http://localhost:8000
```

### Docker Compose

```yaml
# compose.test.yml
services:
  service:
    build: .
    environment:
      - DATABASE_URL=sqlite+aiosqlite:///:memory:
    ports:
      - "8000:8000"
```

```bash
docker compose -f compose.test.yml up -d
chapkit test
docker compose -f compose.test.yml down
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Test ML Service

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv sync

      - name: Run ML service tests
        run: uv run chapkit test --start-service -c 2 -t 2 -p 5
```

## Troubleshooting

### Database Lock Errors

If you see SQLite "database is locked" errors when running many predictions:

1. Use `--start-service` which uses an in-memory database
2. Or manually start with in-memory: `DATABASE_URL="sqlite+aiosqlite:///:memory:"`
3. Increase the delay between jobs: `--delay 2`

### Service Not Ready

If the service takes a long time to start:

1. The default wait timeout is 30 seconds
2. Check service logs for startup errors
3. Ensure all dependencies are installed

### Connection Refused

If you get "Cannot connect" errors:

1. Verify the service is running
2. Check the URL matches the service address
3. Ensure no firewall is blocking the port
