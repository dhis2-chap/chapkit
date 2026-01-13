# Chapkit CLI Roadmap

This document outlines planned enhancements to the chapkit CLI.

## Current Features

- `chapkit init` - Scaffold new chapkit ML service projects
- `chapkit artifact list` - List artifacts from database or service
- `chapkit artifact download` - Download artifact content

## Planned Features

### chapkit test (In Progress)

End-to-end testing command for ML service workflows.

**Features:**
- Creates configs with test parameters
- Runs training jobs with generated synthetic data
- Runs predictions on trained models
- Verifies all operations complete without errors
- Supports service info discovery for `required_covariates`, `requires_geo`, and `allow_free_additional_continuous_covariates`

**Usage:**
```bash
chapkit test [OPTIONS]
```

**Options:**
- `--url, -u` - Base URL of running service (default: http://localhost:8000)
- `--configs, -c` - Number of configs to create
- `--trainings, -t` - Number of training jobs per config
- `--predictions, -p` - Number of predictions per trained model
- `--rows, -r` - Number of rows in generated data
- `--timeout` - Job completion timeout in seconds
- `--verbose, -v` - Show detailed output
- `--cleanup/--no-cleanup` - Delete test resources after completion

## Future Ideas

### Auto-start Service
Automatically start the service if not running when executing test command.

### Schema-aware Data Generation
Fetch config schema from `/api/v1/configs/$schema` to generate data that matches expected types and constraints.

### Parallel Job Execution
Run multiple training/prediction jobs concurrently for faster testing.

### Report Export
Export test results to JSON or HTML reports for CI/CD integration.

### chapkit validate
Validate project structure and configuration without running the service.

### chapkit logs
Stream or tail service logs for debugging.
