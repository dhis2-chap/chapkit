"""Test command implementation."""

import subprocess
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from ulid import ULID

from chapkit.cli.test.generator import TestDataGenerator
from chapkit.cli.test.runner import TestRunner
from chapkit.cli.test.utils import (
    find_project_main,
    save_test_data,
    start_service_subprocess,
    wait_for_service_ready,
)


def test_command(
    url: Annotated[
        str,
        typer.Option("--url", "-u", help="Base URL of running chapkit service"),
    ] = "http://localhost:8000",
    num_configs: Annotated[
        int,
        typer.Option("--configs", "-c", help="Number of configs to create"),
    ] = 1,
    num_trainings: Annotated[
        int,
        typer.Option("--trainings", "-t", help="Number of training jobs per config"),
    ] = 1,
    num_predictions: Annotated[
        int,
        typer.Option("--predictions", "-p", help="Number of predictions per trained model"),
    ] = 1,
    num_rows: Annotated[
        int,
        typer.Option("--rows", "-r", help="Number of rows in generated training data"),
    ] = 100,
    timeout: Annotated[
        float,
        typer.Option("--timeout", help="Job completion timeout in seconds"),
    ] = 60.0,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed output including job statuses"),
    ] = False,
    delay: Annotated[
        float,
        typer.Option("--delay", "-d", help="Delay in seconds between job submissions"),
    ] = 1.0,
    start_service: Annotated[
        bool,
        typer.Option("--start-service", help="Auto-start service with in-memory database"),
    ] = False,
    save_data: Annotated[
        bool,
        typer.Option("--save-data", help="Save generated test data files"),
    ] = False,
    save_data_dir: Annotated[
        str,
        typer.Option("--save-data-dir", help="Directory for saved test data"),
    ] = "target",
    parallel: Annotated[
        int,
        typer.Option("--parallel", help="Number of jobs to run in parallel (experimental)"),
    ] = 1,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Show full stack traces on errors"),
    ] = False,
) -> None:
    """Run end-to-end test of the ML service workflow."""
    service_process: subprocess.Popen[bytes] | None = None
    save_data_path: Path | None = None

    # Handle --save-data
    if save_data:
        save_data_path = Path(save_data_dir)
        typer.echo(f"Saving test data to {save_data_path}/")

    # Handle --start-service
    if start_service:
        main_py = find_project_main(Path.cwd())
        if main_py is None:
            typer.echo("[FAILED] Could not find main.py in project", err=True)
            raise typer.Exit(code=1)

        project_root = main_py.parent
        typer.echo(f"Starting service from {project_root}...")
        service_process = start_service_subprocess(project_root)

        typer.echo(f"  Waiting for service to be ready at {url}...")
        ready, msg = wait_for_service_ready(url)
        if not ready:
            typer.echo(f"  [FAILED] {msg}", err=True)
            service_process.terminate()
            raise typer.Exit(code=1)
        typer.echo(f"  [OK] {msg}")
        typer.echo()

    runner = TestRunner(url, timeout=timeout, verbose=verbose, debug=debug)
    generator = TestDataGenerator(seed=42)  # Reproducible data

    # Statistics tracking
    stats: dict[str, Any] = {
        "configs_created": 0,
        "trainings_completed": 0,
        "trainings_failed": 0,
        "predictions_completed": 0,
        "predictions_failed": 0,
        "start_time": time.time(),
    }

    try:
        # 1. Health check
        typer.echo(f"Connecting to service at {url}...")
        success, message = runner.check_service_health()
        if not success:
            typer.echo(f"  [FAILED] {message}", err=True)
            typer.echo("Tip: Use --start-service to auto-start the service, or run: uv run python main.py")
            raise typer.Exit(code=1)
        typer.echo(f"  [OK] {message}")
        typer.echo()

        # 2. Fetch service info
        success, message = runner.fetch_service_info()
        if not success:
            typer.echo(f"  [WARNING] {message}")
        elif verbose:
            typer.echo(
                f"  Service info: required_covariates={runner.required_covariates}, "
                f"requires_geo={runner.requires_geo}, "
                f"allow_free_additional={runner.allow_free_additional_continuous_covariates}"
            )
            typer.echo()

        # 3. Fetch config schema
        success, message, config_schema = runner.fetch_config_schema()
        if not success or config_schema is None:
            typer.echo(f"  [FAILED] {message}", err=True)
            raise typer.Exit(code=1)
        elif verbose:
            typer.echo("  Config schema fetched")

        # Determine extra covariates to add
        extra_covariates = 2 if runner.allow_free_additional_continuous_covariates else 0

        # Generate geo if required
        geo_data = generator.generate_geo_data() if runner.requires_geo else None
        if save_data_path and geo_data:
            save_test_data(save_data_path, "geo.json", geo_data)

        # 4. Create configs
        typer.echo(f"Creating {num_configs} config(s)...")
        config_ids: list[str] = []
        for i in range(num_configs):
            config_name = f"test_config_{ULID()}"
            config_data = generator.generate_config_data_from_schema(config_schema, variation=i)

            if save_data_path:
                save_test_data(save_data_path, f"config_{i}.json", config_data)

            success, message, config_id = runner.create_config(config_name, config_data)
            if success and config_id:
                config_ids.append(config_id)
                stats["configs_created"] += 1
                if verbose:
                    typer.echo(f"  [OK] {message}")
            else:
                typer.echo(f"  [FAILED] {message}", err=True)
                raise typer.Exit(code=1)

        typer.echo(f"  Created {len(config_ids)} config(s)")
        typer.echo()

        # 5. Run trainings
        total_trainings = num_configs * num_trainings
        typer.echo(
            f"Running {total_trainings} training job(s)..." + (f" (parallel={parallel})" if parallel > 1 else "")
        )

        model_artifacts: list[tuple[str, str]] = []  # List of (config_id, artifact_id) tuples
        training_index = 0
        pending_training_jobs: list[tuple[str, str, str]] = []  # (job_id, artifact_id, config_id)

        def process_training_batch() -> None:
            """Wait for pending training jobs and process results."""
            if not pending_training_jobs:
                return
            job_ids = [j[0] for j in pending_training_jobs]
            results = runner.wait_for_jobs(job_ids)
            for (job_id, artifact_id, cfg_id), (success, msg, _) in zip(pending_training_jobs, results):
                if success:
                    stats["trainings_completed"] += 1
                    model_artifacts.append((cfg_id, artifact_id))
                    if verbose:
                        typer.echo(f"  [OK] Training {artifact_id}: {msg}")
                else:
                    stats["trainings_failed"] += 1
                    typer.echo(f"  [FAILED] Training job {job_id}: {msg}", err=True)

        for config_idx, config_id in enumerate(config_ids):
            for _ in range(num_trainings):
                training_data = generator.generate_training_data(
                    num_rows=num_rows,
                    required_covariates=runner.required_covariates,
                    extra_covariates=extra_covariates,
                )

                if save_data_path:
                    save_test_data(save_data_path, f"training_{config_idx}_{training_index}.json", training_data)
                    training_index += 1

                success, msg, job_id, artifact_id = runner.submit_training(config_id, training_data, geo_data)
                if not success:
                    typer.echo(f"  [FAILED] Submit: {msg}", err=True)
                    stats["trainings_failed"] += 1
                    continue

                if verbose:
                    typer.echo(f"  Submitted training job {job_id}")

                pending_training_jobs.append((job_id, artifact_id, config_id))  # type: ignore[arg-type]

                # Process batch when we reach parallel limit
                if len(pending_training_jobs) >= parallel:
                    process_training_batch()
                    pending_training_jobs.clear()
                    if delay > 0:
                        time.sleep(delay)

        # Process remaining jobs
        process_training_batch()

        typer.echo(f"  Completed: {stats['trainings_completed']}, Failed: {stats['trainings_failed']}")
        typer.echo()

        # 6. Run predictions
        if model_artifacts:
            total_predictions = len(model_artifacts) * num_predictions
            typer.echo(
                f"Running {total_predictions} prediction job(s)..."
                + (f" (parallel={parallel})" if parallel > 1 else "")
            )
            prediction_index = 0
            pending_prediction_jobs: list[tuple[str, str]] = []  # (job_id, pred_artifact_id)

            def process_prediction_batch() -> None:
                """Wait for pending prediction jobs and process results."""
                if not pending_prediction_jobs:
                    return
                job_ids = [j[0] for j in pending_prediction_jobs]
                results = runner.wait_for_jobs(job_ids)
                for (job_id, pred_artifact_id), (success, msg, _) in zip(pending_prediction_jobs, results):
                    if success:
                        stats["predictions_completed"] += 1
                        v_success, _, _ = runner.verify_artifact(pred_artifact_id)
                        if v_success and verbose:
                            typer.echo(f"  [OK] Prediction {pred_artifact_id}: verified")
                    else:
                        stats["predictions_failed"] += 1
                        typer.echo(f"  [FAILED] Prediction job {job_id}: {msg}", err=True)

            for artifact_idx, (_, model_artifact_id) in enumerate(model_artifacts):
                for _ in range(num_predictions):
                    historic, future = generator.generate_prediction_data(
                        num_rows=10,
                        required_covariates=runner.required_covariates,
                        extra_covariates=extra_covariates,
                    )

                    if save_data_path:
                        save_test_data(
                            save_data_path, f"prediction_{artifact_idx}_{prediction_index}_historic.json", historic
                        )
                        save_test_data(
                            save_data_path, f"prediction_{artifact_idx}_{prediction_index}_future.json", future
                        )
                        prediction_index += 1

                    success, msg, job_id, pred_artifact_id = runner.submit_prediction(
                        model_artifact_id, historic, future, geo_data
                    )
                    if not success:
                        typer.echo(f"  [FAILED] Submit: {msg}", err=True)
                        stats["predictions_failed"] += 1
                        continue

                    if verbose:
                        typer.echo(f"  Submitted prediction job {job_id}")

                    pending_prediction_jobs.append((job_id, pred_artifact_id))  # type: ignore[arg-type]

                    # Process batch when we reach parallel limit
                    if len(pending_prediction_jobs) >= parallel:
                        process_prediction_batch()
                        pending_prediction_jobs.clear()
                        if delay > 0:
                            time.sleep(delay)

            # Process remaining jobs
            process_prediction_batch()

            typer.echo(f"  Completed: {stats['predictions_completed']}, Failed: {stats['predictions_failed']}")
            typer.echo()

        # 7. Summary
        elapsed = time.time() - stats["start_time"]

        typer.echo("=" * 50)
        typer.echo("TEST SUMMARY")
        typer.echo("=" * 50)
        typer.echo(f"Service URL:           {url}")
        typer.echo(f"Elapsed time:          {elapsed:.2f}s")
        typer.echo(f"Configs created:       {stats['configs_created']}")
        typer.echo(f"Trainings completed:   {stats['trainings_completed']}")
        typer.echo(f"Trainings failed:      {stats['trainings_failed']}")
        typer.echo(f"Predictions completed: {stats['predictions_completed']}")
        typer.echo(f"Predictions failed:    {stats['predictions_failed']}")
        typer.echo()

        # Determine overall status
        total_failures = stats["trainings_failed"] + stats["predictions_failed"]
        if total_failures == 0 and stats["trainings_completed"] > 0:
            typer.echo("Result: ALL TESTS PASSED")
        elif total_failures > 0:
            typer.echo(f"Result: {total_failures} FAILURE(S)", err=True)
        else:
            typer.echo("Result: NO TESTS RUN", err=True)

        # Exit with appropriate code
        if total_failures > 0:
            raise typer.Exit(code=1)

    finally:
        runner.close()

        # Terminate service if we started it
        if service_process is not None:
            typer.echo()
            typer.echo("Stopping service...")
            service_process.terminate()
            try:
                service_process.wait(timeout=5.0)
                typer.echo("  Service stopped")
            except subprocess.TimeoutExpired:
                service_process.kill()
                typer.echo("  Service killed (did not stop gracefully)")
