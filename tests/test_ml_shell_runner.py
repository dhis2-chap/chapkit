"""Tests for ShellModelRunner implementation."""

import sys
import tempfile
from pathlib import Path

import pytest

from chapkit import BaseConfig
from chapkit.data import DataFrame
from chapkit.ml import ShellModelRunner


class MockConfig(BaseConfig):
    """Mock config for testing."""

    threshold: float = 0.5
    features: list[str] = ["feature1", "feature2"]


@pytest.mark.asyncio
async def test_shell_runner_train_basic() -> None:
    """Test basic training with shell runner using echo command."""
    # Create a pickled model file using python
    train_command = f"{sys.executable} -c \"import pickle; pickle.dump('trained_model', open('{{model_file}}', 'wb'))\""

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command="echo 'predictions' > {output_file}",
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1", "target"], data=[[1, 0], [2, 1], [3, 0]])

    # Train should execute command and load model
    model = await runner.on_train(config, data)

    # Model should be a string "trained_model" from the pickled data
    assert model == "trained_model"


@pytest.mark.asyncio
async def test_shell_runner_predict_basic() -> None:
    """Test basic prediction with shell runner."""
    # Create a simple script that writes CSV output
    predict_command = 'echo "feature1,prediction\\n1,0.5\\n2,0.6" > {output_file}'

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command="echo 'model' > {model_file}",
        predict_command=predict_command,
    )

    config = MockConfig()
    model = "mock_model"
    historic = DataFrame(columns=["feature1"], data=[])
    future = DataFrame(columns=["feature1"], data=[[1], [2]])

    # Predict should execute command and load results
    predictions = await runner.on_predict(config, model, historic, future)

    assert len(predictions.data) == 2
    assert "prediction" in predictions.columns
    pred_idx = predictions.columns.index("prediction")
    assert float(predictions.data[0][pred_idx]) == 0.5


@pytest.mark.asyncio
async def test_shell_runner_train_with_real_script() -> None:
    """Test training with actual Python script."""
    # Create a temp script that trains a simple model
    script = """
import sys
import pickle
import csv
import yaml

# Read config
with open(sys.argv[1]) as f:
    config = yaml.safe_load(f)

# Read data from CSV
with open(sys.argv[2]) as f:
    reader = csv.DictReader(f)
    feature1_values = [float(row['feature1']) for row in reader]

# Create simple model (just store mean of feature1)
mean_value = sum(feature1_values) / len(feature1_values)
model = {"mean": mean_value, "config": config}

# Save model
with open(sys.argv[3], "wb") as f:
    pickle.dump(model, f)

print("Training completed")
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        train_command = f"{sys.executable} {script_path} {{config_file}} {{data_file}} {{model_file}}"

        runner: ShellModelRunner[MockConfig] = ShellModelRunner(
            train_command=train_command,
            predict_command="echo 'predictions' > {output_file}",
        )

        config = MockConfig()
        data = DataFrame(columns=["feature1", "target"], data=[[10, 1], [20, 2], [30, 3]])

        model = await runner.on_train(config, data)

        # Check model contains expected data
        assert isinstance(model, dict)
        assert model["mean"] == 20.0
        assert model["config"]["threshold"] == 0.5

    finally:
        Path(script_path).unlink()


@pytest.mark.asyncio
async def test_shell_runner_predict_with_real_script() -> None:
    """Test prediction with actual Python script."""
    # Create a temp script that makes predictions
    script = """
import sys
import pickle
import csv

# Load model
with open(sys.argv[1], "rb") as f:
    model = pickle.load(f)

# Read future data
with open(sys.argv[2]) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Make predictions (just add model value to feature1)
with open(sys.argv[3], "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["feature1", "prediction"])
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "feature1": row["feature1"],
            "prediction": float(row["feature1"]) + model
        })

print("Prediction completed")
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        predict_command = f"{sys.executable} {script_path} {{model_file}} {{future_file}} {{output_file}}"

        runner: ShellModelRunner[MockConfig] = ShellModelRunner(
            train_command="echo 'model' > {model_file}",
            predict_command=predict_command,
        )

        config = MockConfig()
        model = 100  # Model is just a number
        historic = DataFrame(columns=["feature1"], data=[])
        future = DataFrame(columns=["feature1"], data=[[1], [2], [3]])

        predictions = await runner.on_predict(config, model, historic, future)

        assert len(predictions.data) == 3
        assert "prediction" in predictions.columns
        pred_idx = predictions.columns.index("prediction")
        assert float(predictions.data[0][pred_idx]) == 101  # 1 + 100

    finally:
        Path(script_path).unlink()


@pytest.mark.asyncio
async def test_shell_runner_train_failure() -> None:
    """Test handling of training script failure."""
    # Command that will fail
    train_command = "exit 1"

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command="echo 'predictions' > {output_file}",
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1"], data=[[1], [2], [3]])

    with pytest.raises(RuntimeError, match="Training script failed with exit code 1"):
        await runner.on_train(config, data)


@pytest.mark.asyncio
async def test_shell_runner_predict_failure() -> None:
    """Test handling of prediction script failure."""
    # Command that will fail
    predict_command = "exit 2"

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command="echo 'model' > {model_file}",
        predict_command=predict_command,
    )

    config = MockConfig()
    model = "mock_model"
    historic = DataFrame(columns=["feature1"], data=[])
    future = DataFrame(columns=["feature1"], data=[[1], [2]])

    with pytest.raises(RuntimeError, match="Prediction script failed with exit code 2"):
        await runner.on_predict(config, model, historic, future)


@pytest.mark.asyncio
async def test_shell_runner_missing_model_file() -> None:
    """Test error when training script doesn't create model file."""
    # Command that doesn't create model file
    train_command = "echo 'no model created'"

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command="echo 'predictions' > {output_file}",
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1"], data=[[1], [2], [3]])

    with pytest.raises(RuntimeError, match="Training script did not create model file"):
        await runner.on_train(config, data)


@pytest.mark.asyncio
async def test_shell_runner_missing_output_file() -> None:
    """Test error when prediction script doesn't create output file."""
    # Command that doesn't create output file
    predict_command = "echo 'no predictions created'"

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command="echo 'model' > {model_file}",
        predict_command=predict_command,
    )

    config = MockConfig()
    model = "mock_model"
    historic = DataFrame(columns=["feature1"], data=[])
    future = DataFrame(columns=["feature1"], data=[[1], [2]])

    with pytest.raises(RuntimeError, match="Prediction script did not create output file"):
        await runner.on_predict(config, model, historic, future)


@pytest.mark.asyncio
async def test_shell_runner_variable_substitution() -> None:
    """Test that all variables are properly substituted in commands."""
    # Test that the runner creates the expected files with substituted paths
    # This is implicitly tested by the other tests, but we verify explicitly here

    train_command = f"{sys.executable} -c \"import pickle; pickle.dump('model', open('{{model_file}}', 'wb'))\""

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command='echo "feature1,prediction\\n1,0.5" > {output_file}',
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1"], data=[[1], [2], [3]])

    # Train - this will verify {model_file} substitution works
    model = await runner.on_train(config, data)
    assert model == "model"

    # Predict - this will verify {output_file} substitution works
    historic = DataFrame(columns=["feature1"], data=[])
    future = DataFrame(columns=["feature1"], data=[[1]])
    predictions = await runner.on_predict(config, model, historic, future)
    assert len(predictions.data) == 1
    assert "prediction" in predictions.columns


@pytest.mark.asyncio
async def test_shell_runner_cleanup_temp_files() -> None:
    """Test that temporary files are cleaned up after execution."""

    temp_dirs_before = len(list(Path(tempfile.gettempdir()).glob("chapkit_ml_*")))

    train_command = f"{sys.executable} -c \"import pickle; pickle.dump('model', open('{{model_file}}', 'wb'))\""
    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command="echo 'predictions' > {output_file}",
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1"], data=[[1], [2], [3]])

    await runner.on_train(config, data)

    # Check that temp dirs are cleaned up
    temp_dirs_after = len(list(Path(tempfile.gettempdir()).glob("chapkit_ml_*")))
    assert temp_dirs_after == temp_dirs_before
