"""Tests for ShellModelRunner implementation."""

import shutil
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
    """Test handling when training script doesn't create model file."""
    # Command that doesn't create model file
    train_command = "echo 'no model created'"

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command="echo 'predictions' > {output_file}",
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1"], data=[[1], [2], [3]])

    # Should return placeholder dict instead of raising error
    model = await runner.on_train(config, data)
    assert isinstance(model, dict)
    assert model["model_type"] == "no_file"
    assert "stdout" in model
    assert "stderr" in model


@pytest.mark.asyncio
async def test_shell_runner_predict_with_placeholder_model() -> None:
    """Test prediction with placeholder model (no model file created during training)."""
    # Create a predict script that doesn't need a model file
    predict_script = """
import sys
import csv

# Read future data (note: no model file to load)
with open(sys.argv[1]) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Make predictions without model (simple rule-based)
with open(sys.argv[2], "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["feature1", "prediction"])
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "feature1": row["feature1"],
            "prediction": float(row["feature1"]) * 2  # Simple rule
        })

print("Prediction completed without model file")
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(predict_script)
        script_path = f.name

    try:
        # Note: predict command doesn't use {model_file}
        predict_command = f"{sys.executable} {script_path} {{future_file}} {{output_file}}"

        runner: ShellModelRunner[MockConfig] = ShellModelRunner(
            train_command="echo 'no model file'",
            predict_command=predict_command,
        )

        config = MockConfig()

        # Simulate placeholder model from training without model file
        placeholder_model = {"model_type": "no_file", "stdout": "training log"}

        historic = DataFrame(columns=["feature1"], data=[])
        future = DataFrame(columns=["feature1"], data=[[5], [10]])

        predictions = await runner.on_predict(config, placeholder_model, historic, future)

        # Should successfully predict without model file
        assert len(predictions.data) == 2
        assert "prediction" in predictions.columns
        pred_idx = predictions.columns.index("prediction")
        assert float(predictions.data[0][pred_idx]) == 10  # 5 * 2
        assert float(predictions.data[1][pred_idx]) == 20  # 10 * 2

    finally:
        Path(script_path).unlink()


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


@pytest.mark.asyncio
async def test_shell_runner_copies_project_directory(tmp_path: Path) -> None:
    """Test that current working directory is copied to temp directory."""
    # Create a test project structure
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Create a main script
    train_script = project_dir / "train.py"
    train_script.write_text("""
import pickle
import sys

# Write a simple model
with open('model.pickle', 'wb') as f:
    pickle.dump('test_model', f)
""")

    # Create a utility file that the script might import
    utils_file = project_dir / "utils.py"
    utils_file.write_text("def helper(): return 'helper_result'")

    # Change to project directory
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(project_dir)

        train_command = f"{sys.executable} train.py"
        runner: ShellModelRunner[MockConfig] = ShellModelRunner(
            train_command=train_command,
            predict_command="echo 'done' > {output_file}",
        )

        config = MockConfig()
        data = DataFrame(columns=["feature1"], data=[[1], [2]])

        # Train should copy project directory to temp
        model = await runner.on_train(config, data)

        # Model should be successfully created
        assert model == "test_model"

    finally:
        os.chdir(original_cwd)


@pytest.mark.asyncio
async def test_shell_runner_excludes_venv_and_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that .venv, .git, and other excluded directories are not copied."""
    # Create a test project with excluded directories
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Create excluded directories
    (project_dir / ".venv").mkdir()
    (project_dir / ".git").mkdir()
    (project_dir / "__pycache__").mkdir()
    (project_dir / "node_modules").mkdir()

    # Add files to excluded directories
    (project_dir / ".venv" / "lib").mkdir()
    (project_dir / ".venv" / "lib" / "package.py").write_text("# venv package")
    (project_dir / ".git" / "config").write_text("# git config")

    # Create a train script that checks if excluded dirs exist
    train_script = project_dir / "check_excludes.py"
    train_script.write_text("""
import pickle
import os
from pathlib import Path

# Check that excluded directories don't exist in current directory
assert not Path('.venv').exists(), '.venv should not be copied'
assert not Path('.git').exists(), '.git should not be copied'
assert not Path('__pycache__').exists(), '__pycache__ should not be copied'
assert not Path('node_modules').exists(), 'node_modules should not be copied'

# Write model to confirm script ran
with open('model.pickle', 'wb') as f:
    pickle.dump('exclusion_test_passed', f)
""")

    # Change to project directory
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(project_dir)

        train_command = f"{sys.executable} check_excludes.py"
        runner: ShellModelRunner[MockConfig] = ShellModelRunner(
            train_command=train_command,
            predict_command="echo 'done' > {output_file}",
        )

        config = MockConfig()
        data = DataFrame(columns=["feature1"], data=[[1], [2]])

        # Train should copy project but exclude certain directories
        model = await runner.on_train(config, data)

        # Model should confirm exclusions worked
        assert model == "exclusion_test_passed"

    finally:
        os.chdir(original_cwd)


@pytest.mark.asyncio
async def test_shell_runner_relative_paths_work(tmp_path: Path) -> None:
    """Test that scripts can use relative imports after directory copying."""
    # Create a test project with module structure
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Create a helper module
    helpers_file = project_dir / "helpers.py"
    helpers_file.write_text("""
def get_multiplier():
    return 2.5
""")

    # Create a train script that imports the helper
    train_script = project_dir / "train_with_import.py"
    train_script.write_text("""
import pickle
import sys

# Import from local module - this will only work if cwd contains the module
from helpers import get_multiplier

multiplier = get_multiplier()

# Write model with the multiplier value
with open('model.pickle', 'wb') as f:
    pickle.dump({'multiplier': multiplier}, f)
""")

    # Change to project directory
    import os

    original_cwd = os.getcwd()
    try:
        os.chdir(project_dir)

        train_command = f"{sys.executable} train_with_import.py"
        runner: ShellModelRunner[MockConfig] = ShellModelRunner(
            train_command=train_command,
            predict_command="echo 'done' > {output_file}",
        )

        config = MockConfig()
        data = DataFrame(columns=["feature1"], data=[[1], [2]])

        # Train should successfully import local module
        model = await runner.on_train(config, data)

        # Model should contain the value from the imported module
        assert isinstance(model, dict)
        assert model["multiplier"] == 2.5

    finally:
        os.chdir(original_cwd)


@pytest.mark.asyncio
async def test_shell_runner_cleanup_policy_always() -> None:
    """Test that cleanup_policy='always' always deletes temp directory."""
    train_command = f"{sys.executable} -c \"import pickle; pickle.dump('model', open('model.pickle', 'wb'))\""

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command="echo 'done' > {output_file}",
        cleanup_policy="always",
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1"], data=[[1], [2]])

    # Track temp dirs before
    temp_dirs_before = set(Path(tempfile.gettempdir()).glob("chapkit_ml_train_*"))

    # Run training (success case)
    await runner.on_train(config, data)

    # Temp dir should be deleted
    temp_dirs_after = set(Path(tempfile.gettempdir()).glob("chapkit_ml_train_*"))
    assert temp_dirs_after == temp_dirs_before


@pytest.mark.asyncio
async def test_shell_runner_cleanup_policy_never() -> None:
    """Test that cleanup_policy='never' keeps temp directory."""
    train_command = f"{sys.executable} -c \"import pickle; pickle.dump('model', open('model.pickle', 'wb'))\""

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command="echo 'done' > {output_file}",
        cleanup_policy="never",
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1"], data=[[1], [2]])

    # Run training
    await runner.on_train(config, data)

    # Find the temp directory that was created
    temp_dirs = list(Path(tempfile.gettempdir()).glob("chapkit_ml_train_*"))
    assert len(temp_dirs) > 0, "Temp directory should be preserved"

    # Cleanup manually
    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_shell_runner_cleanup_policy_on_success_with_success() -> None:
    """Test that cleanup_policy='on_success' deletes temp directory when operation succeeds."""
    train_command = f"{sys.executable} -c \"import pickle; pickle.dump('model', open('model.pickle', 'wb'))\""

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command="echo 'done' > {output_file}",
        cleanup_policy="on_success",
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1"], data=[[1], [2]])

    # Track temp dirs before
    temp_dirs_before = set(Path(tempfile.gettempdir()).glob("chapkit_ml_train_*"))

    # Run training (success case)
    await runner.on_train(config, data)

    # Temp dir should be deleted
    temp_dirs_after = set(Path(tempfile.gettempdir()).glob("chapkit_ml_train_*"))
    assert temp_dirs_after == temp_dirs_before


@pytest.mark.asyncio
async def test_shell_runner_cleanup_policy_on_success_with_failure() -> None:
    """Test that cleanup_policy='on_success' keeps temp directory when operation fails."""
    train_command = "exit 1"

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command="echo 'done' > {output_file}",
        cleanup_policy="on_success",
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1"], data=[[1], [2]])

    # Try to run training (will fail)
    with pytest.raises(RuntimeError, match="Training script failed"):
        await runner.on_train(config, data)

    # Find the temp directory that was preserved
    temp_dirs = list(Path(tempfile.gettempdir()).glob("chapkit_ml_train_*"))
    assert len(temp_dirs) > 0, "Temp directory should be preserved on failure"

    # Cleanup manually
    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir, ignore_errors=True)
