"""Tests for ShellModelRunner implementation."""

import shutil
import tempfile
from pathlib import Path
from typing import TypedDict

import pytest

from chapkit import BaseConfig
from chapkit.data import DataFrame
from chapkit.ml import ShellModelRunner


class MockConfig(BaseConfig):
    """Mock config for testing."""

    threshold: float = 0.5
    features: list[str] = ["feature1", "feature2"]


class WorkspaceArtifact(TypedDict):
    """Type for workspace artifact returned by ShellModelRunner.on_train()."""

    workspace_dir: str
    exit_code: int
    stdout: str
    stderr: str


def create_mock_workspace() -> WorkspaceArtifact:
    """Create a mock workspace artifact for testing on_predict()."""
    temp_dir = Path(tempfile.mkdtemp(prefix="chapkit_test_workspace_"))
    return {
        "workspace_dir": str(temp_dir),
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
    }


@pytest.mark.asyncio
async def test_shell_runner_train_basic() -> None:
    """Test basic training with shell runner using echo command."""
    # Create a pickled model file using python
    train_command = "python -c \"import pickle; pickle.dump('trained_model', open('model.pickle', 'wb'))\""

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command="echo 'predictions' > {output_file}",
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1", "target"], data=[[1, 0], [2, 1], [3, 0]])

    # Train should execute command and return workspace info
    result = await runner.on_train(config, data)

    # Result should be workspace dict (v0.10.0+)
    assert isinstance(result, dict)
    assert "workspace_dir" in result
    assert "exit_code" in result
    assert result["exit_code"] == 0
    assert Path(result["workspace_dir"]).exists()

    # Verify model file exists in workspace
    workspace_dir = Path(result["workspace_dir"])
    model_file = workspace_dir / "model.pickle"
    assert model_file.exists()

    # Verify model content is correct
    with open(model_file, "rb") as f:
        import pickle

        model = pickle.load(f)
        assert model == "trained_model"


@pytest.mark.asyncio
async def test_shell_runner_predict_basic() -> None:
    """Test basic prediction with shell runner."""
    # Create a simple script that writes CSV output
    predict_command = 'echo "feature1,prediction\\n1,0.5\\n2,0.6" > {output_file}'

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command="echo 'model' > model.pickle",
        predict_command=predict_command,
    )

    config = MockConfig()
    model = create_mock_workspace()  # Use workspace artifact
    historic = DataFrame(columns=["feature1"], data=[])
    future = DataFrame(columns=["feature1"], data=[[1], [2]])

    try:
        # Predict should execute command and load results
        predictions = await runner.on_predict(config, model, historic, future)

        assert len(predictions.data) == 2
        assert "prediction" in predictions.columns
        pred_idx = predictions.columns.index("prediction")
        assert float(predictions.data[0][pred_idx]) == 0.5
    finally:
        # Cleanup mock workspace
        shutil.rmtree(model["workspace_dir"], ignore_errors=True)


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
with open("config.yml") as f:
    config = yaml.safe_load(f)

# Read data from CSV
with open(sys.argv[1]) as f:
    reader = csv.DictReader(f)
    feature1_values = [float(row['feature1']) for row in reader]

# Create simple model (just store mean of feature1)
mean_value = sum(feature1_values) / len(feature1_values)
model = {"mean": mean_value, "config": config}

# Save model
with open("model.pickle", "wb") as f:
    pickle.dump(model, f)

print("Training completed")
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        train_command = f"python {script_path} {{data_file}}"

        runner: ShellModelRunner[MockConfig] = ShellModelRunner(
            train_command=train_command,
            predict_command="echo 'predictions' > {output_file}",
        )

        config = MockConfig()
        data = DataFrame(columns=["feature1", "target"], data=[[10, 1], [20, 2], [30, 3]])

        result = await runner.on_train(config, data)

        # Check result is workspace dict (v0.10.0+)
        assert isinstance(result, dict)
        assert "workspace_dir" in result
        assert result["exit_code"] == 0

        # Load model from workspace
        workspace_dir = Path(result["workspace_dir"])
        model_file = workspace_dir / "model.pickle"
        assert model_file.exists()

        with open(model_file, "rb") as f:
            import pickle

            model = pickle.load(f)

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
with open("model.pickle", "rb") as f:
    model = pickle.load(f)

# Read future data
with open(sys.argv[1]) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Make predictions (just add model value to feature1)
with open(sys.argv[2], "w", newline='') as f:
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
        predict_command = f"python {script_path} {{future_file}} {{output_file}}"

        runner: ShellModelRunner[MockConfig] = ShellModelRunner(
            train_command="echo 'model' > model.pickle",
            predict_command=predict_command,
        )

        config = MockConfig()
        model = create_mock_workspace()  # Create workspace artifact

        # Create model file in workspace (script expects to load it)
        import pickle

        workspace_dir = Path(model["workspace_dir"])
        model_file = workspace_dir / "model.pickle"
        with open(model_file, "wb") as f:
            pickle.dump(100, f)  # Model is just a number

        historic = DataFrame(columns=["feature1"], data=[])
        future = DataFrame(columns=["feature1"], data=[[1], [2], [3]])

        predictions = await runner.on_predict(config, model, historic, future)

        assert len(predictions.data) == 3
        assert "prediction" in predictions.columns
        pred_idx = predictions.columns.index("prediction")
        assert float(predictions.data[0][pred_idx]) == 101  # 1 + 100

        # Cleanup workspace
        shutil.rmtree(workspace_dir, ignore_errors=True)

    finally:
        Path(script_path).unlink()


@pytest.mark.asyncio
async def test_shell_runner_train_failure() -> None:
    """Test handling of training script failure (v0.10.0+: preserves workspace on failure)."""
    # Command that will fail
    train_command = "exit 1"

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command="echo 'predictions' > {output_file}",
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1"], data=[[1], [2], [3]])

    # v0.10.0+: Failed training returns workspace dict (doesn't raise)
    result = await runner.on_train(config, data)

    assert isinstance(result, dict)
    assert "workspace_dir" in result
    assert "exit_code" in result
    assert result["exit_code"] == 1  # Non-zero exit code
    assert Path(result["workspace_dir"]).exists()  # Workspace preserved for debugging


@pytest.mark.asyncio
async def test_shell_runner_predict_failure() -> None:
    """Test handling of prediction script failure."""
    # Command that will fail
    predict_command = "exit 2"

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command="echo 'model' > model.pickle",
        predict_command=predict_command,
    )

    config = MockConfig()
    model = create_mock_workspace()  # Use workspace artifact
    historic = DataFrame(columns=["feature1"], data=[])
    future = DataFrame(columns=["feature1"], data=[[1], [2]])

    try:
        with pytest.raises(RuntimeError, match="Prediction script failed with exit code 2"):
            await runner.on_predict(config, model, historic, future)
    finally:
        # Cleanup mock workspace
        shutil.rmtree(model["workspace_dir"], ignore_errors=True)


@pytest.mark.asyncio
async def test_shell_runner_missing_model_file() -> None:
    """Test handling when training script doesn't create model file (v0.10.0+: still creates workspace)."""
    # Command that doesn't create model file
    train_command = "echo 'no model created'"

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command="echo 'predictions' > {output_file}",
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1"], data=[[1], [2], [3]])

    # v0.10.0+: Returns workspace dict even without model file
    result = await runner.on_train(config, data)
    assert isinstance(result, dict)
    assert "workspace_dir" in result
    assert "exit_code" in result
    assert result["exit_code"] == 0
    assert "stdout" in result
    assert "stderr" in result

    # Workspace exists but no model file
    workspace_dir = Path(result["workspace_dir"])
    assert workspace_dir.exists()
    model_file = workspace_dir / "model.pickle"
    assert not model_file.exists()  # No model file created


@pytest.mark.asyncio
async def test_shell_runner_predict_with_placeholder_model() -> None:
    """Test prediction without model file (workspace contains no model.pickle)."""
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
        predict_command = f"python {script_path} {{future_file}} {{output_file}}"

        runner: ShellModelRunner[MockConfig] = ShellModelRunner(
            train_command="echo 'no model file'",
            predict_command=predict_command,
        )

        config = MockConfig()

        # Create workspace artifact without model file (simulates training that didn't create one)
        model = create_mock_workspace()

        historic = DataFrame(columns=["feature1"], data=[])
        future = DataFrame(columns=["feature1"], data=[[5], [10]])

        predictions = await runner.on_predict(config, model, historic, future)

        # Should successfully predict without model file
        assert len(predictions.data) == 2
        assert "prediction" in predictions.columns
        pred_idx = predictions.columns.index("prediction")
        assert float(predictions.data[0][pred_idx]) == 10  # 5 * 2
        assert float(predictions.data[1][pred_idx]) == 20  # 10 * 2

        # Cleanup workspace
        shutil.rmtree(model["workspace_dir"], ignore_errors=True)

    finally:
        Path(script_path).unlink()


@pytest.mark.asyncio
async def test_shell_runner_missing_output_file() -> None:
    """Test error when prediction script doesn't create output file."""
    # Command that doesn't create output file
    predict_command = "echo 'no predictions created'"

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command="echo 'model' > model.pickle",
        predict_command=predict_command,
    )

    config = MockConfig()
    model = create_mock_workspace()  # Use workspace artifact
    historic = DataFrame(columns=["feature1"], data=[])
    future = DataFrame(columns=["feature1"], data=[[1], [2]])

    try:
        with pytest.raises(RuntimeError, match="Prediction script did not create output file"):
            await runner.on_predict(config, model, historic, future)
    finally:
        # Cleanup mock workspace
        shutil.rmtree(model["workspace_dir"], ignore_errors=True)


@pytest.mark.asyncio
async def test_shell_runner_variable_substitution() -> None:
    """Test that all variables are properly substituted in commands."""
    # Test that the runner creates the expected files with substituted paths
    # This is implicitly tested by the other tests, but we verify explicitly here

    train_command = "python -c \"import pickle; pickle.dump('model', open('model.pickle', 'wb'))\""

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command='echo "feature1,prediction\\n1,0.5" > {output_file}',
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1"], data=[[1], [2], [3]])

    # Train - this will verify {model_file} substitution works
    result = await runner.on_train(config, data)
    assert isinstance(result, dict)
    assert "workspace_dir" in result
    assert result["exit_code"] == 0

    # Verify model was created
    workspace_dir = Path(result["workspace_dir"])
    model_file = workspace_dir / "model.pickle"
    assert model_file.exists()

    # Predict - this will verify {output_file} substitution works
    # Use workspace from training
    historic = DataFrame(columns=["feature1"], data=[])
    future = DataFrame(columns=["feature1"], data=[[1]])
    predictions = await runner.on_predict(config, result, historic, future)
    assert len(predictions.data) == 1
    assert "prediction" in predictions.columns

    # Cleanup workspace
    shutil.rmtree(workspace_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_shell_runner_cleanup_temp_files() -> None:
    """Test workspace preservation (v0.10.0+: workspace NOT cleaned up by runner)."""

    temp_dirs_before = len(list(Path(tempfile.gettempdir()).glob("chapkit_ml_*")))

    train_command = "python -c \"import pickle; pickle.dump('model', open('model.pickle', 'wb'))\""
    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command="echo 'predictions' > {output_file}",
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1"], data=[[1], [2], [3]])

    result = await runner.on_train(config, data)

    # v0.10.0+: Workspace is NOT cleaned up by runner (manager will cleanup after storing artifact)
    temp_dirs_after = len(list(Path(tempfile.gettempdir()).glob("chapkit_ml_*")))
    assert temp_dirs_after == temp_dirs_before + 1  # One workspace dir exists

    # Verify workspace exists and contains model
    workspace_dir = Path(result["workspace_dir"])
    assert workspace_dir.exists()
    model_file = workspace_dir / "model.pickle"
    assert model_file.exists()

    # Manual cleanup for test
    shutil.rmtree(workspace_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_copies_entire_project_directory() -> None:
    """Test that entire project directory is copied to temp workspace."""
    # Use Python inline command to verify project files and create model
    train_command = (
        'python -c "'
        "import sys; "
        "from pathlib import Path; "
        "import pickle; "
        "expected = ['pyproject.toml', 'src', 'tests']; "
        "missing = [f for f in expected if not Path(f).exists()]; "
        "sys.exit(1) if missing else None; "
        "pickle.dump('success', open('model.pickle', 'wb'))"
        '"'
    )

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command="echo 'predictions' > {output_file}",
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1"], data=[[1], [2]])

    # Should succeed if project files are copied
    result = await runner.on_train(config, data)
    assert isinstance(result, dict)
    assert "workspace_dir" in result
    assert result["exit_code"] == 0  # Should succeed


@pytest.mark.asyncio
async def test_ignores_venv_directory() -> None:
    """Test that .venv directory is not copied to workspace."""
    train_command = (
        'python -c "'
        "import sys; "
        "from pathlib import Path; "
        "import pickle; "
        "sys.exit(1) if Path('.venv').exists() else None; "
        "pickle.dump('venv_ignored', open('model.pickle', 'wb'))"
        '"'
    )

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command="echo 'predictions' > {output_file}",
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1"], data=[[1], [2]])

    result = await runner.on_train(config, data)
    assert isinstance(result, dict)
    assert "workspace_dir" in result
    assert result["exit_code"] == 0  # Should succeed (.venv not copied)


@pytest.mark.asyncio
async def test_ignores_node_modules() -> None:
    """Test that node_modules directory is not copied to workspace."""
    train_command = (
        'python -c "'
        "import sys; "
        "from pathlib import Path; "
        "import pickle; "
        "sys.exit(1) if Path('node_modules').exists() else None; "
        "pickle.dump('node_modules_ignored', open('model.pickle', 'wb'))"
        '"'
    )

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command="echo 'predictions' > {output_file}",
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1"], data=[[1], [2]])

    result = await runner.on_train(config, data)
    assert isinstance(result, dict)
    assert "workspace_dir" in result
    assert result["exit_code"] == 0  # Should succeed (node_modules not copied)


@pytest.mark.asyncio
async def test_ignores_pycache() -> None:
    """Test that __pycache__ directories are not copied to workspace."""
    train_command = (
        'python -c "'
        "import sys; "
        "from pathlib import Path; "
        "import pickle; "
        "pycache = list(Path('.').rglob('__pycache__')); "
        "sys.exit(1) if pycache else None; "
        "pickle.dump('pycache_ignored', open('model.pickle', 'wb'))"
        '"'
    )

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command="echo 'predictions' > {output_file}",
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1"], data=[[1], [2]])

    result = await runner.on_train(config, data)
    assert isinstance(result, dict)
    assert "workspace_dir" in result
    assert result["exit_code"] == 0  # Should succeed (__pycache__ not copied)


@pytest.mark.asyncio
async def test_project_structure_preserved() -> None:
    """Test that project directory structure is preserved in workspace."""
    train_command = (
        'python -c "'
        "import sys; "
        "from pathlib import Path; "
        "import pickle; "
        "missing = []; "
        "missing.append('src') if not Path('src').exists() else None; "
        "missing.append('src/chapkit') if not Path('src/chapkit').exists() else None; "
        "sys.exit(1) if missing else None; "
        "pickle.dump('structure_preserved', open('model.pickle', 'wb'))"
        '"'
    )

    runner: ShellModelRunner[MockConfig] = ShellModelRunner(
        train_command=train_command,
        predict_command="echo 'predictions' > {output_file}",
    )

    config = MockConfig()
    data = DataFrame(columns=["feature1"], data=[[1], [2]])

    result = await runner.on_train(config, data)
    assert isinstance(result, dict)
    assert "workspace_dir" in result
    assert result["exit_code"] == 0  # Should succeed (structure preserved)


@pytest.mark.asyncio
async def test_uses_relative_paths() -> None:
    """Test that variable substitution uses relative paths and supports relative imports."""
    # Create a helper module in the project root temporarily
    lib_content = "def process_data(value):\n    return value * 2\n"
    lib_file = Path.cwd() / "test_helper_lib.py"
    lib_file.write_text(lib_content)

    try:
        # Command that imports the helper module and creates model
        train_command = (
            'python -c "'
            "import pickle; "
            "from test_helper_lib import process_data; "
            "result = process_data(42); "
            "pickle.dump(result, open('model.pickle', 'wb'))"
            '"'
        )

        runner: ShellModelRunner[MockConfig] = ShellModelRunner(
            train_command=train_command,
            predict_command="echo 'predictions' > {output_file}",
        )

        config = MockConfig()
        data = DataFrame(columns=["feature1"], data=[[1], [2]])

        # Should succeed with relative imports (lib file copied to workspace)
        result = await runner.on_train(config, data)
        assert isinstance(result, dict)
        assert "workspace_dir" in result
        assert result["exit_code"] == 0  # Should succeed

        # Verify model file was created with correct value
        workspace_dir = Path(result["workspace_dir"])
        model_file = workspace_dir / "model.pickle"
        assert model_file.exists()

        with open(model_file, "rb") as f:
            import pickle

            model = pickle.load(f)
            assert model == 84  # 42 * 2

    finally:
        # Cleanup
        if lib_file.exists():
            lib_file.unlink()
