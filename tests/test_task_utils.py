"""Tests for task utility functions."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from chapkit import run_shell


@pytest.mark.asyncio
async def test_run_shell_basic_command():
    """Test basic shell command execution."""
    result = await run_shell("echo 'hello world'")

    assert result["command"] == "echo 'hello world'"
    assert result["stdout"].strip() == "hello world"
    assert result["stderr"] == ""
    assert result["returncode"] == 0


@pytest.mark.asyncio
async def test_run_shell_command_with_stderr():
    """Test shell command that writes to stderr."""
    # Use a command that writes to stderr but succeeds
    result = await run_shell("echo 'error message' >&2")

    assert result["returncode"] == 0
    assert result["stderr"].strip() == "error message"
    assert result["stdout"] == ""


@pytest.mark.asyncio
async def test_run_shell_non_zero_exit():
    """Test shell command with non-zero exit code."""
    result = await run_shell("exit 42")

    assert result["returncode"] == 42
    assert result["command"] == "exit 42"


@pytest.mark.asyncio
async def test_run_shell_with_timeout():
    """Test shell command with timeout."""
    # Command that sleeps for 5 seconds should timeout with 1 second limit
    result = await run_shell("sleep 5", timeout=1.0)

    assert result["returncode"] == -1
    assert "timed out" in result["stderr"].lower()


@pytest.mark.asyncio
async def test_run_shell_with_cwd(tmp_path: Path):
    """Test shell command with custom working directory."""
    # Create a test file in tmp directory
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    # List files in that directory
    result = await run_shell("ls", cwd=str(tmp_path))

    assert result["returncode"] == 0
    assert "test.txt" in result["stdout"]


@pytest.mark.asyncio
async def test_run_shell_with_env():
    """Test shell command with custom environment variables."""
    result = await run_shell("echo $MY_VAR", env={"MY_VAR": "test_value"})

    assert result["returncode"] == 0
    assert result["stdout"].strip() == "test_value"


@pytest.mark.asyncio
async def test_run_shell_complex_output():
    """Test shell command with multi-line output."""
    result = await run_shell("printf 'line1\\nline2\\nline3'")

    assert result["returncode"] == 0
    assert "line1" in result["stdout"]
    assert "line2" in result["stdout"]
    assert "line3" in result["stdout"]


@pytest.mark.asyncio
async def test_run_shell_empty_output():
    """Test shell command with no output."""
    result = await run_shell("true")

    assert result["returncode"] == 0
    assert result["stdout"] == ""
    assert result["stderr"] == ""


@pytest.mark.asyncio
async def test_run_shell_concurrent_execution():
    """Test multiple concurrent shell commands."""
    # Run multiple commands concurrently
    commands = [
        run_shell("echo 'cmd1'"),
        run_shell("echo 'cmd2'"),
        run_shell("echo 'cmd3'"),
    ]

    results = await asyncio.gather(*commands)

    assert len(results) == 3
    assert all(r["returncode"] == 0 for r in results)
    assert results[0]["stdout"].strip() == "cmd1"
    assert results[1]["stdout"].strip() == "cmd2"
    assert results[2]["stdout"].strip() == "cmd3"
