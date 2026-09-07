"""Utility functions for task execution."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import Any


async def run_shell(
    command: str,
    *,
    timeout: float | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute shell command with asyncio subprocess.

    Args:
        command: Shell command to execute
        timeout: Optional timeout in seconds
        cwd: Optional working directory
        env: Optional environment variables (merged with current env)

    Returns:
        Dict with command, stdout, stderr, and returncode keys

    Example:
        >>> result = await run_shell("echo 'hello'")
        >>> print(result["stdout"])
        hello
        >>> result["returncode"]
        0

        >>> result = await run_shell("ls -la", cwd="/tmp", timeout=5.0)
        >>> if result["returncode"] != 0:
        ...     print(f"Command failed: {result['stderr']}")
    """
    # A new session puts the shell and everything it spawns in one process group, so a
    # timeout or cancellation can take down the whole tree rather than just the shell.
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.CancelledError:
        # The job was canceled: stop the computation instead of letting it run on unattended.
        await _kill_process_group(proc)
        raise
    except asyncio.TimeoutError:
        await _kill_process_group(proc)
        return {
            "command": command,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "returncode": -1,
        }

    return {
        "command": command,
        "stdout": stdout.decode() if stdout else "",
        "stderr": stderr.decode() if stderr else "",
        "returncode": proc.returncode or 0,
    }


async def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """Kill the process group started for a shell command and reap the shell."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        proc.kill()
    await proc.wait()
