"""Utility functions for task execution."""

from __future__ import annotations

import asyncio
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
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
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
