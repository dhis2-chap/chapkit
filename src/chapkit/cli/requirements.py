"""Helpers for the chapkit dependency requirement rendered into scaffolded projects."""

from __future__ import annotations

import re

#: Matches the leading release segment of a PEP 440 version, e.g. "1.2.0" in "1.2.0.dev0".
_RELEASE_PATTERN = re.compile(r"^\s*v?(\d+(?:\.\d+)*)")


def base_version(version: str) -> str | None:
    """Return the release segment of a version, dropping any pre-release or dev suffix."""
    match = _RELEASE_PATTERN.match(version)
    if match is None:
        return None
    return match.group(1)


def chapkit_requirement(version: str) -> str:
    """Build a floor-and-ceiling chapkit requirement, e.g. "chapkit>=1.2.0,<2"."""
    floor = base_version(version)
    if floor is None:
        # The installed version is unreadable (e.g. "unknown" when chapkit is not
        # installed from metadata); emit an unbounded requirement rather than a
        # requirement that cannot be satisfied.
        return "chapkit"
    major = int(floor.split(".")[0])
    return f"chapkit>={floor},<{major + 1}"
