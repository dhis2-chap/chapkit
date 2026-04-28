#!/usr/bin/env python3
"""Benchmark idle and peak memory for chapkit init templates, per chapkit version.

Runs inside an active chapkit venv (uses subprocess to drive chapkit, uvx, docker,
and uv). For each (template, chapkit version):
1. chapkit init in a temp directory
   - "published" -> uvx chapkit init  (latest PyPI; pin via --published-pin)
   - "local"     -> chapkit init      (the chapkit on PATH, i.e. the active venv)
2. If local install is a pre-release that PyPI cannot resolve, build a wheel of
   the local chapkit source and vendor it into the scaffolded project (rewrite
   pyproject.toml chapkit dep + add COPY to Dockerfile).
3. uv lock, then docker compose up -d --build, wait for /health.
4. Sample idle memory two ways: docker stats and cgroup memory.current.
5. Reset cgroup memory.peak, run chapkit test while polling docker stats,
   then read memory.peak for the exact high-water.
6. docker compose down -v.

Outputs a markdown table to stdout and writes it to --output-file.

Requires: docker (compose v2), uv, chapkit + pydantic in the active venv.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATES = ["fn-py", "shell-py", "shell-r"]
DEFAULT_VERSIONS = ["published", "local"]


def log(msg: str) -> None:
    """Emit a timestamped progress line to stderr (line-flushed for live tail)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class BenchResult(BaseModel):
    """One row of the benchmark table."""

    template: str
    chapkit_version: str
    image_size_mb: int | None = None
    idle_stats_mib: float | None = None
    idle_cgroup_mib: float | None = None
    peak_stats_mib: float | None = None
    peak_cgroup_mib: float | None = None
    test_duration_sec: int = 0
    status: str = "PASS"

    def row(self) -> str:
        """Render as a markdown table row."""

        def cell(value: float | int | None, suffix: str = "") -> str:
            return "n/a" if value is None else f"{value}{suffix}"

        return (
            f"| {self.template} | {self.chapkit_version} "
            f"| {cell(self.image_size_mb, ' MB')} "
            f"| {cell(self.idle_stats_mib, ' MiB')} "
            f"| {cell(self.idle_cgroup_mib, ' MiB')} "
            f"| {cell(self.peak_stats_mib, ' MiB')} "
            f"| {cell(self.peak_cgroup_mib, ' MiB')} "
            f"| {self.test_duration_sec}s "
            f"| {self.status} |"
        )


class BenchConfig(BaseModel):
    """CLI/runtime configuration for the benchmark."""

    templates: list[str] = Field(default_factory=lambda: list(DEFAULT_TEMPLATES))
    versions: list[str] = Field(default_factory=lambda: list(DEFAULT_VERSIONS))
    published_pin: str = ""
    root_dir: Path = Path("/tmp/chapkit-bench")
    host_port: int = 9090
    poll_interval_sec: float = 0.2
    idle_settle_sec: int = 10
    health_timeout_sec: int = 600
    test_timeout_sec: int = 300
    output_file: Path = REPO_ROOT / "target" / "memory-bench.md"


def parse_mem_to_mib(s: str) -> float | None:
    """Parse a docker stats memory string ('123.4MiB', '1.2GiB', ...) to MiB."""
    m = re.match(r"([\d.]+)\s*(GiB|MiB|KiB|B)?", s.strip())
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2) or "MiB"
    if unit == "GiB":
        return round(val * 1024, 1)
    if unit == "KiB":
        return round(val / 1024, 1)
    if unit == "B":
        return round(val / (1024 * 1024), 1)
    return round(val, 1)


def bytes_to_mib(b: int) -> float:
    """Convert a byte count to MiB rounded to 1 decimal."""
    return round(b / (1024 * 1024), 1)


def chapkit_cmd(version_kind: str, published_pin: str) -> list[str]:
    """Return the argv prefix for invoking chapkit (init or test) for a version_kind."""
    if version_kind == "published":
        if published_pin:
            return ["uvx", "--from", f"chapkit=={published_pin}", "chapkit"]
        return ["uvx", "chapkit"]
    if version_kind == "local":
        return ["chapkit"]
    raise ValueError(f"unknown version_kind: {version_kind}")


def parse_scaffolded_chapkit_version(proj_dir: Path) -> str | None:
    """Return the version string from `"chapkit>=X"` in a freshly scaffolded pyproject.toml."""
    m = re.search(r'"chapkit>=([^"]+)"', (proj_dir / "pyproject.toml").read_text())
    return m.group(1) if m else None


_PRERELEASE_RE = re.compile(r"\.dev|a\d+|b\d+|rc\d+")


def is_prerelease(version: str) -> bool:
    """True when the version string carries a PEP 440 pre-release suffix."""
    return bool(_PRERELEASE_RE.search(version))


def vendor_local_chapkit_wheel(proj_dir: Path, pinned: str) -> str:
    """Build a wheel of REPO_ROOT chapkit and wire it into the scaffolded project.

    Uses `[tool.uv.sources]` to point chapkit at the vendored wheel — uv resolves
    the path relative to pyproject.toml, which works both for `uv lock` here and
    for `uv sync --frozen` inside the docker build (vendor/ is COPYed in too).
    """
    vendor = proj_dir / "vendor"
    vendor.mkdir(exist_ok=True)
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(vendor), "--quiet"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wheels = sorted(vendor.glob("chapkit-*.whl"))
    if not wheels:
        raise RuntimeError("uv build produced no chapkit wheel")
    wheel_name = wheels[0].name

    pp = proj_dir / "pyproject.toml"
    text = pp.read_text()
    # Drop the version constraint - we'll satisfy chapkit via [tool.uv.sources].
    text = text.replace(f'"chapkit>={pinned}"', '"chapkit"')
    text += (
        f"\n[tool.uv.sources]\n"
        f'chapkit = {{ path = "./vendor/{wheel_name}" }}\n'
    )
    pp.write_text(text)

    df = proj_dir / "Dockerfile"
    df.write_text(
        df.read_text().replace(
            "COPY pyproject.toml uv.lock ./\n",
            "COPY pyproject.toml uv.lock ./\nCOPY vendor/ ./vendor/\n",
        )
    )
    return wheel_name


def wait_for_health(url: str, timeout_sec: float) -> bool:
    """Poll url every second until it returns 200 or the deadline passes."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310 - localhost
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(1)
    return False


def docker_stats_mib(container_id: str) -> float | None:
    """Return current container memory in MiB via `docker stats --no-stream`."""
    result = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return parse_mem_to_mib(result.stdout.strip().split("/", 1)[0])


def docker_exec_read_int(container_id: str, path: str) -> int | None:
    """Cat a file inside the container and parse it as an int."""
    result = subprocess.run(
        ["docker", "exec", container_id, "sh", "-c", f"cat {path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def reset_cgroup_peak(container_id: str) -> None:
    """Best-effort reset of cgroup v2 memory.peak; silently ignored on read-only kernels."""
    subprocess.run(
        ["docker", "exec", container_id, "sh", "-c", "echo 0 > /sys/fs/cgroup/memory.peak"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


class PollMaxWorker(threading.Thread):
    """Background thread that samples `docker stats`, tracks the max, and emits periodic status."""

    def __init__(self, container_id: str, interval_sec: float, status_every_sec: float = 10.0) -> None:
        """Capture polling target, cadence, and status emit interval; thread is daemonic."""
        super().__init__(daemon=True)
        self.container_id = container_id
        self.interval_sec = interval_sec
        self.status_every_sec = status_every_sec
        self.max_mib = 0.0
        self._stop = threading.Event()

    def stop(self) -> None:
        """Signal the polling loop to exit."""
        self._stop.set()

    def run(self) -> None:
        """Sample loop until stop is set; emits a status line every status_every_sec."""
        started = time.monotonic()
        last_status = started
        last_mib: float | None = None
        while not self._stop.is_set():
            mib = docker_stats_mib(self.container_id)
            if mib is not None:
                last_mib = mib
                if mib > self.max_mib:
                    self.max_mib = mib
            now = time.monotonic()
            if now - last_status >= self.status_every_sec:
                last_status = now
                elapsed = int(now - started)
                cur = f"{last_mib:.0f} MiB" if last_mib is not None else "?"
                log(f"  ... chapkit test running, {elapsed}s elapsed, current={cur}, peak={self.max_mib:.0f} MiB")
            self._stop.wait(self.interval_sec)


def compose(
    args: Sequence[str],
    cwd: Path,
    *,
    check: bool = True,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run `docker compose <args>` in cwd."""
    kwargs: dict[str, object] = {"cwd": cwd, "check": check, "text": True}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    return subprocess.run(["docker", "compose", *args], **kwargs)  # type: ignore[arg-type]


def get_compose_container_id(cwd: Path) -> str | None:
    """Return the first container ID for the compose project rooted at cwd."""
    result = subprocess.run(
        ["docker", "compose", "ps", "-q"], cwd=cwd, capture_output=True, text=True, check=True
    )
    ids = [line for line in result.stdout.splitlines() if line.strip()]
    return ids[0] if ids else None


def get_compose_image_size_mb(cwd: Path) -> int | None:
    """Return the on-disk size of the first compose-built image, in MB."""
    listing = subprocess.run(
        ["docker", "compose", "images", "-q"], cwd=cwd, capture_output=True, text=True, check=True
    )
    ids = [line for line in listing.stdout.splitlines() if line.strip()]
    if not ids:
        return None
    inspect = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Size}}", ids[0]],
        capture_output=True,
        text=True,
        check=True,
    )
    try:
        return round(int(inspect.stdout.strip()) / (1024 * 1024))
    except ValueError:
        return None


def bench_one(template: str, version_kind: str, cfg: BenchConfig) -> BenchResult:
    """Run the full bench cycle for a single (template, version) pair."""
    proj_name = f"bench_{template.replace('-', '_')}_{version_kind}"
    proj_dir = cfg.root_dir / proj_name

    print(f"\n{'=' * 64}", file=sys.stderr, flush=True)
    log(f"Template: {template}  | chapkit: {version_kind}  | dir: {proj_dir}")
    print("=" * 64, file=sys.stderr, flush=True)

    if proj_dir.exists():
        shutil.rmtree(proj_dir)

    init_argv = chapkit_cmd(version_kind, cfg.published_pin)
    log(f"step 1/6: chapkit init via {' '.join(init_argv)}")
    try:
        subprocess.run(
            [*init_argv, "init", proj_name, "--path", str(cfg.root_dir), "--template", template],
            check=True,
        )
    except subprocess.CalledProcessError:
        return BenchResult(template=template, chapkit_version=f"{version_kind}?", status="INIT_FAIL")

    pinned = parse_scaffolded_chapkit_version(proj_dir)
    if not pinned:
        return BenchResult(template=template, chapkit_version=f"{version_kind}?", status="INIT_PARSE_FAIL")
    log(f"  scaffolded pyproject pins chapkit>={pinned}")

    if version_kind == "local" and is_prerelease(pinned):
        log("  pre-release pin detected; building local chapkit wheel and vendoring it (slower the first time)")
        try:
            wheel = vendor_local_chapkit_wheel(proj_dir, pinned)
            log(f"  vendored local chapkit wheel: {wheel}")
        except (subprocess.CalledProcessError, RuntimeError) as e:
            log(f"  ERROR: failed to vendor wheel: {e}")
            return BenchResult(template=template, chapkit_version=pinned, status="WHEEL_BUILD_FAIL")

    try:
        try:
            log("step 2/6: uv lock")
            subprocess.run(["uv", "lock"], cwd=proj_dir, check=True)
            log("step 3/6: docker compose up -d --build (first build can be minutes; subsequent are cached)")
            compose(["up", "-d", "--build"], cwd=proj_dir)
        except subprocess.CalledProcessError:
            return BenchResult(template=template, chapkit_version=pinned, status="BUILD_FAIL")

        log(f"step 4/6: waiting for /health on port {cfg.host_port} (timeout {cfg.health_timeout_sec}s)")
        if not wait_for_health(
            f"http://localhost:{cfg.host_port}/health", cfg.health_timeout_sec
        ):
            log("FAIL: never became healthy")
            compose(["logs", "--tail", "100"], cwd=proj_dir, check=False)
            return BenchResult(template=template, chapkit_version=pinned, status="UNHEALTHY")

        container_id = get_compose_container_id(proj_dir)
        if container_id is None:
            return BenchResult(template=template, chapkit_version=pinned, status="NO_CONTAINER")

        log(f"  /health green; settling {cfg.idle_settle_sec}s before idle sample")
        time.sleep(cfg.idle_settle_sec)

        idle_stats = docker_stats_mib(container_id)
        idle_cur = docker_exec_read_int(container_id, "/sys/fs/cgroup/memory.current")
        idle_cgroup = bytes_to_mib(idle_cur) if idle_cur is not None else None
        log(f"  idle: stats={idle_stats} MiB / cgroup.current={idle_cgroup} MiB")

        reset_cgroup_peak(container_id)
        log(f"step 5/6: chapkit test (2 configs x 5 trainings x 5 predictions x 2000 rows; expect ~1-2 min)")

        poller = PollMaxWorker(container_id, cfg.poll_interval_sec)
        poller.start()

        test_argv = chapkit_cmd(version_kind, cfg.published_pin)
        test_log_fd, test_log_name = tempfile.mkstemp(prefix="chapkit-test-", suffix=".log")
        test_log = Path(test_log_name)
        os.close(test_log_fd)
        test_start = time.monotonic()
        with test_log.open("ab") as log_f:
            try:
                subprocess.run(
                    [
                        *test_argv,
                        "test",
                        "--url",
                        f"http://localhost:{cfg.host_port}",
                        "--timeout",
                        str(cfg.test_timeout_sec),
                        "--configs",
                        "2",
                        "--trainings",
                        "5",
                        "--predictions",
                        "5",
                        "--rows",
                        "2000",
                    ],
                    cwd=proj_dir,
                    stdout=log_f,
                    stderr=log_f,
                    timeout=cfg.test_timeout_sec + 60,
                    check=True,
                )
                test_status = "PASS"
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                test_status = "FAIL"
        test_duration = int(time.monotonic() - test_start)

        poller.stop()
        poller.join(timeout=5)

        peak_stats_value = round(poller.max_mib, 1)
        peak_stats: float | None = peak_stats_value if peak_stats_value > 0 else None
        peak_cur = docker_exec_read_int(container_id, "/sys/fs/cgroup/memory.peak")
        peak_cgroup = bytes_to_mib(peak_cur) if peak_cur is not None else None

        image_size = get_compose_image_size_mb(proj_dir)

        log(
            f"  test {test_status} in {test_duration}s; "
            f"peak stats={peak_stats} MiB / cgroup={peak_cgroup} MiB; image={image_size} MB"
        )

        if test_status == "FAIL":
            log("--- chapkit test log (tail) ---")
            sys.stderr.write(test_log.read_text()[-4000:])
            log("------------------------------")
        test_log.unlink(missing_ok=True)

        return BenchResult(
            template=template,
            chapkit_version=pinned,
            image_size_mb=image_size,
            idle_stats_mib=idle_stats,
            idle_cgroup_mib=idle_cgroup,
            peak_stats_mib=peak_stats,
            peak_cgroup_mib=peak_cgroup,
            test_duration_sec=test_duration,
            status=test_status,
        )
    finally:
        log("step 6/6: docker compose down -v")
        compose(["down", "-v"], cwd=proj_dir, check=False, quiet=True)


def render_table(results: list[BenchResult], cfg: BenchConfig) -> str:
    """Build the full markdown report (table + how / what / caveats sections)."""
    uname = os.uname()
    host = f"{uname.sysname} {uname.release} {uname.machine}"
    parts = [
        "# chapkit memory benchmark",
        "",
        f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"Host: {host}",
        f"Settle before idle sample: {cfg.idle_settle_sec}s",
        f"Poll interval during chapkit test: {cfg.poll_interval_sec}s",
        "",
        "| Template | chapkit | Image size | Idle (docker stats) | Idle (cgroup current) "
        "| Peak (docker stats poll) | Peak (cgroup memory.peak) | chapkit test | Result |",
        "|---|---|---|---|---|---|---|---|---|",
        *[r.row() for r in results],
        "",
        "## How this was measured",
        "",
        "For each (template, chapkit version) pair the script ran, in sequence:",
        "",
        "1. `chapkit init <name> --template <t>` into a temp project directory.",
        "   - **published** uses `uvx chapkit init` (latest PyPI by default; pin via "
        "`--published-pin`).",
        "   - **local** uses the `chapkit` already on PATH (typically an active venv installed "
        "from source). If that install is a pre-release that PyPI cannot resolve, the script "
        "builds a wheel of the local chapkit source and vendors it into the scaffolded project "
        "- dropping the version constraint and adding a `[tool.uv.sources]` entry that points "
        "chapkit at `./vendor/<wheel>`, plus a `COPY vendor/ ./vendor/` line in the Dockerfile "
        "so `uv sync --frozen` inside the docker build resolves the same path.",
        "2. `uv lock`, then `docker compose up -d --build`.",
        f"3. Polled `GET /health` until 200, then slept {cfg.idle_settle_sec}s for the service "
        "to reach steady state.",
        "4. Sampled idle memory two ways: a single `docker stats --no-stream` reading and "
        "`cat /sys/fs/cgroup/memory.current` via `docker exec`.",
        "5. Reset the cgroup high-water mark "
        "(`echo 0 > /sys/fs/cgroup/memory.peak`; silently ignored if the kernel is read-only "
        "on that file), then ran `chapkit test` against the container while a background "
        f"thread sampled `docker stats` every {cfg.poll_interval_sec}s and kept the max. "
        "Afterwards read `memory.peak` for the exact high-water.",
        "6. `docker compose down -v` and removed the temp project.",
        "",
        "## What the numbers mean",
        "",
        "- **Template** - the `chapkit init` template under test (`fn-py`, `shell-py`, "
        "`shell-r`).",
        "- **chapkit** - the chapkit version actually pinned in the scaffolded "
        "`pyproject.toml`. Compare rows pairwise to see how the published vs local-dev release "
        "differ for the same template.",
        "- **Image size** - uncompressed image size from `docker image inspect .Size`. On-disk "
        "footprint, not the over-the-wire pull size.",
        "- **Idle (docker stats)** vs **Idle (cgroup current)** - same physical quantity "
        "(container RSS) read two ways. `docker stats` rounds to a coarser unit; the raw "
        "cgroup byte count is more precise. They should agree to within a few MiB.",
        f"- **Peak (docker stats poll)** - max of samples taken every {cfg.poll_interval_sec}s "
        "while `chapkit test` ran. Portable across cgroup versions but **undersamples short "
        "spikes**: the polling cadence can miss sub-second peaks.",
        "- **Peak (cgroup memory.peak)** - exact high-water mark recorded by the kernel since "
        "the reset call. Always >= the polled peak. Requires cgroup v2 (modern Linux, including "
        "Docker Desktop's VM); n/a on cgroup v1 hosts.",
        "- **chapkit test** - wall-clock duration of the test workload: 2 configs x 5 trainings "
        "x 5 predictions over 2000 rows of synthetic data. Heavier than the CLI default "
        "(1 x 1 x 1 x 250) so the peak has more headroom to manifest. Still synthetic - to "
        "benchmark against your actual model code, swap the scaffolded stub train/predict "
        "scripts.",
        "",
        "## Caveats",
        "",
        f"- The numbers reflect this host (`{host}`). `shell-r` uses the amd64-only "
        "`chapkit-r-inla` base and runs under Rosetta on Apple Silicon; expect both higher RSS "
        "and slower test times than a native amd64 host.",
        "- The scaffolded `shell-r` and `shell-py` templates ship with stub train/predict "
        'scripts that do not actually fit a model. The `shell-r` peak in particular is "R '
        'interpreter + tiny script", not "INLA fit".',
        "- Idle is sampled once. Long-running services may grow due to caches and connection "
        "pools - re-sample after the workload of interest if you care about steady-state under "
        "load, not cold idle.",
        "- The **local** version is whatever `chapkit` is on PATH when the script runs. If you "
        "switch venvs mid-session, re-run from a clean shell.",
        "",
    ]
    return "\n".join(parts)


def parse_args() -> BenchConfig:
    """Parse CLI arguments into a BenchConfig."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--templates",
        default=",".join(DEFAULT_TEMPLATES),
        help=f"comma-separated subset of {DEFAULT_TEMPLATES}",
    )
    parser.add_argument(
        "--versions",
        default=",".join(DEFAULT_VERSIONS),
        help=f"comma-separated subset of {DEFAULT_VERSIONS}",
    )
    parser.add_argument(
        "--published-pin",
        default="",
        help="pin the published chapkit version (e.g. 0.23.0); empty -> uvx default (latest)",
    )
    parser.add_argument("--root-dir", default="/tmp/chapkit-bench")
    parser.add_argument("--host-port", type=int, default=9090)
    parser.add_argument("--poll-interval-sec", type=float, default=0.2)
    parser.add_argument("--idle-settle-sec", type=int, default=10)
    parser.add_argument("--health-timeout-sec", type=int, default=600)
    parser.add_argument("--test-timeout-sec", type=int, default=300)
    parser.add_argument(
        "--output-file",
        default=str(REPO_ROOT / "target" / "memory-bench.md"),
    )
    args = parser.parse_args()
    return BenchConfig(
        templates=[t.strip() for t in args.templates.split(",") if t.strip()],
        versions=[v.strip() for v in args.versions.split(",") if v.strip()],
        published_pin=args.published_pin,
        root_dir=Path(args.root_dir),
        host_port=args.host_port,
        poll_interval_sec=args.poll_interval_sec,
        idle_settle_sec=args.idle_settle_sec,
        health_timeout_sec=args.health_timeout_sec,
        test_timeout_sec=args.test_timeout_sec,
        output_file=Path(args.output_file),
    )


def check_deps() -> None:
    """Exit with a clear error if any required CLI is missing from PATH."""
    missing = [c for c in ("chapkit", "docker", "uv", "uvx") if shutil.which(c) is None]
    if missing:
        print(f"Required commands not in PATH: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    """Run the full benchmark and emit the markdown table."""
    cfg = parse_args()
    check_deps()
    cfg.root_dir.mkdir(parents=True, exist_ok=True)

    cells = [(t, v) for t in cfg.templates for v in cfg.versions]
    log(
        f"benchmarking {len(cells)} cell(s) "
        f"({len(cfg.templates)} template(s) x {len(cfg.versions)} version(s)); "
        f"each takes ~2-4 min depending on docker cache"
    )

    results: list[BenchResult] = []
    interrupted = False
    try:
        for i, (template, version_kind) in enumerate(cells, start=1):
            log(f"=== cell {i}/{len(cells)}: {template} @ {version_kind} ===")
            results.append(bench_one(template, version_kind, cfg))
    except KeyboardInterrupt:
        interrupted = True
        log("interrupted - writing partial results table")

    text = render_table(results, cfg)
    cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_file.write_text(text)
    sys.stdout.write(text)
    log(f"wrote {cfg.output_file}")
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
