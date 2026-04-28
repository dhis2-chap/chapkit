#!/usr/bin/env bash
# Benchmark idle and peak memory for each chapkit init template.
#
# For each of fn-py, shell-py, shell-r:
#   1. chapkit init in a temp directory
#   2. uv lock (Dockerfile expects uv.lock)
#   3. docker compose up -d --build, wait for /health
#   4. Sample idle memory after a settle period
#      - Approach 1: docker stats --no-stream (RSS via cgroups)
#      - Approach 2: cat /sys/fs/cgroup/memory.current via docker exec
#   5. Run `chapkit test` while polling docker stats; also read cgroup memory.peak
#   6. docker compose down -v
#
# Outputs a markdown table to stdout and writes it to OUTPUT_FILE.
#
# Requirements: docker (compose v2), uv, chapkit, bash 4+, awk, curl.

set -euo pipefail

if [[ -n "${TEMPLATES_OVERRIDE:-}" ]]; then
    IFS=',' read -ra TEMPLATES <<< "$TEMPLATES_OVERRIDE"
else
    TEMPLATES=(fn-py shell-py shell-r)
fi
ROOT_DIR="${ROOT_DIR:-/tmp/chapkit-bench}"
HOST_PORT="${HOST_PORT:-9090}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-0.2}"
IDLE_SETTLE_SEC="${IDLE_SETTLE_SEC:-10}"
HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-600}"
TEST_TIMEOUT_SEC="${TEST_TIMEOUT_SEC:-300}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_FILE="${OUTPUT_FILE:-${REPO_ROOT}/target/memory-bench.md}"

declare -a RESULTS=()
CURRENT_PROJ_DIR=""

cleanup() {
    if [[ -n "$CURRENT_PROJ_DIR" && -d "$CURRENT_PROJ_DIR" ]]; then
        ( cd "$CURRENT_PROJ_DIR" && docker compose down -v >/dev/null 2>&1 ) || true
    fi
}
trap cleanup EXIT INT TERM

# Convert a "docker stats" memory string ("123.4MiB", "1.2GiB", etc.) to MiB.
parse_mem_to_mib() {
    awk -v s="$1" 'BEGIN {
        unit = s; sub(/^[0-9.]+/, "", unit)
        val  = s; sub(/[A-Za-z]+$/, "", val)
        v = val + 0
        if (unit == "GiB")      v *= 1024
        else if (unit == "KiB") v /= 1024
        else if (unit == "B")   v /= 1024 * 1024
        printf "%.1f", v
    }'
}

bytes_to_mib() {
    awk -v b="$1" 'BEGIN { printf "%.1f", b / 1024 / 1024 }'
}

wait_for_health() {
    local url="http://localhost:${HOST_PORT}/health"
    local deadline=$((SECONDS + HEALTH_TIMEOUT_SEC))
    while (( SECONDS < deadline )); do
        if curl -fsS "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

sample_docker_stats_mib() {
    local cid="$1"
    local raw
    raw=$(docker stats --no-stream --format '{{.MemUsage}}' "$cid" 2>/dev/null \
        | awk -F'/' '{ gsub(/ /, "", $1); print $1 }')
    [[ -z "$raw" ]] && { echo ""; return; }
    parse_mem_to_mib "$raw"
}

read_cgroup_file_bytes() {
    local cid="$1" file="$2"
    docker exec "$cid" sh -c "cat /sys/fs/cgroup/${file} 2>/dev/null" 2>/dev/null || true
}

# cgroup v2 lets you reset the high-water mark by writing 0 to memory.peak (kernel >= 6.7).
# On older kernels this is read-only and the call silently fails - we then just live with
# any memory spent during startup being included in the peak.
reset_cgroup_peak() {
    local cid="$1"
    docker exec "$cid" sh -c 'echo 0 > /sys/fs/cgroup/memory.peak' >/dev/null 2>&1 || true
}

poll_max_mib_into() {
    local cid="$1" stop_file="$2" out_file="$3"
    local max=0 mib
    while [[ ! -f "$stop_file" ]]; do
        mib=$(sample_docker_stats_mib "$cid")
        if [[ -n "$mib" ]] && awk -v m="$mib" -v mx="$max" 'BEGIN { exit !(m > mx) }'; then
            max="$mib"
        fi
        sleep "$POLL_INTERVAL_SEC"
    done
    echo "$max" > "$out_file"
}

bench_template() {
    local template="$1"
    local proj_name="bench_${template//-/_}"
    local proj_dir="$ROOT_DIR/$proj_name"

    echo "================================================================" >&2
    echo "Template: $template -> $proj_dir" >&2
    echo "================================================================" >&2

    rm -rf "$proj_dir"
    chapkit init "$proj_name" --path "$ROOT_DIR" --template "$template" >&2
    CURRENT_PROJ_DIR="$proj_dir"
    cd "$proj_dir"

    uv lock >&2
    docker compose up -d --build >&2

    if ! wait_for_health; then
        echo "FAIL: $template never became healthy" >&2
        docker compose logs --tail 100 >&2 || true
        docker compose down -v >&2 || true
        cd "$REPO_ROOT"
        CURRENT_PROJ_DIR=""
        RESULTS+=("| $template | n/a | n/a | n/a | n/a | n/a | n/a | UNHEALTHY |")
        return
    fi

    local cid
    cid=$(docker compose ps -q | head -n1)

    sleep "$IDLE_SETTLE_SEC"

    local idle_stats_mib idle_cgroup_bytes idle_cgroup_mib
    idle_stats_mib=$(sample_docker_stats_mib "$cid")
    idle_cgroup_bytes=$(read_cgroup_file_bytes "$cid" memory.current)
    if [[ -n "$idle_cgroup_bytes" ]]; then
        idle_cgroup_mib=$(bytes_to_mib "$idle_cgroup_bytes")
    else
        idle_cgroup_mib="n/a"
    fi

    reset_cgroup_peak "$cid"

    local stop_file poll_out test_log
    stop_file=$(mktemp -u)
    poll_out=$(mktemp)
    test_log=$(mktemp)

    poll_max_mib_into "$cid" "$stop_file" "$poll_out" &
    local poll_pid=$!

    local test_status="PASS" test_start test_end test_dur
    test_start=$(date +%s)
    if ! chapkit test --url "http://localhost:${HOST_PORT}" --timeout "$TEST_TIMEOUT_SEC" \
            --configs 2 --trainings 5 --predictions 5 --rows 2000 \
            > "$test_log" 2>&1; then
        test_status="FAIL"
    fi
    test_end=$(date +%s)
    test_dur=$((test_end - test_start))

    touch "$stop_file"
    wait "$poll_pid" || true

    local peak_stats_mib peak_cgroup_bytes peak_cgroup_mib
    peak_stats_mib=$(cat "$poll_out")
    peak_cgroup_bytes=$(read_cgroup_file_bytes "$cid" memory.peak)
    if [[ -n "$peak_cgroup_bytes" ]]; then
        peak_cgroup_mib=$(bytes_to_mib "$peak_cgroup_bytes")
    else
        peak_cgroup_mib="n/a"
    fi

    local image_id image_size_mb
    image_id=$(docker compose images -q | head -n1)
    if [[ -n "$image_id" ]]; then
        image_size_mb=$(docker image inspect --format '{{.Size}}' "$image_id" 2>/dev/null \
            | awk '{ printf "%.0f", $1 / 1024 / 1024 }')
        image_size_mb="${image_size_mb} MB"
    else
        image_size_mb="n/a"
    fi

    if [[ "$test_status" == "FAIL" ]]; then
        echo "--- chapkit test log (tail) ---" >&2
        tail -n 40 "$test_log" >&2
        echo "------------------------------" >&2
    fi

    docker compose down -v >&2 || true
    cd "$REPO_ROOT"
    CURRENT_PROJ_DIR=""
    rm -f "$stop_file" "$poll_out" "$test_log"

    RESULTS+=("| $template | $image_size_mb | ${idle_stats_mib} MiB | ${idle_cgroup_mib} MiB | ${peak_stats_mib} MiB | ${peak_cgroup_mib} MiB | ${test_dur}s | $test_status |")
}

write_table() {
    mkdir -p "$(dirname "$OUTPUT_FILE")"
    {
        echo "# chapkit memory benchmark"
        echo
        echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "Host: $(uname -srm)"
        echo "Settle before idle sample: ${IDLE_SETTLE_SEC}s"
        echo "Poll interval during chapkit test: ${POLL_INTERVAL_SEC}s"
        echo
        echo "| Template | Image size | Idle (docker stats) | Idle (cgroup current) | Peak (docker stats poll) | Peak (cgroup memory.peak) | chapkit test | Result |"
        echo "|---|---|---|---|---|---|---|---|"
        printf '%s\n' "${RESULTS[@]}"
        echo
        echo "## How this was measured"
        echo
        echo "For each template the script ran, in sequence:"
        echo
        echo "1. \`chapkit init <name> --template <t>\` into a temp project directory."
        echo "2. \`uv lock\` (the scaffolded Dockerfile expects \`uv.lock\`), then \`docker compose up -d --build\`."
        echo "3. Polled \`GET /health\` until 200, then slept ${IDLE_SETTLE_SEC}s for the service to reach steady state."
        echo "4. Sampled idle memory two ways: a single \`docker stats --no-stream\` reading and \`cat /sys/fs/cgroup/memory.current\` via \`docker exec\`."
        echo "5. Reset the cgroup high-water mark (\`echo 0 > /sys/fs/cgroup/memory.peak\`; silently ignored if the kernel is read-only on that file), then ran \`chapkit test\` against the container while a background loop sampled \`docker stats\` every ${POLL_INTERVAL_SEC}s and kept the max. Afterwards read \`memory.peak\` for the exact high-water."
        echo "6. \`docker compose down -v\` and removed the temp project."
        echo
        echo "## What the numbers mean"
        echo
        echo "- **Image size** — uncompressed image size from \`docker image inspect .Size\`. On-disk footprint, not the over-the-wire pull size."
        echo "- **Idle (docker stats)** vs **Idle (cgroup current)** — same physical quantity (container RSS) read two ways. \`docker stats\` rounds to a coarser unit; the raw cgroup byte count is more precise. They should agree to within a few MiB."
        echo "- **Peak (docker stats poll)** — max of samples taken every ${POLL_INTERVAL_SEC}s while \`chapkit test\` ran. Portable across cgroup versions but **undersamples short spikes**: if the test finishes in a few seconds the poller may miss the actual peak."
        echo "- **Peak (cgroup memory.peak)** — exact high-water mark recorded by the kernel since the reset call. Always >= the polled peak. Requires cgroup v2 (modern Linux, including Docker Desktop's VM); n/a on cgroup v1 hosts."
        echo "- **chapkit test** — wall-clock duration of the test workload: 2 configs x 5 trainings x 5 predictions over 2000 rows of synthetic data. Heavier than the CLI default (1 x 1 x 1 x 250) so the peak has more headroom to manifest. Still synthetic — to benchmark against your actual model code, swap the scaffolded stub train/predict scripts."
        echo
        echo "## Caveats"
        echo
        echo "- The numbers reflect this host (\`$(uname -srm)\`). \`shell-r\` uses the amd64-only \`chapkit-r-inla\` base and runs under Rosetta on Apple Silicon; expect both higher RSS and slower test times than a native amd64 host."
        echo "- The scaffolded \`shell-r\` and \`shell-py\` templates ship with stub train/predict scripts that don't actually fit a model. The \`shell-r\` peak in particular is \"R interpreter + tiny script\", not \"INLA fit\"."
        echo "- Idle is sampled once. Long-running services may grow due to caches and connection pools — re-sample after the workload of interest if you care about steady-state under load, not cold idle."
    } | tee "$OUTPUT_FILE"
}

main() {
    for cmd in chapkit docker uv curl awk; do
        if ! command -v "$cmd" >/dev/null; then
            echo "Required command not in PATH: $cmd" >&2
            exit 1
        fi
    done

    mkdir -p "$ROOT_DIR"

    for t in "${TEMPLATES[@]}"; do
        bench_template "$t"
    done

    write_table
}

main "$@"
