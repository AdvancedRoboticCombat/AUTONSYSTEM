# Sourced by every script in this folder. Sets up the one shared environment:
# repo root, ROS 2, the built workspace (if any) and the unified .venv.
set -eo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export NEMESIS_ROOT="$ROOT"
# ~/.local may hold a numpy 2 that breaks the system (GStreamer) OpenCV.
export PYTHONNOUSERSITE=1
cd "$ROOT"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -x .venv/bin/python ]] || die "No .venv yet. Run scripts/setup_env.sh first."

ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
[[ -f $ROS_SETUP ]] || die "ROS 2 not found at $ROS_SETUP (set ROS_SETUP=/path/to/setup.bash)."
source "$ROS_SETUP"
if [[ -f install/setup.bash ]]; then
    source install/setup.bash
fi
source .venv/bin/activate

# build_frontend_if_stale: builds the dashboard page (nemesis_dashboard/dist,
# served by main.py) when it is missing or older than its sources.
build_frontend_if_stale() {
    local dir="$ROOT/nemesis_dashboard"
    if [[ -f $dir/dist/index.html ]] && [[ -z $(find "$dir/src" "$dir/public" "$dir/index.html" \
            "$dir/package-lock.json" "$dir/vite.config.ts" "$dir/tailwind.config.ts" \
            -newer "$dir/dist/index.html" -print -quit 2>/dev/null) ]]; then
        return 0
    fi
    command -v npm >/dev/null || die "The dashboard page needs building and npm is missing (Node 18+)."
    [[ -d $dir/node_modules ]] || die "Frontend dependencies missing. Run scripts/setup_env.sh."
    echo ">>> Building the dashboard page"
    (cd "$dir" && npm run build --silent >/dev/null) || die "Building the dashboard page failed (cd nemesis_dashboard && npm run build)."
}

need_build() {
    [[ -f install/setup.bash ]] || die "ROS packages are not built. Run scripts/build.sh first."
}

# run_in_background <log file> <command...>
# Starts the command in its own process group, output to the terminal and
# the log file. Sets BG_PID. Stop it with stop_background.
run_in_background() {
    local log="$1"; shift
    mkdir -p "$(dirname "$log")"
    setsid "$@" > >(tee "$log") 2>&1 &
    BG_PID=$!
}

# stop_background <pid>: Ctrl-C the whole group, wait, then force.
stop_background() {
    local pid="$1"
    [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null || return 0
    kill -INT -- "-$pid" 2>/dev/null || true
    for _ in $(seq 50); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 0.2
    done
    kill -KILL -- "-$pid" 2>/dev/null || true
}
