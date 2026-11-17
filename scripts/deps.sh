#!/usr/bin/env bash
# Python dependencies: locked, checked, and undoable.
#
#   scripts/deps.sh lock                write requirements.lock: the newest versions of what
#                                       requirements.txt lists that were released at least
#                                       COOLDOWN_DAYS (default 7) days ago, every file hashed.
#                                       torch, torchvision, ultralytics and numpy stay at the
#                                       versions .venv has now (a new torch can need a new
#                                       NVIDIA driver and quietly fall back to the CPU)
#   scripts/deps.sh lock --upgrade-gpu  the same, but those four may move too
#   scripts/deps.sh check               is .venv built from the current requirements.lock?
#   scripts/deps.sh audit               known vulnerabilities in requirements.lock and in the
#                                       dashboard page's npm packages (needs the internet)
#   scripts/deps.sh rollback            put back the environment from before the last
#                                       scripts/setup_env.sh (.venv.prev) and its lock
#
# scripts/setup_env.sh installs exactly requirements.lock. Update before an
# event, not the night before: run scripts/test.sh and a dry run afterwards.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONNOUSERSITE=1

LOCK=requirements.lock
STAMP=.venv/.nemesis-lock-sha256         # hash of the lock .venv was built from
LOCK_COPY=.venv/nemesis-requirements.lock  # that lock itself, for rollback
COOLDOWN_DAYS="${COOLDOWN_DAYS:-7}"
HELD=(torch torchvision ultralytics numpy)
PIP_AUDIT=pip-audit==2.10.1

die() { echo "ERROR: $*" >&2; exit 1; }
sha256() { { sha256sum "$1" 2>/dev/null || shasum -a 256 "$1"; } | cut -d' ' -f1; }
command -v uv >/dev/null || die "uv not found: https://docs.astral.sh/uv/"

# The machine the lock is for: this one on Linux, else the club PC (x86_64 Linux)
default_platform() {
    if [[ $(uname -s) == Linux ]]; then echo "$(uname -m)-unknown-linux-gnu"; else echo x86_64-unknown-linux-gnu; fi
}

cutoff() {
    date -u -d "$COOLDOWN_DAYS days ago" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null \
        || date -u -v-"$COOLDOWN_DAYS"d +%Y-%m-%dT%H:%M:%SZ
}

# name==version of each HELD package .venv has
held_versions() {
    [[ -x .venv/bin/python ]] || return 0
    .venv/bin/python -I - "${HELD[@]}" <<'EOF'
import sys
from importlib.metadata import PackageNotFoundError, version
for name in sys.argv[1:]:
    try:
        print(f"{name}=={version(name)}")
    except PackageNotFoundError:
        pass
EOF
}

# Package lines of a lock, "name==version", sorted
pins() { [[ -f $1 ]] && grep -E '^[A-Za-z0-9._-]+==' "$1" | sed 's/ .*//' | sort || true; }

show_changes() {  # show_changes OLD NEW
    python3 - "$1" "$2" <<'EOF'
import sys
def read(path):
    out = {}
    for line in open(path):
        name, _, version = line.strip().partition("==")
        if version:
            out[name.lower()] = version
    return out
old, new = read(sys.argv[1]), read(sys.argv[2])
lines = [f"  {n}: {old[n]} -> {new[n]}" for n in sorted(old.keys() & new.keys()) if old[n] != new[n]]
lines += [f"  {n}: new, {new[n]}" for n in sorted(new.keys() - old.keys())]
lines += [f"  {n}: removed (was {old[n]})" for n in sorted(old.keys() - new.keys())]
print("\n".join(lines) if lines else "  (no version changes)")
EOF
}

cmd_lock() {
    local upgrade_gpu=0 platform
    platform="$(default_platform)"
    while [[ $# -gt 0 ]]; do
        case $1 in
            --upgrade-gpu) upgrade_gpu=1 ;;
            --platform) [[ $# -ge 2 ]] || die "--platform needs a value"; platform="$2"; shift ;;
            *) die "Unknown option for lock: $1" ;;
        esac
        shift
    done

    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    local args=(requirements.txt --override requirements-overrides.txt
                --generate-hashes --exclude-newer "$(cutoff)"
                --python-version 3.12 --python-platform "$platform"
                --custom-compile-command "scripts/deps.sh lock"
                --output-file "$tmp/lock" --quiet)

    if [[ $upgrade_gpu == 0 ]]; then
        held_versions > "$tmp/held"
        if grep -q '+' "$tmp/held"; then
            die "$(grep '+' "$tmp/held" | tr '\n' ' ')came from outside PyPI (e.g. download.pytorch.org), \
so the lock cannot reproduce it. Decide on purpose: scripts/deps.sh lock --upgrade-gpu, then check CUDA."
        fi
        if [[ -s $tmp/held ]]; then
            echo ">>> Keeping $(tr '\n' ' ' < "$tmp/held")(--upgrade-gpu to move them)"
            args+=(--constraint "$tmp/held")
        else
            echo ">>> No .venv to read versions from: torch and friends get their newest versions too"
        fi
    fi

    echo ">>> Resolving for $platform, Python 3.12, releases older than $COOLDOWN_DAYS days"
    uv pip compile "${args[@]}" || die "Could not resolve requirements.txt (nothing changed)."

    pins "$LOCK" > "$tmp/old"
    pins "$tmp/lock" > "$tmp/new"
    echo ">>> Changes against the current $LOCK:"
    show_changes "$tmp/old" "$tmp/new"
    mv "$tmp/lock" "$LOCK"
    echo ">>> Wrote $LOCK. Install it with scripts/setup_env.sh (the current .venv is kept for rollback)."
}

cmd_check() {
    [[ -f $LOCK ]] || die "No $LOCK. Run scripts/setup_env.sh."
    [[ -x .venv/bin/python ]] || die "No .venv. Run scripts/setup_env.sh."
    if [[ $(sha256 "$LOCK") == "$(cat "$STAMP" 2>/dev/null)" ]]; then
        echo ".venv is built from the current $LOCK."
    elif [[ -f .venv/ROLLED_BACK ]]; then
        echo "WARNING: .venv was rolled back and is not built from the current $LOCK ($(cat .venv/ROLLED_BACK))."
    else
        die ".venv is not built from the current $LOCK. Run scripts/setup_env.sh."
    fi
}

cmd_audit() {
    local status=0
    [[ -f $LOCK ]] || die "No $LOCK. Run scripts/setup_env.sh."
    echo "=== Python packages ($LOCK)"
    uvx --quiet --from "$PIP_AUDIT" pip-audit --requirement "$LOCK" --require-hashes --disable-pip \
        --progress-spinner off || status=1
    echo
    echo "=== npm packages of the dashboard page (critical ones fail this check)"
    echo "    Still listed and accepted (see nemesis_dashboard/README.md): build tools only"
    echo "    (Tailwind's file matching, Vite's dev server) and react-router's <Link>/SSR paths, which"
    echo "    the page does not use."
    (cd nemesis_dashboard && npm audit --omit=dev --audit-level=critical) || status=1
    return $status
}

cmd_rollback() {
    [[ -d .venv.prev ]] || die "No .venv.prev: nothing to roll back to."
    rm -rf .venv.rejected
    [[ -d .venv ]] && mv .venv .venv.rejected
    mv .venv.prev .venv
    echo "rolled back $(date '+%Y-%m-%d %H:%M')" > .venv/ROLLED_BACK
    echo ">>> Restored the previous .venv. The one it replaced is in .venv.rejected."
    if [[ -f $LOCK_COPY ]]; then
        cp "$LOCK_COPY" "$LOCK"
        echo ">>> Restored the $LOCK it was built from (git now shows it as changed)."
    else
        echo ">>> That environment predates requirements.lock; scripts/setup_env.sh leaves it alone"
        echo "    until you run scripts/setup_env.sh --fresh."
    fi
}

case "${1:-}" in
    lock) shift; cmd_lock "$@" ;;
    check) cmd_check ;;
    audit) cmd_audit ;;
    rollback) cmd_rollback ;;
    -h|--help|"") sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
    *) die "Unknown command: $1 (lock, check, audit, rollback)" ;;
esac
