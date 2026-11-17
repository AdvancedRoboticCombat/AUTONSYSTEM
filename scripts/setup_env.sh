#!/usr/bin/env bash
# Build the single Python environment (./.venv) used by every component from
# requirements.lock, and install and build the dashboard page.
#
#   scripts/setup_env.sh            install requirements.lock, if .venv is not built from it yet
#   scripts/setup_env.sh --fresh    rebuild .venv from requirements.lock anyway
#
# A changed lock is installed into a new .venv; the old one is kept as
# .venv.prev and scripts/deps.sh rollback puts it back. The first run writes
# requirements.lock from what .venv has now (scripts/deps.sh lock).
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON=/usr/bin/python3.12   # ROS 2 Jazzy is built for the system Python 3.12
LOCK=requirements.lock
STAMP=.venv/.nemesis-lock-sha256
LOCK_COPY=.venv/nemesis-requirements.lock

command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/" >&2; exit 1; }
[[ -x $PYTHON ]] || { echo "$PYTHON not found" >&2; exit 1; }

FRESH=0
[[ ${1:-} == --fresh ]] && FRESH=1

# ~/.local may hold a numpy 2 that breaks the system cv2; never look there.
export PYTHONNOUSERSITE=1

sha256() { { sha256sum "$1" 2>/dev/null || shasum -a 256 "$1"; } | cut -d' ' -f1; }
cuda() { "$1" -c 'import torch; print("yes" if torch.cuda.is_available() else "no")' 2>/dev/null || echo unknown; }

if [[ ! -f $LOCK ]]; then
    echo ">>> No $LOCK yet: writing it"
    scripts/deps.sh lock
fi

OLD_CUDA=""
if [[ -f .venv/ROLLED_BACK && $FRESH == 0 && $(sha256 "$LOCK") != "$(cat "$STAMP" 2>/dev/null)" ]]; then
    echo ">>> .venv was rolled back ($(cat .venv/ROLLED_BACK)); leaving it as it is."
    echo "    scripts/setup_env.sh --fresh installs $LOCK again."
elif [[ $FRESH == 1 || $(sha256 "$LOCK") != "$(cat "$STAMP" 2>/dev/null)" ]]; then
    if [[ -x .venv/bin/python ]]; then
        OLD_CUDA="$(cuda .venv/bin/python)"
        rm -rf .venv.prev
        mv .venv .venv.prev
        echo ">>> Kept the current environment as .venv.prev (scripts/deps.sh rollback restores it)"
        trap 'echo "Install failed. scripts/deps.sh rollback puts the previous environment back." >&2' ERR
    fi
    # System site packages give the venv rclpy, the GStreamer OpenCV and gi.
    uv venv --quiet --python "$PYTHON" --system-site-packages .venv
    touch .venv/COLCON_IGNORE
    # Exactly the lock: every file checked against its hash, nothing resolved now
    uv pip sync --python .venv/bin/python --require-hashes "$LOCK"
    cp "$LOCK" "$LOCK_COPY"
    sha256 "$LOCK" > "$STAMP"
    if [[ -x .venv.prev/bin/python ]]; then
        echo ">>> Changes against .venv.prev:"
        python3 - <(uv pip freeze --python .venv.prev/bin/python 2>/dev/null) \
                  <(uv pip freeze --python .venv/bin/python 2>/dev/null) <<'EOF'
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
    fi
else
    echo ">>> .venv is already built from $LOCK"
fi

.venv/bin/python - "$OLD_CUDA" <<'EOF'
import re, sys
import cv2, numpy, torch

old_cuda = sys.argv[1]
cuda = torch.cuda.is_available()
problems = []
if not re.search(r"GStreamer:\s+YES", cv2.getBuildInformation()):
    problems.append(f"cv2 at {cv2.__file__} has no GStreamer support "
                    "(sudo apt install python3-opencv, and make sure no "
                    "opencv-python wheel is installed in .venv)")
if int(numpy.__version__.split(".")[0]) >= 2:
    problems.append(f"numpy {numpy.__version__} is too new for the system cv2")
if old_cuda == "yes" and not cuda:
    problems.append(f"CUDA worked in the previous environment and does not with torch {torch.__version__}: "
                    "run scripts/deps.sh rollback")

print(f"python {sys.version.split()[0]} | numpy {numpy.__version__} | "
      f"cv2 {cv2.__version__} (GStreamer) | torch {torch.__version__} | "
      f"CUDA {'yes' if cuda else 'NO (CPU only)'}")
for problem in problems:
    print("PROBLEM:", problem)
sys.exit(1 if problems else 0)
EOF

# The dashboard page: exactly package-lock.json (install scripts are off, see
# nemesis_dashboard/.npmrc), then the production build main.py serves.
if command -v npm >/dev/null; then
    (
        cd nemesis_dashboard
        if [[ ! node_modules/.package-lock.json -nt package-lock.json ]]; then
            npm ci --no-audit --no-fund
        fi
        npm run build --silent
    )
else
    echo "npm not found: the dashboard page is not built (Node 18+ is needed once, to build it)" >&2
fi

echo "Environment ready: $ROOT/.venv"
