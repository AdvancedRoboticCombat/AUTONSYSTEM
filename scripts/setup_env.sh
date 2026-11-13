#!/usr/bin/env bash
# Build the single Python environment (./.venv) used by every component, and
# install the dashboard's node modules.
#
#   scripts/setup_env.sh            create or update .venv
#   scripts/setup_env.sh --fresh    delete .venv first
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON=/usr/bin/python3.12   # ROS 2 Jazzy is built for the system Python 3.12

command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/" >&2; exit 1; }
[[ -x $PYTHON ]] || { echo "$PYTHON not found" >&2; exit 1; }

if [[ ${1:-} == --fresh ]]; then
    rm -rf .venv
fi

if [[ ! -x .venv/bin/python ]]; then
    # System site packages give the venv rclpy, the GStreamer OpenCV and gi.
    uv venv --python "$PYTHON" --system-site-packages .venv
    touch .venv/COLCON_IGNORE
fi

uv pip install --python .venv/bin/python \
    -r requirements.txt --override requirements-overrides.txt

# ~/.local may hold a numpy 2 that breaks the system cv2; never look there.
export PYTHONNOUSERSITE=1

.venv/bin/python - <<'EOF'
import re, sys
import cv2, numpy, torch

problems = []
if not re.search(r"GStreamer:\s+YES", cv2.getBuildInformation()):
    problems.append(f"cv2 at {cv2.__file__} has no GStreamer support "
                    "(sudo apt install python3-opencv, and make sure no "
                    "opencv-python wheel is installed in .venv)")
if int(numpy.__version__.split(".")[0]) >= 2:
    problems.append(f"numpy {numpy.__version__} is too new for the system cv2")

print(f"python {sys.version.split()[0]} | numpy {numpy.__version__} | "
      f"cv2 {cv2.__version__} (GStreamer) | torch {torch.__version__} | "
      f"CUDA {'yes' if torch.cuda.is_available() else 'NO (CPU only)'}")
for problem in problems:
    print("PROBLEM:", problem)
sys.exit(1 if problems else 0)
EOF

if [[ ! -d nemesis_dashboard/node_modules ]]; then
    command -v npm >/dev/null || { echo "npm not found; skipping the dashboard frontend" >&2; exit 0; }
    (cd nemesis_dashboard && npm install)
fi

echo "Environment ready: $ROOT/.venv"
