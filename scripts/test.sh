#!/usr/bin/env bash
# Checks that need no robot, camera or radio.
#
#   scripts/test.sh              all of the below
#   scripts/test.sh env          the Python environment (imports, GStreamer, CUDA)
#   scripts/test.sh frontend     dashboard TypeScript type check
#   scripts/test.sh pipeline     fake camera -> Kalman -> policy -> /control/channels,
#                                manual override, and vision-loss stop
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
set +e
steps=("$@"); [[ ${#steps[@]} -eq 0 ]] && steps=(env frontend pipeline)
failed=()

run() {   # run <name> <command...>
    local name="$1"; shift
    echo; echo "=== $name"
    if "$@"; then echo "--- ok: $name"; else echo "--- FAILED: $name"; failed+=("$name"); fi
}

check_env() {
    python - <<'PY'
import importlib, re, sys
bad = []
for name in ["rclpy", "cv2", "gi", "tkinter", "torch", "ultralytics", "aiohttp",
             "PIL.ImageTk", "scipy", "pygame", "serial", "pySerialTransfer", "sounddevice",
             "matplotlib", "numpy"]:
    try:
        importlib.import_module(name)
    except Exception as error:
        bad.append(f"{name}: {error}")
import cv2, numpy, torch
if not re.search(r"GStreamer:\s+YES", cv2.getBuildInformation()):
    bad.append("cv2 has no GStreamer support")
print(f"numpy {numpy.__version__} | cv2 {cv2.__version__} | torch {torch.__version__} | "
      f"CUDA {'yes' if torch.cuda.is_available() else 'no'}")
for line in bad:
    print("PROBLEM:", line)
sys.exit(1 if bad else 0)
PY
}

for step in "${steps[@]}"; do
    case "$step" in
        env)
            run "environment" check_env ;;
        frontend)
            run "frontend type check" bash -c "cd nemesis_dashboard && npx tsc --noEmit -p tsconfig.app.json" ;;
        pipeline)
            if [[ -f install/setup.bash ]]; then
                mkdir -p logs
                run "autonomy pipeline" env NEMESIS_LOG_DIR="$ROOT/logs" python scripts/tools/pipeline_smoke_test.py
            else
                echo; echo "=== autonomy pipeline"; echo "--- FAILED: not built (run scripts/build.sh)"; failed+=("autonomy pipeline")
            fi ;;
        *)
            die "Unknown step: $step (env, frontend, pipeline)" ;;
    esac
done

echo
if [[ ${#failed[@]} -eq 0 ]]; then
    echo "ALL CHECKS PASSED"
else
    echo "FAILED: ${failed[*]}"
    exit 1
fi
