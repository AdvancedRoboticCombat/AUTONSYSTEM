#!/usr/bin/env bash
# Camera + vision + web dashboard, without the autonomy nodes. The backend still
# publishes centroids and the override switch to ROS, so this also works next
# to scripts/autonomy.sh in another terminal.
#
#   scripts/dashboard.sh               camera setup window, then the dashboard
#   scripts/dashboard.sh --skip-setup  no setup window: use the camera as it is
#                                      configured now (backend + web UI only)
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
cd nemesis_dashboard
[[ -d node_modules ]] || die "Frontend dependencies missing. Run scripts/setup_env.sh."

if [[ ${1:-} != --skip-setup ]]; then
    exec python camera_setup.py
fi

run_in_background "$ROOT/logs/frontend.log" npm run dev -- --host 127.0.0.1 --port 8080 --strictPort
FRONTEND_PID=$BG_PID
trap 'stop_background "$FRONTEND_PID"' EXIT
trap 'exit 130' INT TERM
echo ">>> Dashboard: http://localhost:8080"
python main.py --vision yolo
