#!/usr/bin/env bash
# Camera + vision + web dashboard, without the autonomy nodes. The backend still
# publishes centroids and the override switch to ROS, so this also works next
# to scripts/autonomy.sh in another terminal.
#
#   scripts/dashboard.sh               camera setup window, then the dashboard
#   scripts/dashboard.sh --skip-setup  no setup window: use the camera as it is
#                                      configured now (backend + web UI only)
#   scripts/dashboard.sh --fight-video [VIDEO]
#                                      no camera: play a recorded fight through
#                                      the detector (VIDEO: absolute path)
#   scripts/dashboard.sh [--skip-setup | --fight-video] --record DIR
#                                      also save every camera frame to
#                                      DIR/video.mp4 and its time to DIR/frames.csv
#                                      (--record-scale 0.5: at half size)
#   Without --record the camera is still recorded, to
#   nemesis_dashboard/saved_camera_recordings/auto/<date_time>/  (--no-record: don't)
START_DIR=$PWD  # --record DIR is relative to where the script was called from
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
cd nemesis_dashboard
[[ -d node_modules ]] || die "Frontend dependencies missing. Run scripts/setup_env.sh."

SETUP_WINDOW=1
AUTO_RECORD=1
BACKEND_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-setup) SETUP_WINDOW=0 ;;
        --no-record) AUTO_RECORD=0 ;;
        --fight-video)
            SETUP_WINDOW=0
            AUTO_RECORD=0
            BACKEND_ARGS+=(--fight-video)
            if [[ $# -ge 2 && $2 != --* ]]; then BACKEND_ARGS+=("$2"); shift; fi ;;
        --record)
            [[ $# -ge 2 ]] || die "--record needs a folder"
            AUTO_RECORD=0
            BACKEND_ARGS+=(--record "$(cd "$START_DIR" && realpath -m -- "$2")"); shift ;;
        --record-scale)
            [[ $# -ge 2 ]] || die "--record-scale needs a number"
            BACKEND_ARGS+=(--record-scale "$2"); shift ;;
        *) die "Unknown option: $1" ;;
    esac
    shift
done
if [[ $AUTO_RECORD == 1 ]]; then
    REC_DIR="$PWD/saved_camera_recordings/auto/$(date +%Y%m%d_%H%M%S)"
    BACKEND_ARGS+=(--record "$REC_DIR")
    echo ">>> Recording the camera to $REC_DIR"
fi
if [[ $SETUP_WINDOW == 1 ]]; then
    exec python camera_setup.py "${BACKEND_ARGS[@]}"
fi

run_in_background "$ROOT/logs/frontend.log" npm run dev -- --host 127.0.0.1 --port 8080 --strictPort
FRONTEND_PID=$BG_PID
trap 'stop_background "$FRONTEND_PID"' EXIT
trap 'exit 130' INT TERM
echo ">>> Dashboard: http://localhost:8080"
python main.py --vision yolo "${BACKEND_ARGS[@]}"
