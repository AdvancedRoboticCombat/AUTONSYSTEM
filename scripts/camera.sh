#!/usr/bin/env bash
# Camera only. No dashboard, no ROS nodes.
#
#   scripts/camera.sh           camera window: settings, live preview, YOLO
#                               overlay, photos, video, calibration
#   scripts/camera.sh --check   no window: find the camera, grab frames for a
#                               few seconds and print resolution and FPS
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

if [[ ${1:-} == --check ]]; then
    exec python scripts/tools/camera_check.py "${@:2}"
fi
cd nemesis_dashboard
exec python camera_setup.py --preview-only "$@"
