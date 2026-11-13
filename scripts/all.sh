#!/usr/bin/env bash
# Launch the whole robot stack at once:
#   - ROS autonomy nodes (Kalman filter, policy engine, channel mux, radio
#     reader, IMU receiver, PPM output)          -> background, logged to logs/
#   - camera setup window, then the dashboard (vision backend + web UI)
#
#   scripts/all.sh                 everything
#   scripts/all.sh --manual        start in manual override; press the dashboard
#                                  OVERRIDE button to hand control to the policy
#   scripts/all.sh --no-policy     everything except the policy (dry run)
#   scripts/all.sh --no-ppm        nothing is sent to the transmitter
#   scripts/all.sh --weapon        also send the policy's blade and arm commands
#                                  (default: drive only)
#   scripts/all.sh --checkpoint ppo_models/my_run/SAW_upd100.pt
#                                  another saw policy (path from auton_training/)
#   scripts/all.sh --no-imu  --no-radio  --plotter
#
# The policy's settings (arena, robot sizes, opponent card) are in
# nemesis_policy_engine/config/saw_policy.yaml.
#
# Ctrl-C (or closing the dashboard) stops everything.
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
need_build

launch_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --manual)        export NEMESIS_START_MANUAL=1 ;;
        --no-policy)     launch_args+=(policy:=false) ;;
        --no-ppm)        launch_args+=(ppm:=false) ;;
        --weapon)        launch_args+=(weapon:=true) ;;
        --checkpoint)    [[ $# -ge 2 ]] || die "--checkpoint needs a path"; launch_args+=("checkpoint:=$2"); shift ;;
        --no-imu)        launch_args+=(imu:=false) ;;
        --no-radio)      launch_args+=(radio:=false) ;;
        --plotter)       launch_args+=(plotter:=true) ;;
        -h|--help)       sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)               die "Unknown option: $1 (see --help)" ;;
    esac
    shift
done

if [[ ${NEMESIS_START_MANUAL:-} == 1 ]]; then
    echo ">>> Starting in MANUAL override. Use the dashboard OVERRIDE button to go autonomous."
else
    echo ">>> Starting in AUTONOMOUS mode: the policy drives as soon as vision sees both robots."
    echo ">>> Use the dashboard OVERRIDE button for manual control (or start with --manual)."
fi

LOG="logs/autonomy_$(date +%Y%m%d_%H%M%S).log"
run_in_background "$LOG" ros2 launch nemesis_gazebo nemesis.launch.py "${launch_args[@]}"
AUTONOMY_PID=$BG_PID
trap 'stop_background "$AUTONOMY_PID"' EXIT
trap 'exit 130' INT TERM
echo ">>> Autonomy nodes starting (log: $LOG)"

cd nemesis_dashboard
python camera_setup.py
