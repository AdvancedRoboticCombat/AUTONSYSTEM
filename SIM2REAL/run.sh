#!/usr/bin/env bash
# One SIM2REAL session, recorded to SIM2REAL/runs/<date_time>_<name>/ (see README.md).
#
#   SIM2REAL/run.sh --dry                  dry run: fake robot + recorder + test pilot on a private
#                                          ROS domain. Nothing leaves the PC: no PPM, radio or camera.
#   SIM2REAL/run.sh --dry --policy         dry run of the real saw_policy_node against a phantom opponent
#   SIM2REAL/run.sh                        THE REAL ROBOT: the existing stack with the policy off, the
#                                          dashboard recording video, the recorder and the test pilot.
#                                          Asks before it starts anything. UNTESTED ON HARDWARE.
#   SIM2REAL/run.sh --policy               the real robot flown by the policy against a phantom opponent
#
#   --name N             what the session is (folder name and meta.json). Default: dry, policy or robot
#   --maneuvers LIST     a session (standard, quick) or maneuver names separated by commas
#   --no-imu             no IMU: the pilot takes its heading from --heading and the direction of travel
#   --heading DEG        the way the robot faces when the pilot first gets it (default: settings.yaml)
#   --settings FILE      a YAML whose lines replace those of SIM2REAL/settings.yaml
#   --seconds S          with --policy: how long the policy flies (default 30)
#   --phantom "ARGS"     with --policy: arguments of phantom_opponent.py (default: --at 34 27)
#   --fake "ARGS"        with --dry: more arguments for fake_robot.py, e.g. "--plant v_max=0.7 --fault no_imu:5:2"
#   --skip-setup         real robot: no camera setup window (the camera is already set up)
#
# Ctrl-C stops everything this script started, and nothing else.
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/_common.sh"

DRY=0 POLICY=0 IMU=1 NAME="" MANEUVERS=standard HEADING="" SETTINGS="" SECONDS_TO_FLY=30
PHANTOM_ARGS="--at 34 27" FAKE_ARGS="" SKIP_SETUP=0
PARAMS="$ROOT/nemesis_policy_engine/config/saw_policy.yaml"
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry)        DRY=1 ;;
        --policy)     POLICY=1 ;;
        --no-imu)     IMU=0 ;;
        --skip-setup) SKIP_SETUP=1 ;;
        --name|--maneuvers|--heading|--settings|--seconds|--phantom|--fake)
            [[ $# -ge 2 ]] || die "$1 needs a value"
            case $1 in
                --name) NAME=$2 ;; --maneuvers) MANEUVERS=$2 ;; --heading) HEADING=$2 ;;
                --settings) SETTINGS=$(realpath -- "$2") ;; --seconds) SECONDS_TO_FLY=$2 ;;
                --phantom) PHANTOM_ARGS=$2 ;; --fake) FAKE_ARGS=$2 ;;
            esac
            shift ;;
        -h|--help)    sed -n '2,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)            die "Unknown option: $1 (see --help)" ;;
    esac
    shift
done
[[ $POLICY == 1 && $IMU == 0 ]] && die "The policy cannot fly without the IMU (--policy with --no-imu)."
if [[ -z $NAME ]]; then
    NAME=robot
    if [[ $DRY == 1 ]]; then NAME=dry; fi
    if [[ $POLICY == 1 ]]; then NAME+=_policy; fi
fi
RUN_DIR="$ROOT/SIM2REAL/runs/$(date +%Y%m%d_%H%M%S)_$NAME"

PILOT=(python SIM2REAL/test_pilot.py --maneuvers "$MANEUVERS")
RECORDER=(python SIM2REAL/recorder.py --dir "$RUN_DIR" --name "$NAME")
FAKE=(python SIM2REAL/fake_robot.py)
[[ $POLICY == 0 ]] && FAKE+=(--driver)        # the stand-in driver makes room when the pilot asks for it
[[ -n $SETTINGS ]] && { PILOT+=(--settings "$SETTINGS"); FAKE+=(--settings "$SETTINGS"); }
[[ -n $HEADING ]] && PILOT+=(--heading "$HEADING")
[[ $IMU == 0 ]] && { PILOT+=(--no-imu); FAKE+=(--no-imu); }
[[ $POLICY == 1 ]] && RECORDER+=(--policy-params "$PARAMS" --note "phantom opponent $PHANTOM_ARGS")
# shellcheck disable=SC2206   # the two ARGS strings are meant to be split into words
FAKE+=($FAKE_ARGS); PHANTOM=(python SIM2REAL/phantom_opponent.py $PHANTOM_ARGS)

# start <log name> <command...>: in the background, in a process group of its own (so it and its
# children can be stopped by PID), Ctrl-C not ignored, output to the run folder's logs/.
PIDS=()
start() {
    local log="$RUN_DIR/logs/$1.log"; shift
    setsid env --default-signal=INT,QUIT "$@" > "$log" 2>&1 &
    PIDS+=($!)
    echo ">>> started (pid $!): $*"
}
stop_all() {
    trap - EXIT
    local i
    for (( i = ${#PIDS[@]} - 1; i >= 0; i-- )); do stop_background "${PIDS[i]}"; done
    echo ">>> Stopped everything. Run folder: $RUN_DIR"
}

if [[ $DRY == 1 ]]; then
    # ── Dry run ───────────────────────────────────────────────────────────
    need_build
    export ROS_DOMAIN_ID="${SIM2REAL_DOMAIN_ID:-88}"         # a domain of its own: cannot reach a live robot stack
    export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST           # and nothing leaves this PC
    export DODGE_DEVICE="${DODGE_DEVICE:-cpu}"               # the GPU may be busy training
    mkdir -p "$RUN_DIR/logs"
    trap stop_all EXIT
    trap 'exit 130' INT TERM
    echo ">>> DRY RUN on ROS_DOMAIN_ID=$ROS_DOMAIN_ID (no PPM output, radio, camera or IMU receiver is started)"
    start recorder "${RECORDER[@]}" --source fake
    start fake_robot "${FAKE[@]}"
    if [[ $POLICY == 1 ]]; then
        start phantom_opponent "${PHANTOM[@]}"
        start saw_policy ros2 run nemesis_policy_engine saw_policy_node --ros-args \
            --params-file "$PARAMS" -p "auton_dir:=$ROOT/auton_training"
        echo ">>> Waiting for the policy to start driving (it loads for a few seconds) ..."
        commands() { if [[ -f $RUN_DIR/commands.csv ]]; then wc -l < "$RUN_DIR/commands.csv"; else echo 0; fi; }
        for _ in $(seq 120); do
            [[ $(commands) -gt 1 ]] && break
            sleep 0.5
        done
        [[ $(commands) -gt 1 ]] || die "The policy never drove: see $RUN_DIR/logs/saw_policy.log"
        echo ">>> The policy is driving. Recording $SECONDS_TO_FLY s ..."
        sleep "$SECONDS_TO_FLY"
    else
        "${PILOT[@]}" --give-up 20 2>&1 | tee "$RUN_DIR/logs/test_pilot.log"
    fi
    exit 0
fi

# ── The real robot ────────────────────────────────────────────────────────
# UNTESTED ON HARDWARE: written without a robot, camera, radio or IMU connected.
need_build
LAUNCH=(scripts/autonomy.sh "policy:=$([[ $POLICY == 1 ]] && echo true || echo false)")
[[ $IMU == 0 ]] && LAUNCH+=(imu:=false)
DASHBOARD=(scripts/dashboard.sh --record "$RUN_DIR")
[[ $SKIP_SETUP == 1 ]] && DASHBOARD+=(--skip-setup)

cat <<EOF

THE REAL ROBOT. This script has never been run on hardware. It will start, in this order:

  1. ${RECORDER[*]} --source robot
  2. ${LAUNCH[*]}
       channel_mux, radio reader,$([[ $IMU == 1 ]] && echo " IMU receiver,") PPM OUTPUT TO THE TRANSMITTER$([[ $POLICY == 1 ]] && echo ", saw_policy_node")
  3. NEMESIS_START_MANUAL=1 ${DASHBOARD[*]}
       $([[ $SKIP_SETUP == 1 ]] || echo "camera setup window, then ")camera, detector and the web dashboard at http://localhost:8080,
       video to the run folder. It comes up in OVERRIDE (the driver has the robot).
$(if [[ $POLICY == 1 ]]; then
    echo "  4. ${PHANTOM[*]}"
    echo "       The policy drives at the phantom as soon as OVERRIDE is released, for as long as you leave it."
    echo "       Set the robot down FACING the phantom (initial_heading in saw_policy.yaml), and make sure the"
    echo "       dashboard sees no second robot: it would publish on /opponent/centroid as well."
  else
    echo "  4. ${PILOT[*]}"
    echo "       Maneuvers start by themselves once OVERRIDE is released, the robot stands still and the"
    echo "       next maneuver fits the floor. OVERRIDE takes the robot back at any moment."
  fi)

Before you go on: weapon off and locked, nobody in the arena, the driver at the transmitter,
SIM2REAL/settings.yaml checked against this floor (arena, keep-out boxes, margin_ft, heading).
Run folder: $RUN_DIR
EOF
read -r -p 'Type "drive" to start, anything else to leave: ' answer
[[ $answer == drive ]] || { echo "Nothing was started."; exit 0; }

mkdir -p "$RUN_DIR/logs"
trap stop_all EXIT
trap 'exit 130' INT TERM
export NEMESIS_START_MANUAL=1
start recorder "${RECORDER[@]}" --source robot
start autonomy "${LAUNCH[@]}"
start dashboard "${DASHBOARD[@]}"
if [[ $POLICY == 1 ]]; then
    start phantom_opponent "${PHANTOM[@]}"
    echo ">>> Release OVERRIDE in the dashboard to let the policy drive. Ctrl-C here ends the session."
    tail -f "$RUN_DIR/logs/autonomy.log"
else
    echo ">>> Release OVERRIDE in the dashboard to let the pilot drive. Ctrl-C here ends the session."
    "${PILOT[@]}" 2>&1 | tee "$RUN_DIR/logs/test_pilot.log"
fi
