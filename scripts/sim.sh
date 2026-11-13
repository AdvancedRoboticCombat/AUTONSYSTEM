#!/usr/bin/env bash
# Gazebo simulations (need Gazebo Harmonic + ros-jazzy-ros-gz).
#
#   scripts/sim.sh              nn: Nemesis flown by the hammer-saw policy vs a radio-driven opponent
#   scripts/sim.sh nn weapon:=true headless:=true checkpoint:=ppo_models/my_run/SAW_upd100.pt
#   scripts/sim.sh sim          drive Nemesis by radio
#   scripts/sim.sh playground   drive Nemesis against a ragdoll horizontal spinner
#   scripts/sim.sh horiz_test   drive the generic horizontal spinner
#   scripts/sim.sh vert_test    drive the generic vertical spinner
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
need_build
name="${1:-nn}"
[[ -f nemesis_gazebo/launch/$name.launch.py ]] || die "No such sim: $name (see the top of this script)"
command -v gz >/dev/null || die "Gazebo (gz) not found."
exec ros2 launch nemesis_gazebo "$name.launch.py" "${@:2}"
