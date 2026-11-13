#!/usr/bin/env bash
# Build every ROS 2 package in the repo into ./build and ./install.
# Runs colcon with the .venv Python, so the Python nodes get the venv (torch)
# on their shebang line. Extra arguments go to colcon, e.g.
#   scripts/build.sh --packages-select nemesis_control
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
python -m colcon build --symlink-install "$@"
echo "Built. The other scripts source install/setup.bash by themselves."
