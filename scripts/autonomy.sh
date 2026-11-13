#!/usr/bin/env bash
# ROS autonomy nodes only (no camera, no dashboard): Kalman filter, policy
# engine, channel mux, radio reader, IMU receiver, PPM output.
# Arguments are passed to the launch file, e.g.
#   scripts/autonomy.sh policy:=false ppm:=false plotter:=true
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
need_build
exec ros2 launch nemesis_gazebo nemesis.launch.py "$@"
