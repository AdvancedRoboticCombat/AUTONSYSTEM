#!/usr/bin/env bash
# Record every ROS topic of a match to ~/match_data_logs/<name>.
#   MATCH_NAME=semifinal scripts/record.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
MATCH_NAME="${MATCH_NAME:-match_$(date +%Y%m%d_%H%M%S)}"
exec ros2 bag record -a -o "$HOME/match_data_logs/$MATCH_NAME" --max-bag-size 1000000000
