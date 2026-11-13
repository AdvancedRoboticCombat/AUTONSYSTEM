#!/usr/bin/env bash
# Web viewer for the hammer-saw training sim (auton_training): training
# curves, yardstick scores, replays, and "Fight the policy" (you drive the
# opponent with WASD or the arrow keys). Open http://127.0.0.1:8765.
#
#   scripts/viewer.sh
#   scripts/viewer.sh --fight ppo_models/my_run            fight that run's newest checkpoint
#   scripts/viewer.sh --fight ppo_models/my_run/SAW_upd100.pt
#   scripts/viewer.sh --port 9000
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fight) [[ $# -ge 2 ]] || die "--fight needs a checkpoint or run folder"
                 export DODGE_PLAY_CHECKPOINT="$2"; shift ;;
        *)       args+=("$1") ;;
    esac
    shift
done
cd auton_training
echo "Viewer: http://127.0.0.1:8765  (Replays tab -> Fight the policy)"
exec python web/serve.py "${args[@]}"
