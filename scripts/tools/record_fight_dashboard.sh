#!/usr/bin/env bash
# Records the screen while the dashboard plays the fight video.
#
#   scripts/tools/record_fight_dashboard.sh [OUT.mp4] [SECONDS]
#
# The dashboard must already be running (scripts/dashboard.sh --fight-video).
# Opens it full screen in its own Firefox with the live feed enlarged, starts
# recording, restarts the fight from its first frame, records SECONDS (default
# 175; the fight is 170 s) and closes that Firefox again. It takes over the
# screen while it records.
set -euo pipefail

OUT="$(realpath -m "${1:-fight_dashboard.mp4}")"
DURATION="${2:-175}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"

curl -sf -o /dev/null http://localhost:8080 || {
    echo "ERROR: the dashboard is not running. Start scripts/dashboard.sh --fight-video first." >&2
    exit 1
}

# A profile of its own, so this Firefox does not join one that is already open.
# Inside the repository because the Firefox snap cannot read /tmp.
mkdir -p "$ROOT/logs"
PROFILE="$(mktemp -d "$ROOT/logs/firefox-profile.XXXXXX")"
firefox --no-remote --profile "$PROFILE" --kiosk "http://localhost:8080/?expand" >/dev/null 2>&1 &
FIREFOX_PID=$!
FFMPEG_PID=
trap 'kill $FIREFOX_PID $FFMPEG_PID 2>/dev/null || true; wait; rm -rf "$PROFILE"' EXIT
trap 'exit 130' INT TERM

sleep 10  # let the page load

# No video size given: x11grab records the whole screen.
ffmpeg -loglevel error -y -f x11grab -framerate 30 -draw_mouse 0 -i "$DISPLAY" -t "$DURATION" \
    -c:v libx264 -preset veryfast -pix_fmt yuv420p "$OUT" &
FFMPEG_PID=$!
sleep 1

# Press "Run fight video" without a mouse: send what the button sends.
(cd "$ROOT/nemesis_dashboard" && node -e '
    const socket = require("socket.io-client").io("http://localhost:3001", { transports: ["websocket"] });
    socket.on("connect", () => {
        socket.emit("trigger-action", { key: "fight_video_btn_pressed" });
        setTimeout(() => process.exit(), 500);
    });
')

wait "$FFMPEG_PID"
echo "Saved $OUT"
