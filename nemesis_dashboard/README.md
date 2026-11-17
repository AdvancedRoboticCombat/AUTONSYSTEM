# Nemesis Dashboard

Camera, vision and the web dashboard for Nemesis. One Python process reads the
overhead GigE camera, finds both robots, turns pixels into arena positions,
publishes them to ROS, and serves the dashboard page with the live feed and the
manual-override button.

```
GigE camera ──► main.py ──► ROS 2: /nemesis/centroid, /opponent/centroid, /nemesis/override
                   │
                   └──► http://localhost:8080 ──► the page (built React app + one WebSocket)
```

## Run

From the repository root (the environment comes from `scripts/setup_env.sh`,
which also builds the page):

```sh
scripts/camera.sh               # camera only: settings, preview, YOLO overlay, photos, video, calibration
scripts/camera.sh --check       # no window: find the camera and print FPS
scripts/dashboard.sh            # camera settings window, then the dashboard at http://localhost:8080
scripts/dashboard.sh --skip-setup
scripts/dashboard.sh --fight-video     # no camera: play a recorded fight through the detector
scripts/all.sh                  # the dashboard together with the autonomy nodes
```

There is no Node server at run time: `main.py` serves the built page from
`dist/`. The scripts rebuild it when `src/` is newer, so a `git pull` with page
changes needs nothing extra (Node 18+ must be installed for that rebuild).

`--fight-video` (or the "Run fight video" button on the live feed, which also
restarts it) plays `FIGHT_VIDEO_PATH` at its own frame rate through the
two-class model `FIGHT_MODEL_PATH`, which tells Nemesis from the opponent by
class. Arena positions come from `fight_video_homography.json` when that file
exists. `scripts/tools/record_fight_dashboard.sh OUT.mp4` records the screen
while the fight plays (the dashboard must already be running).

`--record DIR` (alone or with `--skip-setup` / `--fight-video`) also saves every
grabbed frame to `DIR/video.mp4` and, per saved frame, a row `frame, t` in
`DIR/frames.csv` (`t` = `time.time()` at the grab). Frames the encoder cannot keep
up with are dropped and counted in the terminal; `--record-scale 0.5` halves the size.

Other `main.py` options: `--stream-width PX` (default 1280; frames are shrunk to
this before they go to the page, 0 = full size), `--port`, `--no-ros` (positions
to the page only), and `--vision none` (no detector). The page is only reachable
from this computer; see Security below before using `--host`.

### Working on the page

```sh
python main.py --no-ros --vision none --fight-video /path/to/video.mp4   # a backend to talk to
npm run dev                                                              # http://localhost:5173, hot reload
npm test                                                                 # type check
```

The dev server passes `/ws` on to the backend on :8080.

## Layout

```
main.py              Command line -> backend.config.Config -> backend.app.run
backend/config.py    Every setting: camera pipeline, models, homography points, stream, port
backend/app.py       Dashboard: threads, detector, homography; carries out OVERRIDE, swap, fight video, clicks
backend/pipeline.py  Capture, fight-video, vision and stream loops (one thread each)
backend/server.py    The page and its WebSocket (message formats at the top of the file)
backend/ros_bridge.py  ROS node: centroid/override publishers, robot subscribers (only module importing rclpy)
backend/state.py     Newest frame, and the numbers the page shows
backend/tracker.py   Constant-velocity tracker in arena coordinates
backend/recorder.py  --record
camera_setup.py      Camera settings window (Aravis); "Done" starts the dashboard
vision/              Detectors behind one interface (base.py): yolo.py, and "none"
vision/camera_calibration.npz   Camera intrinsics, written by the Calibration tab
utils/               homography.py (pixels to arena), camera_intrinsics.py (checkerboard calibration), geometry.py
YOLO_models/         all_fixes_1280.pt (two classes: robot, nemesis) is the model the dashboard loads (VISION_MODEL_PATH)
src/                 React + TypeScript + Tailwind page; src/lib/link.ts is its one connection
```

## Latency

- Vision wakes the moment a frame lands and publishes to ROS before doing
  anything for the page.
- Video is credit-based: the page gets the next frame only after it has drawn
  the last one, so the picture cannot fall behind, and nothing is encoded while
  no page is open.
- Dashboard numbers go out at most 10 times a second, only when they changed.
- The terminal prints camera FPS, vision FPS and grab-to-result milliseconds
  every 5 s; the page shows the same numbers.

## Things to set in `backend/config.py`

- `HOMOG_DEST`: the arena positions of the points you click in the live feed, in
  order. Until it is filled in, no positions are published. The repository
  README says which frame the policy expects.
- `DEFAULT_CLICKED_POINTS`: the matching pixel positions, to skip the clicking.
- `VISION_MODEL_PATH`: the YOLO checkpoint.

## What it publishes

| Topic | Type | Meaning |
| --- | --- | --- |
| `/nemesis/centroid`, `/opponent/centroid` | `geometry_msgs/PointStamped` | Arena position. `point.z` is 1 when the tracker is extrapolating a robot it did not detect in this frame |
| `/nemesis/override` | `std_msgs/Bool` | The dashboard's OVERRIDE button, at 20 Hz. True = the driver has the robot |
| `/camera/image_raw` | `sensor_msgs/Image` | Raw frames, only while something subscribes |

## Security

The page has no password and carries the OVERRIDE switch, so:

- It listens on `127.0.0.1` only. `--host` with anything else is refused unless
  `--allow-host NAME` names each address the page will be opened under, and
  then anyone who can reach that address can press OVERRIDE.
- Requests must be addressed to an allowed host name (stops DNS rebinding), a
  WebSocket opened by another site's page is refused, and the page cannot be
  framed by another site (Content-Security-Policy, X-Frame-Options).
- Messages from the page are size-capped (64 KB), checked field by field and
  limited to 10 actions a second; at most 8 pages at once.
- Every OVERRIDE change, connection and refused request goes to
  `logs/dashboard.log` with its time.
- Ultralytics runs offline (`YOLO_OFFLINE=1`: no update checks, analytics or
  downloads) and never installs packages (`YOLO_AUTOINSTALL=false`); a missing
  model file is an error, not a download.
- npm packages cannot run install scripts (`.npmrc`).

`scripts/deps.sh audit` lists known vulnerabilities. The npm ones it still
shows are accepted: Tailwind's and ESLint's file matching (`picomatch`, no fixed
version exists; build time only, on our own files), Vite's dev server (fixed only
in Vite 8; the dev server is not used to run the dashboard and listens on this
computer only), and `react-router` advisories for `<Link>`, `useNavigate` and
server rendering, none of which the page uses (fixed only in react-router 7).

## System packages

`python3-opencv` (the build with GStreamer; the PyPI wheel has none),
`gstreamer1.0-plugins-bad`, `gstreamer1.0-aravis`, `aravis-tools`, `python3-tk`,
Node 18+ (to build the page).
