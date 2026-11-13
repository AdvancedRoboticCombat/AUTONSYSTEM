# Nemesis Dashboard

Camera, vision and the web dashboard for Nemesis. A Python backend reads the
overhead GigE camera, finds both robots, turns pixels into arena positions and
publishes them to ROS; a React page shows the live feed and carries the
manual-override button.

```
GigE camera ──► main.py (Python backend) ──► Socket.IO (:3001) ──► Web GUI (Vite, :8080)
                     │
                     └──► ROS 2: /nemesis/centroid, /opponent/centroid, /nemesis/override
```

## Run

From the repository root (the environment comes from `scripts/setup_env.sh`):

```sh
scripts/camera.sh               # camera only: settings, preview, YOLO overlay, photos, video, calibration
scripts/camera.sh --check       # no window: find the camera and print FPS
scripts/dashboard.sh            # camera settings window, then backend + web UI at http://localhost:8080
scripts/dashboard.sh --skip-setup
scripts/all.sh                  # the dashboard together with the autonomy nodes
```

## Layout

```
main.py              Backend: capture, detection, homography, tracker, Socket.IO server, ROS node
camera_setup.py      Camera settings window (Aravis); "Done" starts the backend and the web UI
vision/              Detectors behind one interface (base.py); yolo.py is the only one
vision/camera_calibration.npz   Camera intrinsics, written by the Calibration tab
utils/               homography.py (pixels to arena), camera_intrinsics.py (checkerboard calibration), geometry.py
YOLO_models/         MediumBest.pt is the model main.py loads (VISION_MODEL_PATH)
src/                 React + TypeScript + Tailwind frontend
```

## Things to set in `main.py`

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

## System packages

`python3-opencv` (the build with GStreamer; the PyPI wheel has none),
`gstreamer1.0-plugins-bad`, `gstreamer1.0-aravis`, `aravis-tools`, `python3-tk`,
Node 18+.
