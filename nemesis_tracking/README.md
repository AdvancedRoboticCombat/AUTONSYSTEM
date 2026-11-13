# nemesis_tracking

ROS 2 (ament_python) package that fuses the camera centroids and the IMU into
smooth pose and velocity estimates for both robots.

| Node | Run | Purpose |
| --- | --- | --- |
| `kalman_node` | `ros2 run nemesis_tracking kalman_node` | EKF for Nemesis (`[x, y, vx, vy, yaw]`), constant-velocity KF for the opponent |
| `kalman_plotter.py` | `python nemesis_tracking/kalman_plotter.py` | Live matplotlib view of camera vs filtered position and velocity |

Subscribes: `/nemesis/centroid`, `/opponent/centroid` (PointStamped, from the
dashboard backend), `/nemesis/imu` (optional; without it the filter runs
camera-only), `/nemesis/pose` and `/opponent/rpy` (optional 6D-pose yaw
sources, currently unused).

Publishes: `/nemesis/kalman/pos`, `/opponent/kalman/pos` (PoseStamped),
`/nemesis/kalman/twist`, `/opponent/kalman/twist` (TwistStamped).

All tunable numbers live in `KalmanTuning` at the top of
`nemesis_tracking/kalman_node.py`; `tune.md` says how to measure each one and
`../nemesis_control/nemesis_tuning.py` automates the measurements.
