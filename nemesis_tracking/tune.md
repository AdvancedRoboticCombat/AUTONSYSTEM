Everything in KalmanTuning that depends on hardware/physics — measure, don't guess:

gyro_bias (kalman_node.py:37) — sit the bot still, log imu.angular_velocity.z, take the mean. Sim ≈ 0, real is usually 0.005–0.05 rad/s. Biggest single source of yaw drift if wrong.

imu_gyro_sigma (:38) — std of gz at rest. Real IMUs: ~0.005–0.05 rad/s. Sim is often zero.

cam_pos_sigma (:39) — std of /nemesis/centroid with bot stationary. Drives the trust in camera vs. predicted position.

nem_process_vel_sigma (:40) — how aggressively your real bot actually changes velocity (m/s per √s). If sim bot has punchier accel than reality (or vice versa), retune this. Too low → lag; too high → noisy.

opp_process_vel_sigma (:49) — same for opponent; usually higher since opponent behaves erratically.

nem_pose_yaw_sigma (:42) — trust in the 6D-pose yaw source (ArUco/whatever). Sim is much cleaner than real.

reset_movement_threshold / reset_gyro_threshold (:45-46) — stationary detector thresholds; need to exceed real noise floor or you never detect rest (and zero_reset never fires).

max_dt_sec / min_dt_sec (:33-34) — clamp to your real camera/IMU publish rate. If real IMU is 100 Hz not 200 Hz, max_dt_sec=0.05 may be too tight.
