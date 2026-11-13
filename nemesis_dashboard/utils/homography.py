"""Pixel ↔ arena coordinate homography.

Encapsulates the homography state (forward/inverse matrices and, for the 3D
pose variant, the camera intrinsics + pose) so main.py can keep its
calibration config variables but delegate the math here.

The destination point sets and center height are supplied by the caller
(main.py) at construction time.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# The calibration matrix lives in the vision/ package, regardless of CWD.
_DEFAULT_CALIBRATION_PATH = str(Path(__file__).resolve().parent.parent / "vision" / "camera_calibration.npz")


class Homography:
    """Computes and stores the pixel ↔ arena homography.

    Args:
        homog_dest: arena-space destination points for the planar homography.
        homog_dest_3d: arena-space destination points for the 3D-pose variant.
        center_height: robot center height used by the 3D-pose variant.
        calibration_path: path to the saved camera intrinsics ``.npz``.
    """

    def __init__(self, homog_dest, homog_dest_3d, center_height, calibration_path=_DEFAULT_CALIBRATION_PATH):
        self.homog_dest = np.asarray(homog_dest, dtype=np.float32)
        self.homog_dest_3d = np.asarray(homog_dest_3d, dtype=np.float32)
        self.center_height = center_height
        self.calibration_path = calibration_path

        self.matrix = None       # forward: pixel → arena
        self.inv_matrix = None   # inverse: arena → pixel
        self.K = None            # camera matrix
        self.D = None            # distortion coefficients
        self.rvec = None         # rotation vector
        self.tvec = None         # translation vector

    def compute(self, src_points):
        """Compute homography + inverse from pixel-space source points."""
        if self.homog_dest is None or len(self.homog_dest) < 4:
            print("⚠️ Homography destination points not set — skipping compute.")
            return
        src = np.array(src_points, dtype=np.float32)
        H, _ = cv2.findHomography(src, self.homog_dest, cv2.RANSAC, 5.0)
        self.matrix = H
        self.inv_matrix = np.linalg.inv(H) if H is not None else None
        print(f"✅ Homography computed (src points: {len(src_points)})")

    def compute_3d_2d(self, src_points):
        """Compute homography + inverse from pixel-space source points (3D pose)."""
        if self.homog_dest_3d is None or len(self.homog_dest_3d) < 4:
            print("⚠️ Homography 3D destination points not set — skipping compute.")
            return
        data = np.load(self.calibration_path)
        self.K = data["camera_matrix"].astype(np.float32)
        self.D = data["dist_coeffs"].astype(np.float32)
        src = np.array(src_points, dtype=np.float32)
        ok, self.rvec, self.tvec = cv2.solvePnP(self.homog_dest_3d, src, self.K, self.D)
        if not ok:
            raise RuntimeError("solvePnP failed to find camera pose.")
        R, _ = cv2.Rodrigues(self.rvec)
        r1 = R[:, 0:1]
        r2 = R[:, 1:2]
        r3 = R[:, 2:3]
        t = self.tvec.reshape(3, 1)
        t_elev = self.center_height * r3 + t
        self.matrix = self.K @ np.hstack([r1, r2, t_elev])
        self.inv_matrix = np.linalg.inv(self.matrix)
        print(f"Homography computed (src points: {len(src_points)})")

    def pixel_to_arena(self, u, v):
        pts = np.array([[[u, v]]], dtype=np.float32)
        ud = cv2.undistortPoints(pts, self.K, self.D, P=self.K)[0, 0]
        p = np.array([ud[0], ud[1], 1.0], dtype=np.float64)
        q = self.inv_matrix @ p
        return q[0] / q[2], q[1] / q[2]

    def arena_to_pixel(self, x_arena, y_arena):
        target_3d = np.array([[[x_arena, y_arena, self.center_height]]], dtype=np.float32)
        pixels_2d, _ = cv2.projectPoints(target_3d, self.rvec, self.tvec, self.K, self.D)
        u = float(pixels_2d[0, 0, 0])
        v = float(pixels_2d[0, 0, 1])
        return u, v
