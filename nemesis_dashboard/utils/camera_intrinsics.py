"""Chessboard-based camera intrinsic calibration.

Detects chessboard corners across a directory of images and computes the
camera matrix + distortion coefficients, saving them to a ``.npz`` consumed by
main.py's 3D homography (``_compute_homography_3D_2D``).

Can be used as a library (``compute_intrinsics(...)``) or run standalone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Defaults (overridable via function args / CLI)
DEFAULT_IMAGE_DIR = Path.cwd() / "camera_calibration_images"
# The calibration matrix lives in the vision/ package, regardless of CWD.
DEFAULT_OUTPUT_NPZ = str(Path(__file__).resolve().parent.parent / "vision" / "camera_calibration.npz")
# inner corner counts of black/white square corners
DEFAULT_PATTERN_COLS = 8
DEFAULT_PATTERN_ROWS = 5
# size of each square length in mm
DEFAULT_SQUARE_SIZE = 21.61
DEFAULT_MIN_IMAGES = 10
SUBPIX_WIN = 11
IMAGE_EXTENSIONS = {".jpg", ".png"}


def compute_intrinsics(
    image_dir=DEFAULT_IMAGE_DIR,
    pattern_cols: int = DEFAULT_PATTERN_COLS,
    pattern_rows: int = DEFAULT_PATTERN_ROWS,
    square_size: float = DEFAULT_SQUARE_SIZE,
    output_npz: str = DEFAULT_OUTPUT_NPZ,
    min_images: int = DEFAULT_MIN_IMAGES,
):
    """Compute and save camera intrinsics from chessboard images.

    Args:
        image_dir: directory containing calibration images.
        pattern_cols: number of inner corners along the board width.
        pattern_rows: number of inner corners along the board height.
        square_size: edge length of one square, in mm.
        output_npz: path to write the calibration result to.
        min_images: minimum number of usable images required.

    Returns:
        dict with keys ``camera_matrix``, ``dist_coeffs``, ``rms``,
        ``image_size`` and ``images_used``.
    """
    image_dir = Path(image_dir)

    # Gather coordinates of the chessboard corners in the images
    paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise SystemExit(f"No images in {image_dir}")

    objp = np.zeros((pattern_rows * pattern_cols, 3), dtype=np.float32)
    grid = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
    objp[:, :2] = grid * square_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    win = (SUBPIX_WIN, SUBPIX_WIN)
    pattern_size = (pattern_cols, pattern_rows)
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip unreadable: {path.name}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        wh = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = wh
        elif wh != image_size:
            print(f"skip wrong size {path.name}")
            continue
        ok, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not ok:
            print(f"skip no board: {path.name}")
            continue
        corners = cv2.cornerSubPix(gray, corners, win, (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)

    # Calibrate the camera
    n = len(objpoints)
    if n < min_images:
        raise SystemExit(
            f"Need >= {min_images} images with detected board; got {n}. "
            "Add photos or fix pattern_cols / pattern_rows."
        )
    assert image_size is not None
    w, h = image_size
    rms, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    np.savez_compressed(
        output_npz,
        camera_matrix=K.astype(np.float64),
        dist_coeffs=dist.astype(np.float64),
        image_width=np.int32(w),
        image_height=np.int32(h),
        rms_reprojection_error=np.float64(float(rms)),
    )
    print(f"Saved {output_npz}")
    print(f"  images used: {n}  RMS reprojection error: {rms:.4f} px")
    print(f"  size {w}×{h}  fx={K[0, 0]:.2f} fy={K[1, 1]:.2f}")
    print(K)

    return {
        "camera_matrix": K,
        "dist_coeffs": dist,
        "rms": float(rms),
        "image_size": image_size,
        "images_used": n,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compute camera intrinsics from chessboard images.")
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR), help="Directory of calibration images.")
    parser.add_argument("--cols", type=int, default=DEFAULT_PATTERN_COLS, help="Inner corners along board width.")
    parser.add_argument("--rows", type=int, default=DEFAULT_PATTERN_ROWS, help="Inner corners along board height.")
    parser.add_argument("--square-size", type=float, default=DEFAULT_SQUARE_SIZE, help="Square edge length (mm).")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_NPZ, help="Output .npz path.")
    parser.add_argument("--min-images", type=int, default=DEFAULT_MIN_IMAGES, help="Minimum usable images required.")
    return parser.parse_args()


def main():
    args = parse_args()
    compute_intrinsics(
        image_dir=args.image_dir,
        pattern_cols=args.cols,
        pattern_rows=args.rows,
        square_size=args.square_size,
        output_npz=args.output,
        min_images=args.min_images,
    )


if __name__ == "__main__":
    main()
