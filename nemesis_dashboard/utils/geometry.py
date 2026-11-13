"""Small geometry / math helpers used by the backend pipeline."""

import math


def quat_to_yaw_deg(x, y, z, w) -> float:
    """Convert a quaternion to a yaw angle in degrees."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def adjust_center(cent_x, cent_y, box_y1, box_y2, frame_height):
    """Nudge a detection's center downward to compensate for camera perspective.

    The further up the frame an object is, the more its mask centroid sits
    above the true ground-contact point, so we bias the center toward the
    bottom of the box by a factor that decreases with vertical position.
    """
    y_norm = cent_y / frame_height
    k = max(0.0, min(1.0, -1.61647 * y_norm + 0.5))
    return (cent_x, cent_y + k * ((box_y2 - box_y1) / 2))
