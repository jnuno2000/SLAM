"""
Utility functions for SLAM system.

Provides coordinate transformations and geometric calculations.
"""

from typing import Tuple
import numpy as np


def polar_to_cartesian(r: float, theta: float) -> Tuple[float, float]:
    """Convert polar coordinates to Cartesian.

    Args:
        r: Range (distance).
        theta: Angle in radians.

    Returns:
        Tuple of (x, y) Cartesian coordinates.
    """
    return (r * np.cos(theta), r * np.sin(theta))


def cartesian_to_polar(x: float, y: float) -> Tuple[float, float]:
    """Convert Cartesian coordinates to polar.

    Args:
        x: X coordinate.
        y: Y coordinate.

    Returns:
        Tuple of (range, angle) where angle is in radians.
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return (r, theta)


def transform_to_global(
    local_x: float,
    local_y: float,
    robot_x: float,
    robot_y: float,
    robot_theta: float
) -> Tuple[float, float]:
    """Transform point from robot frame to global frame.

    Args:
        local_x: X coordinate in robot frame.
        local_y: Y coordinate in robot frame.
        robot_x: Robot X position in global frame.
        robot_y: Robot Y position in global frame.
        robot_theta: Robot orientation in global frame (radians).

    Returns:
        Tuple of (global_x, global_y).
    """
    # Rotation matrix
    cos_t = np.cos(robot_theta)
    sin_t = np.sin(robot_theta)

    global_x = robot_x + local_x * cos_t - local_y * sin_t
    global_y = robot_y + local_x * sin_t + local_y * cos_t

    return (global_x, global_y)


def compute_range_bearing(
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    landmark_x: float,
    landmark_y: float
) -> Tuple[float, float]:
    """Compute range and bearing from robot to landmark.

    Args:
        robot_x: Robot X position.
        robot_y: Robot Y position.
        robot_theta: Robot orientation (radians).
        landmark_x: Landmark X position.
        landmark_y: Landmark Y position.

    Returns:
        Tuple of (range, bearing) where bearing is relative to robot heading.
    """
    dx = landmark_x - robot_x
    dy = landmark_y - robot_y

    range_to_landmark = np.sqrt(dx**2 + dy**2)
    bearing = np.arctan2(dy, dx) - robot_theta

    return (range_to_landmark, bearing)


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi] range.

    Args:
        angle: Angle in radians.

    Returns:
        Normalized angle in radians.
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle
