"""
Feature detection algorithms for SLAM.

Provides corner detection from 2D LIDAR scans using geometric methods.
"""

from typing import List, Optional, Tuple
import numpy as np

from slam.config import SLAMConfig, DEFAULT_CONFIG


class CornerDetector:
    """Detects corners (room landmarks) from 2D LIDAR scans.

    Uses a split-and-fit algorithm to identify corners where two
    walls meet at approximately 90 degrees.

    Attributes:
        config: SLAM configuration parameters.
        angles: Array of LIDAR beam angles in radians.
    """

    def __init__(self, config: SLAMConfig = DEFAULT_CONFIG):
        """Initialize corner detector.

        Args:
            config: SLAM configuration parameters.
        """
        self.config = config
        self.angles = config.lidar_angles_rad

    def detect(self, lidar_ranges: np.ndarray) -> Optional[Tuple[float, float]]:
        """Detect corner in a single LIDAR scan.

        Args:
            lidar_ranges: Array of range measurements (meters).

        Returns:
            Corner position (x, y) in sensor frame, or None if no corner detected.
        """
        # Convert to Cartesian coordinates
        x_points, y_points = self._polar_to_cartesian(lidar_ranges)

        if len(x_points) < 2 * self.config.corner_margin + 1:
            return None

        # Find best split point
        best_residual = float('inf')
        best_index = -1
        vectors_left: List[np.ndarray] = []
        vectors_right: List[np.ndarray] = []

        margin = self.config.corner_margin

        for j in range(margin, len(x_points) - margin):
            # Fit lines to left and right segments
            vec_left, res_left = self._fit_line(
                x_points[:j], y_points[:j]
            )
            vec_right, res_right = self._fit_line(
                x_points[j:], y_points[j:]
            )

            vectors_left.append(vec_left)
            vectors_right.append(vec_right)

            total_residual = res_left + res_right

            if total_residual < best_residual:
                best_residual = total_residual
                best_index = j - margin

        if best_index < 0 or best_index >= len(vectors_left):
            return None

        # Verify corner using cross product of normal vectors
        cross_product = np.cross(
            vectors_left[best_index],
            vectors_right[best_index]
        )

        if abs(cross_product) > self.config.corner_threshold:
            corner_idx = best_index + margin
            return (x_points[corner_idx], y_points[corner_idx])

        return None

    def _polar_to_cartesian(
        self,
        ranges: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        """Convert polar LIDAR readings to Cartesian coordinates.

        Args:
            ranges: Array of range measurements.

        Returns:
            Tuple of (x_coordinates, y_coordinates) lists.
        """
        x_points: List[float] = []
        y_points: List[float] = []

        for i, r in enumerate(ranges):
            if r > 0:
                x_points.append(r * np.cos(self.angles[i]))
                y_points.append(r * np.sin(self.angles[i]))

        return x_points, y_points

    def _fit_line(
        self,
        x: List[float],
        y: List[float]
    ) -> Tuple[np.ndarray, float]:
        """Fit a line to points and return normal vector and residual.

        Chooses orientation based on data spread to avoid numerical issues.

        Args:
            x: X coordinates.
            y: Y coordinates.

        Returns:
            Tuple of (normal_vector, residual_sum).
        """
        if len(x) < 2:
            return np.array([1.0, 0.0]), 0.0

        dx = abs(x[-1] - x[0])
        dy = abs(y[-1] - y[0])

        if dx > dy:
            # Fit y = mx + b
            coeffs, residuals, *_ = np.polyfit(x, y, 1, full=True)
            normal = np.array([1.0, coeffs[0]])
        else:
            # Fit x = my + b
            coeffs, residuals, *_ = np.polyfit(y, x, 1, full=True)
            normal = np.array([coeffs[0], 1.0])

        residual = residuals[0] if len(residuals) > 0 else 0.0
        return normal, residual

    @staticmethod
    def get_corner_angle_index(
        corner_x: float,
        corner_y: float,
        angle_offset: float = 30.0
    ) -> int:
        """Convert corner position to LIDAR angle index.

        Args:
            corner_x: Corner X coordinate in sensor frame.
            corner_y: Corner Y coordinate in sensor frame.
            angle_offset: Offset to convert angle to array index.

        Returns:
            Index into LIDAR array corresponding to corner direction.
        """
        angle_rad = np.arctan2(corner_y, corner_x)
        angle_deg = np.rad2deg(angle_rad)
        return int(round(angle_deg + angle_offset))


class LandmarkManager:
    """Manages known landmarks and data association.

    Tracks discovered landmarks and associates new observations
    with existing landmarks or creates new ones.

    Attributes:
        landmarks: List of known landmark positions [(x, y), ...].
        threshold: Distance threshold for landmark association.
    """

    def __init__(self, association_threshold: float = 3.0):
        """Initialize landmark manager.

        Args:
            association_threshold: Maximum distance to associate observation
                with existing landmark (meters).
        """
        self.landmarks: List[Tuple[float, float]] = []
        self.threshold = association_threshold

    def associate(
        self,
        observed_x: float,
        observed_y: float
    ) -> Tuple[int, bool]:
        """Associate observation with existing landmark or create new one.

        Args:
            observed_x: Observed landmark X position.
            observed_y: Observed landmark Y position.

        Returns:
            Tuple of (landmark_index, is_new).
            landmark_index is 1-indexed for existing landmarks, 0 for new.
        """
        for i, (lx, ly) in enumerate(self.landmarks):
            if (abs(lx - observed_x) < self.threshold and
                abs(ly - observed_y) < self.threshold):
                # Update landmark position
                self.landmarks[i] = (observed_x, observed_y)
                return (i + 1, False)

        # New landmark
        self.landmarks.append((observed_x, observed_y))
        return (0, True)

    def get_landmark(self, index: int) -> Tuple[float, float]:
        """Get landmark position by 1-based index.

        Args:
            index: 1-based landmark index.

        Returns:
            Landmark position (x, y).
        """
        return self.landmarks[index - 1]

    @property
    def count(self) -> int:
        """Get number of known landmarks."""
        return len(self.landmarks)
