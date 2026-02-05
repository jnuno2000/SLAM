"""
Configuration management for SLAM system.

Centralizes all tunable parameters and magic numbers for easy modification.
"""

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


@dataclass
class SLAMConfig:
    """Configuration parameters for the SLAM system.

    Attributes:
        lidar_fov: LIDAR field of view in degrees (min, max).
        lidar_rays: Number of LIDAR rays.
        corner_threshold: Cross-product threshold for corner detection (0-1).
        corner_margin: Margin of rays to skip at scan edges for corner detection.
        landmark_association_threshold: Distance threshold for landmark matching (meters).
        process_noise_std: Standard deviation for process noise (odometry uncertainty).
        measurement_noise_std: Standard deviation for measurement noise (LIDAR uncertainty).
    """

    # LIDAR configuration
    lidar_fov: Tuple[float, float] = (-30.0, 30.0)
    lidar_rays: int = 61

    # Corner detection parameters
    corner_threshold: float = 0.95
    corner_margin: int = 5

    # Data association
    landmark_association_threshold: float = 3.0

    # Kalman filter noise parameters
    process_noise_std: float = 4.0
    measurement_noise_std: float = 5.0

    # Visualization
    plot_xlim: Tuple[float, float] = (-3.0, 18.0)
    plot_ylim: Tuple[float, float] = (-5.0, 18.0)

    @property
    def lidar_angles(self) -> np.ndarray:
        """Generate array of LIDAR beam angles in degrees."""
        return np.linspace(self.lidar_fov[0], self.lidar_fov[1], self.lidar_rays)

    @property
    def lidar_angles_rad(self) -> np.ndarray:
        """Generate array of LIDAR beam angles in radians."""
        return np.deg2rad(self.lidar_angles)

    def get_process_noise_matrix(self, state_size: int) -> np.ndarray:
        """Generate process noise covariance matrix Q.

        Args:
            state_size: Current size of state vector.

        Returns:
            Process noise covariance matrix of shape (state_size, state_size).
        """
        return np.eye(state_size) * (self.process_noise_std ** 2)

    def get_measurement_noise_matrix(self) -> np.ndarray:
        """Generate measurement noise covariance matrix R.

        Returns:
            2x2 measurement noise covariance matrix.
        """
        return np.eye(2) * (self.measurement_noise_std ** 2)


# Default configuration instance
DEFAULT_CONFIG = SLAMConfig()
