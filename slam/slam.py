"""
Main SLAM system implementation.

Integrates all components for simultaneous localization and mapping.
"""

from typing import List, Optional, Tuple
import numpy as np

from slam.config import SLAMConfig, DEFAULT_CONFIG
from slam.filters import ExtendedKalmanFilter
from slam.features import CornerDetector, LandmarkManager
from slam.utils import compute_range_bearing
from slam.visualization import SLAMVisualizer


class SLAM:
    """Simultaneous Localization and Mapping system.

    Integrates EKF-based state estimation with corner detection
    for feature-based SLAM.

    Attributes:
        config: SLAM configuration parameters.
        ekf: Extended Kalman Filter for state estimation.
        detector: Corner detector for feature extraction.
        landmarks: Landmark manager for data association.
        visualizer: Optional visualization handler.
    """

    def __init__(
        self,
        initial_pose: Tuple[float, float, float],
        config: SLAMConfig = DEFAULT_CONFIG,
        enable_visualization: bool = False
    ):
        """Initialize SLAM system.

        Args:
            initial_pose: Initial robot pose (x, y, theta).
            config: SLAM configuration parameters.
            enable_visualization: Whether to enable real-time visualization.
        """
        self.config = config

        # Initialize state [x, y, theta]
        initial_state = np.array([
            [initial_pose[0]],
            [initial_pose[1]],
            [initial_pose[2]]
        ])

        # Initialize covariance (zero - known initial pose)
        initial_covariance = np.zeros((3, 3))

        self.ekf = ExtendedKalmanFilter(initial_state, initial_covariance)
        self.detector = CornerDetector(config)
        self.landmarks = LandmarkManager(config.landmark_association_threshold)

        # Control input vector (grows with state)
        self._control = np.zeros((3, 1))

        # Process noise matrix (grows with state)
        self._process_noise = config.get_process_noise_matrix(3)

        # Trajectory history
        self.trajectory: List[Tuple[float, float]] = []

        # Visualization
        self.visualizer: Optional[SLAMVisualizer] = None
        if enable_visualization:
            self.visualizer = SLAMVisualizer(config)
            self.visualizer.setup_plot()

    @property
    def pose(self) -> Tuple[float, float, float]:
        """Get current estimated robot pose."""
        return self.ekf.robot_pose

    @property
    def landmark_positions(self) -> List[Tuple[float, float]]:
        """Get all landmark positions."""
        return self.landmarks.landmarks.copy()

    def process_step(
        self,
        lidar_scan: np.ndarray,
        odometry: Tuple[float, float, float]
    ) -> None:
        """Process one SLAM iteration.

        Args:
            lidar_scan: Array of LIDAR range measurements.
            odometry: Odometry increment (dx, dy, dtheta).
        """
        angles = self.config.lidar_angles

        # Try to detect corner
        corner = self.detector.detect(lidar_scan)

        if corner is not None:
            corner_index = CornerDetector.get_corner_angle_index(
                corner[0], corner[1]
            )

            # Process corner for each LIDAR ray
            for i, angle in enumerate(angles):
                if i != corner_index:
                    continue

                # Transform corner to global frame
                rx, ry, rtheta = self.pose
                angle_rad = np.deg2rad(angle)

                global_x = rx + lidar_scan[i] * np.cos(rtheta + angle_rad)
                global_y = ry + lidar_scan[i] * np.sin(rtheta + angle_rad)

                # Data association
                landmark_idx, is_new = self.landmarks.associate(global_x, global_y)

                if is_new:
                    # Add new landmark to state
                    self._add_landmark(global_x, global_y)
                else:
                    # Update with known landmark observation
                    self._update_with_landmark(
                        lidar_scan[i],
                        angle_rad,
                        landmark_idx
                    )

        # Record trajectory
        self.trajectory.append((self.pose[0], self.pose[1]))

        # Prediction step with odometry
        self._predict(odometry)

    def _add_landmark(self, x: float, y: float) -> None:
        """Add new landmark to EKF state.

        Args:
            x: Landmark X position.
            y: Landmark Y position.
        """
        landmark_pos = np.array([[x], [y]])
        self.ekf.augment_state(
            landmark_pos,
            self.config.process_noise_std ** 2
        )

        # Expand control and process noise matrices
        self._control = np.vstack([self._control, np.zeros((2, 1))])

        n = self._process_noise.shape[0]
        self._process_noise = np.pad(
            self._process_noise,
            ((0, 2), (0, 2)),
            mode='constant'
        )
        self._process_noise[n, n] = self.config.process_noise_std ** 2
        self._process_noise[n + 1, n + 1] = self.config.process_noise_std ** 2

    def _update_with_landmark(
        self,
        measured_range: float,
        measured_bearing: float,
        landmark_index: int
    ) -> None:
        """Perform EKF update with landmark observation.

        Args:
            measured_range: Measured range to landmark.
            measured_bearing: Measured bearing to landmark (radians).
            landmark_index: Index of observed landmark (1-based).
        """
        # Get landmark position
        lx, ly = self.landmarks.get_landmark(landmark_index)
        rx, ry, rtheta = self.pose

        # Compute expected measurement
        expected_range, expected_bearing = compute_range_bearing(
            rx, ry, rtheta, lx, ly
        )

        # Build measurement vectors
        measurement = np.array([[measured_range], [measured_bearing]])
        expected = np.array([[expected_range], [expected_bearing]])

        # Compute Jacobian
        jacobian = ExtendedKalmanFilter.compute_measurement_jacobian(
            (rx, ry, rtheta),
            (lx, ly),
            expected_range,
            self.landmarks.count,
            landmark_index
        )

        # EKF update
        self.ekf.update(
            measurement,
            expected,
            jacobian,
            self.config.get_measurement_noise_matrix()
        )

    def _predict(self, odometry: Tuple[float, float, float]) -> None:
        """Perform EKF prediction step.

        Args:
            odometry: Odometry increment (dx, dy, dtheta).
        """
        # Update control vector
        self._control[0, 0] = odometry[0]
        self._control[1, 0] = odometry[1]
        self._control[2, 0] = odometry[2]

        self.ekf.predict(self._control, self._process_noise)

    def run(
        self,
        lidar_data: np.ndarray,
        odometry_data: np.ndarray,
        ground_truth: Optional[np.ndarray] = None
    ) -> dict:
        """Run SLAM on complete dataset.

        Args:
            lidar_data: LIDAR scans of shape (num_steps, num_rays).
            odometry_data: Odometry of shape (num_steps, 3) for (dx, dy, dtheta).
            ground_truth: Optional ground truth poses of shape (num_steps, 3).

        Returns:
            Dictionary containing results:
                - 'trajectory': List of estimated poses
                - 'landmarks': List of landmark positions
                - 'errors': Position errors if ground truth provided
        """
        num_steps = len(lidar_data)

        for t in range(num_steps):
            odometry = (
                odometry_data[t, 0],
                odometry_data[t, 1],
                odometry_data[t, 2]
            )
            self.process_step(lidar_data[t], odometry)

        results = {
            'trajectory': self.trajectory.copy(),
            'landmarks': self.landmark_positions
        }

        # Compute errors if ground truth available
        if ground_truth is not None:
            errors = []
            for i, (ex, ey) in enumerate(self.trajectory):
                if i < len(ground_truth):
                    gtx, gty = ground_truth[i, 0], ground_truth[i, 1]
                    error = np.sqrt((ex - gtx)**2 + (ey - gty)**2)
                    errors.append(error)
            results['errors'] = errors
            results['mean_error'] = np.mean(errors)
            results['max_error'] = np.max(errors)

        return results
