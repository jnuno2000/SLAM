"""
Kalman Filter implementations for state estimation.

Provides Extended Kalman Filter (EKF) for robot localization and mapping.
"""

from typing import Tuple
import numpy as np


class ExtendedKalmanFilter:
    """Extended Kalman Filter for SLAM state estimation.

    Implements the prediction and update steps of the EKF algorithm
    for simultaneous localization and mapping.

    Attributes:
        state: Current state estimate [x, y, theta, x1, y1, ...].
        covariance: Current state covariance matrix P.
    """

    def __init__(self, initial_state: np.ndarray, initial_covariance: np.ndarray):
        """Initialize the Extended Kalman Filter.

        Args:
            initial_state: Initial state vector of shape (n, 1).
            initial_covariance: Initial covariance matrix of shape (n, n).
        """
        self.state = initial_state.copy()
        self.covariance = initial_covariance.copy()

    @property
    def robot_pose(self) -> Tuple[float, float, float]:
        """Get current robot pose (x, y, theta)."""
        return (
            float(self.state[0, 0]),
            float(self.state[1, 0]),
            float(self.state[2, 0])
        )

    @property
    def num_landmarks(self) -> int:
        """Get number of landmarks in state vector."""
        return (len(self.state) - 3) // 2

    def predict(self, control_input: np.ndarray, process_noise: np.ndarray) -> None:
        """Perform prediction step using motion model.

        Updates state estimate based on odometry control input.

        Args:
            control_input: Control vector [dx, dy, dtheta] of shape (n, 1).
            process_noise: Process noise covariance matrix Q of shape (n, n).
        """
        self.state = self.state + control_input
        self.covariance = self.covariance + process_noise

    def update(
        self,
        measurement: np.ndarray,
        expected_measurement: np.ndarray,
        jacobian: np.ndarray,
        measurement_noise: np.ndarray
    ) -> None:
        """Perform update step using measurement model.

        Refines state estimate based on sensor observation.

        Args:
            measurement: Actual measurement [range, bearing] of shape (2, 1).
            expected_measurement: Expected measurement h(x) of shape (2, 1).
            jacobian: Measurement Jacobian H of shape (2, n).
            measurement_noise: Measurement noise covariance R of shape (2, 2).
        """
        # Innovation (measurement residual)
        innovation = measurement - expected_measurement

        # Normalize bearing angle to [-pi, pi]
        innovation[1, 0] = self._normalize_angle(innovation[1, 0])

        # Innovation covariance
        S = jacobian @ self.covariance @ jacobian.T + measurement_noise

        # Kalman gain
        K = self.covariance @ jacobian.T @ np.linalg.inv(S)

        # State update
        self.state = self.state + K @ innovation

        # Covariance update (Joseph form for numerical stability)
        self.covariance = self.covariance - K @ S @ K.T

    def augment_state(
        self,
        landmark_position: np.ndarray,
        process_noise_var: float
    ) -> None:
        """Add a new landmark to the state vector.

        Args:
            landmark_position: Position [x, y] of new landmark, shape (2, 1).
            process_noise_var: Variance for new landmark state.
        """
        # Augment state vector
        self.state = np.vstack([self.state, landmark_position])

        # Augment covariance matrix
        n = self.covariance.shape[0]

        # Add columns of zeros
        self.covariance = np.hstack([
            self.covariance,
            np.zeros((n, 2))
        ])

        # Add rows with initial variance on diagonal
        new_rows = np.zeros((2, n + 2))
        new_rows[0, n] = self.covariance[0, 0]
        new_rows[1, n + 1] = self.covariance[1, 1]

        self.covariance = np.vstack([self.covariance, new_rows])

    @staticmethod
    def _normalize_angle(angle: float) -> float:
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

    @staticmethod
    def compute_measurement_jacobian(
        robot_pose: Tuple[float, float, float],
        landmark_pos: Tuple[float, float],
        distance: float,
        num_landmarks: int,
        landmark_index: int
    ) -> np.ndarray:
        """Compute measurement Jacobian matrix H.

        Args:
            robot_pose: Robot pose (x, y, theta).
            landmark_pos: Landmark position (x, y).
            distance: Distance from robot to landmark.
            num_landmarks: Total number of landmarks in state.
            landmark_index: Index of observed landmark (1-indexed).

        Returns:
            Jacobian matrix H of shape (2, 3 + 2*num_landmarks).
        """
        rx, ry, _ = robot_pose
        lx, ly = landmark_pos

        dx = lx - rx
        dy = ly - ry
        d2 = distance ** 2

        # Jacobian w.r.t. robot pose
        H_robot = np.array([
            [-dx / distance, -dy / distance, 0],
            [dy / d2, -dx / d2, -1]
        ])

        # Jacobian w.r.t. landmark position
        H_landmark = np.array([
            [dx / distance, dy / distance],
            [-dy / d2, dx / d2]
        ])

        # Build full Jacobian
        state_size = 3 + 2 * num_landmarks
        H = np.zeros((2, state_size))
        H[:, :3] = H_robot

        # Place landmark Jacobian at correct position
        landmark_start = 3 + 2 * (landmark_index - 1)
        H[:, landmark_start:landmark_start + 2] = H_landmark

        return H
