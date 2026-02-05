"""Unit tests for Kalman Filter implementation."""

import numpy as np
import pytest

from slam.filters import ExtendedKalmanFilter


class TestExtendedKalmanFilter:
    """Tests for ExtendedKalmanFilter class."""

    def test_initialization(self):
        """Test EKF initializes with correct state and covariance."""
        initial_state = np.array([[1.0], [2.0], [0.5]])
        initial_cov = np.eye(3) * 0.1

        ekf = ExtendedKalmanFilter(initial_state, initial_cov)

        assert ekf.state.shape == (3, 1)
        assert ekf.covariance.shape == (3, 3)
        np.testing.assert_array_equal(ekf.state, initial_state)
        np.testing.assert_array_equal(ekf.covariance, initial_cov)

    def test_robot_pose_property(self):
        """Test robot_pose returns correct values."""
        state = np.array([[1.5], [2.5], [0.3]])
        ekf = ExtendedKalmanFilter(state, np.eye(3))

        x, y, theta = ekf.robot_pose

        assert x == pytest.approx(1.5)
        assert y == pytest.approx(2.5)
        assert theta == pytest.approx(0.3)

    def test_predict_updates_state(self):
        """Test prediction step correctly updates state."""
        initial_state = np.array([[0.0], [0.0], [0.0]])
        initial_cov = np.eye(3) * 0.1
        ekf = ExtendedKalmanFilter(initial_state, initial_cov)

        control = np.array([[1.0], [0.5], [0.1]])
        process_noise = np.eye(3) * 0.01

        ekf.predict(control, process_noise)

        np.testing.assert_array_almost_equal(
            ekf.state,
            np.array([[1.0], [0.5], [0.1]])
        )

    def test_predict_increases_uncertainty(self):
        """Test prediction increases covariance."""
        initial_cov = np.eye(3) * 0.1
        ekf = ExtendedKalmanFilter(np.zeros((3, 1)), initial_cov)

        process_noise = np.eye(3) * 0.05
        ekf.predict(np.zeros((3, 1)), process_noise)

        # Covariance should increase by process noise
        expected_cov = initial_cov + process_noise
        np.testing.assert_array_almost_equal(ekf.covariance, expected_cov)

    def test_augment_state(self):
        """Test state augmentation adds landmark correctly."""
        state = np.array([[1.0], [2.0], [0.5]])
        cov = np.eye(3) * 0.1
        ekf = ExtendedKalmanFilter(state, cov)

        landmark = np.array([[5.0], [6.0]])
        ekf.augment_state(landmark, 0.1)

        assert ekf.state.shape == (5, 1)
        assert ekf.covariance.shape == (5, 5)
        assert ekf.num_landmarks == 1

        # Check landmark was added
        np.testing.assert_array_almost_equal(
            ekf.state[3:],
            landmark
        )

    def test_normalize_angle(self):
        """Test angle normalization to [-pi, pi]."""
        # Test various angles
        assert ExtendedKalmanFilter._normalize_angle(0) == pytest.approx(0)
        assert ExtendedKalmanFilter._normalize_angle(np.pi) == pytest.approx(np.pi)
        assert ExtendedKalmanFilter._normalize_angle(-np.pi) == pytest.approx(-np.pi)
        assert ExtendedKalmanFilter._normalize_angle(3 * np.pi) == pytest.approx(np.pi)
        assert ExtendedKalmanFilter._normalize_angle(-3 * np.pi) == pytest.approx(-np.pi)

    def test_compute_measurement_jacobian_shape(self):
        """Test Jacobian has correct shape."""
        robot_pose = (1.0, 2.0, 0.5)
        landmark_pos = (5.0, 6.0)
        distance = 5.0

        H = ExtendedKalmanFilter.compute_measurement_jacobian(
            robot_pose, landmark_pos, distance,
            num_landmarks=2, landmark_index=1
        )

        # 2 measurements, 3 robot state + 2*2 landmarks = 7
        assert H.shape == (2, 7)


class TestEKFUpdate:
    """Tests for EKF update step."""

    def test_update_reduces_uncertainty(self):
        """Test that update step reduces covariance."""
        state = np.array([[1.0], [2.0], [0.5]])
        cov = np.eye(3) * 1.0  # High initial uncertainty
        ekf = ExtendedKalmanFilter(state, cov)

        # Simple measurement model
        measurement = np.array([[1.0], [0.0]])
        expected = np.array([[1.0], [0.0]])
        H = np.array([[1, 0, 0], [0, 1, 0]])
        R = np.eye(2) * 0.1  # Low measurement noise

        initial_trace = np.trace(ekf.covariance)
        ekf.update(measurement, expected, H, R)
        final_trace = np.trace(ekf.covariance)

        # Uncertainty should decrease (or stay same if measurement matches)
        assert final_trace <= initial_trace
