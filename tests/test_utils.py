"""Unit tests for utility functions."""

import numpy as np
import pytest

from slam.utils import (
    polar_to_cartesian,
    cartesian_to_polar,
    transform_to_global,
    compute_range_bearing,
    normalize_angle
)


class TestCoordinateTransforms:
    """Tests for coordinate transformation functions."""

    def test_polar_to_cartesian_zero_angle(self):
        """Test polar to Cartesian at zero angle."""
        x, y = polar_to_cartesian(5.0, 0.0)

        assert x == pytest.approx(5.0)
        assert y == pytest.approx(0.0)

    def test_polar_to_cartesian_90_degrees(self):
        """Test polar to Cartesian at 90 degrees."""
        x, y = polar_to_cartesian(5.0, np.pi / 2)

        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(5.0)

    def test_cartesian_to_polar_positive_x(self):
        """Test Cartesian to polar on positive X axis."""
        r, theta = cartesian_to_polar(5.0, 0.0)

        assert r == pytest.approx(5.0)
        assert theta == pytest.approx(0.0)

    def test_cartesian_to_polar_quadrant1(self):
        """Test Cartesian to polar in first quadrant."""
        r, theta = cartesian_to_polar(3.0, 4.0)

        assert r == pytest.approx(5.0)
        assert theta == pytest.approx(np.arctan2(4.0, 3.0))

    def test_polar_cartesian_roundtrip(self):
        """Test roundtrip conversion preserves values."""
        original_r, original_theta = 7.5, np.pi / 6

        x, y = polar_to_cartesian(original_r, original_theta)
        r, theta = cartesian_to_polar(x, y)

        assert r == pytest.approx(original_r)
        assert theta == pytest.approx(original_theta)


class TestGlobalTransform:
    """Tests for robot-to-global frame transformation."""

    def test_transform_no_rotation(self):
        """Test transformation with no robot rotation."""
        global_x, global_y = transform_to_global(
            local_x=1.0, local_y=0.0,
            robot_x=5.0, robot_y=3.0, robot_theta=0.0
        )

        assert global_x == pytest.approx(6.0)
        assert global_y == pytest.approx(3.0)

    def test_transform_90_degree_rotation(self):
        """Test transformation with 90 degree robot rotation."""
        global_x, global_y = transform_to_global(
            local_x=1.0, local_y=0.0,
            robot_x=0.0, robot_y=0.0, robot_theta=np.pi / 2
        )

        assert global_x == pytest.approx(0.0, abs=1e-10)
        assert global_y == pytest.approx(1.0)

    def test_transform_origin(self):
        """Test transformation of origin point."""
        global_x, global_y = transform_to_global(
            local_x=0.0, local_y=0.0,
            robot_x=5.0, robot_y=3.0, robot_theta=0.5
        )

        assert global_x == pytest.approx(5.0)
        assert global_y == pytest.approx(3.0)


class TestRangeBearing:
    """Tests for range and bearing computation."""

    def test_range_bearing_same_position(self):
        """Test range/bearing when landmark at robot position."""
        r, bearing = compute_range_bearing(
            robot_x=5.0, robot_y=5.0, robot_theta=0.0,
            landmark_x=5.0, landmark_y=5.0
        )

        assert r == pytest.approx(0.0)

    def test_range_bearing_ahead(self):
        """Test range/bearing for landmark directly ahead."""
        r, bearing = compute_range_bearing(
            robot_x=0.0, robot_y=0.0, robot_theta=0.0,
            landmark_x=5.0, landmark_y=0.0
        )

        assert r == pytest.approx(5.0)
        assert bearing == pytest.approx(0.0)

    def test_range_bearing_left(self):
        """Test range/bearing for landmark to the left."""
        r, bearing = compute_range_bearing(
            robot_x=0.0, robot_y=0.0, robot_theta=0.0,
            landmark_x=0.0, landmark_y=5.0
        )

        assert r == pytest.approx(5.0)
        assert bearing == pytest.approx(np.pi / 2)


class TestNormalizeAngle:
    """Tests for angle normalization."""

    def test_normalize_zero(self):
        """Test normalizing zero."""
        assert normalize_angle(0.0) == pytest.approx(0.0)

    def test_normalize_pi(self):
        """Test normalizing pi."""
        assert normalize_angle(np.pi) == pytest.approx(np.pi)

    def test_normalize_negative_pi(self):
        """Test normalizing negative pi."""
        assert normalize_angle(-np.pi) == pytest.approx(-np.pi)

    def test_normalize_large_positive(self):
        """Test normalizing large positive angle."""
        result = normalize_angle(5 * np.pi)
        assert -np.pi <= result <= np.pi

    def test_normalize_large_negative(self):
        """Test normalizing large negative angle."""
        result = normalize_angle(-5 * np.pi)
        assert -np.pi <= result <= np.pi

    def test_normalize_two_pi(self):
        """Test normalizing 2*pi."""
        assert normalize_angle(2 * np.pi) == pytest.approx(0.0, abs=1e-10)
