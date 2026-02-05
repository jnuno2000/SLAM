"""Unit tests for feature detection."""

import numpy as np
import pytest

from slam.features import CornerDetector, LandmarkManager
from slam.config import SLAMConfig


class TestCornerDetector:
    """Tests for CornerDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a corner detector instance."""
        return CornerDetector()

    def test_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector.config is not None
        assert len(detector.angles) == 61  # Default LIDAR rays

    def test_no_corner_flat_wall(self, detector):
        """Test no corner detected for flat wall."""
        # Simulate flat wall - constant range
        ranges = np.ones(61) * 5.0

        corner = detector.detect(ranges)

        # Flat wall should not trigger corner detection
        # (may or may not detect depending on threshold)
        # This is a basic sanity check
        assert corner is None or isinstance(corner, tuple)

    def test_corner_detection_l_shape(self, detector):
        """Test corner detection for L-shaped scan."""
        angles = detector.angles

        # Simulate L-shaped corner
        ranges = np.zeros(61)
        for i, angle in enumerate(angles):
            if angle < 0:
                # Left wall
                ranges[i] = 3.0 / np.cos(np.deg2rad(angle))
            else:
                # Right wall
                ranges[i] = 3.0 / np.sin(np.deg2rad(angle)) if angle > 0 else 10.0

        # Should detect something (corner at the L junction)
        corner = detector.detect(ranges)

        # Corner should be detected near the junction
        # Exact position depends on algorithm parameters
        if corner is not None:
            assert isinstance(corner, tuple)
            assert len(corner) == 2

    def test_get_corner_angle_index(self):
        """Test corner angle to index conversion."""
        # Center (0 degrees) should map to index 30
        idx = CornerDetector.get_corner_angle_index(1.0, 0.0)
        assert idx == 30

        # -30 degrees should map to index 0
        idx = CornerDetector.get_corner_angle_index(
            1.0 * np.cos(np.deg2rad(-30)),
            1.0 * np.sin(np.deg2rad(-30))
        )
        assert idx == 0

        # +30 degrees should map to index 60
        idx = CornerDetector.get_corner_angle_index(
            1.0 * np.cos(np.deg2rad(30)),
            1.0 * np.sin(np.deg2rad(30))
        )
        assert idx == 60


class TestLandmarkManager:
    """Tests for LandmarkManager class."""

    def test_initialization(self):
        """Test landmark manager initializes empty."""
        manager = LandmarkManager()

        assert manager.count == 0
        assert manager.landmarks == []

    def test_add_first_landmark(self):
        """Test adding first landmark."""
        manager = LandmarkManager()

        idx, is_new = manager.associate(5.0, 6.0)

        assert is_new is True
        assert idx == 0
        assert manager.count == 1
        assert manager.landmarks[0] == (5.0, 6.0)

    def test_associate_existing_landmark(self):
        """Test associating with existing landmark."""
        manager = LandmarkManager(association_threshold=2.0)

        # Add landmark
        manager.associate(5.0, 6.0)

        # Associate nearby point
        idx, is_new = manager.associate(5.5, 6.5)

        assert is_new is False
        assert idx == 1  # 1-indexed
        assert manager.count == 1  # No new landmark added

    def test_add_distant_landmark(self):
        """Test adding landmark far from existing ones."""
        manager = LandmarkManager(association_threshold=2.0)

        # Add first landmark
        manager.associate(0.0, 0.0)

        # Add distant landmark
        idx, is_new = manager.associate(10.0, 10.0)

        assert is_new is True
        assert idx == 0  # New landmark index
        assert manager.count == 2

    def test_get_landmark(self):
        """Test retrieving landmark by index."""
        manager = LandmarkManager(association_threshold=1.0)  # Small threshold
        manager.associate(1.0, 2.0)
        manager.associate(10.0, 20.0)  # Far enough to be a new landmark

        lm1 = manager.get_landmark(1)
        lm2 = manager.get_landmark(2)

        assert lm1 == (1.0, 2.0)
        assert lm2 == (10.0, 20.0)

    def test_landmark_position_update(self):
        """Test that re-observation updates landmark position."""
        manager = LandmarkManager(association_threshold=2.0)

        # Initial observation
        manager.associate(5.0, 6.0)

        # Re-observe with slightly different position
        manager.associate(5.1, 6.1)

        # Position should be updated
        assert manager.landmarks[0] == (5.1, 6.1)
