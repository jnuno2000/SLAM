"""
SLAM: Simultaneous Localization and Mapping

A feature-based EKF-SLAM implementation using 2D LIDAR and odometry.
"""

from slam.config import SLAMConfig
from slam.filters import ExtendedKalmanFilter
from slam.features import CornerDetector
from slam.slam import SLAM

__version__ = "1.0.0"
__author__ = "Tiago Pereira Correia, Joao Nuno Leitao"
__all__ = ["SLAM", "SLAMConfig", "ExtendedKalmanFilter", "CornerDetector"]
