"""
Visualization utilities for SLAM system.

Provides plotting functions for robot trajectory, map, and landmarks.
"""

from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from slam.config import SLAMConfig, DEFAULT_CONFIG


class SLAMVisualizer:
    """Visualization handler for SLAM results.

    Creates plots of robot trajectory, environment map, and landmarks.

    Attributes:
        config: SLAM configuration parameters.
        fig: Matplotlib figure.
        ax: Matplotlib axes.
    """

    def __init__(self, config: SLAMConfig = DEFAULT_CONFIG):
        """Initialize visualizer.

        Args:
            config: SLAM configuration parameters.
        """
        self.config = config
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes] = None

    def setup_plot(self) -> Tuple[Figure, Axes]:
        """Create and configure plot figure.

        Returns:
            Tuple of (figure, axes).
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_aspect('equal')
        self.ax.set_xlim(self.config.plot_xlim)
        self.ax.set_ylim(self.config.plot_ylim)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        return self.fig, self.ax

    def plot_frame(
        self,
        ground_truth_x: List[float],
        ground_truth_y: List[float],
        estimated_x: List[float],
        estimated_y: List[float],
        border_x: List[float],
        border_y: List[float],
        landmarks: List[Tuple[float, float]],
        odometry_x: Optional[List[float]] = None,
        odometry_y: Optional[List[float]] = None,
        title: str = "EKF-SLAM: Localization and Mapping"
    ) -> None:
        """Plot a single frame of SLAM visualization.

        Args:
            ground_truth_x: Ground truth X positions.
            ground_truth_y: Ground truth Y positions.
            estimated_x: EKF estimated X positions.
            estimated_y: EKF estimated Y positions.
            border_x: Environment border X coordinates.
            border_y: Environment border Y coordinates.
            landmarks: List of landmark positions.
            odometry_x: Optional odometry-only X positions.
            odometry_y: Optional odometry-only Y positions.
            title: Plot title.
        """
        if self.ax is None:
            self.setup_plot()

        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(self.config.plot_xlim)
        self.ax.set_ylim(self.config.plot_ylim)
        self.ax.grid(True, alpha=0.3)

        # Plot environment borders
        self.ax.scatter(
            border_x, border_y,
            c='lightblue', s=1, alpha=0.5, label='Environment'
        )

        # Plot ground truth trajectory
        self.ax.plot(
            ground_truth_x, ground_truth_y,
            'g-', linewidth=2, alpha=0.7, label='Ground Truth'
        )

        # Plot estimated trajectory
        self.ax.plot(
            estimated_x, estimated_y,
            'r-', linewidth=2, label='EKF Estimate'
        )

        # Plot odometry-only trajectory if provided
        if odometry_x and odometry_y:
            self.ax.plot(
                odometry_x, odometry_y,
                'gray', linewidth=1, alpha=0.5,
                linestyle='--', label='Odometry Only'
            )

        # Plot landmarks
        if landmarks:
            lx = [l[0] for l in landmarks]
            ly = [l[1] for l in landmarks]
            self.ax.scatter(
                lx, ly,
                c='orange', s=200, marker='*',
                edgecolors='black', linewidths=1,
                label=f'Landmarks ({len(landmarks)})', zorder=10
            )

        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title(title)
        self.ax.legend(loc='upper left')

    def save_figure(self, filepath: str, dpi: int = 150) -> None:
        """Save current figure to file.

        Args:
            filepath: Output file path.
            dpi: Resolution in dots per inch.
        """
        if self.fig is not None:
            self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')

    def show(self) -> None:
        """Display the plot."""
        plt.show()

    def close(self) -> None:
        """Close the plot."""
        plt.close(self.fig)


def plot_corner_detection(
    lidar_ranges: np.ndarray,
    angles: np.ndarray,
    corner: Optional[Tuple[float, float]] = None,
    title: str = "Corner Detection"
) -> None:
    """Plot LIDAR scan with detected corner.

    Args:
        lidar_ranges: Array of range measurements.
        angles: Array of beam angles in radians.
        corner: Optional detected corner position (x, y).
        title: Plot title.
    """
    # Convert to Cartesian
    x = lidar_ranges * np.cos(angles)
    y = lidar_ranges * np.sin(angles)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    # Plot LIDAR points
    ax.scatter(x, y, c='blue', s=20, label='LIDAR Points')

    # Plot corner if detected
    if corner is not None:
        ax.scatter(
            corner[0], corner[1],
            c='red', s=200, marker='X',
            edgecolors='black', linewidths=2,
            label='Detected Corner', zorder=10
        )

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.show()
