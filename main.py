#!/usr/bin/env python3
"""
SLAM: Simultaneous Localization and Mapping

Command-line interface for running EKF-SLAM on sensor data.

Usage:
    python main.py data.csv
    python main.py data.csv --visualize
    python main.py data.csv --output results.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from slam import SLAM, SLAMConfig
from slam.visualization import SLAMVisualizer


def load_data(filepath: str) -> tuple:
    """Load sensor data from CSV file.

    Args:
        filepath: Path to CSV file.

    Returns:
        Tuple of (lidar_data, odometry_data, ground_truth).
    """
    data = pd.read_csv(
        filepath,
        delimiter=';',
        header=None,
        comment='#',
        na_values=['Nothing']
    )

    # Extract columns
    # 0-2: Robot pose (ground truth)
    # 3-5: Odometry deltas
    # 6-66: LIDAR measurements (61 rays)

    ground_truth = data.iloc[:, 0:3].values  # px, py, theta
    odometry = data.iloc[:, 3:6].values  # dx, dy, dtheta
    lidar = data.iloc[:, 6:67].values  # 61 LIDAR rays

    return lidar, odometry, ground_truth


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='EKF-SLAM: Simultaneous Localization and Mapping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py data.csv                    # Run SLAM on data
  python main.py data.csv --visualize        # Run with visualization
  python main.py data.csv -o results.png     # Save output plot

For more information, see: https://github.com/jnuno2000/SLAM
        """
    )

    parser.add_argument(
        'data_file',
        type=str,
        help='Path to CSV file with sensor data'
    )

    parser.add_argument(
        '-v', '--visualize',
        action='store_true',
        help='Show real-time visualization'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Save final plot to file'
    )

    parser.add_argument(
        '--corner-threshold',
        type=float,
        default=0.95,
        help='Corner detection threshold (default: 0.95)'
    )

    parser.add_argument(
        '--landmark-threshold',
        type=float,
        default=3.0,
        help='Landmark association threshold in meters (default: 3.0)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output messages'
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.data_file).exists():
        print(f"Error: File not found: {args.data_file}", file=sys.stderr)
        sys.exit(1)

    # Load data
    if not args.quiet:
        print(f"Loading data from {args.data_file}...")

    try:
        lidar_data, odometry_data, ground_truth = load_data(args.data_file)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"  Loaded {len(lidar_data)} timesteps")
        print(f"  LIDAR rays: {lidar_data.shape[1]}")

    # Configure SLAM
    config = SLAMConfig(
        corner_threshold=args.corner_threshold,
        landmark_association_threshold=args.landmark_threshold
    )

    # Initialize SLAM with first ground truth pose
    initial_pose = (
        ground_truth[0, 0],
        ground_truth[0, 1],
        ground_truth[0, 2]
    )

    if not args.quiet:
        print(f"\nInitializing SLAM at pose: ({initial_pose[0]:.2f}, {initial_pose[1]:.2f}, {initial_pose[2]:.2f})")

    slam = SLAM(initial_pose, config)

    # Run SLAM
    if not args.quiet:
        print("\nRunning EKF-SLAM...")

    results = slam.run(lidar_data, odometry_data, ground_truth)

    # Print results
    if not args.quiet:
        print("\n" + "=" * 50)
        print("SLAM Results")
        print("=" * 50)
        print(f"  Landmarks detected: {len(results['landmarks'])}")
        print(f"  Trajectory points:  {len(results['trajectory'])}")

        if 'mean_error' in results:
            print(f"\n  Position Error:")
            print(f"    Mean:  {results['mean_error']:.4f} m")
            print(f"    Max:   {results['max_error']:.4f} m")

        print("\n  Landmark positions:")
        for i, (lx, ly) in enumerate(results['landmarks'], 1):
            print(f"    {i}. ({lx:.2f}, {ly:.2f})")

    # Visualization
    if args.visualize or args.output:
        visualizer = SLAMVisualizer(config)
        visualizer.setup_plot()

        # Compute border points for visualization
        border_x, border_y = [], []
        angles = config.lidar_angles

        for t in range(len(lidar_data)):
            for i, angle in enumerate(angles):
                if lidar_data[t, i] > 0:
                    border_x.append(
                        ground_truth[t, 0] +
                        lidar_data[t, i] * np.cos(ground_truth[t, 2] + np.deg2rad(angle))
                    )
                    border_y.append(
                        ground_truth[t, 1] +
                        lidar_data[t, i] * np.sin(ground_truth[t, 2] + np.deg2rad(angle))
                    )

        # Plot final result
        traj_x = [p[0] for p in results['trajectory']]
        traj_y = [p[1] for p in results['trajectory']]

        visualizer.plot_frame(
            ground_truth_x=ground_truth[:, 0].tolist(),
            ground_truth_y=ground_truth[:, 1].tolist(),
            estimated_x=traj_x,
            estimated_y=traj_y,
            border_x=border_x,
            border_y=border_y,
            landmarks=results['landmarks']
        )

        if args.output:
            visualizer.save_figure(args.output)
            if not args.quiet:
                print(f"\nPlot saved to: {args.output}")

        if args.visualize:
            visualizer.show()

    if not args.quiet:
        print("\nDone.")


if __name__ == '__main__':
    main()
