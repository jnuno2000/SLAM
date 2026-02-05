# SLAM: Simultaneous Localization and Mapping

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Robotics](https://img.shields.io/badge/Domain-Robotics-red.svg)]()
[![Kalman Filter](https://img.shields.io/badge/Algorithm-EKF--SLAM-purple.svg)]()

A real-time **Simultaneous Localization and Mapping (SLAM)** system that enables autonomous robots to build maps of unknown environments while simultaneously tracking their own position. This implementation uses **Extended Kalman Filtering** for sensor fusion, combining 2D LIDAR measurements with odometry data.

## Overview

SLAM is one of the fundamental challenges in autonomous robotics. This project demonstrates a complete feature-based SLAM pipeline that:

- **Detects landmarks** (room corners) from 2D LIDAR scans using polynomial fitting and geometric analysis
- **Estimates robot pose** by fusing noisy odometry with LIDAR observations
- **Builds a consistent map** of the environment in real-time
- **Handles data association** between new observations and known landmarks

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SLAM SYSTEM ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐         ┌──────────────────────────────────────┐        │
│   │   SENSORS    │         │         KALMAN FILTER CORE           │        │
│   ├──────────────┤         ├──────────────────────────────────────┤        │
│   │              │         │                                      │        │
│   │  ┌────────┐  │  Pose   │  ┌────────────┐    ┌─────────────┐  │        │
│   │  │Odometry│──┼─Deltas──┼─►│  PREDICT   │───►│ State X_k|k-1│  │        │
│   │  └────────┘  │  (u_k)  │  │  X = X + U │    └──────┬──────┘  │        │
│   │              │         │  │  P = P + Q │           │         │        │
│   │  ┌────────┐  │         │  └────────────┘           ▼         │        │
│   │  │ LIDAR  │  │         │                    ┌─────────────┐  │        │
│   │  │(61 rays)│ │         │  ┌────────────┐   │   UPDATE    │  │        │
│   │  └────┬───┘  │         │  │  Corner    │   │  K = PH'S⁻¹ │  │        │
│   │       │      │         │  │  Detection │──►│  X = X + Ky │  │        │
│   └───────┼──────┘         │  └────────────┘   │  P = P-KSK' │  │        │
│           │                │                    └──────┬──────┘  │        │
│           ▼                │                           │         │        │
│   ┌──────────────┐         │                           ▼         │        │
│   │   FEATURE    │         │  ┌──────────────────────────────┐   │        │
│   │  EXTRACTION  │         │  │      STATE VECTOR X          │   │        │
│   ├──────────────┤         │  │  [x, y, θ, x₁, y₁, ..., xₙ, yₙ] │        │
│   │ • Polar→Cart │         │  │   └──┬──┘  └───────┬────────┘│   │        │
│   │ • Line Fit   │         │  │   Robot    Landmark          │   │        │
│   │ • Corner Det │         │  │   Pose     Positions         │   │        │
│   └──────────────┘         │  └──────────────────────────────┘   │        │
│                            └──────────────────────────────────────┘        │
│                                          │                                  │
│                                          ▼                                  │
│                            ┌──────────────────────────┐                    │
│                            │    OUTPUT: 2D Map +      │                    │
│                            │    Robot Trajectory      │                    │
│                            └──────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Modular Architecture** | Clean separation of concerns with dedicated modules for filtering, features, and visualization |
| **Corner Detection** | Geometric algorithm using least-squares line fitting to detect room corners from LIDAR scans |
| **EKF-SLAM** | Full Extended Kalman Filter with dynamic state augmentation for discovered landmarks |
| **Sensor Fusion** | Combines odometry (motion model) with LIDAR observations (measurement model) |
| **Data Association** | Distance-threshold matching to associate observations with known landmarks |
| **CLI Interface** | Command-line tool with configurable parameters for easy experimentation |
| **Comprehensive Tests** | Unit tests for all core algorithms ensuring reliability |

## Installation

```bash
# Clone the repository
git clone https://github.com/jnuno2000/SLAM.git
cd SLAM

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Development Installation

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Or use make
make install-dev
```

## Quick Start

### Command Line Interface

```bash
# Run SLAM with visualization
python main.py data.csv --visualize

# Save output plot
python main.py data.csv --output results.png

# Customize parameters
python main.py data.csv --corner-threshold 0.95 --landmark-threshold 3.0
```

### Python API

```python
from slam import SLAM, SLAMConfig

# Configure SLAM parameters
config = SLAMConfig(
    corner_threshold=0.95,
    landmark_association_threshold=3.0
)

# Initialize with starting pose
slam = SLAM(initial_pose=(0, 0, 0), config=config)

# Process sensor data
for lidar_scan, odometry in sensor_stream:
    slam.process_step(lidar_scan, odometry)

# Get results
print(f"Estimated pose: {slam.pose}")
print(f"Landmarks found: {slam.landmark_positions}")
```

### Jupyter Notebook

```bash
jupyter notebook SLAM_localization.ipynb
```

## Project Structure

```
SLAM/
├── slam/                       # Core SLAM package
│   ├── __init__.py            # Package exports
│   ├── config.py              # Configuration management
│   ├── filters.py             # Extended Kalman Filter
│   ├── features.py            # Corner detection & landmarks
│   ├── slam.py                # Main SLAM class
│   ├── utils.py               # Coordinate transforms
│   └── visualization.py       # Plotting utilities
├── tests/                     # Unit tests
│   ├── test_filters.py        # EKF tests
│   ├── test_features.py       # Feature detection tests
│   └── test_utils.py          # Utility function tests
├── main.py                    # CLI entry point
├── SLAM_localization.ipynb    # Interactive notebook
├── SLAM_localization.py       # Legacy script
├── data.csv                   # Sample sensor data
├── requirements.txt           # Dependencies
├── pyproject.toml            # Package configuration
├── Makefile                   # Development commands
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## Development

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
python -m pytest tests/test_filters.py -v
```

### Code Quality

```bash
# Format code
make format

# Run linter
make lint

# Type checking
make typecheck
```

### Available Make Commands

```
make install      Install dependencies
make install-dev  Install dev dependencies
make test         Run unit tests
make test-cov     Run tests with coverage
make lint         Run code linting
make format       Format code with black
make clean        Remove build artifacts
make run          Run SLAM on sample data
```

## Technical Implementation

### Corner Detection Algorithm

The system identifies room corners from 2D LIDAR data using a split-and-fit approach:

1. **Coordinate Transformation**: Convert polar LIDAR readings (r, θ) to Cartesian (x, y)
2. **Line Fitting**: For each potential split point, fit lines to left/right segments
3. **Corner Verification**: Compute cross-product of normal vectors; if |v₁ × v₂| > threshold, corner exists
4. **Optimal Split**: Select the split point minimizing total residual error

```python
# From slam/features.py
class CornerDetector:
    def detect(self, lidar_ranges: np.ndarray) -> Optional[Tuple[float, float]]:
        """Detect corner in a single LIDAR scan."""
        x_points, y_points = self._polar_to_cartesian(lidar_ranges)

        for j in range(margin, len(x_points) - margin):
            vec_left, res_left = self._fit_line(x_points[:j], y_points[:j])
            vec_right, res_right = self._fit_line(x_points[j:], y_points[j:])

            if abs(np.cross(vec_left, vec_right)) > self.config.corner_threshold:
                return (x_points[j], y_points[j])
        return None
```

### Extended Kalman Filter

The EKF maintains a state vector that grows dynamically as new landmarks are discovered:

**State Vector:**
```
X = [x, y, θ, x₁, y₁, x₂, y₂, ..., xₙ, yₙ]ᵀ
     └─┬─┘   └────────────┬────────────┘
    Robot        n Landmark positions
    Pose
```

**Prediction Step** (Motion Model):
```
X_{k|k-1} = X_{k-1} + U_k
P_{k|k-1} = P_{k-1} + Q
```

**Update Step** (Measurement Model):
```
K = P · Hᵀ · (H · P · Hᵀ + R)⁻¹
X_k = X_{k|k-1} + K · (z - h(X_{k|k-1}))
P_k = (I - K · H) · P_{k|k-1}
```

### Configuration Parameters

All tunable parameters are centralized in `SLAMConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `corner_threshold` | 0.95 | Cross-product threshold for corner detection |
| `corner_margin` | 5 | Rays to skip at scan edges |
| `landmark_association_threshold` | 3.0 | Distance for landmark matching (meters) |
| `process_noise_std` | 4.0 | Odometry uncertainty |
| `measurement_noise_std` | 5.0 | LIDAR measurement uncertainty |

## Data Format

The system expects a CSV file with the following structure:

| Columns | Description |
|---------|-------------|
| 0-2 | Robot pose: `px, py, theta` (ground truth) |
| 3-5 | Odometry deltas: `dx, dy, dtheta` |
| 6-66 | LIDAR readings: 61 range measurements (-30° to +30°) |

## Results

The system successfully:

- Detects and tracks **4 room corners** as persistent landmarks
- Maintains **consistent pose estimation** despite odometry drift
- Produces a **coherent 2D map** matching ground truth
- Demonstrates **EKF improvement over pure odometry**

### Performance Comparison

| Method | Position Error | Map Consistency |
|--------|----------------|-----------------|
| Pure Odometry | Accumulates unbounded drift | No map building |
| EKF-SLAM | Bounded, corrected by observations | Consistent landmark positions |

## Algorithm Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Prediction | O(n²) | O(n²) |
| Update | O(n²) | O(n²) |
| Corner Detection | O(m²) | O(m) |

Where n = number of landmarks, m = number of LIDAR rays (61)

## Technologies

- **Python 3.7+** - Core implementation
- **NumPy** - Linear algebra and matrix operations
- **Matplotlib** - Visualization and animation
- **Pandas** - Data loading and manipulation
- **SciPy** - Signal processing utilities
- **pytest** - Unit testing framework

## Future Improvements

- [ ] Implement loop closure detection for large-scale mapping
- [ ] Add scan matching (ICP) for more robust localization
- [ ] Extend to 3D SLAM using RGB-D sensors
- [ ] Implement particle filter (FastSLAM) for comparison
- [ ] Add occupancy grid mapping alongside feature-based approach
- [ ] GPU acceleration for real-time performance

## References

1. Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
2. Durrant-Whyte, H., & Bailey, T. (2006). Simultaneous Localization and Mapping: Part I. *IEEE Robotics & Automation Magazine*.
3. Smith, R., Self, M., & Cheeseman, P. (1990). Estimating Uncertain Spatial Relationships in Robotics. *Autonomous Robot Vehicles*.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Tiago Pereira Correia** - [up201806248@up.pt](mailto:up201806248@up.pt)
- **Joao Nuno Leitao** - [up201806619@up.pt](mailto:up201806619@up.pt)

*Faculty of Engineering, University of Porto (FEUP) - Perception and Mapping Course, 2022/23*
