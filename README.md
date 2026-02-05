# SLAM: Simultaneous Localization and Mapping

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Robotics](https://img.shields.io/badge/Domain-Robotics-red.svg)]()
[![Kalman Filter](https://img.shields.io/badge/Algorithm-Kalman%20Filter-purple.svg)]()

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
| **Corner Detection** | Geometric algorithm using least-squares line fitting to detect room corners from LIDAR scans |
| **Kalman Filter SLAM** | Full EKF implementation with dynamic state augmentation for discovered landmarks |
| **Sensor Fusion** | Combines odometry (motion model) with LIDAR observations (measurement model) |
| **Data Association** | Distance-threshold matching to associate observations with known landmarks |
| **Real-time Visualization** | Frame-by-frame animation of robot trajectory and map building |

## Technical Implementation

### Corner Detection Algorithm

The system identifies room corners (landmarks) from 2D LIDAR data using a split-and-fit approach:

1. **Coordinate Transformation**: Convert polar LIDAR readings (r, θ) to Cartesian (x, y)
2. **Line Fitting**: For each potential split point, fit lines to left/right segments using least-squares
3. **Corner Verification**: Compute cross-product of line normal vectors; if |v₁ × v₂| > 0.95, a corner exists
4. **Optimal Split**: Select the split point that minimizes total residual error

```python
# Corner detection via perpendicular line intersection
for split_point in range(5, len(scan) - 5):
    line_left = polyfit(points[:split_point])
    line_right = polyfit(points[split_point:])

    if |cross(normal_left, normal_right)| > threshold:
        corner_detected = True
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

Where:
- **X**: State estimate (robot pose + landmark positions)
- **P**: State covariance matrix (uncertainty)
- **Q**: Process noise covariance (odometry uncertainty)
- **R**: Measurement noise covariance (LIDAR uncertainty)
- **H**: Jacobian of measurement function
- **K**: Kalman gain (optimal weighting)

### Measurement Model

Observations are made in polar form relative to the robot:

```
z = [r, α]ᵀ = [√((xᵢ-x)² + (yᵢ-y)²), atan2(yᵢ-y, xᵢ-x) - θ]ᵀ
```

The Jacobian H is computed analytically for each observed landmark:

```
H = │ ∂r/∂x   ∂r/∂y   ∂r/∂θ   ...  ∂r/∂xᵢ   ∂r/∂yᵢ   ... │
    │ ∂α/∂x   ∂α/∂y   ∂α/∂θ   ...  ∂α/∂xᵢ   ∂α/∂yᵢ   ... │
```

## Installation

```bash
# Clone the repository
git clone https://github.com/jnuno2000/SLAM.git
cd SLAM

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook (Recommended)
```bash
jupyter notebook SLAM_localization.ipynb
```

### Python Script
```bash
python SLAM_localization.py
```

### Data Format

The system expects a CSV file (`data.csv`) with the following structure:

| Columns | Description |
|---------|-------------|
| 0-2 | Robot pose: `px, py, theta` |
| 3-5 | Odometry deltas: `dx, dy, dtheta` |
| 6-66 | LIDAR readings: 61 range measurements (-30° to +30°) |

## Project Structure

```
SLAM/
├── SLAM_localization.ipynb  # Main implementation with visualizations
├── SLAM_localization.py     # Python script version
├── data.csv                 # Sensor measurements dataset
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
└── README.md               # This file
```

## Results

The system successfully:

- Detects and tracks **4 room corners** as persistent landmarks
- Maintains **consistent pose estimation** despite odometry drift
- Produces a **coherent 2D map** that matches ground truth
- Demonstrates the **improvement of EKF over pure odometry**

### Comparison: Odometry vs. Kalman Filter SLAM

| Method | Position Error | Map Consistency |
|--------|----------------|-----------------|
| Pure Odometry | Accumulates unbounded drift | No map building |
| EKF-SLAM | Bounded, corrected by observations | Consistent landmark positions |

## Technologies

- **Python 3.7+** - Core implementation
- **NumPy** - Linear algebra and matrix operations
- **Matplotlib** - Visualization and animation
- **Pandas** - Data loading and manipulation
- **SciPy** - Signal processing utilities

## Algorithm Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Prediction | O(n²) | O(n²) |
| Update | O(n²) | O(n²) |
| Corner Detection | O(m²) | O(m) |

Where n = number of landmarks, m = number of LIDAR rays (61)

## Future Improvements

- [ ] Implement loop closure detection for large-scale mapping
- [ ] Add scan matching (ICP) for more robust localization
- [ ] Extend to 3D SLAM using RGB-D sensors
- [ ] Implement particle filter (FastSLAM) for comparison
- [ ] Add occupancy grid mapping alongside feature-based approach

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
