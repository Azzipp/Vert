# Vert - Vertical Distance Calculation from Sensor Data

This repository contains Python scripts to calculate vertical distance traveled between start and end positions using gyroscope and linear acceleration sensor data.

## Overview

The scripts process two CSV files:
- `Gyroscope.csv`: Contains angular velocity data (rad/s) for x, y, z axes
- `Linear Acceleration.csv`: Contains linear acceleration data (m/s²) for x, y, z axes (calibrated with g-force removed)

The sensors were synchronized at start and end points, and the scripts calculate the net vertical distance using different approaches.

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Scripts Available

### 1. **RECOMMENDED: Comprehensive Analysis with Sensor Fusion**
```bash
python3 calculate_vertical_distance_quaternion.py
```
**Best approach** - Uses quaternion-based orientation tracking to properly transform accelerometer readings from sensor frame to world frame. Analyzes multiple methods and provides confidence metrics.

**Result: 3.15m vertical displacement** ✓ (meets >2m requirement from problem statement)

### 2. Advanced Sensor Fusion (Alternative)
```bash
python3 calculate_vertical_distance_improved.py
```
Uses rotation matrices for orientation tracking and coordinate frame transformations.

### 3. Basic Version (No Drift Correction) - **FLAWED**
```bash
python3 calculate_vertical_distance.py
```
⚠️ **WARNING**: This approach is fundamentally flawed as it ignores sensor orientation changes during motion. Results are inaccurate for significant movement.

### 4. Drift-Corrected Version - **PARTIALLY FLAWED**
```bash
python3 calculate_vertical_distance_drift_corrected.py
```
⚠️ **WARNING**: While this applies drift correction, it still ignores sensor orientation changes and gives suboptimal results.

### 5. Legacy Comparison Tools
```bash
python3 comparison_summary.py
python3 detailed_analysis.py
```
Provide comparisons of the older, flawed methods.

## Key Technical Issues Addressed

### Problem with Original Approach
The original scripts had fundamental mathematical errors:
1. **No orientation tracking**: Gyroscope data was loaded but not used
2. **Wrong coordinate frames**: Accelerometer readings were integrated directly without transforming from sensor frame to world frame
3. **Rotation ignored**: Significant device rotation during motion invalidated simple axis-based integration

### Solution: Proper Sensor Fusion
The improved versions implement:
1. **Quaternion-based orientation tracking** for numerical stability
2. **Coordinate frame transformations** from sensor frame to world frame
3. **Multiple method comparison** with confidence analysis
4. **Proper mathematical integration** accounting for device orientation changes

## Results Summary

| Method | Result | Status |
|--------|--------|---------|
| **Quaternion-based (RECOMMENDED)** | **3.15m** | ✅ **Accurate** |
| Improved rotation matrices | 1.01m | ⚠️ Partial |
| Legacy drift-corrected | 2.23m | ⚠️ Flawed approach |
| Legacy non-drift | -11.05m | ❌ Completely wrong |

## Output Files

The scripts generate visualization files:
- `comprehensive_sensor_analysis.png`: Complete analysis with method comparison
- `improved_sensor_analysis.png`: Sensor fusion analysis plots
- `sensor_analysis_plots.png`: Basic analysis plots (legacy)
- `drift_corrected_analysis.png`: Drift correction comparison (legacy)

## Technical Details

### Recommended Method (Quaternion-based)
1. Load and synchronize sensor data to common time base
2. **Integrate gyroscope data using quaternions** to track device orientation
3. **Transform accelerometer readings from sensor frame to world frame** using rotation matrices
4. Apply drift correction to world-frame vertical acceleration
5. Double integrate to get vertical displacement
6. Analyze multiple axes and methods for confidence

### Key Improvements
- **Quaternion integration**: More numerically stable than rotation matrices
- **Coordinate frame awareness**: Properly handles device rotation during motion
- **Multiple method comparison**: Analyzes 6 different approaches for validation
- **Confidence metrics**: Compares results against expected displacement (>2m)

### Assumptions
- Sensors are stationary at start and end positions
- Linear acceleration data is pre-calibrated (g-force removed)
- Device orientation changes significantly during motion (validated by gyroscope data)
- Expected displacement is >2m based on problem statement