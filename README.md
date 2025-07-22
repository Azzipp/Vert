# Vert - Vertical Distance Calculation from Sensor Data

This repository contains Python scripts to calculate vertical distance traveled between start and end positions using gyroscope and linear acceleration sensor data.

## Overview

The scripts process two CSV files:
- `Gyroscope.csv`: Contains angular velocity data (rad/s) for x, y, z axes
- `Linear Acceleration.csv`: Contains linear acceleration data (m/s²) for x, y, z axes (calibrated with g-force removed)

The sensors were synchronized at start and end points, and the scripts calculate the net vertical distance by double integrating the acceleration data.

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Scripts Available

### 1. Basic Version (No Drift Correction)
```bash
python3 calculate_vertical_distance.py
```
Calculates vertical distance without any drift correction. Useful for understanding the raw sensor integration behavior.

### 2. Drift-Corrected Version (Recommended)
```bash
python3 calculate_vertical_distance_drift_corrected.py
```
Applies linear drift correction to eliminate velocity drift, assuming sensors are stationary at start and end.

### 3. Comparison Summary
```bash
python3 comparison_summary.py
```
Runs both methods and provides a detailed comparison of results.

### 4. Detailed Analysis
```bash
python3 detailed_analysis.py
```
Provides additional statistical analysis and validation of the calculation methods.

## Results Summary

Based on the provided sensor data:

**Without Drift Correction:**
- Vertical axis: x
- Net vertical distance: -11.05 m (-1105 cm)
- Shows significant drift effects

**With Drift Correction (Recommended):**
- Vertical axis: z
- Net vertical distance: +2.23 m (+223 cm)
- Eliminates velocity drift for more accurate results

## Output Files

The scripts generate several visualization files:
- `sensor_analysis_plots.png`: Basic analysis plots
- `drift_corrected_analysis.png`: Drift correction comparison
- `comparison_drift_correction.png`: Side-by-side comparison
- `detailed_analysis_plots.png`: Additional statistical plots

## Technical Details

### Method
1. Load and synchronize sensor data to common time base
2. Double integrate acceleration: acceleration → velocity → position
3. Apply drift correction (in corrected version) by linear detrending
4. Identify vertical axis as the one with largest displacement
5. Calculate net vertical distance

### Drift Correction
The drift-corrected version applies linear detrending to velocity, ensuring final velocity is zero since sensors are stationary at start and end. This eliminates the accumulation of integration errors that cause unrealistic displacement values.

### Assumptions
- Sensors are stationary at start and end positions
- Linear acceleration data is pre-calibrated (g-force removed)
- The axis with largest displacement represents vertical movement
- Linear drift model is appropriate for the ~20-second recording duration