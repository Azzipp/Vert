# Vert - Vertical Distance Calculation from Sensor Data

This repository contains a Python script to calculate vertical distance traveled between start and end positions using gyroscope and linear acceleration sensor data.

## Overview

The script processes two CSV files:
- `Gyroscope.csv`: Contains angular velocity data (rad/s) for x, y, z axes
- `Linear Acceleration.csv`: Contains linear acceleration data (m/s²) for x, y, z axes (calibrated with g-force removed)

The sensors were synchronized at start and end points, and the script calculates the net vertical distance by double integrating the acceleration data.

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Simply run the calculation script:

```bash
python3 calculate_vertical_distance.py
```

## Output

The script will:
1. Load and synchronize the sensor data
2. Calculate position by double integrating acceleration
3. Determine which axis represents vertical movement (axis with largest displacement)
4. Generate visualization plots saved as `sensor_analysis_plots.png`
5. Display the net vertical distance

## Results

The script outputs:
- Vertical axis identification
- Net vertical distance in meters and centimeters
- Visualization plots showing:
  - Raw acceleration data
  - Gyroscope data
  - Calculated velocity (integrated acceleration)
  - Calculated position (double integrated acceleration)

## Note

This version does not include drift correction. The calculation assumes:
- Sensors are stationary at start and end
- Linear acceleration data is already calibrated (g-force removed)
- The largest displacement axis represents vertical movement