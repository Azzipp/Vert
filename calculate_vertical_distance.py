#!/usr/bin/env python3
"""
Calculate vertical distance from gyroscope and linear acceleration sensor data.
Version without drift correction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d

def load_sensor_data():
    """Load gyroscope and linear acceleration data from CSV files."""
    # Load gyroscope data
    gyro_df = pd.read_csv('Gyroscope.csv')
    print("Gyroscope data shape:", gyro_df.shape)
    print("Gyroscope time range: {:.3f}s to {:.3f}s".format(
        gyro_df['Time (s)'].min(), gyro_df['Time (s)'].max()))
    
    # Load linear acceleration data
    accel_df = pd.read_csv('Linear Acceleration.csv')
    print("Acceleration data shape:", accel_df.shape)
    print("Acceleration time range: {:.3f}s to {:.3f}s".format(
        accel_df['Time (s)'].min(), accel_df['Time (s)'].max()))
    
    return gyro_df, accel_df

def synchronize_data(gyro_df, accel_df):
    """
    Synchronize the two datasets to a common time base.
    Use the overlapping time range and interpolate to the acceleration timestamps.
    """
    # Find overlapping time range
    start_time = max(gyro_df['Time (s)'].min(), accel_df['Time (s)'].min())
    end_time = min(gyro_df['Time (s)'].max(), accel_df['Time (s)'].max())
    print(f"Overlapping time range: {start_time:.3f}s to {end_time:.3f}s")
    
    # Filter acceleration data to overlapping range
    accel_sync = accel_df[
        (accel_df['Time (s)'] >= start_time) & 
        (accel_df['Time (s)'] <= end_time)
    ].copy().reset_index(drop=True)
    
    # Filter gyroscope data to overlapping range
    gyro_sync = gyro_df[
        (gyro_df['Time (s)'] >= start_time) & 
        (gyro_df['Time (s)'] <= end_time)
    ].copy().reset_index(drop=True)
    
    # Interpolate gyroscope data to acceleration timestamps
    time_accel = accel_sync['Time (s)'].values
    time_gyro = gyro_sync['Time (s)'].values
    
    gyro_interpolated = pd.DataFrame({'Time (s)': time_accel})
    
    for col in ['Gyroscope x (rad/s)', 'Gyroscope y (rad/s)', 'Gyroscope z (rad/s)']:
        interp_func = interp1d(time_gyro, gyro_sync[col].values, 
                              kind='linear', bounds_error=False, fill_value='extrapolate')
        gyro_interpolated[col] = interp_func(time_accel)
    
    return gyro_interpolated, accel_sync

def calculate_position_from_acceleration(accel_df):
    """
    Calculate position by double integrating acceleration.
    Returns velocity and position for each axis.
    
    NOTE: This simple approach does NOT account for sensor orientation changes
    and will give incorrect results for significant motion. For accurate results,
    use the improved versions with gyroscope-based orientation tracking.
    """
    time = accel_df['Time (s)'].values
    
    # Initialize results
    results = {}
    
    for axis in ['x', 'y', 'z']:
        accel_col = f'Linear Acceleration {axis} (m/s^2)'
        accel = accel_df[accel_col].values
        
        # First integration: acceleration -> velocity
        # Use scipy's cumulative trapezoidal integration (more accurate)
        velocity = integrate.cumulative_trapezoid(accel, time, initial=0)
        
        # Second integration: velocity -> position  
        position = integrate.cumulative_trapezoid(velocity, time, initial=0)
        
        results[axis] = {
            'acceleration': accel,
            'velocity': velocity,
            'position': position
        }
    
    return results, time

def determine_vertical_axis(position_results):
    """
    Determine which axis represents vertical movement.
    Use the axis with the largest absolute displacement.
    """
    max_displacement = {}
    final_displacement = {}
    
    for axis in ['x', 'y', 'z']:
        position = position_results[axis]['position']
        max_displacement[axis] = np.max(np.abs(position))
        final_displacement[axis] = position[-1] - position[0]
    
    # Find axis with largest absolute displacement
    vertical_axis = max(max_displacement.keys(), key=lambda k: max_displacement[k])
    
    print(f"Maximum absolute displacements:")
    for axis in ['x', 'y', 'z']:
        print(f"  {axis}: {max_displacement[axis]:.6f} m")
    
    print(f"Final net displacements:")
    for axis in ['x', 'y', 'z']:
        print(f"  {axis}: {final_displacement[axis]:.6f} m")
    
    print(f"Determined vertical axis: {vertical_axis}")
    
    return vertical_axis, final_displacement[vertical_axis]

def create_plots(time, gyro_df, accel_df, position_results, vertical_axis):
    """Create visualization plots for verification."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Sensor Data Analysis', fontsize=16)
    
    # Plot 1: Raw acceleration data
    ax1 = axes[0, 0]
    for axis in ['x', 'y', 'z']:
        accel_col = f'Linear Acceleration {axis} (m/s^2)'
        ax1.plot(time, accel_df[accel_col], label=f'{axis}-axis', alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_title('Linear Acceleration Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gyroscope data
    ax2 = axes[0, 1]
    for axis in ['x', 'y', 'z']:
        gyro_col = f'Gyroscope {axis} (rad/s)'
        ax2.plot(time, gyro_df[gyro_col], label=f'{axis}-axis', alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.set_title('Gyroscope Data')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Velocity (integrated acceleration)
    ax3 = axes[1, 0]
    for axis in ['x', 'y', 'z']:
        color = 'red' if axis == vertical_axis else None
        linewidth = 2 if axis == vertical_axis else 1
        ax3.plot(time, position_results[axis]['velocity'], 
                label=f'{axis}-axis', alpha=0.7, color=color, linewidth=linewidth)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity (Integrated Acceleration)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Position (double integrated acceleration)
    ax4 = axes[1, 1]
    for axis in ['x', 'y', 'z']:
        color = 'red' if axis == vertical_axis else None
        linewidth = 2 if axis == vertical_axis else 1
        ax4.plot(time, position_results[axis]['position'], 
                label=f'{axis}-axis', alpha=0.7, color=color, linewidth=linewidth)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position (m)')
    ax4.set_title('Position (Double Integrated Acceleration)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sensor_analysis_plots.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main function to calculate vertical distance."""
    print("=== Vertical Distance Calculation (No Drift Correction) ===")
    print("WARNING: This simple method does NOT account for sensor orientation")
    print("changes and will give inaccurate results for significant motion.")
    print("For accurate results, use calculate_vertical_distance_quaternion.py\n")
    
    # Load data
    gyro_df, accel_df = load_sensor_data()
    print()
    
    # Synchronize data
    gyro_sync, accel_sync = synchronize_data(gyro_df, accel_df)
    print(f"Synchronized data points: {len(accel_sync)}")
    print()
    
    # Calculate position from acceleration
    position_results, time = calculate_position_from_acceleration(accel_sync)
    print("Position calculation completed.")
    print()
    
    # Determine vertical axis and calculate vertical distance
    vertical_axis, vertical_distance = determine_vertical_axis(position_results)
    print()
    
    # Create plots
    create_plots(time, gyro_sync, accel_sync, position_results, vertical_axis)
    
    # Final results
    print("=" * 50)
    print("RESULTS (SIMPLE METHOD - LIKELY INACCURATE):")
    print(f"Vertical axis: {vertical_axis}")
    print(f"Net vertical distance: {vertical_distance:.6f} m")
    print(f"Net vertical distance: {vertical_distance * 100:.2f} cm")
    print("=" * 50)
    print()
    print("IMPORTANT: This result is likely inaccurate because it ignores")
    print("sensor orientation changes during motion. The gyroscope data")
    print("shows significant rotation, which invalidates this simple approach.")
    print()
    print("For accurate results, run:")
    print("  python3 calculate_vertical_distance_quaternion.py")
    print("=" * 50)
    
    # Additional information
    print("\nMovement summary:")
    print(f"Recording duration: {time[-1] - time[0]:.2f} seconds")
    print(f"Data points: {len(time)}")
    print(f"Average sampling rate: {len(time) / (time[-1] - time[0]):.1f} Hz")
    
    return vertical_distance

if __name__ == "__main__":
    vertical_distance = main()