#!/usr/bin/env python3
"""
Calculate vertical distance with basic drift correction.
This version applies linear detrending to eliminate velocity drift.
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

def apply_drift_correction(velocity, time):
    """
    Apply linear drift correction to velocity.
    Assumes sensor is still at start and end, so final velocity should be zero.
    """
    # Linear detrend: remove linear trend to make final velocity zero
    final_velocity = velocity[-1]
    total_time = time[-1] - time[0]
    
    # Create linear drift correction
    drift_rate = final_velocity / total_time
    drift_correction = drift_rate * (time - time[0])
    
    corrected_velocity = velocity - drift_correction
    
    return corrected_velocity, drift_correction

def calculate_position_with_drift_correction(accel_df):
    """
    Calculate position by double integrating acceleration with drift correction.
    Returns velocity and position for each axis.
    """
    time = accel_df['Time (s)'].values
    
    # Initialize results
    results = {}
    
    for axis in ['x', 'y', 'z']:
        accel_col = f'Linear Acceleration {axis} (m/s^2)'
        accel = accel_df[accel_col].values
        
        # First integration: acceleration -> velocity
        velocity_raw = integrate.cumulative_trapezoid(accel, time, initial=0)
        
        # Apply drift correction to velocity
        velocity_corrected, drift_correction = apply_drift_correction(velocity_raw, time)
        
        # Second integration: corrected velocity -> position
        position = integrate.cumulative_trapezoid(velocity_corrected, time, initial=0)
        
        results[axis] = {
            'acceleration': accel,
            'velocity_raw': velocity_raw,
            'velocity_corrected': velocity_corrected,
            'drift_correction': drift_correction,
            'position': position
        }
        
        print(f"{axis}-axis drift correction:")
        print(f"  Raw final velocity: {velocity_raw[-1]:.6f} m/s")
        print(f"  Corrected final velocity: {velocity_corrected[-1]:.6f} m/s")
        print(f"  Drift rate: {drift_correction[-1]/(time[-1]-time[0]):.6f} m/s²")
    
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
    
    print(f"\nMaximum absolute displacements:")
    for axis in ['x', 'y', 'z']:
        print(f"  {axis}: {max_displacement[axis]:.6f} m")
    
    print(f"Final net displacements:")
    for axis in ['x', 'y', 'z']:
        print(f"  {axis}: {final_displacement[axis]:.6f} m")
    
    print(f"Determined vertical axis: {vertical_axis}")
    
    return vertical_axis, final_displacement[vertical_axis]

def create_plots(time, gyro_df, accel_df, position_results, vertical_axis):
    """Create visualization plots for verification."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Sensor Data Analysis with Drift Correction', fontsize=16)
    
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
    
    # Plot 3: Raw vs Corrected Velocity (vertical axis only)
    ax3 = axes[0, 2]
    ax3.plot(time, position_results[vertical_axis]['velocity_raw'], 
             label='Raw velocity', alpha=0.7, color='red')
    ax3.plot(time, position_results[vertical_axis]['velocity_corrected'], 
             label='Drift corrected', alpha=0.7, color='blue')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title(f'{vertical_axis.upper()}-axis Velocity Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 4: Velocity (drift corrected)
    ax4 = axes[1, 0]
    for axis in ['x', 'y', 'z']:
        color = 'red' if axis == vertical_axis else None
        linewidth = 2 if axis == vertical_axis else 1
        ax4.plot(time, position_results[axis]['velocity_corrected'], 
                label=f'{axis}-axis', alpha=0.7, color=color, linewidth=linewidth)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_title('Drift-Corrected Velocity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 5: Position (double integrated acceleration)
    ax5 = axes[1, 1]
    for axis in ['x', 'y', 'z']:
        color = 'red' if axis == vertical_axis else None
        linewidth = 2 if axis == vertical_axis else 1
        ax5.plot(time, position_results[axis]['position'], 
                label=f'{axis}-axis', alpha=0.7, color=color, linewidth=linewidth)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Position (m)')
    ax5.set_title('Position (Drift-Corrected)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 6: Drift correction applied
    ax6 = axes[1, 2]
    for axis in ['x', 'y', 'z']:
        color = 'red' if axis == vertical_axis else None
        linewidth = 2 if axis == vertical_axis else 1
        ax6.plot(time, position_results[axis]['drift_correction'], 
                label=f'{axis}-axis', alpha=0.7, color=color, linewidth=linewidth)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Drift Correction (m/s)')
    ax6.set_title('Applied Drift Correction')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('drift_corrected_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main function to calculate vertical distance with drift correction."""
    print("=== Vertical Distance Calculation (With Drift Correction) ===\n")
    
    # Load data
    gyro_df, accel_df = load_sensor_data()
    print()
    
    # Synchronize data
    gyro_sync, accel_sync = synchronize_data(gyro_df, accel_df)
    print(f"Synchronized data points: {len(accel_sync)}")
    print()
    
    # Calculate position from acceleration with drift correction
    position_results, time = calculate_position_with_drift_correction(accel_sync)
    print("\nPosition calculation with drift correction completed.")
    print()
    
    # Determine vertical axis and calculate vertical distance
    vertical_axis, vertical_distance = determine_vertical_axis(position_results)
    print()
    
    # Create plots
    create_plots(time, gyro_sync, accel_sync, position_results, vertical_axis)
    
    # Compare with non-corrected version
    print("=" * 60)
    print("COMPARISON OF METHODS:")
    print("=" * 60)
    
    # Calculate non-corrected for comparison
    for axis in ['x', 'y', 'z']:
        accel = accel_sync[f'Linear Acceleration {axis} (m/s^2)'].values
        velocity_raw = integrate.cumulative_trapezoid(accel, time, initial=0)
        position_raw = integrate.cumulative_trapezoid(velocity_raw, time, initial=0)
        position_corrected = position_results[axis]['position'][-1]
        
        print(f"{axis}-axis:")
        print(f"  Without drift correction: {position_raw[-1]:.6f} m")
        print(f"  With drift correction:    {position_corrected:.6f} m")
        print(f"  Difference:               {abs(position_raw[-1] - position_corrected):.6f} m")
        print()
    
    # Final results
    print("=" * 60)
    print("FINAL RESULTS (DRIFT CORRECTED):")
    print(f"Vertical axis: {vertical_axis}")
    print(f"Net vertical distance: {vertical_distance:.6f} m")
    print(f"Net vertical distance: {vertical_distance * 100:.2f} cm")
    print("=" * 60)
    
    # Additional information
    print("\nMovement summary:")
    print(f"Recording duration: {time[-1] - time[0]:.2f} seconds")
    print(f"Data points: {len(time)}")
    print(f"Average sampling rate: {len(time) / (time[-1] - time[0]):.1f} Hz")
    
    return vertical_distance

if __name__ == "__main__":
    vertical_distance = main()