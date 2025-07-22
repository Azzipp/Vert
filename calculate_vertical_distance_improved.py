#!/usr/bin/env python3
"""
Improved vertical distance calculation using proper sensor fusion.
This version uses gyroscope data for orientation tracking and transforms
accelerometer readings from sensor frame to world frame before integration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

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

def integrate_orientation(gyro_df):
    """
    Integrate gyroscope data to estimate orientation over time.
    Returns rotation matrices for each time step.
    """
    time = gyro_df['Time (s)'].values
    dt = np.diff(time)
    
    # Extract angular velocities
    omega_x = gyro_df['Gyroscope x (rad/s)'].values
    omega_y = gyro_df['Gyroscope y (rad/s)'].values  
    omega_z = gyro_df['Gyroscope z (rad/s)'].values
    
    # Initialize rotation matrices (start with identity - sensor initially aligned with world frame)
    rotations = np.zeros((len(time), 3, 3))
    rotations[0] = np.eye(3)
    
    # Integrate angular velocities to get orientation changes
    for i in range(1, len(time)):
        # Angular velocity vector at current time step
        omega = np.array([omega_x[i-1], omega_y[i-1], omega_z[i-1]])
        
        # Rotation angle during this time step
        angle = np.linalg.norm(omega) * dt[i-1]
        
        if angle > 1e-8:  # Avoid division by zero
            # Rotation axis (normalized angular velocity vector)
            axis = omega / np.linalg.norm(omega)
            
            # Create rotation matrix for this step using Rodrigues' formula
            # R = I + sin(θ)*K + (1-cos(θ))*K²
            # where K is the skew-symmetric matrix of the axis
            K = np.array([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]])
            
            R_step = (np.eye(3) + 
                     np.sin(angle) * K + 
                     (1 - np.cos(angle)) * np.dot(K, K))
        else:
            R_step = np.eye(3)
        
        # Update cumulative rotation
        rotations[i] = np.dot(rotations[i-1], R_step)
    
    return rotations

def transform_accelerations_to_world_frame(accel_df, rotations):
    """
    Transform accelerometer readings from sensor frame to world frame.
    """
    # Extract accelerations in sensor frame
    accel_sensor = np.column_stack([
        accel_df['Linear Acceleration x (m/s^2)'].values,
        accel_df['Linear Acceleration y (m/s^2)'].values,
        accel_df['Linear Acceleration z (m/s^2)'].values
    ])
    
    # Transform to world frame
    accel_world = np.zeros_like(accel_sensor)
    for i in range(len(accel_sensor)):
        # Transform acceleration vector to world frame
        accel_world[i] = np.dot(rotations[i], accel_sensor[i])
    
    return accel_world

def calculate_vertical_displacement(accel_world, time, apply_drift_correction=True):
    """
    Calculate vertical displacement from world-frame accelerations.
    """
    # Vertical acceleration is the z-component in world frame
    accel_vertical = accel_world[:, 2]  # z-axis is vertical
    
    # First integration: acceleration -> velocity
    velocity = integrate.cumulative_trapezoid(accel_vertical, time, initial=0)
    
    # Apply drift correction if requested
    if apply_drift_correction:
        # Linear detrending - assume stationary at start and end
        final_velocity = velocity[-1]
        total_time = time[-1] - time[0]
        drift_rate = final_velocity / total_time
        drift_correction = drift_rate * (time - time[0])
        velocity_corrected = velocity - drift_correction
        
        print(f"Vertical velocity drift correction:")
        print(f"  Raw final velocity: {final_velocity:.6f} m/s")
        print(f"  Drift rate: {drift_rate:.6f} m/s²")
        print(f"  Corrected final velocity: {velocity_corrected[-1]:.6f} m/s")
        
        velocity = velocity_corrected
    
    # Second integration: velocity -> position
    position = integrate.cumulative_trapezoid(velocity, time, initial=0)
    
    # Net vertical displacement
    vertical_displacement = position[-1] - position[0]
    
    return velocity, position, vertical_displacement, accel_vertical

def create_improved_plots(time, gyro_df, accel_df, rotations, accel_world, 
                         velocity, position, vertical_displacement):
    """Create comprehensive visualization plots."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Improved Sensor Data Analysis with Orientation Tracking', fontsize=16)
    
    # Plot 1: Raw acceleration data (sensor frame)
    ax1 = axes[0, 0]
    for i, axis in enumerate(['x', 'y', 'z']):
        accel_col = f'Linear Acceleration {axis} (m/s^2)'
        ax1.plot(time, accel_df[accel_col], label=f'{axis}-axis', alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_title('Raw Acceleration (Sensor Frame)')
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
    
    # Plot 3: Orientation angles (extracted from rotation matrices)
    ax3 = axes[0, 2]
    # Convert rotation matrices to Euler angles for visualization
    euler_angles = np.zeros((len(rotations), 3))
    for i, R in enumerate(rotations):
        rot = Rotation.from_matrix(R)
        euler_angles[i] = rot.as_euler('xyz', degrees=True)
    
    for i, axis in enumerate(['Roll', 'Pitch', 'Yaw']):
        ax3.plot(time, euler_angles[:, i], label=axis, alpha=0.7)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angle (degrees)')
    ax3.set_title('Estimated Orientation (Euler Angles)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Transformed acceleration (world frame)
    ax4 = axes[1, 0]
    for i, axis in enumerate(['X', 'Y', 'Z']):
        color = 'red' if i == 2 else None  # Highlight Z (vertical)
        linewidth = 2 if i == 2 else 1
        ax4.plot(time, accel_world[:, i], label=f'{axis}-axis', alpha=0.7, 
                color=color, linewidth=linewidth)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Acceleration (m/s²)')
    ax4.set_title('Transformed Acceleration (World Frame)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Vertical velocity
    ax5 = axes[1, 1]
    ax5.plot(time, velocity, 'b-', linewidth=2, alpha=0.8)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Velocity (m/s)')
    ax5.set_title('Vertical Velocity (Drift Corrected)')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 6: Vertical position
    ax6 = axes[1, 2]
    ax6.plot(time, position, 'r-', linewidth=2, alpha=0.8)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Position (m)')
    ax6.set_title('Vertical Position')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax6.axhline(y=vertical_displacement, color='g', linestyle='--', alpha=0.7, 
               label=f'Final: {vertical_displacement:.3f}m')
    ax6.legend()
    
    # Plot 7: Comparison of approaches (sensor frame vs world frame)
    ax7 = axes[2, 0]
    # Calculate simple z-axis integration for comparison
    accel_z_sensor = accel_df['Linear Acceleration z (m/s^2)'].values
    vel_simple = integrate.cumulative_trapezoid(accel_z_sensor, time, initial=0)
    pos_simple = integrate.cumulative_trapezoid(vel_simple, time, initial=0)
    
    ax7.plot(time, pos_simple, label='Sensor Z-axis only', alpha=0.7, linestyle='--')
    ax7.plot(time, position, label='World frame vertical', linewidth=2, alpha=0.8)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Position (m)')
    ax7.set_title('Comparison: Sensor vs World Frame')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 8: Rotation matrix determinants (should be close to 1)
    ax8 = axes[2, 1]
    determinants = [np.linalg.det(R) for R in rotations]
    ax8.plot(time, determinants, 'g-', alpha=0.8)
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Determinant')
    ax8.set_title('Rotation Matrix Determinants')
    ax8.grid(True, alpha=0.3)
    ax8.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Ideal (1.0)')
    ax8.legend()
    
    # Plot 9: Summary statistics
    ax9 = axes[2, 2]
    ax9.axis('off')
    summary_text = f"""
IMPROVED CALCULATION RESULTS

Duration: {time[-1] - time[0]:.2f} seconds
Sampling rate: {len(time)/(time[-1] - time[0]):.1f} Hz

World Frame Accelerations:
  X (lateral): {accel_world[:, 0].std():.3f} m/s² RMS
  Y (lateral): {accel_world[:, 1].std():.3f} m/s² RMS  
  Z (vertical): {accel_world[:, 2].std():.3f} m/s² RMS

Orientation Changes:
  Roll range: {euler_angles[:, 0].max() - euler_angles[:, 0].min():.1f}°
  Pitch range: {euler_angles[:, 1].max() - euler_angles[:, 1].min():.1f}°
  Yaw range: {euler_angles[:, 2].max() - euler_angles[:, 2].min():.1f}°

Final Results:
  Net vertical displacement: {vertical_displacement:.3f} m
  Net vertical displacement: {vertical_displacement*100:.1f} cm

Comparison:
  Simple sensor Z-axis: {pos_simple[-1]:.3f} m
  World frame vertical: {vertical_displacement:.3f} m
  Improvement: {abs(vertical_displacement - pos_simple[-1]):.3f} m
"""
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('improved_sensor_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main function for improved vertical distance calculation."""
    print("=== Improved Vertical Distance Calculation ===")
    print("Using sensor fusion with gyroscope orientation tracking\n")
    
    # Load data
    gyro_df, accel_df = load_sensor_data()
    print()
    
    # Synchronize data
    gyro_sync, accel_sync = synchronize_data(gyro_df, accel_df)
    print(f"Synchronized data points: {len(accel_sync)}")
    print()
    
    # Integrate gyroscope data to get orientation over time
    print("Integrating gyroscope data for orientation tracking...")
    rotations = integrate_orientation(gyro_sync)
    print(f"Computed {len(rotations)} rotation matrices")
    
    # Transform accelerations to world frame
    print("Transforming accelerations to world coordinate frame...")
    accel_world = transform_accelerations_to_world_frame(accel_sync, rotations)
    
    # Calculate vertical displacement
    print("Calculating vertical displacement...")
    time = accel_sync['Time (s)'].values
    velocity, position, vertical_displacement, accel_vertical = calculate_vertical_displacement(
        accel_world, time, apply_drift_correction=True)
    print()
    
    # Create comprehensive plots
    create_improved_plots(time, gyro_sync, accel_sync, rotations, accel_world, 
                         velocity, position, vertical_displacement)
    
    # Final results
    print("=" * 60)
    print("IMPROVED CALCULATION RESULTS:")
    print(f"Net vertical displacement: {vertical_displacement:.6f} m")
    print(f"Net vertical displacement: {vertical_displacement * 100:.2f} cm")
    print("=" * 60)
    
    # Compare with simple approaches for context
    print("\nComparison with simpler methods:")
    
    # Simple Z-axis integration (no orientation correction)
    accel_z_simple = accel_sync['Linear Acceleration z (m/s^2)'].values
    vel_simple = integrate.cumulative_trapezoid(accel_z_simple, time, initial=0)
    pos_simple = integrate.cumulative_trapezoid(vel_simple, time, initial=0)
    print(f"Simple sensor Z-axis integration: {pos_simple[-1]:.6f} m")
    
    # Check if improvement is significant
    improvement = abs(vertical_displacement - pos_simple[-1])
    print(f"Difference from simple method: {improvement:.6f} m")
    
    return vertical_displacement

if __name__ == "__main__":
    vertical_distance = main()