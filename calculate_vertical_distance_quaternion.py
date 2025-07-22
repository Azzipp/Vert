#!/usr/bin/env python3
"""
Advanced vertical distance calculation using quaternion-based orientation tracking.
This version uses more robust numerical methods and analyzes different coordinate frames.
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

def quaternion_multiply(q1, q2):
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ])

def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to rotation matrix."""
    w, x, y, z = q
    
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm > 0:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

def integrate_orientation_quaternion(gyro_df):
    """
    Integrate gyroscope data using quaternions for stable orientation tracking.
    """
    time = gyro_df['Time (s)'].values
    dt = np.diff(time)
    
    # Extract angular velocities
    omega_x = gyro_df['Gyroscope x (rad/s)'].values
    omega_y = gyro_df['Gyroscope y (rad/s)'].values  
    omega_z = gyro_df['Gyroscope z (rad/s)'].values
    
    # Initialize quaternions (start with identity quaternion)
    quaternions = np.zeros((len(time), 4))
    quaternions[0] = [1, 0, 0, 0]  # [w, x, y, z]
    
    # Integrate angular velocities using quaternions
    for i in range(1, len(time)):
        # Angular velocity vector at current time step
        omega = np.array([omega_x[i-1], omega_y[i-1], omega_z[i-1]])
        
        # Angular velocity magnitude
        omega_magnitude = np.linalg.norm(omega)
        
        if omega_magnitude > 1e-8:
            # Rotation quaternion for this time step
            # q = [cos(θ/2), sin(θ/2)*axis]
            angle = omega_magnitude * dt[i-1]
            axis = omega / omega_magnitude
            
            q_delta = np.array([
                np.cos(angle/2),                    # w
                np.sin(angle/2) * axis[0],          # x
                np.sin(angle/2) * axis[1],          # y
                np.sin(angle/2) * axis[2]           # z
            ])
        else:
            q_delta = np.array([1, 0, 0, 0])  # Identity quaternion
        
        # Update cumulative quaternion
        quaternions[i] = quaternion_multiply(quaternions[i-1], q_delta)
        
        # Normalize to prevent drift
        quaternions[i] = quaternions[i] / np.linalg.norm(quaternions[i])
    
    # Convert quaternions to rotation matrices
    rotations = np.array([quaternion_to_rotation_matrix(q) for q in quaternions])
    
    return rotations, quaternions

def analyze_multiple_vertical_axes(accel_df, rotations, time):
    """
    Try different assumptions about which axis is vertical and compare results.
    """
    # Extract accelerations in sensor frame
    accel_sensor = np.column_stack([
        accel_df['Linear Acceleration x (m/s^2)'].values,
        accel_df['Linear Acceleration y (m/s^2)'].values,
        accel_df['Linear Acceleration z (m/s^2)'].values
    ])
    
    results = {}
    
    # Method 1: Direct sensor axes (no orientation correction)
    for i, axis_name in enumerate(['x', 'y', 'z']):
        accel_axis = accel_sensor[:, i]
        velocity = integrate.cumulative_trapezoid(accel_axis, time, initial=0)
        
        # Apply drift correction
        final_velocity = velocity[-1]
        total_time = time[-1] - time[0]
        drift_rate = final_velocity / total_time
        drift_correction = drift_rate * (time - time[0])
        velocity_corrected = velocity - drift_correction
        
        position = integrate.cumulative_trapezoid(velocity_corrected, time, initial=0)
        
        results[f'sensor_{axis_name}'] = {
            'displacement': position[-1],
            'max_displacement': np.max(np.abs(position)),
            'velocity': velocity_corrected,
            'position': position,
            'acceleration': accel_axis
        }
    
    # Method 2: World frame axes after orientation correction
    accel_world = np.zeros_like(accel_sensor)
    for i in range(len(accel_sensor)):
        accel_world[i] = np.dot(rotations[i], accel_sensor[i])
    
    for i, axis_name in enumerate(['X', 'Y', 'Z']):
        accel_axis = accel_world[:, i]
        velocity = integrate.cumulative_trapezoid(accel_axis, time, initial=0)
        
        # Apply drift correction
        final_velocity = velocity[-1]
        total_time = time[-1] - time[0]
        drift_rate = final_velocity / total_time
        drift_correction = drift_rate * (time - time[0])
        velocity_corrected = velocity - drift_correction
        
        position = integrate.cumulative_trapezoid(velocity_corrected, time, initial=0)
        
        results[f'world_{axis_name}'] = {
            'displacement': position[-1],
            'max_displacement': np.max(np.abs(position)),
            'velocity': velocity_corrected,
            'position': position,
            'acceleration': accel_axis
        }
    
    return results, accel_world

def create_comprehensive_analysis_plots(time, gyro_df, accel_df, rotations, 
                                      quaternions, results, accel_world):
    """Create comprehensive visualization and analysis plots."""
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle('Comprehensive Sensor Analysis - Multiple Methods Comparison', fontsize=16)
    
    # Plot 1: Raw acceleration data
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
    
    # Plot 3: Quaternion components
    ax3 = axes[0, 2]
    for i, component in enumerate(['w', 'x', 'y', 'z']):
        ax3.plot(time, quaternions[:, i], label=component, alpha=0.7)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Quaternion Component')
    ax3.set_title('Orientation Quaternions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: World frame accelerations
    ax4 = axes[1, 0]
    for i, axis in enumerate(['X', 'Y', 'Z']):
        ax4.plot(time, accel_world[:, i], label=f'{axis}-axis', alpha=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Acceleration (m/s²)')
    ax4.set_title('World Frame Accelerations')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Sensor frame position comparison
    ax5 = axes[1, 1]
    for axis in ['x', 'y', 'z']:
        ax5.plot(time, results[f'sensor_{axis}']['position'], 
                label=f'Sensor {axis}', alpha=0.7)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Position (m)')
    ax5.set_title('Position - Sensor Frame')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 6: World frame position comparison
    ax6 = axes[1, 2]
    for axis in ['X', 'Y', 'Z']:
        ax6.plot(time, results[f'world_{axis}']['position'], 
                label=f'World {axis}', alpha=0.7)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Position (m)')
    ax6.set_title('Position - World Frame')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 7: Displacement comparison
    ax7 = axes[2, 0]
    methods = list(results.keys())
    displacements = [results[method]['displacement'] for method in methods]
    max_displacements = [results[method]['max_displacement'] for method in methods]
    
    x_pos = np.arange(len(methods))
    ax7.bar(x_pos - 0.2, displacements, 0.4, label='Final Displacement', alpha=0.7)
    ax7.bar(x_pos + 0.2, max_displacements, 0.4, label='Max |Displacement|', alpha=0.7)
    ax7.set_xlabel('Method')
    ax7.set_ylabel('Displacement (m)')
    ax7.set_title('Displacement Comparison')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(methods, rotation=45)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Best candidates based on maximum displacement
    ax8 = axes[2, 1]
    # Find the method with largest absolute displacement
    best_method = max(results.keys(), key=lambda k: results[k]['max_displacement'])
    ax8.plot(time, results[best_method]['position'], 'r-', linewidth=2, 
            label=f'Best: {best_method}')
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Position (m)')
    ax8.set_title(f'Best Candidate: {best_method}')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 9: Acceleration magnitude comparison
    ax9 = axes[2, 2]
    # Calculate acceleration magnitudes
    accel_mag_sensor = np.linalg.norm(np.column_stack([
        accel_df['Linear Acceleration x (m/s^2)'],
        accel_df['Linear Acceleration y (m/s^2)'],
        accel_df['Linear Acceleration z (m/s^2)']
    ]), axis=1)
    accel_mag_world = np.linalg.norm(accel_world, axis=1)
    
    ax9.plot(time, accel_mag_sensor, label='Sensor Frame', alpha=0.7)
    ax9.plot(time, accel_mag_world, label='World Frame', alpha=0.7)
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Acceleration Magnitude (m/s²)')
    ax9.set_title('Acceleration Magnitude')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # Plot 10: Summary statistics
    ax10 = axes[3, 0]
    ax10.axis('off')
    summary_text = f"""
COMPREHENSIVE ANALYSIS RESULTS

Data Summary:
  Duration: {time[-1] - time[0]:.2f} seconds
  Sampling rate: {len(time)/(time[-1] - time[0]):.1f} Hz
  Data points: {len(time)}

Displacement Results:
"""
    for method, result in results.items():
        summary_text += f"  {method}: {result['displacement']:.3f} m\n"
    
    summary_text += f"""
Best Candidate: {best_method}
  Final displacement: {results[best_method]['displacement']:.3f} m
  Max |displacement|: {results[best_method]['max_displacement']:.3f} m
"""
    
    ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot 11: Method confidence analysis
    ax11 = axes[3, 1]
    # Analyze which methods give consistent results
    sensor_displacements = [results[f'sensor_{axis}']['displacement'] 
                           for axis in ['x', 'y', 'z']]
    world_displacements = [results[f'world_{axis}']['displacement'] 
                          for axis in ['X', 'Y', 'Z']]
    
    ax11.plot(['x', 'y', 'z'], sensor_displacements, 'bo-', label='Sensor Frame', markersize=8)
    ax11.plot(['X', 'Y', 'Z'], world_displacements, 'ro-', label='World Frame', markersize=8)
    ax11.set_xlabel('Axis')
    ax11.set_ylabel('Final Displacement (m)')
    ax11.set_title('Frame Comparison')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    ax11.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 12: Confidence metrics
    ax12 = axes[3, 2]
    ax12.axis('off')
    
    # Calculate some confidence metrics
    expected_min = 2.0  # From problem statement: "more than 2m"
    
    confidence_text = f"""
CONFIDENCE ANALYSIS

Expected: > 2.0 m descent
(Based on problem statement)

Methods meeting expectation:
"""
    
    meeting_expectation = []
    for method, result in results.items():
        displacement = abs(result['displacement'])
        if displacement >= expected_min:
            meeting_expectation.append(method)
            confidence_text += f"  ✓ {method}: {result['displacement']:.3f} m\n"
        else:
            confidence_text += f"  ✗ {method}: {result['displacement']:.3f} m\n"
    
    if meeting_expectation:
        confidence_text += f"\nRECOMMENDED: {meeting_expectation[0]}"
        recommended_displacement = results[meeting_expectation[0]]['displacement']
        confidence_text += f"\nFinal answer: {recommended_displacement:.3f} m"
        confidence_text += f"\nFinal answer: {recommended_displacement*100:.1f} cm"
    else:
        # Find closest to expectation
        closest_method = min(results.keys(), 
                           key=lambda k: abs(abs(results[k]['displacement']) - expected_min))
        confidence_text += f"\nCLOSEST TO EXPECTATION: {closest_method}"
        confidence_text += f"\nValue: {results[closest_method]['displacement']:.3f} m"
    
    ax12.text(0.05, 0.95, confidence_text, transform=ax12.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('comprehensive_sensor_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return best_method

def main():
    """Main function for comprehensive vertical distance analysis."""
    print("=== Comprehensive Vertical Distance Analysis ===")
    print("Using quaternion-based orientation tracking and multiple methods\n")
    
    # Load data
    gyro_df, accel_df = load_sensor_data()
    print()
    
    # Synchronize data
    gyro_sync, accel_sync = synchronize_data(gyro_df, accel_df)
    print(f"Synchronized data points: {len(accel_sync)}")
    print()
    
    # Integrate gyroscope data using quaternions
    print("Integrating gyroscope data using quaternions...")
    rotations, quaternions = integrate_orientation_quaternion(gyro_sync)
    print(f"Computed {len(rotations)} rotation matrices from quaternions")
    print()
    
    # Analyze multiple vertical axes approaches
    print("Analyzing multiple approaches for vertical displacement...")
    time = accel_sync['Time (s)'].values
    results, accel_world = analyze_multiple_vertical_axes(accel_sync, rotations, time)
    print()
    
    # Print results summary
    print("=" * 70)
    print("DISPLACEMENT RESULTS SUMMARY:")
    print("=" * 70)
    
    for method, result in results.items():
        print(f"{method:12s}: {result['displacement']:8.3f} m "
              f"(max |disp|: {result['max_displacement']:6.3f} m)")
    
    print()
    
    # Create comprehensive plots and analysis
    best_method = create_comprehensive_analysis_plots(
        time, gyro_sync, accel_sync, rotations, quaternions, results, accel_world)
    
    # Final recommendation
    print("=" * 70)
    print("FINAL ANALYSIS:")
    print("=" * 70)
    
    # Problem statement says "more than 2m descent"
    expected_min = 2.0
    meeting_expectation = []
    
    for method, result in results.items():
        displacement = abs(result['displacement'])
        if displacement >= expected_min:
            meeting_expectation.append((method, result['displacement']))
    
    if meeting_expectation:
        # Sort by displacement magnitude
        meeting_expectation.sort(key=lambda x: abs(x[1]), reverse=True)
        recommended_method, recommended_displacement = meeting_expectation[0]
        
        print(f"RECOMMENDED METHOD: {recommended_method}")
        print(f"Vertical displacement: {recommended_displacement:.6f} m")
        print(f"Vertical displacement: {recommended_displacement * 100:.2f} cm")
        print(f"Direction: {'downward' if recommended_displacement < 0 else 'upward'}")
        
        if len(meeting_expectation) > 1:
            print(f"\nOther methods meeting expectation (>= {expected_min}m):")
            for method, disp in meeting_expectation[1:]:
                print(f"  {method}: {disp:.3f} m")
    else:
        # Find the largest displacement
        best_method = max(results.keys(), key=lambda k: abs(results[k]['displacement']))
        best_displacement = results[best_method]['displacement']
        
        print(f"WARNING: No method meets expected displacement (>= {expected_min}m)")
        print(f"BEST AVAILABLE: {best_method}")
        print(f"Vertical displacement: {best_displacement:.6f} m")
        print(f"Vertical displacement: {best_displacement * 100:.2f} cm")
        
        recommended_displacement = best_displacement
    
    print("=" * 70)
    
    return recommended_displacement

if __name__ == "__main__":
    vertical_distance = main()