#!/usr/bin/env python3
"""
Summary script to compare vertical distance calculations with and without drift correction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d

def load_and_sync_data():
    """Load and synchronize sensor data."""
    # Load data
    gyro_df = pd.read_csv('Gyroscope.csv')
    accel_df = pd.read_csv('Linear Acceleration.csv')
    
    # Find overlapping time range
    start_time = max(gyro_df['Time (s)'].min(), accel_df['Time (s)'].min())
    end_time = min(gyro_df['Time (s)'].max(), accel_df['Time (s)'].max())
    
    # Filter to overlapping range
    accel_sync = accel_df[
        (accel_df['Time (s)'] >= start_time) & 
        (accel_df['Time (s)'] <= end_time)
    ].copy().reset_index(drop=True)
    
    return accel_sync

def calculate_without_drift_correction(accel_df):
    """Calculate position without drift correction."""
    time = accel_df['Time (s)'].values
    results = {}
    
    for axis in ['x', 'y', 'z']:
        accel_col = f'Linear Acceleration {axis} (m/s^2)'
        accel = accel_df[accel_col].values
        
        velocity = integrate.cumulative_trapezoid(accel, time, initial=0)
        position = integrate.cumulative_trapezoid(velocity, time, initial=0)
        
        results[axis] = {
            'velocity': velocity,
            'position': position,
            'final_position': position[-1],
            'final_velocity': velocity[-1]
        }
    
    return results, time

def calculate_with_drift_correction(accel_df):
    """Calculate position with drift correction."""
    time = accel_df['Time (s)'].values
    results = {}
    
    for axis in ['x', 'y', 'z']:
        accel_col = f'Linear Acceleration {axis} (m/s^2)'
        accel = accel_df[accel_col].values
        
        # First integration
        velocity_raw = integrate.cumulative_trapezoid(accel, time, initial=0)
        
        # Drift correction
        final_velocity = velocity_raw[-1]
        total_time = time[-1] - time[0]
        drift_rate = final_velocity / total_time
        drift_correction = drift_rate * (time - time[0])
        velocity_corrected = velocity_raw - drift_correction
        
        # Second integration
        position = integrate.cumulative_trapezoid(velocity_corrected, time, initial=0)
        
        results[axis] = {
            'velocity': velocity_corrected,
            'position': position,
            'final_position': position[-1],
            'final_velocity': velocity_corrected[-1],
            'drift_rate': drift_rate
        }
    
    return results, time

def determine_vertical_axis(results):
    """Determine vertical axis based on maximum displacement."""
    max_displacements = {}
    for axis in ['x', 'y', 'z']:
        max_displacements[axis] = np.max(np.abs(results[axis]['position']))
    
    return max(max_displacements.keys(), key=lambda k: max_displacements[k])

def create_comparison_plot(time, results_no_drift, results_with_drift):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comparison: With vs Without Drift Correction', fontsize=16)
    
    colors = {'x': 'red', 'y': 'green', 'z': 'blue'}
    
    # Row 1: Without drift correction
    # Velocity
    ax1 = axes[0, 0]
    for axis in ['x', 'y', 'z']:
        ax1.plot(time, results_no_drift[axis]['velocity'], 
                color=colors[axis], label=f'{axis}-axis', alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Velocity - No Drift Correction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Position
    ax2 = axes[0, 1]
    for axis in ['x', 'y', 'z']:
        ax2.plot(time, results_no_drift[axis]['position'], 
                color=colors[axis], label=f'{axis}-axis', alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position - No Drift Correction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Final values bar chart
    ax3 = axes[0, 2]
    axes_list = ['x', 'y', 'z']
    final_positions_no_drift = [results_no_drift[axis]['final_position'] for axis in axes_list]
    bars = ax3.bar(axes_list, final_positions_no_drift, color=[colors[axis] for axis in axes_list], alpha=0.7)
    ax3.set_ylabel('Final Position (m)')
    ax3.set_title('Final Positions - No Drift Correction')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, final_positions_no_drift):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.5),
                f'{value:.2f}m', ha='center', va='bottom' if height > 0 else 'top')
    
    # Row 2: With drift correction
    # Velocity
    ax4 = axes[1, 0]
    for axis in ['x', 'y', 'z']:
        ax4.plot(time, results_with_drift[axis]['velocity'], 
                color=colors[axis], label=f'{axis}-axis', alpha=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_title('Velocity - With Drift Correction')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Position
    ax5 = axes[1, 1]
    for axis in ['x', 'y', 'z']:
        ax5.plot(time, results_with_drift[axis]['position'], 
                color=colors[axis], label=f'{axis}-axis', alpha=0.7)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Position (m)')
    ax5.set_title('Position - With Drift Correction')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Final values bar chart
    ax6 = axes[1, 2]
    final_positions_with_drift = [results_with_drift[axis]['final_position'] for axis in axes_list]
    bars = ax6.bar(axes_list, final_positions_with_drift, color=[colors[axis] for axis in axes_list], alpha=0.7)
    ax6.set_ylabel('Final Position (m)')
    ax6.set_title('Final Positions - With Drift Correction')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, final_positions_with_drift):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.1),
                f'{value:.2f}m', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('comparison_drift_correction.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main comparison function."""
    print("=== VERTICAL DISTANCE CALCULATION COMPARISON ===\n")
    
    # Load data
    accel_df = load_and_sync_data()
    print(f"Data points: {len(accel_df)}")
    print(f"Duration: {accel_df['Time (s)'].iloc[-1] - accel_df['Time (s)'].iloc[0]:.2f} seconds\n")
    
    # Calculate without drift correction
    results_no_drift, time = calculate_without_drift_correction(accel_df)
    vertical_axis_no_drift = determine_vertical_axis(results_no_drift)
    
    # Calculate with drift correction
    results_with_drift, _ = calculate_with_drift_correction(accel_df)
    vertical_axis_with_drift = determine_vertical_axis(results_with_drift)
    
    # Create comparison plots
    create_comparison_plot(time, results_no_drift, results_with_drift)
    
    # Print detailed comparison
    print("=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    
    print("Final velocities:")
    print("Without drift correction:")
    for axis in ['x', 'y', 'z']:
        print(f"  {axis}-axis: {results_no_drift[axis]['final_velocity']:8.6f} m/s")
    print("With drift correction:")
    for axis in ['x', 'y', 'z']:
        print(f"  {axis}-axis: {results_with_drift[axis]['final_velocity']:8.6f} m/s")
    print()
    
    print("Final positions:")
    print("Without drift correction:")
    for axis in ['x', 'y', 'z']:
        print(f"  {axis}-axis: {results_no_drift[axis]['final_position']:8.6f} m")
    print("With drift correction:")
    for axis in ['x', 'y', 'z']:
        print(f"  {axis}-axis: {results_with_drift[axis]['final_position']:8.6f} m")
    print()
    
    print("Drift rates applied:")
    for axis in ['x', 'y', 'z']:
        print(f"  {axis}-axis: {results_with_drift[axis]['drift_rate']:8.6f} m/s²")
    print()
    
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Without drift correction:")
    print(f"  Vertical axis: {vertical_axis_no_drift}")
    print(f"  Vertical distance: {results_no_drift[vertical_axis_no_drift]['final_position']:.6f} m")
    print(f"  Vertical distance: {results_no_drift[vertical_axis_no_drift]['final_position']*100:.2f} cm")
    print()
    print(f"With drift correction:")
    print(f"  Vertical axis: {vertical_axis_with_drift}")
    print(f"  Vertical distance: {results_with_drift[vertical_axis_with_drift]['final_position']:.6f} m")
    print(f"  Vertical distance: {results_with_drift[vertical_axis_with_drift]['final_position']*100:.2f} cm")
    print("=" * 80)
    
    return results_no_drift, results_with_drift

if __name__ == "__main__":
    results_no_drift, results_with_drift = main()