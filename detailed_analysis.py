#!/usr/bin/env python3
"""
Additional analysis script to verify the vertical distance calculation.
This script provides more detailed analysis and validation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def detailed_analysis():
    """Perform detailed analysis of the sensor data and calculations."""
    
    # Load the data
    accel_df = pd.read_csv('Linear Acceleration.csv')
    
    print("=== Detailed Analysis ===\n")
    
    # Basic statistics
    print("Acceleration data statistics:")
    for axis in ['x', 'y', 'z']:
        col = f'Linear Acceleration {axis} (m/s^2)'
        data = accel_df[col]
        print(f"{axis}-axis: mean={data.mean():.6f}, std={data.std():.6f}, "
              f"min={data.min():.6f}, max={data.max():.6f}")
    print()
    
    # Check if sensors are truly still at start and end
    time = accel_df['Time (s)'].values
    print("Start/End stillness check (first/last 1 second):")
    start_mask = time <= (time[0] + 1.0)
    end_mask = time >= (time[-1] - 1.0)
    
    for axis in ['x', 'y', 'z']:
        col = f'Linear Acceleration {axis} (m/s^2)'
        start_std = accel_df.loc[start_mask, col].std()
        end_std = accel_df.loc[end_mask, col].std()
        print(f"{axis}-axis: start_std={start_std:.6f}, end_std={end_std:.6f}")
    print()
    
    # Calculate position with different integration methods
    print("Comparing integration methods:")
    
    dt = np.diff(time)
    dt_avg = np.mean(dt)
    print(f"Average time step: {dt_avg:.6f} seconds")
    print(f"Sampling rate: {1/dt_avg:.1f} Hz")
    
    for axis in ['x', 'y', 'z']:
        col = f'Linear Acceleration {axis} (m/s^2)'
        accel = accel_df[col].values
        
        # Method 1: Cumulative trapezoidal (current method)
        velocity1 = np.zeros_like(accel)
        for i in range(1, len(accel)):
            velocity1[i] = velocity1[i-1] + 0.5 * (accel[i] + accel[i-1]) * dt[i-1]
        
        position1 = np.zeros_like(velocity1)
        for i in range(1, len(velocity1)):
            position1[i] = position1[i-1] + 0.5 * (velocity1[i] + velocity1[i-1]) * dt[i-1]
        
        # Method 2: Using scipy integrate
        velocity2 = integrate.cumulative_trapezoid(accel, time, initial=0)
        position2 = integrate.cumulative_trapezoid(velocity2, time, initial=0)
        
        print(f"{axis}-axis:")
        print(f"  Method 1 final position: {position1[-1]:.6f} m")
        print(f"  Method 2 final position: {position2[-1]:.6f} m")
        print(f"  Difference: {abs(position1[-1] - position2[-1]):.6f} m")
    
    print()
    
    # Check for drift by examining velocity at end
    print("Velocity drift check (should be near zero at end if truly still):")
    for axis in ['x', 'y', 'z']:
        col = f'Linear Acceleration {axis} (m/s^2)'
        accel = accel_df[col].values
        velocity = integrate.cumulative_trapezoid(accel, time, initial=0)
        final_velocity = velocity[-1]
        print(f"{axis}-axis final velocity: {final_velocity:.6f} m/s")
    
    print()
    
    # Create a detailed plot focusing on the dominant axis
    create_detailed_plot(accel_df)

def create_detailed_plot(accel_df):
    """Create detailed plots for verification."""
    time = accel_df['Time (s)'].values
    
    # Calculate position for x-axis (determined as vertical)
    accel_x = accel_df['Linear Acceleration x (m/s^2)'].values
    velocity_x = integrate.cumulative_trapezoid(accel_x, time, initial=0)
    position_x = integrate.cumulative_trapezoid(velocity_x, time, initial=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Detailed Analysis - X-axis (Vertical)', fontsize=16)
    
    # Plot 1: Raw acceleration
    ax1 = axes[0, 0]
    ax1.plot(time, accel_x, 'b-', alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_title('X-axis Acceleration')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 2: Velocity
    ax2 = axes[0, 1]
    ax2.plot(time, velocity_x, 'g-', alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('X-axis Velocity (Integrated Acceleration)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 3: Position
    ax3 = axes[1, 0]
    ax3.plot(time, position_x, 'r-', alpha=0.7)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('X-axis Position (Double Integrated Acceleration)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 4: Moving statistics
    ax4 = axes[1, 1]
    # Calculate rolling standard deviation of acceleration
    window = 100  # points
    if len(accel_x) > window:
        rolling_std = pd.Series(accel_x).rolling(window).std()
        ax4.plot(time, rolling_std, 'purple', alpha=0.7)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Rolling Std (m/s²)')
        ax4.set_title(f'X-axis Acceleration Rolling Std ({window} points)')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detailed_analysis_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print("=== Summary Statistics ===")
    print(f"Maximum absolute position: {np.max(np.abs(position_x)):.6f} m")
    print(f"Final position: {position_x[-1]:.6f} m")
    print(f"Final velocity: {velocity_x[-1]:.6f} m/s")
    print(f"RMS acceleration: {np.sqrt(np.mean(accel_x**2)):.6f} m/s²")

if __name__ == "__main__":
    detailed_analysis()