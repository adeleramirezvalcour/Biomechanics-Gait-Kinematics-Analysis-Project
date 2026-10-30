import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 1. Load exported kinematics CSV
csv_filename = "C:/Users/adele rv/Downloads/Group4_c3d/Group4_c3d/shoes/002-Camera/pose_filt_0.csv"
df = pd.read_csv(csv_filename)

# Time step from frame rate
time = df["Time_s"].to_numpy()
dt = time[1] - time[0]

# 2. Select joint angle of interest
# Column options include: R_Knee_FlexExt, R_Hip_FlexExt, R_Ankle_FlexExt, Lumbar_Spine_FlexExt, etc.
angle_raw = df["R_Knee_FlexExt"].to_numpy()

# 3. Calculate smoothed angle, angular velocity, and acceleration
# Using a Savitzky-Golay filter to avoid raw finite-difference noise:
# window_length must be an odd integer, polyorder must be < window_length
window = 15
poly = 3

angle_smooth = savgol_filter(
    angle_raw, window_length=window, polyorder=poly, deriv=0)
ang_velocity = savgol_filter(
    angle_raw, window_length=window, polyorder=poly, deriv=1, delta=dt)
ang_accel = savgol_filter(angle_raw, window_length=window,
                          polyorder=poly, deriv=2, delta=dt)

# 4. Plot Angle, Velocity, and Acceleration
fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

# Angle
axs[0].plot(time, angle_raw, color="silver", label="Raw Signal")
axs[0].plot(time, angle_smooth, color="#0055b8",
            linewidth=2, label="Smoothed Angle")
axs[0].set_ylabel("Angle (deg)")
axs[0].set_title("Right Knee Flexion / Extension Kinematics")
axs[0].legend(loc="upper right")
axs[0].grid(True, linestyle="--", alpha=0.6)

# Angular Velocity
axs[1].plot(time, ang_velocity, color="#d9534f", linewidth=1.5)
axs[1].set_ylabel("Velocity (deg/s)")
axs[1].grid(True, linestyle="--", alpha=0.6)

# Angular Acceleration
axs[2].plot(time, ang_accel, color="#2ca02c", linewidth=1.5)
axs[2].set_ylabel("Accel (deg/s²)")
axs[2].set_xlabel("Time (s)")
axs[2].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

# 5. (Optional) Example: Plotting vertical foot position (Z-axis in meters)
if "Pos_r_foot_Z" in df.columns:
    plt.figure(figsize=(8, 3))
    plt.plot(time, df["Pos_r_foot_Z"], color="purple")
    plt.ylabel("Vertical Position (m)")
    plt.xlabel("Time (s)")
    plt.title("Right Foot Vertical Trajectory")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
