import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


CWD = os.getcwd()

possible_bases = [
    os.path.join(CWD, "Group4_c3d"),
    CWD,
    r"C:\Users\adele rv\Downloads\Group4_c3d\Group4_c3d",
    r"C:\Users\adele rv\Downloads\Group4_c3d"
]

BASE_DIR = None
for path in possible_bases:
    if os.path.exists(os.path.join(path, "shoes")):
        BASE_DIR = path
        break

if BASE_DIR is None:
    print("ERROR: Could not locate 'shoes' folder in any expected directory!")
    print(f"Checked paths: {possible_bases}")
    exit()

SHOES_DIR = os.path.join(BASE_DIR, "shoes")
HEELS_DIR = os.path.join(BASE_DIR, "heels")

print(f"--> Located BASE_DIR: {BASE_DIR}")
print(f"--> Checking SHOES_DIR exists: {os.path.exists(SHOES_DIR)}")
print(f"--> Checking HEELS_DIR exists: {os.path.exists(HEELS_DIR)}")


def find_csv_files(root_folder):
    """Recursively finds all pose CSV files across camera subfolders."""
    csv_files = []
    if not os.path.exists(root_folder):
        return csv_files

    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files


def extract_single_stride(time_vec, angle_vec):
    """Segment data from 1st detected local minimum (heel strike) to 2nd."""
    inverted_angle = -angle_vec
    peaks, _ = find_peaks(inverted_angle, distance=25, prominence=2)

    if len(peaks) >= 2:
        start_idx = peaks[0]
        end_idx = peaks[1]
        return time_vec[start_idx:end_idx], angle_vec[start_idx:end_idx]
    else:
        return time_vec, angle_vec


def process_gait_trials(folder_path, condition_label):
    csv_files = find_csv_files(folder_path)

    if not csv_files:
        print(f"\n[!] Warning: Zero CSV files found in {folder_path}")
        return None, None, None

    print(
        f"\n[+] Processing {len(csv_files)} trial CSV file(s) for '{condition_label}'...")

    cycle_points = np.linspace(0, 100, 101)
    normalized_angles = []
    metrics = []

    for file in csv_files:
        df = pd.read_csv(file)

        knee_col = next(
            (col for col in df.columns if "Knee" in col and "Flex" in col), None)

        if "Time_s" not in df.columns or knee_col is None:
            print(
                f"    [X] Missing required column in {os.path.basename(file)}.")
            continue

        print(
            f"    [✓] Processing {os.path.basename(file)} using column: '{knee_col}'")

        raw_time = df["Time_s"].values
        raw_angle = df[knee_col].values

        stride_time, stride_angle = extract_single_stride(raw_time, raw_angle)
        velocity = np.gradient(stride_angle, stride_time) if len(
            stride_time) > 1 else np.zeros_like(stride_angle)

        time_pct = (stride_time - stride_time[0]) / \
            (stride_time[-1] - stride_time[0]) * 100.0

        interp_angle_fn = interp1d(
            time_pct, stride_angle, kind='linear', fill_value='extrapolate')
        norm_angle = interp_angle_fn(cycle_points)

        normalized_angles.append(norm_angle)

        ic_angle = norm_angle[0]
        stance_peak_flex = np.max(norm_angle[0:35])
        swing_peak_flex = np.max(norm_angle[60:95])
        rom = np.max(norm_angle) - np.min(norm_angle)
        peak_vel = np.max(np.abs(velocity))

        metrics.append({
            'Condition': condition_label,
            'File': os.path.basename(file),
            'IC_Angle': ic_angle,
            'Stance_Peak_Flex': stance_peak_flex,
            'Swing_Peak_Flex': swing_peak_flex,
            'ROM': rom,
            'Peak_Velocity': peak_vel
        })

    if not normalized_angles:
        return None, None, None

    angles_arr = np.array(normalized_angles)
    metrics_df = pd.DataFrame(metrics)

    return cycle_points, angles_arr, metrics_df


def safe_save_fig(fig_path):

    try:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"[+] Saved plot to '{fig_path}'")
    except (OSError, PermissionError):
        dir_name, file_name = os.path.split(fig_path)
        name, ext = os.path.splitext(file_name)
        timestamp = int(time.time())
        fallback_path = os.path.join(dir_name, f"{name}_{timestamp}{ext}")
        plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
        print(
            f"[!] '{file_name}' was locked/unavailable. Saved to fallback: '{fallback_path}'")


print("\nStarting Single-Stride Gait Analysis...")
cycle, norm_angles, metrics_norm = process_gait_trials(SHOES_DIR, "Gym Shoes")
cycle, heels_angles, metrics_heels = process_gait_trials(
    HEELS_DIR, "High Heels")

if metrics_norm is None or metrics_heels is None:
    print("\n[!] Analysis stopped: CSV files could not be processed successfully.")
else:
    all_metrics = pd.concat([metrics_norm, metrics_heels], ignore_index=True)
    summary_path = os.path.join(BASE_DIR, "gait_metrics_summary.csv")

    try:
        all_metrics.to_csv(summary_path, index=False)
        print(f"\n[+] Saved summary stats to '{summary_path}'")
    except (PermissionError, OSError):
        summary_path = os.path.join(
            BASE_DIR, "gait_metrics_summary_updated.csv")
        all_metrics.to_csv(summary_path, index=False)
        print(
            f"\n[!] 'gait_metrics_summary.csv' was locked. Saved to '{summary_path}' instead.")

    mean_norm = np.mean(norm_angles, axis=0)
    std_norm = np.std(norm_angles, axis=0)

    mean_heels = np.mean(heels_angles, axis=0)
    std_heels = np.std(heels_angles, axis=0)

    fig1_path = os.path.join(BASE_DIR, 'single_stride_knee_flexion.png')
    plt.figure(figsize=(9, 5), dpi=120)
    plt.plot(cycle, mean_norm, label='Gym Shoes (Mean)',
             color='#1f77b4', linewidth=2.5)
    plt.fill_between(cycle, mean_norm - std_norm, mean_norm + std_norm,
                     color='#1f77b4', alpha=0.25, label='Gym Shoes (±1 SD)')

    plt.plot(cycle, mean_heels, label='High Heels (Mean)',
             color='#d62728', linewidth=2.5)
    plt.fill_between(cycle, mean_heels - std_heels, mean_heels + std_heels,
                     color='#d62728', alpha=0.25, label='High Heels (±1 SD)')

    plt.title('Single-Stride Knee Flexion Angle (0–100% Gait Cycle)',
              fontsize=13, fontweight='bold')
    plt.xlabel('Gait Cycle (%)', fontsize=11)
    plt.ylabel('Knee Flexion Angle (°)', fontsize=11)
    plt.xlim([0, 100])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    safe_save_fig(fig1_path)
    plt.close()

    metrics_to_plot = ['IC_Angle',
                       'Stance_Peak_Flex', 'Swing_Peak_Flex', 'ROM']
    titles = ['Initial Contact (°)', 'Peak Stance Flexion (°)',
              'Peak Swing Flexion (°)', 'Range of Motion (°)']

    fig2_path = os.path.join(
        BASE_DIR, 'discrete_metrics_stride_comparison.png')
    fig, axes = plt.subplots(1, 4, figsize=(15, 5.0), dpi=120)

    for idx, metric in enumerate(metrics_to_plot):
        norm_mean = metrics_norm[metric].mean()
        norm_std = metrics_norm[metric].std()

        heel_mean = metrics_heels[metric].mean()
        heel_std = metrics_heels[metric].std()

        ax = axes[idx]
        bars = ax.bar(['Gym Shoes', 'Heels'], [norm_mean, heel_mean],
                      yerr=[norm_std, heel_std], capsize=6,
                      color=['#1f77b4', '#d62728'], alpha=0.85, edgecolor='black')

        ax.set_title(titles[idx], fontsize=11, fontweight='bold', pad=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        max_val = max(norm_mean + norm_std, heel_mean + heel_std)
        min_val = min(0, norm_mean - norm_std, heel_mean - heel_std)

        upper_limit = max_val * 1.40 if max_val > 0 else max_val * 0.70
        lower_limit = min_val * 1.20 if min_val < 0 else 0
        ax.set_ylim([lower_limit, upper_limit])

    fig.suptitle('Kinematic Parameters Comparison (Mean ± SD)',
                 fontsize=14, fontweight='bold', y=0.96)

    plt.subplots_adjust(top=0.80, bottom=0.15, wspace=0.35)
    safe_save_fig(fig2_path)
    plt.close()

    print("\n[✓] Analysis execution complete! Check your project folder for plots and CSV.")
