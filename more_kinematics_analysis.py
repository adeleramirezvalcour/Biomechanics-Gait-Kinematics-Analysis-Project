import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(SCRIPT_DIR, "shoes")):
    BASE_DIR = SCRIPT_DIR
elif os.path.exists(os.path.join(SCRIPT_DIR, "Group4_c3d", "shoes")):
    BASE_DIR = os.path.join(SCRIPT_DIR, "Group4_c3d")
else:
    BASE_DIR = r"C:\Users\adele rv\Downloads\Group4_c3d\Group4_c3d"

SHOES_DIR = os.path.join(BASE_DIR, "shoes")
HEELS_DIR = os.path.join(BASE_DIR, "heels")

print(f"--> Script is looking in BASE_DIR: {BASE_DIR}")
print(f"--> Checking SHOES_DIR exists: {os.path.exists(SHOES_DIR)}")
print(f"--> Checking HEELS_DIR exists: {os.path.exists(HEELS_DIR)}")


def find_csv_files(root_folder):

    csv_files = []
    if not os.path.exists(root_folder):
        print(f"ERROR: Folder does not exist -> {root_folder}")
        return csv_files

    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files


def process_gait_trials(folder_path, condition_label):
    csv_files = find_csv_files(folder_path)

    if not csv_files:
        print(f"\n[!] Warning: Zero CSV files found in {folder_path}")
        return None, None, None

    print(
        f"\n[+] Found {len(csv_files)} trial CSV file(s) for '{condition_label}':")
    for f in csv_files:
        print(f"    - {f}")

    cycle_points = np.linspace(0, 100, 101)
    normalized_angles = []
    metrics = []

    for file in csv_files:
        df = pd.read_csv(file)

        knee_col = next(
            (col for col in df.columns if "Knee" in col and "Flex" in col), None)

        if "Time_s" not in df.columns or knee_col is None:
            print(
                f"    Column error in {os.path.basename(file)}. Could not find Knee Flexion column.")
            continue

        print(
            f"    Processing {os.path.basename(file)} using column: '{knee_col}'")

        time = df["Time_s"].values
        angle = df[knee_col].values

        velocity = np.gradient(angle, time) if len(
            time) > 1 else np.zeros_like(angle)

        time_pct = (time - time[0]) / (time[-1] - time[0]) * \
            100.0 if (time[-1] - time[0]) != 0 else np.zeros_like(time)

        interp_angle_fn = interp1d(
            time_pct, angle, kind='linear', fill_value='extrapolate')
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


print("\nStarting Gait Analysis Process...")
cycle, norm_angles, metrics_norm = process_gait_trials(
    SHOES_DIR, "Normal Shoes")
cycle, heels_angles, metrics_heels = process_gait_trials(
    HEELS_DIR, "High Heels")

if metrics_norm is None or metrics_heels is None:
    print("\n[!] Analysis stopped: CSV files could not be processed successfully.")
else:
    all_metrics = pd.concat([metrics_norm, metrics_heels], ignore_index=True)
    all_metrics.to_csv(os.path.join(
        BASE_DIR, "gait_metrics_summary.csv"), index=False)
    print("\n[+] Saved extracted trial metrics to 'gait_metrics_summary.csv'")

    mean_norm = np.mean(norm_angles, axis=0)
    std_norm = np.std(norm_angles, axis=0)

    mean_heels = np.mean(heels_angles, axis=0)
    std_heels = np.std(heels_angles, axis=0)

    plt.figure(figsize=(9, 5), dpi=120)
    plt.plot(cycle, mean_norm, label='Normal Shoes (Mean)',
             color='#1f77b4', linewidth=2.5)
    plt.fill_between(cycle, mean_norm - std_norm, mean_norm + std_norm,
                     color='#1f77b4', alpha=0.25, label='Normal Shoes (±1 SD)')

    plt.plot(cycle, mean_heels, label='High Heels (Mean)',
             color='#d62728', linewidth=2.5)
    plt.fill_between(cycle, mean_heels - std_heels, mean_heels + std_heels,
                     color='#d62728', alpha=0.25, label='High Heels (±1 SD)')

    plt.title('Knee Flexion Angle Across Gait Cycles (Heels vs. Gym Shoes)',
              fontsize=13, fontweight='bold')
    plt.xlabel('Gait Cycle (%)', fontsize=11)
    plt.ylabel('Knee Flexion Angle (°)', fontsize=11)
    plt.xlim([0, 100])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(
        BASE_DIR, 'knee_flexion_variability_envelope.png'), dpi=300)
    plt.show()

    metrics_to_plot = ['IC_Angle',
                       'Stance_Peak_Flex', 'Swing_Peak_Flex', 'ROM']
    titles = ['Initial Contact (°)', 'Peak Stance Flexion (°)',
              'Peak Swing Flexion (°)', 'Range of Motion (°)']

    fig2_path = os.path.join(
        BASE_DIR, 'discrete_metrics_stride_comparison.png')
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), dpi=120)

    for idx, metric in enumerate(metrics_to_plot):
        norm_mean = metrics_norm[metric].mean()
        norm_std = metrics_norm[metric].std()

        heel_mean = metrics_heels[metric].mean()
        heel_std = metrics_heels[metric].std()

        ax = axes[idx]
        bars = ax.bar(['Gym Shoes', 'Heels'], [norm_mean, heel_mean],
                      yerr=[norm_std, heel_std], capsize=5,
                      color=['#1f77b4', '#d62728'], alpha=0.85, edgecolor='black')

        ax.set_title(titles[idx], fontsize=10, fontweight='bold', pad=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        max_val = max(norm_mean + norm_std, heel_mean + heel_std)
        min_val = min(0, norm_mean - norm_std, heel_mean - heel_std)
        ax.set_ylim([min_val * 1.1 if min_val < 0 else 0, max_val * 1.25])

    plt.suptitle('Kinematic Parameters Comparison (Mean ± SD)',
                 fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved updated bar comparison plot to '{fig2_path}'")
