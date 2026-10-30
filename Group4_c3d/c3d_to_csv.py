"""
Boston College - Department of Engineering
ENGR 4230: Biomechanics
Project 1: 3D Body Biomechanics - Theia3D C3D to CSV Parser
"""

import argparse
import ezc3d
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from pathlib import Path

def parse_c3d_to_csv(c3d_path: str, output_csv: str) -> None:
    # Load C3D binary structure
    c = ezc3d.c3d(c3d_path)

    # Validate presence of 4x4 pose group
    if "rotations" not in c["data"] or c["data"]["rotations"].size == 0:
        raise ValueError("No 4x4 rotation/pose matrix group found in this C3D file.")

    # ezc3d rotation tensor shape: (4, 4, num_segments, num_frames)
    rot_data = c["data"]["rotations"]
    raw_labels = c["parameters"]["ROTATION"]["LABELS"]["value"]
    frame_rate = float(c["parameters"]["ROTATION"]["RATE"]["value"][0])
    total_frames = rot_data.shape[3]

    # Normalize segment labels
    clean_labels = [
        lbl.strip().replace("_4X4", "").lower()
        for lbl in raw_labels
    ]
    segment_idx = {name: idx for idx, name in enumerate(clean_labels)}

    # Kinematic definition: (Parent, Child, Column Prefix)
    kinematic_chain = [
        # Spine & Trunk
        ("worldbody", "pelvis", "Pelvis_Global"),
        ("pelvis", "torso", "Lumbar_Spine"),
        ("torso", "head", "Cervical_Spine"),

        # Right Lower Extremity
        ("pelvis", "r_thigh", "R_Hip"),
        ("r_thigh", "r_shank", "R_Knee"),
        ("r_shank", "r_foot", "R_Ankle"),

        # Left Lower Extremity
        ("pelvis", "l_thigh", "L_Hip"),
        ("l_thigh", "l_shank", "L_Knee"),
        ("l_shank", "l_foot", "L_Ankle"),

        # Right Upper Extremity
        ("torso", "r_uarm", "R_Shoulder"),
        ("r_uarm", "r_larm", "R_Elbow"),
        ("r_larm", "r_hand", "R_Wrist"),

        # Left Upper Extremity
        ("torso", "l_uarm", "L_Shoulder"),
        ("l_uarm", "l_larm", "L_Elbow"),
        ("l_larm", "l_hand", "L_Wrist"),
    ]

    time_vector = np.arange(total_frames) / frame_rate
    results = {
        "Frame": np.arange(total_frames),
        "Time_s": time_vector,
    }

    # 1. Compute Relative Joint Angles (Cardan Z-X-Y Sequence)
    for parent, child, prefix in kinematic_chain:
        col_fe = f"{prefix}_FlexExt"
        col_aa = f"{prefix}_AbdAdd"
        col_rot = f"{prefix}_IntExtRot"

        if parent not in segment_idx or child not in segment_idx:
            results[col_fe] = np.full(total_frames, np.nan)
            results[col_aa] = np.full(total_frames, np.nan)
            results[col_rot] = np.full(total_frames, np.nan)
            continue

        p_idx = segment_idx[parent]
        c_idx = segment_idx[child]

        # Reshape segment transforms to (N_frames, 4, 4)
        T_p = np.moveaxis(rot_data[:, :, p_idx, :], -1, 0)
        T_c = np.moveaxis(rot_data[:, :, c_idx, :], -1, 0)

        angles = np.full((total_frames, 3), np.nan)

        for i in range(total_frames):
            R_p = T_p[i, :3, :3]
            R_c = T_c[i, :3, :3]

            if not (np.isfinite(R_p).all() and np.isfinite(R_c).all()):
                continue
            if abs(np.linalg.det(R_p) - 1.0) > 1e-2 or abs(np.linalg.det(R_c) - 1.0) > 1e-2:
                continue

            R_rel = np.dot(R_p.T, R_c)

            try:
                rot_obj = R.from_matrix(R_rel)
                angles[i] = rot_obj.as_euler("zxy", degrees=True)
            except ValueError:
                continue

        # In Theia's segment basis:
        # angles[:, 1] is mediolateral X (Flexion/Extension)
        # angles[:, 0] is anteroposterior Z (Abduction/Adduction)
        # angles[:, 2] is longitudinal Y (Internal/External Rotation)
        results[col_fe] = -angles[:, 1]
        results[col_aa] = angles[:, 0]
        results[col_rot] = angles[:, 2]

    # 2. Extract Global 3D Segment Positions (Translational vector in 4th column)
    tracked_positions = [
        "pelvis", "torso", "head",
        "r_thigh", "r_shank", "r_foot",
        "l_thigh", "l_shank", "l_foot",
        "r_uarm", "r_larm", "r_hand",
        "l_uarm", "l_larm", "l_hand",
    ]

    for seg in tracked_positions:
        col_x = f"Pos_{seg}_X"
        col_y = f"Pos_{seg}_Y"
        col_z = f"Pos_{seg}_Z"

        if seg not in segment_idx:
            results[col_x] = np.full(total_frames, np.nan)
            results[col_y] = np.full(total_frames, np.nan)
            results[col_z] = np.full(total_frames, np.nan)
            continue

        s_idx = segment_idx[seg]
        T_seg = np.moveaxis(rot_data[:, :, s_idx, :], -1, 0)
        pos = T_seg[:, :3, 3]

        valid_mask = np.isfinite(pos).all(axis=1) & (
            np.abs(np.linalg.det(T_seg[:, :3, :3]) - 1.0) <= 1e-2
        )
        pos_cleaned = np.where(valid_mask[:, None], pos, np.nan)

        results[col_x] = pos_cleaned[:, 0]
        results[col_y] = pos_cleaned[:, 1]
        results[col_z] = pos_cleaned[:, 2]

    # Define the output file path
    output_path = Path(output_csv)

    # Ensure parent directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Export completed: '{output_csv}' ({total_frames} frames, {len(df.columns) - 2} channels).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Theia3D C3D into joint angle and position CSV."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input pose_filt_0.c3d")
    parser.add_argument("--output", "-o", default="joint_angles_and_positions.csv", help="Path to output CSV")
    args = parser.parse_args()

    parse_c3d_to_csv(args.input, args.output)