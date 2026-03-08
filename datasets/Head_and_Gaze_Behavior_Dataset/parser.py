import os
import pandas as pd
import numpy as np
import math
from typing import Generator

def parse_metadata(dataset_path: str) -> dict:
    """
    Parses and returns metadata about the dataset.
    """
    return {
        "Name": "Head and Gaze Behavior Dataset",
        "Corpus": "Head_and_Gaze_Behavior_Dataset",
        "Format": "CSV",
        "Description": "Dataset containing head tracking and gaze tracking data for 360-degree videos."
    }

def parse(dataset_path) -> Generator[tuple[str, str, pd.DataFrame], None, None]:
    """
    Parses all CSV files in the Head_and_Gaze_Behavior_Dataset.
    Maps data appropriately based on Version (V1 or V2).
    """
    dataset_path = str(dataset_path)
    # Version 1 parsing
    v1_path = os.path.join(dataset_path, "Version1")
    if os.path.exists(v1_path):
        for user_folder in os.listdir(v1_path):
            user_dir = os.path.join(v1_path, user_folder)
            if not os.path.isdir(user_dir):
                continue
            
            for file_name in os.listdir(user_dir):
                if not file_name.endswith(".csv"):
                    continue
                
                # Extract task name (e.g. video_10)
                parts = file_name.split("_")
                if len(parts) >= 3:
                    task = parts[1] + "_" + parts[2]
                else:
                    task = file_name.replace(".csv", "")
                
                file_path = os.path.join(user_dir, file_name)
                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
                
                if df.empty:
                    continue
                
                parsed_df = pd.DataFrame()
                parsed_df["SessionTime"] = df["AdjustedTime"].astype(float)
                
                # Setup base constants for mapping
                # Mapping from analysis: yaw = A * x + B, pitch = C * y + D
                # We use the constants generated from looking at the equirectangular bounds
                A_yaw = -0.004669541820760188
                B_yaw = 3.6661146747551003
                C_pitch = 0.004909337583683526
                D_pitch = -1.0026794896605285
                
                # Calculate Pose
                pose_x = df["Pose_Point_x"].astype(float)
                pose_y = df["Pose_Point_y"].astype(float)
                
                pose_yaw = A_yaw * pose_x + B_yaw
                pose_pitch = C_pitch * pose_y + D_pitch
                
                # Calculate Cartesian 3D (assuming standard spherical projection: x=sin(yaw)cos(pitch), y=sin(pitch), z=cos(yaw)cos(pitch))
                # For this dataset mapping specific formula:
                parsed_df["HmdPosition.x"] = np.sin(pose_yaw) * np.cos(pose_pitch)
                parsed_df["HmdPosition.y"] = np.sin(pose_pitch)
                parsed_df["HmdPosition.z"] = np.cos(pose_yaw) * np.cos(pose_pitch)
                
                # We do not have rotation or translation components for V1 except implied ray
                parsed_df["IsEyeTrackingSample"] = 1
                
                # Gaze map
                gaze_x = df["GazePoint_x"].astype(float)
                gaze_y = df["GazePoint_y"].astype(float)
                
                gaze_yaw = A_yaw * gaze_x + B_yaw
                gaze_pitch = C_pitch * gaze_y + D_pitch
                
                parsed_df["GazeRay.x"] = np.sin(gaze_yaw) * np.cos(gaze_pitch)
                parsed_df["GazeRay.y"] = np.sin(gaze_pitch)
                parsed_df["GazeRay.z"] = np.cos(gaze_yaw) * np.cos(gaze_pitch)
                
                yield user_folder, f"V1_{task}", parsed_df
                
    # Version 2 parsing
    v2_path = os.path.join(dataset_path, "Version2")
    if os.path.exists(v2_path):
        for user_folder in os.listdir(v2_path):
            user_dir = os.path.join(v2_path, user_folder)
            if not os.path.isdir(user_dir):
                continue
                
            # user_folder is like "V2 (1)", extract "1"
            clean_user = user_folder.replace("V2 (", "").replace(")", "")
            
            for file_name in os.listdir(user_dir):
                if not file_name.endswith(".csv"):
                    continue
                
                # Extract task name
                parts = file_name.split("_")
                if len(parts) >= 3:
                    task = parts[1] + "_" + parts[2]
                else:
                    task = file_name.replace(".csv", "")
                
                file_path = os.path.join(user_dir, file_name)
                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
                
                if df.empty:
                    continue
                
                parsed_df = pd.DataFrame()
                parsed_df["SessionTime"] = df["AdjustedTime"].astype(float)
                
                # Parse Pose_Position (x,y,z)
                def extract_tuple_col_multi(series, n_elements=3):
                    """Extracts a tuple formatted column into a numpy mxN array"""
                    import re
                    def safe_extract(x):
                        if pd.isna(x):
                            return [np.nan] * n_elements
                        try:
                            # Strip parentheses and split by comma
                            vals = [float(v) for v in x.strip(" \t()").split(",")]
                            return vals if len(vals) == n_elements else [np.nan] * n_elements
                        except:
                            return [np.nan] * n_elements
                            
                    return np.array(series.apply(safe_extract).tolist())
                
                pose_pos = extract_tuple_col_multi(df["Pose_Position"], 3)
                parsed_df["HmdPosition.x"] = pose_pos[:, 0]
                parsed_df["HmdPosition.y"] = pose_pos[:, 1]
                parsed_df["HmdPosition.z"] = pose_pos[:, 2]
                
                # Parse Pose_Rotation (quaternion x,y,z,w)
                pose_rot = extract_tuple_col_multi(df["Pose_Rotation"], 4)
                parsed_df["UnitQuaternion.x"] = pose_rot[:, 0]
                parsed_df["UnitQuaternion.y"] = pose_rot[:, 1]
                parsed_df["UnitQuaternion.z"] = pose_rot[:, 2]
                parsed_df["UnitQuaternion.w"] = pose_rot[:, 3]
                
                parsed_df["IsEyeTrackingSample"] = 1
                
                # Gaze direction calculation: LeftGazeDirection + RightGazeDirection vectors
                # Fallback to Unit_Vector if left/right are absent/malformed
                left_gaze = extract_tuple_col_multi(df["LeftGazeDirection"], 3)
                right_gaze = extract_tuple_col_multi(df["RightGazeDirection"], 3)
                
                with np.errstate(invalid='ignore'):
                    combined_gaze = (left_gaze + right_gaze) / 2.0
                    # Normalize
                    norms = np.linalg.norm(combined_gaze, axis=1, keepdims=True)
                    combined_gaze = np.where(norms == 0, np.nan, combined_gaze / norms)
                    
                parsed_df["GazeRay.x"] = combined_gaze[:, 0]
                parsed_df["GazeRay.y"] = combined_gaze[:, 1]
                parsed_df["GazeRay.z"] = combined_gaze[:, 2]
                
                yield clean_user, f"V2_{task}", parsed_df
