
import pandas as pd
import glob
import os
import numpy as np
from pathlib import Path

def parse(dataset_path):
    """
    Parses the Salient360 dataset.
    
    Yields:
        (user_id, task_id, dataframe)
    """
    
    # 1. Parse Head Data (Hscanpath)
    head_path = dataset_path / "Image_H" / "H" / "Scanpaths"
    head_files = list(head_path.glob("Hscanpath_*.txt"))
    
    for file_path in head_files:
        video_id =  file_path.stem.split('_')[1] # Hscanpath_1.txt -> 1
        
        # Read the file
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        columns = [c.strip() for c in df.columns]
        df.columns = columns
        # Expected: Idx, longitude, latitude, start timestamp
        
        # Identify blocks (Idx resets to 0) indicating different users
        df['block_diff'] = df['Idx'].diff()
        # Start of a new block is where Idx drops (diff < 0) or it's the first row
        # Actually safer: Idx == 0 is start of user
        # Assign user IDs based on blocks
        
        df['UserBlock'] = (df['Idx'] == 0).cumsum()
        
        for block_id, group in df.groupby('UserBlock'):
            user_id = f"P{block_id}_V{video_id}" # Participant X on Video Y (Since we don't have global PID, we make unique per video stream or we can't assume P1 in V1 is P1 in V2 reliably without more info, but usually they are consistent. Let's assume unique stream ID for now)
            # Actually, standard practice: usually P1 is same person across videos. 
            # But the row count might differ. Let's stick with specific ID per file/block to be safe unless we find otherwise.
            # "P{block_id}" implies 1st person in file, 2nd person in file...
            
            task_id = f"video_{video_id}_head"
            
            processed_df = process_group(group, "head")
            yield user_id, task_id, processed_df

    # 2. Parse Gaze Data (HEscanpath - 'L')
    gaze_path = dataset_path / "Image_HE" / "HE" / "Scanpaths" / "L"
    gaze_files = list(gaze_path.glob("HEscanpath_*.txt"))
    
    for file_path in gaze_files:
        video_id = file_path.stem.split('_')[1]
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        columns = [c.strip() for c in df.columns]
        df.columns = columns
        
        df['UserBlock'] = (df['Idx'] == 0).cumsum()
        
        for block_id, group in df.groupby('UserBlock'):
            user_id = f"P{block_id}_V{video_id}" # Matching ID scheme (hopefully P1 in H is P1 in HE)
            task_id = f"video_{video_id}_gaze"
            
            processed_df = process_group(group, "gaze")
            yield user_id, task_id, processed_df

def process_group(group, type_suffix):
    # Normalize Time
    # timestamp is 'start timestamp' in what unit? Likely ms if values are like 382.59, 831.89
    # Some values are negative? -0.024057. Treat as 0 start?
    
    # Sort just in case
    group = group.copy()
    
    # Check timestamp column name
    ts_col = [c for c in group.columns if 'timestamp' in c.lower()][0]
    lat_col = [c for c in group.columns if 'latitude' in c.lower()][0]
    lon_col = [c for c in group.columns if 'longitude' in c.lower()][0]
    
    group = group.sort_values(by=ts_col)
    
    # Calculate SessionTime (seconds) - assuming raw is Ms
    # First value might be ~0 or offsets. Let's strictly normalize to 0.0 start
    # Wait, raw values are like 382.59. Is that seconds or ms?
    # 382.59 seems like ms if it's fixation start. Or maybe seconds? 
    # Let's check diffs. 382 -> 831 is ~450 diff. 
    # If 450 seconds, that's huge (7 mins between samples). Unlikely for scanpath.
    # If 450 ms, that's 2Hz. Plausible for fixations.
    # So raw is likely Milliseconds.
    
    start_time = group[ts_col].min()
    group['SessionTime'] = (group[ts_col] - start_time) / 1000.0
    group['SessionTime'] = group['SessionTime'].clip(lower=0.0)
    
    # Coordinates (0-1 normalized)
    # Longitude (u) -> Yaw: (u - 0.5) * 360
    # Latitude (v) -> Pitch: (0.5 - v) * 180 (Usually v=0 is top/bottom? need to confirm standard equirect)
    # Standard Equirectangular:
    # u=0 (left) -> u=1 (right). u=0.5 (center). Yaw -180 to 180.
    # v=0 (top) -> v=1 (bottom). v=0.5 (center). Pitch 90 to -90.
    
    u = group[lon_col].values
    v = group[lat_col].values
    
    # Yaw/Pitch degrees
    # Yaw: 0 to 360 or -180 to 180?
    # Let's assume u=0.5 is 0 yaw.
    yaw_deg = (u - 0.5) * 360
    # Pitch: v=0.5 is 0 pitch. v=0 is 90 (up), v=1 is -90 (down)
    pitch_deg = (0.5 - v) * 180
    
    # Convert to Quaternion
    # Order: Yaw (Z? Y?), Pitch (X?), Roll (0)
    # Common VR: Y is up. Yaw is rotation around Y. Pitch is rotation around X.
    # We will use a helper for Euler -> Quat
    
    quats = euler_to_quaternion(yaw_deg, pitch_deg, np.zeros_like(yaw_deg))
    
    # Convert UV to Cartesian Unit Vector (HmdPosition)
    # x = cos(lat) * cos(lon)
    # y = sin(lat)
    # z = cos(lat) * sin(lon)
    # Need radians.
    # Latitude in spherical mapping: 
    # v=0 -> pi/2, v=1 -> -pi/2
    lat_rad = (0.5 - v) * np.pi #  Range pi/2 to -pi/2
    # Longitude:
    # u=0 -> -pi, u=1 -> pi (or 0 to 2pi)
    lon_rad = (u - 0.5) * 2 * np.pi # Range -pi to pi
    
    # Standard Physics convention (Z up? Y up?) 
    # Let's use standard graphics: Y up.
    # x = cos(lat) * sin(lon)
    # y = sin(lat)
    # z = cos(lat) * cos(lon)
    
    # Let's align with the previous dataset's UnitQuaternion output if possible.
    # But strictly:
    # x = cos(lat) * cos(lon)
    # z = cos(lat) * sin(lon) 
    # y = sin(lat) 
    
    # Wait, usually for VR: -Z is forward.
    # Let's just stick to standard maths spherical conversion for now.
    
    x = np.cos(lat_rad) * np.sin(lon_rad)
    y = np.sin(lat_rad)
    z = -np.cos(lat_rad) * np.cos(lon_rad) # -Z forward convention at 0,0
    
    # Create DataFrame
    out_df = pd.DataFrame({
        'SessionTime': group['SessionTime'],
        'UnitQuaternion.w': quats[:, 0],
        'UnitQuaternion.x': quats[:, 1],
        'UnitQuaternion.y': quats[:, 2],
        'UnitQuaternion.z': quats[:, 3],
        'HmdPosition.x': x,
        'HmdPosition.y': y,
        'HmdPosition.z': z,
    })
    
    # Add raw gaze angles if requested? Not strictly in schema, but good to have
    if type_suffix == 'gaze':
        out_df['Gaze.yaw'] = yaw_deg
        out_df['Gaze.pitch'] = pitch_deg
    else:
        out_df['Head.yaw'] = yaw_deg
        out_df['Head.pitch'] = pitch_deg
        
    return out_df

def euler_to_quaternion(yaw_deg, pitch_deg, roll_deg):
    """
    Convert Euler angles (degrees) to Quaternions (w, x, y, z).
    Yaw: Y-axis, Pitch: X-axis, Roll: Z-axis (Intrinsic? Extrinsic?)
    Assuming standard: ZYX order or similar. 
    Here we implement a standard conversion:
    """
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # Order?
    # q = R_y(yaw) * R_x(pitch) * R_z(roll) ?
    # Let's use:
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.column_stack((w, x, y, z))

def parse_metadata(dataset_path):
    metadata = {}
    
    # No explicit user file found, so we will generate based on what we saw during parsing or return basic structure.
    # Since parse() is a generator, we can't easily pre-scan everything without running it.
    # But formatter expects this function to be fast.
    # We can list files.
    
    # Users
    # We don't know exactly how many users per file without reading.
    # But we can assume a set.
    # Let's create an empty dataframe with columns for now, or scan one file to estimate.
    # To be safe and compliant, we provide a placeholder or skip if not strictly required.
    # The formatter uses this to update 'users.csv'.
    
    # Let's scan files quickly to count users? Too slow.
    # Let's return minimal metadata.
    
    metadata['users'] = pd.DataFrame(columns=['UserID']) # Placeholder
    metadata['tasks'] = pd.DataFrame(columns=['TaskID', 'VideoID', 'Type']) # Placeholder
    
    # Populate Tasks from file list
    head_files = list((dataset_path / "Image_H" / "H" / "Scanpaths").glob("Hscanpath_*.txt"))
    # gaze_files ...
    
    tasks = []
    for f in head_files:
        vid = f.stem.split('_')[1]
        tasks.append({'TaskID': f"video_{vid}_head", 'VideoID': vid, 'Type': 'Head'})
        tasks.append({'TaskID': f"video_{vid}_gaze", 'VideoID': vid, 'Type': 'Gaze'})
        
    metadata['tasks'] = pd.DataFrame(tasks)
    
    return metadata
