import os
import pandas as pd
import numpy as np
from typing import Generator

def parse_metadata(dataset_path: str) -> dict:
    """
    Parses and returns metadata about the dataset.
    """
    return {
        "Name": "Head and Eye Movements for 360-degree Videos (Salient360!)",
        "Corpus": "Head_and_Eye_Movements_for_360-degree_Videos_(Salient360!)",
        "Format": "TXT (CSV Format)",
        "Description": "Dataset of head and eye movements for 360 videos."
    }

def convert_normalized_to_unit_vector(lon: float, lat: float) -> tuple[float, float, float]:
    """
    Convert normalized longitude [0, 1] and latitude [0, 1] to a 3D cartesian unit vector.
    Assumes equirectangular projection where:
    - longitude 0.5 is forward (yaw=0), 0 is -pi, 1 is +pi
    - latitude 0.5 is forward (pitch=0), 0 is +pi/2, 1 is -pi/2
    This maps:
    x = sin(theta) * cos(phi)
    y = sin(phi)
    z = cos(theta) * cos(phi)
    """
    theta = lon * 2 * np.pi - np.pi
    phi = (0.5 - lat) * np.pi
    
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(theta)
    return x, y, z

def _parse_scanpath_file(file_path: str, is_eye_tracking: bool, user_id: str, task_name: str) -> pd.DataFrame:
    """
    Helper function to parse a single scanpath txt file.
    """
    # The files are comma separated with a header
    # "Idx, longitude, latitude, start timestamp"
    try:
        # Use skipinitialspace because the header has spaces " longitude", " latitude", " start timestamp"
        df = pd.read_csv(file_path, skipinitialspace=True)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()
        
    if df.empty:
        return pd.DataFrame()

    # Column names might have leading/trailing spaces
    df.columns = [c.strip() for c in df.columns]

    parsed_df = pd.DataFrame()
    
    if "start timestamp" in df.columns:
        parsed_df["SessionTime"] = df["start timestamp"].astype(float) / 1000.0
    else:
        # If the column name differs or is misspelled
        time_col = [c for c in df.columns if "time" in c.lower()][0]
        parsed_df["SessionTime"] = df[time_col].astype(float) / 1000.0

    # Ensure latitude and longitude are float
    lon_col = [c for c in df.columns if "lon" in c.lower()][0]
    lat_col = [c for c in df.columns if "lat" in c.lower()][0]
    
    lon = df[lon_col].astype(float)
    lat = df[lat_col].astype(float)
    
    # Vectorize the spherical to cartesian conversion
    theta = lon * 2 * np.pi - np.pi
    phi = (0.5 - lat) * np.pi
    
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(theta)
    
    if is_eye_tracking:
        # Typically Hscanpath gives head, HEscanpath gives eye (or head+eye)
        # Using HE as gaze
        parsed_df["GazeRay.x"] = x
        parsed_df["GazeRay.y"] = y
        parsed_df["GazeRay.z"] = z
        parsed_df["IsEyeTrackingSample"] = 1
        
        # It's unclear if HEscanpath provides ONLY gaze or Head+Gaze. 
        # Since it's eye movement tracking, mapping to GazeRay.
        # We will not populate HmdPosition here to avoid mixing.
    else:
        parsed_df["HmdPosition.x"] = x
        parsed_df["HmdPosition.y"] = y
        parsed_df["HmdPosition.z"] = z
        parsed_df["IsEyeTrackingSample"] = 0

    return parsed_df

def parse(dataset_path) -> Generator[tuple[str, str, pd.DataFrame], None, None]:
    """
    Parses all relevant text files in the dataset.
    Yields (user_id, task_id, dataframe)
    """
    dataset_path = str(dataset_path)
    
    # 1. Parse Image_H (Head tracking only)
    h_scanpaths_dir = os.path.join(dataset_path, "Image_H", "H", "Scanpaths")
    if os.path.exists(h_scanpaths_dir):
        for file_name in os.listdir(h_scanpaths_dir):
            if not file_name.startswith("Hscanpath_") or not file_name.endswith(".txt"):
                continue
                
            file_path = os.path.join(h_scanpaths_dir, file_name)
            
            # Extract User from filename e.g., Hscanpath_1.txt -> 1
            user_id = file_name.replace("Hscanpath_", "").replace(".txt", "")
            task_name = "Image_H"
            
            parsed_df = _parse_scanpath_file(file_path, is_eye_tracking=False, user_id=user_id, task_name=task_name)
            if not parsed_df.empty:
                yield user_id, task_id, parsed_df
                
    # 2. Parse Image_HE (Head and Eye tracking) - Both L and R if they exist
    he_base_dir = os.path.join(dataset_path, "Image_HE", "HE", "Scanpaths")
    if os.path.exists(he_base_dir):
        for eye in ["L", "R"]:
            eye_dir = os.path.join(he_base_dir, eye)
            if not os.path.exists(eye_dir):
                continue
                
            for file_name in os.listdir(eye_dir):
                if not file_name.startswith("HEscanpath_") or not file_name.endswith(".txt"):
                    continue
                    
                file_path = os.path.join(eye_dir, file_name)
                
                # Extract User from filename
                user_id = file_name.replace("HEscanpath_", "").replace(".txt", "")
                task_id = f"Image_HE_{eye}"
                
                parsed_df = _parse_scanpath_file(file_path, is_eye_tracking=True, user_id=user_id, task_name=task_id)
                if not parsed_df.empty:
                    yield user_id, task_id, parsed_df
