import pandas as pd
import glob
import os

def parse(dataset_path):
    """
    Parses the EyeNavGS_6-DoF_Navigation_Dataset.
    Yields: (user_id, task_id, dataframe)
    """
    # Base dataset dir
    data_dir = dataset_path / "main" / "EyeNavGS_Rutgers_Dataset-main" / "dataset"
    
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return
        
    # Get scene directories
    scene_dirs = [d for d in os.listdir(data_dir) if (data_dir / d).is_dir()]
    
    for scene in scene_dirs:
        scene_path = data_dir / scene
        csv_files = glob.glob(str(scene_path / "*.csv"))
        
        for csv_file in csv_files:
            try:
                # Filename is typically like "user101_alameda.csv"
                filename = os.path.basename(csv_file)
                user_id = filename.split("_")[0].replace("user", "") # e.g., 101
                task_id = scene # e.g., alameda
                
                df = pd.read_csv(csv_file)
                
                # Check required columns from original dataset
                col_mapping = {
                    "PositionX": "HmdPosition.x",
                    "PositionY": "HmdPosition.y",
                    "PositionZ": "HmdPosition.z",
                    "QuaternionX": "UnitQuaternion.x",
                    "QuaternionY": "UnitQuaternion.y",
                    "QuaternionZ": "UnitQuaternion.z",
                    "QuaternionW": "UnitQuaternion.w",
                }
                
                # Rename the mapping columns if they exist
                rename_dict = {k: v for k, v in col_mapping.items() if k in df.columns}
                df.rename(columns=rename_dict, inplace=True)
                
                # Compute SessionTime
                if "Timestamp" in df.columns:
                    # Timestamp is in relative milliseconds
                    df["SessionTime"] = df["Timestamp"] / 1000.0
                else:
                    print(f"Timestamp column missing in {csv_file}")
                    continue

                # Select and rename columns
                required_columns = [
                    "SessionTime", 
                    "UnitQuaternion.x", "UnitQuaternion.y", "UnitQuaternion.z", "UnitQuaternion.w",
                    "HmdPosition.x", "HmdPosition.y", "HmdPosition.z"
                ]
                
                # Check if all required columns exist
                if not all(col in df.columns for col in required_columns):
                     missing = [col for col in required_columns if col not in df.columns]
                     print(f"Missing columns in {csv_file}: {missing}")
                     continue
                    
                final_df = df[required_columns]
                
                yield user_id, task_id, final_df
                
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

def parse_metadata(dataset_path):
    """
    Parses metadata for tasks (scenes).
    Returns: {'tasks': task_df}
    """
    metadata = {}
    
    # Parse tasks (Scene metadata)
    scene_settings_path = dataset_path / "main" / "EyeNavGS_Rutgers_Dataset-main" / "scene_setting.csv"
    if scene_settings_path.exists():
        try:
            task_df = pd.read_csv(scene_settings_path)
            # Rename Scene_Name to TaskID
            if "Scene_Name" in task_df.columns:
                task_df.rename(columns={"Scene_Name": "TaskID"}, inplace=True)
            metadata['tasks'] = task_df
        except Exception as e:
            print(f"Error parsing scene_setting.csv: {e}")
            
    return metadata
