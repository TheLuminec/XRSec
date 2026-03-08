import os
import glob
import pandas as pd
from pathlib import Path

def parse(dataset_path):
    """
    Parses the ViewGauss_Head-Movement_Dataset.
    Yields: (user_id, task_id, dataframe)
    """
    ds_dir = dataset_path / "ViewGauss-DataSet" / "dataset"
    if not ds_dir.exists():
        print(f"Directory not found: {ds_dir}")
        return
        
    csv_files = [f for f in glob.glob(str(ds_dir / "*.csv")) if os.path.isfile(f)]
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        task_id = filename.replace(".csv", "") # e.g. "sequence1"
        
        try:
            df_full = pd.read_csv(filepath)
            
            # Identify individual users based on when 'Frame' resets
            df_full['UserBlock'] = (df_full['Frame'] < df_full['Frame'].shift(1).fillna(0)).cumsum()
            
            # The paper mentions a 10Hz sampling rate
            # We can convert 'Frame' into 'SessionTime' in seconds
            
            # Map columns
            # Raw: Frame,PosX,PosY,PosZ,RotX,RotY,RotZ,RotW
            # Target: SessionTime, UnitQuaternion.x, y, z, w, HmdPosition.x, y, z
            
            for block_id, group in df_full.groupby('UserBlock'):
                user_id = str(block_id)
                
                out_df = pd.DataFrame()
                out_df['SessionTime'] = (group['Frame'] - 1) / 10.0 # 0-indexed seconds
                out_df['UnitQuaternion.x'] = group['RotX']
                out_df['UnitQuaternion.y'] = group['RotY']
                out_df['UnitQuaternion.z'] = group['RotZ']
                out_df['UnitQuaternion.w'] = group['RotW']
                out_df['HmdPosition.x'] = group['PosX']
                out_df['HmdPosition.y'] = group['PosY']
                out_df['HmdPosition.z'] = group['PosZ']
                
                yield user_id, task_id, out_df
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

def parse_metadata(dataset_path):
    """
    Parses metadata for tasks.
    Returns: {'tasks': task_df}
    """
    ds_dir = dataset_path / "ViewGauss-DataSet" / "dataset"
    task_df = pd.DataFrame(columns=["TaskID"])
    
    if ds_dir.exists():
        csv_files = [f for f in glob.glob(str(ds_dir / "*.csv")) if os.path.isfile(f)]
        tasks = [os.path.basename(f).replace(".csv", "") for f in csv_files]
        task_df["TaskID"] = tasks

    return {'tasks': task_df}
