import os
import pandas as pd

def parse_metadata(dataset_dir):
    """
    Parses the dataset directory and returns a dictionary of tasks or users.
    For Panonut360, this is derived from directory structure rather than an external metadata file.
    """
    logs_dir = os.path.join(dataset_dir, 'Panonut360', 'Logs')
    tasks = set()
    
    if os.path.exists(logs_dir):
        for user_dir in os.listdir(logs_dir):
            user_path = os.path.join(logs_dir, user_dir)
            if os.path.isdir(user_path):
                for file in os.listdir(user_path):
                    if file.endswith('.csv'):
                        task_name = file.replace('.csv', '')
                        tasks.add(task_name)
                        
    return {
        "tasks": list(tasks)
    }

def parse(dataset_dir):
    """
    Yields dataframes processed for each user and task in the Panonut360 Dataset.
    Each dataframe conforms to formatting requirements (SessionTime, HmdPosition, etc.)
    """
    dataset_dir = str(dataset_dir)
    logs_dir = os.path.join(dataset_dir, 'Panonut360', 'Logs')
    
    if not os.path.exists(logs_dir):
        print(f"Log directory not found at {logs_dir}")
        return

    # Iterate through user directories
    for user_dir in os.listdir(logs_dir):
        user_path = os.path.join(logs_dir, user_dir)
        if not os.path.isdir(user_path):
            continue
            
        user_id = user_dir  # e.g., 'user1'
        
        # Iterate through task CSVs for the user
        for file in os.listdir(user_path):
            if not file.endswith('.csv'):
                continue
                
            task_id = file.replace('.csv', '')
            file_path = os.path.join(user_path, file)
            
            try:
                # Read CSV
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
                
            # player_time is formatted as 00:00:00.069
            # converting to total seconds
            df['SessionTime'] = pd.to_timedelta(df['player_time']).dt.total_seconds()
            
            # Map required columns
            df.rename(columns={
                'head_x': 'HmdPosition.x',
                'head_y': 'HmdPosition.y',
                'head_z': 'HmdPosition.z',
                'head_quaternion_x': 'UnitQuaternion.x',
                'head_quaternion_y': 'UnitQuaternion.y',
                'head_quaternion_z': 'UnitQuaternion.z',
                'head_quaternion_w': 'UnitQuaternion.w',
                
                # Optional eye tracking mapping
                'eye_x_t': 'GazeRay.x',
                'eye_y_t': 'GazeRay.y',
                'eye_z_t': 'GazeRay.z',
            }, inplace=True)
            
            df['IsEyeTrackingSample'] = 1  # Contains eye tracking data from the dataset
            
            # Ensure sorting by time
            df.sort_values(by='SessionTime', inplace=True)
            
            # Yield the dataframe for the core formatter to pick up
            yield user_id, task_id, df
