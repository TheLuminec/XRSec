import os
import pandas as pd

def parse_metadata(dataset_dir):
    """
    Parses the dataset directory and returns a dictionary of tasks.
    """
    data_dir = os.path.join(dataset_dir, '25749378')
    tasks = set()
    
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            user_path = os.path.join(data_dir, item)
            if os.path.isdir(user_path) and item.startswith("User"):
                for task_file in os.listdir(user_path):
                    if task_file.endswith('.csv'):
                        # typical file name User1_ArcSmoothPursuit_0.csv
                        parts = task_file.replace('.csv', '').split('_')
                        if len(parts) >= 3:
                            task_name = f"{parts[1]}_{parts[2]}"
                            tasks.add(task_name)

    return {
        "tasks": list(tasks)
    }

def parse_dataset(dataset_dir):
    """
    Yields dataframes processed for each user and structured properly.
    """
    data_dir = os.path.join(dataset_dir, '25749378')
    
    if not os.path.exists(data_dir):
        print(f"Directory not found at {data_dir}")
        return

    # Iterate through user directories
    for item in os.listdir(data_dir):
        user_path = os.path.join(data_dir, item)
        # Verify it's a User directory
        if not os.path.isdir(user_path) or not item.startswith("User"):
            continue
            
        # Keep numeric ID mapping e.g., 'User1' -> '1' or raw 'User1' depending on system mappings
        # Let's keep 'User1' format string identifier
        user_id = item
        
        # Process each task file inside the User
        for task_file in os.listdir(user_path):
            if not task_file.endswith('.csv'):
                continue
                
            task_file_path = os.path.join(user_path, task_file)
            parts = task_file.replace('.csv', '').split('_')
            
            # parts will be ['User1', 'ArcSmoothPursuit', '0']
            if len(parts) >= 3:
                task_id = f"{parts[1]}_{parts[2]}"
            else:
                task_id = parts[1] if len(parts) > 1 else 'Unknown'

            try:
                # Read CSV
                df = pd.read_csv(task_file_path)
            except Exception as e:
                print(f"Error reading {task_file_path}: {e}")
                continue
                
            # 'time_stamp(ms)' converted to total seconds
            df['SessionTime'] = df['time_stamp(ms)'] / 1000.0
            
            # Map head tracking structure coordinates
            df.rename(columns={
                'head_x': 'HmdPosition.x',
                'head_y': 'HmdPosition.y',
                'head_z': 'HmdPosition.z',
            }, inplace=True)
            
            # The Left and Right vectors averaged to form 'GazeRay' 
            df['GazeRay.x'] = (df['eye_in_head_left_x'] + df['eye_in_head_right_x']) / 2.0
            df['GazeRay.y'] = (df['eye_in_head_left_y'] + df['eye_in_head_right_y']) / 2.0
            df['GazeRay.z'] = (df['eye_in_head_left_z'] + df['eye_in_head_right_z']) / 2.0
            
            # Adding Formatter Identifier Requirements
            df['User'] = user_id
            df['Task'] = task_id
            df['IsEyeTrackingSample'] = 1  # Contains eye tracking data from the dataset
            
            # Ensure sorting by chronological timestamp metrics
            df.sort_values(by='time_stamp(ms)', inplace=True)
            
            # Yield the dataframe for the core formatting operation 
            yield df
