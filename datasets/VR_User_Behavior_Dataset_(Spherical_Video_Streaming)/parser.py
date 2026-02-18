import pandas as pd
import glob
import os

def parse(dataset_path):
    """
    Parses the VR User Behavior Dataset.
    Yields: (user_id, task_id, dataframe)
    """
    experiments = ["Experiment_1", "Experiment_2"]
    
    for exp in experiments:
        exp_path = dataset_path / "Formated_Data" / exp
        if not exp_path.exists():
            continue
            
        # Get user directories (numbered folders)
        user_dirs = [d for d in os.listdir(exp_path) if d.isdigit() and (exp_path / d).is_dir()]
        
        for user_id in user_dirs:
            user_path = exp_path / user_id
            csv_files = glob.glob(str(user_path / "video_*.csv"))
            
            for csv_file in csv_files:
                try:
                    # Extract video ID from filename
                    filename = os.path.basename(csv_file)
                    video_id =  filename.split(".")[0] # e.g., video_0
                    task_id = f"{exp.lower()}_{video_id}"
                    
                    df = pd.read_csv(csv_file)
                    
                    # Compute SessionTime
                    if "Timestamp" in df.columns:
                        try:
                            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                            start_time = df["Timestamp"].min()
                            df["SessionTime"] = (df["Timestamp"] - start_time).dt.total_seconds()
                        except Exception as e:
                             print(f"Error parsing timestamp for {task_id}: {e}")
                             continue
                    else:
                        print(f"Timestamp column missing in {csv_file}")
                        continue

                    # Select and rename columns
                    required_columns = [
                        "SessionTime", 
                        "UnitQuaternion.x", "UnitQuaternion.y", "UnitQuaternion.z", "UnitQuaternion.w",
                        "HmdPosition.x", "HmdPosition.y", "HmdPosition.z"
                    ]
                    
                    # Check if all columns exist
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
    Parses metadata for users and tasks.
    Returns: {'users': user_df, 'tasks': task_df}
    """
    metadata = {}
    
    # 1. Parse Users
    user_demo_path = dataset_path / "Formated_Data" / "userDemo.csv"
    if user_demo_path.exists():
        try:
            users_df = pd.read_csv(user_demo_path)
            # Rename 'No.' to 'UserID' if present
            if "No." in users_df.columns:
                users_df.rename(columns={"No.": "UserID"}, inplace=True)
            metadata['users'] = users_df
        except Exception as e:
            print(f"Error parsing userDemo.csv: {e}")
            
    # 2. Parse Tasks (Video Metadata)
    experiments = ["Experiment_1", "Experiment_2"]
    all_tasks = []
    
    for exp in experiments:
        meta_path = dataset_path / "Formated_Data" / exp / "videoMeta.csv"
        if meta_path.exists():
            try:
                task_df = pd.read_csv(meta_path)
                # Create TaskID: experiment_{N}_video_{VideoNo-1}
                # Assuming VideoNo starts at 1 and maps to video_0, video_1...
                exp_num = exp.split("_")[1]
                
                def generate_task_id(row):
                    if "VideoNo" in row:
                        video_index = int(row["VideoNo"]) - 1
                        return f"experiment_{exp_num}_video_{video_index}"
                    return None

                task_df["TaskID"] = task_df.apply(generate_task_id, axis=1)
                all_tasks.append(task_df)
                
            except Exception as e:
                print(f"Error parsing {exp} metadata: {e}")

    if all_tasks:
        metadata['tasks'] = pd.concat(all_tasks, ignore_index=True)
        
    return metadata
