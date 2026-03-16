import pandas as pd
import os
from train import train, plot_training_history

def validate_dataset(data_dir):
    """
    Validates the dataset to ensure it is properly formatted.
    Samples several files to validate the dataset, assuming it is consistent.

    Columns required:
    - SessionTime
    - UnitQuaternion.x
    - UnitQuaternion.y
    - UnitQuaternion.z
    - UnitQuaternion.w
    - HmdPosition.x
    - HmdPosition.y
    - HmdPosition.z

    Minimum Hertz: 10Hz
    """
    
    # Get all users
    users = os.listdir(data_dir)
    
    # Sample 5 users
    sampled_users = users[:5]
    
    for user in sampled_users:
        user_dir = os.path.join(data_dir, user)
        
        # Get all sessions for the user
        sessions = os.listdir(user_dir)
        
        # Sample 5 sessions
        sampled_sessions = sessions[:5]
        
        for session in sampled_sessions:
            session_path = os.path.join(user_dir, session)
            
            # Read the CSV file
            df = pd.read_csv(session_path)
            
            # Check columns
            required_columns = [
                "SessionTime",
                "UnitQuaternion.x",
                "UnitQuaternion.y",
                "UnitQuaternion.z",
                "UnitQuaternion.w",
                "HmdPosition.x",
                "HmdPosition.y",
                "HmdPosition.z"
            ]
            
            for col in required_columns:
                if col not in df.columns:
                    print(f"Missing column {col} in {session_path}")
                    return False
            
            # Check Hertz
            if len(df) / df["SessionTime"].iloc[-1] < 10:
                print(f"Low Hertz in {session_path}")
                return False
    
    print("Validation complete")
    return True

def save_dataset_profile(dataset_profile, save_path):
    """
    Saves the dataset profile to a CSV file.
    """
    df = pd.DataFrame(dataset_profile)
    df.to_csv(save_path, index=False)
    print(f"Dataset profile saved to {save_path}")

def validate_all_datasets(dataset_dir):
    """
    Validates all datasets in the dataset directory.
    """
    dataset_profile = {
        "name": [],
        "users": [],
        "sessions": [],
        "valid": [],
        "learning_acc": []
    }

    # Get all datasets
    datasets = os.listdir(dataset_dir)
    
    for dataset in datasets:
        dataset_path = os.path.join(dataset_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
        users = os.path.join(dataset_path, "processed_data", "users")
        if not os.path.isdir(users):
            continue
        valid = validate_dataset(users)
        if valid:
            try:
                learning_acc = learning_check(dataset_path)
            except:
                learning_acc = "Error"
        else:
            learning_acc = "Invalid"

        dataset_profile["name"].append(dataset)
        dataset_profile["users"].append(len(os.listdir(users)))
        dataset_profile["sessions"].append(len(os.listdir(os.path.join(users, os.listdir(users)[0]))))
        dataset_profile["valid"].append(valid)
        dataset_profile["learning_acc"].append(learning_acc)
    
    return dataset_profile

def learning_check(data_dir):
    """
    Loads and runs the dataset through the model to check for learning.
    Returns the increase in accuracy over the 5 epochs.
    """
    args = {
        "data_dir": os.path.join(data_dir, "processed_data", "users"),
        "batch_size": 2048,
        "test_dir": None,
        "epochs": 5,
        "save_path": os.path.join(data_dir, "processed_data", "model.pth"),
        "embedding_dim": 128,
        "lr": 0.001
    }
    history = train(args)
    plot_training_history(history, os.path.join(data_dir, "processed_data", "training_history.png"))
    learning_acc = (history["test_acc"][-1] - history["test_acc"][0]) / history["test_acc"][0]
    return learning_acc

if __name__ == "__main__":
    dataset_profile = validate_all_datasets("datasets")
    save_dataset_profile(dataset_profile, "dataset_profile.csv")