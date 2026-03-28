import os
import shutil

def main():
    base_path = r"c:\Users\thelu\Desktop\GIT\XRSec"
    old_datasets_path = os.path.join(base_path, "datasets")
    raw_datasets_path = os.path.join(base_path, "raw_datasets")
    processed_datasets_path = os.path.join(base_path, "processed_datasets")
    
    # Create the new base directories if they don't exist
    if not os.path.exists(raw_datasets_path):
        os.makedirs(raw_datasets_path)
    if not os.path.exists(processed_datasets_path):
        os.makedirs(processed_datasets_path)
        
    if not os.path.exists(old_datasets_path):
        print("old datasets path does not exist.")
        return
        
    for dataset_name in os.listdir(old_datasets_path):
        dataset_path = os.path.join(old_datasets_path, dataset_name)
        
        if not os.path.isdir(dataset_path):
            continue
            
        old_raw_dir = os.path.join(dataset_path, "raw_data")
        old_processed_dir = os.path.join(dataset_path, "processed_data")
        
        new_raw_dir = os.path.join(raw_datasets_path, dataset_name)
        new_processed_dir = os.path.join(processed_datasets_path, dataset_name)
        
        if os.path.isdir(old_raw_dir):
            if not os.path.exists(new_raw_dir):
                os.makedirs(new_raw_dir)
            for item in os.listdir(old_raw_dir):
                shutil.move(os.path.join(old_raw_dir, item), os.path.join(new_raw_dir, item))
            # Remove the now-empty old_raw_dir
            if not os.listdir(old_raw_dir):
                os.rmdir(old_raw_dir)
            
        if os.path.isdir(old_processed_dir):
            if not os.path.exists(new_processed_dir):
                os.makedirs(new_processed_dir)
            for item in os.listdir(old_processed_dir):
                shutil.move(os.path.join(old_processed_dir, item), os.path.join(new_processed_dir, item))
            # Remove the now-empty old_processed_dir
            if not os.listdir(old_processed_dir):
                os.rmdir(old_processed_dir)
            
        # If the dataset_path is now empty, remove it
        if not os.listdir(dataset_path):
            os.rmdir(dataset_path)

    # Finally, if old_datasets_path is completely empty, we can remove it.
    if not os.listdir(old_datasets_path):
        os.rmdir(old_datasets_path)
        
    print("Dataset restructuring to new base folders completed successfully.")

if __name__ == "__main__":
    main()
