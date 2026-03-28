import os
import shutil

def main():
    base_path = r"c:\Users\thelu\Desktop\GIT\XRSec\datasets"
    
    for dataset_name in os.listdir(base_path):
        dataset_path = os.path.join(base_path, dataset_name)
        
        # Make sure it's a directory
        if not os.path.isdir(dataset_path):
            continue
            
        raw_data_path = os.path.join(dataset_path, "raw_data")
        processed_data_path = os.path.join(dataset_path, "processed_data")
        
        # Create raw_data and processed_data if they don't exist
        if not os.path.exists(raw_data_path):
            os.makedirs(raw_data_path)
        if not os.path.exists(processed_data_path):
            os.makedirs(processed_data_path)
            
        # Move everything else in the dataset folder to raw_data
        for item in os.listdir(dataset_path):
            if item in ["raw_data", "processed_data"]:
                continue
                
            src_path = os.path.join(dataset_path, item)
            dst_path = os.path.join(raw_data_path, item)
            
            # Using shutil.move to move data, not copy
            shutil.move(src_path, dst_path)
            
    print("Dataset organization completed successfully.")

if __name__ == "__main__":
    main()
