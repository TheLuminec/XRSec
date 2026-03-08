import os
import pandas as pd
import scipy.io
from scipy.spatial.transform import Rotation as R

def parse_metadata(dataset_dir):
    """
    Returns generic metadata matching the NJIT_6DOF VR traces framework.
    Task is generally mapped universally to a generic 'Navigation'.
    """
    return {
        "tasks": ["Navigation"]
    }

def extract_mat_frame(mat_path):
    """
    Extracts the mobility matrix from a `.mat` filepath targeting dynamic keys.
    """
    mat = scipy.io.loadmat(mat_path)
    for key in mat.keys():
        if not key.startswith('__'):
            matrix = mat[key]
            # Ensure it fits the geometry expected 30000x6
            if hasattr(matrix, 'shape') and len(matrix.shape) == 2 and matrix.shape[1] == 6:
                return matrix
    return None

def parse(dataset_dir):
    """
    Yields dataframes for each user inside the VR traces folder.
    Maps .mat sequences representing coordinates into formatted columns.
    """
    dataset_dir = str(dataset_dir)
    traces_dir = os.path.join(dataset_dir, 'Traces_6DOF_VR_NJIT')
    
    if not os.path.exists(traces_dir):
        print(f"Directory not found at {traces_dir}")
        return

    for file in os.listdir(traces_dir):
        if not file.endswith('.mat'):
            continue
            
        file_path = os.path.join(traces_dir, file)
        
        # Extract user ID, format is node1mobility.mat
        user_id = file.replace('mobility.mat', '').replace('node', '')
        task_id = 'Navigation'
        
        try:
            matrix = extract_mat_frame(file_path)
            if matrix is None:
                continue
                
            # Parse matrix via Pandas wrapper
            # columns = [x, y, z, yaw, pitch, roll]
            df = pd.DataFrame(matrix, columns=['x', 'y', 'z', 'yaw', 'pitch', 'roll'])
            
            # Formulating output parameters according to expected structure
            # Map SessionTime relying on standard 250 Hz increment rate
            df['SessionTime'] = df.index / 250.0
            
            # Standard positional offsets map exactly 1-to-1
            df.rename(columns={
                'x': 'HmdPosition.x',
                'y': 'HmdPosition.y',
                'z': 'HmdPosition.z',
            }, inplace=True)
            
            # Convert yaw-pitch-roll configurations into expected quaternion derivations
            # Assuming typical VR orientations use degrees, applying order ZYX
            euler_angles = df[['yaw', 'pitch', 'roll']].values
            rotations = R.from_euler('ZYX', euler_angles, degrees=True)
            quats = rotations.as_quat() # format defaults to (x, y, z, w)
            
            df['UnitQuaternion.x'] = quats[:, 0]
            df['UnitQuaternion.y'] = quats[:, 1]
            df['UnitQuaternion.z'] = quats[:, 2]
            df['UnitQuaternion.w'] = quats[:, 3]
            
            # Eye tracking dataset feature false flag
            df['IsEyeTrackingSample'] = 0
            
            df.drop(columns=['yaw', 'pitch', 'roll'], inplace=True)
            yield user_id, task_id, df
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
