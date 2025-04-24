import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime, timedelta

# Fix the folder paths to be a list, not a set
folder_paths = ["./data/20380357", "./data/21609641"]
all_files = []

for path in folder_paths:
    try:
        files = os.listdir(path)
        all_files.extend([os.path.join(path, file) for file in files if file.endswith('.csv')])
    except FileNotFoundError:
        print(f"Folder {path} not found. Skipping.")

def load_nwm_predictions(folder_path):
    """Load and combine all NWM prediction files."""
    prediction_files = [f for f in os.listdir(folder_path) if f.startswith('streamflow_')]
    all_predictions = []
    
    for file in prediction_files:
        df = pd.read_csv(os.path.join(folder_path, file))
        all_predictions.append(df)
    
    # Combine all prediction dataframes
    if all_predictions:
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        
        # Convert string timestamps to datetime objects
        predictions_df['model_initialization_time'] = pd.to_datetime(
            predictions_df['model_initialization_time'], format='%Y-%m-%d_%H:%M:%S')
        predictions_df['model_output_valid_time'] = pd.to_datetime(
            predictions_df['model_output_valid_time'], format='%Y-%m-%d_%H:%M:%S')
        
        return predictions_df
    return None

def load_usgs_observations(folder_path):
    """Load and process USGS observation data."""
    obs_files = [f for f in os.listdir(folder_path) if 'Strt' in f and f.endswith('.csv')]
    
    if not obs_files:
        return None
    
    # Load the observation data
    observations_df = pd.read_csv(os.path.join(folder_path, obs_files[0]))
    
    # Convert DateTime to datetime objects
    observations_df['DateTime'] = pd.to_datetime(observations_df['DateTime']).dt.tz_localize(None)

    # Filter for hour timestamps
    observations_df = observations_df[observations_df['DateTime'].dt.minute == 0]
    
    return observations_df


def align_data(predictions_df, observations_df):
    """Align prediction and observation data by timestamp."""
    # Create a key in each dataframe for easy alignment
    predictions_df['timestamp'] = predictions_df['model_output_valid_time']
    
    # Create timestamp column in observations dataframe
    observations_hourly = observations_df.copy()
    observations_hourly['timestamp'] = observations_hourly['DateTime']
    
    # Now drop the original DateTime column
    observations_hourly.drop(columns=['DateTime'], inplace=True)
    
    # Print column names to debug
    print("Prediction columns:", predictions_df.columns.tolist())
    print("Observation columns:", observations_hourly.columns.tolist())
    
    # Merge the dataframes on the aligned timestamps
    merged_data = pd.merge(predictions_df, observations_hourly, on='timestamp', how='inner')
    
    return merged_data

def create_supervised_dataset(merged_data):
    """Create feature-target pairs for supervised learning."""
    # Group by initialization time to get prediction sequences
    feature_list = []
    target_list = []
    
    # Get unique initialization times
    init_times = merged_data['model_initialization_time'].unique()
    
    for init_time in init_times:
        # Get predictions for this initialization time
        predictions = merged_data[merged_data['model_initialization_time'] == init_time]
        
        if len(predictions) > 0:
            # Sort by valid time to ensure proper sequence
            predictions = predictions.sort_values('model_output_valid_time')
            
            # Features: predicted streamflow values
            features = predictions['streamflow_value'].values
            
            # Target: actual observed values
            targets = predictions['USGSFlowValue'].values
            
            # Only use complete sequences
            if len(features) == len(targets) and len(features) > 0:
                feature_list.append(features)
                target_list.append(targets)
    
    # Convert to numpy arrays
    X = np.array(feature_list)
    y = np.array(target_list)
    
    return X, y

def process_data_for_nn(folder_path):
    """Process all data from a folder for neural network training."""
    # Load prediction and observation data
    predictions_df = load_nwm_predictions(folder_path)
    observations_df = load_usgs_observations(folder_path)
    
    if predictions_df is None or observations_df is None:
        print(f"Could not load data from {folder_path}")
        return None, None
    
    # Align data
    merged_data = align_data(predictions_df, observations_df)
    
    # Create supervised dataset
    X, y = create_supervised_dataset(merged_data)
    
    return X, y

# Example usage
if __name__ == "__main__":
    folder_paths = ["./data/20380357", "./data/21609641"]
    for path in folder_paths:
        if os.path.exists(path):
            print(f"Processing data from {path}")
            X, y = process_data_for_nn(path)
            if X is not None and y is not None:
                print(f"Created dataset with {X.shape[0]} sequences")
                print(f"Feature shape: {X.shape}, Target shape: {y.shape}")

