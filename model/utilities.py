import os
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

np.random.seed(7777777)

def load_nwm_predictions(folder_path):
    """
    Reads all prediction files (starting with "streamflow_") and returns a DataFrame with these columns:
      - model_initialization_time
      - model_output_valid_time
      - streamflow_value
      - streamID (mapped to 0 and 1)
      - timestamp (equals model_output_valid_time)
    """
    prediction_files = [f for f in os.listdir(folder_path)
                        if f.startswith("streamflow_") and f.endswith(".csv")]
    if not prediction_files:
        return None
    dfs = []
    for file in prediction_files:
        df = pd.read_csv(os.path.join(folder_path, file))
        dfs.append(df)
    predictions_df = pd.concat(dfs, ignore_index=True)
    
    # Convert to datetime
    predictions_df['model_initialization_time'] = pd.to_datetime(
        predictions_df['model_initialization_time'], format='%Y-%m-%d_%H:%M:%S')
    predictions_df['model_output_valid_time'] = pd.to_datetime(
        predictions_df['model_output_valid_time'], format='%Y-%m-%d_%H:%M:%S')
    
    # Map streamID to numeric values
    unique_ids = sorted(predictions_df['streamID'].unique())
    id_map = {uid: idx for idx, uid in enumerate(unique_ids)}
    predictions_df['streamID'] = predictions_df['streamID'].map(id_map)
    
    # Set timestamp equal to model_output_valid_time
    predictions_df['timestamp'] = predictions_df['model_output_valid_time']
    
    # Keep only desired columns
    predictions_df = predictions_df[['model_initialization_time',
                                     'model_output_valid_time',
                                     'streamflow_value', 'streamID', 'timestamp']]
    predictions_df = predictions_df.sort_values('timestamp')
    return predictions_df

def load_usgs_observations(folder_path):
    """
    Reads the USGS observation file (containing 'Strt' in its name) and:
      - Converts DateTime to datetime (removing timezone)
      - Filters out any observations that are not on a round hour (minute != 0)
      - Returns a DataFrame with only:
            * timestamp (from DateTime)
            * USGSFlowValue
    """
    obs_files = [f for f in os.listdir(folder_path)
                 if "Strt" in f and f.endswith(".csv")]
    if not obs_files:
        return None
    observations_df = pd.read_csv(os.path.join(folder_path, obs_files[0]))
    
    # Convert DateTime to datetime and remove timezone
    observations_df['DateTime'] = pd.to_datetime(observations_df['DateTime']).dt.tz_localize(None)
    
    # Filter to only keep rows where minute is 0 (i.e. round hour)
    observations_df = observations_df[observations_df['DateTime'].dt.minute == 0]
    
    # Use DateTime as timestamp
    observations_df['timestamp'] = observations_df['DateTime']
    
    # Keep only desired columns
    observations_df = observations_df[['timestamp', 'USGSFlowValue']]
    observations_df = observations_df.sort_values('timestamp')
    return observations_df

def create_sliding_windows(data, window_size=8, step=1):
    """Create sliding windows from data sequences for more training examples."""
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i:i+window_size])
    return windows

def prepare_data_for_training(folder_path, window_size=8, step=2, use_sliding_windows=True):
    """
    Loads prediction and observation DataFrames, merges them on timestamp, groups by model_initialization_time
    to create supervised sequences, and converts them to PyTorch tensors.
    
    Features: input_time, output_time, prediction, streamID, and advanced time features
    Target: observed value
    """
    predictions_df = load_nwm_predictions(folder_path)
    observations_df = load_usgs_observations(folder_path)
    if predictions_df is None or observations_df is None:
        print(f"Failed to load data from {folder_path}")
        return None, None

    # Remove timezone info to ensure matching
    predictions_df['timestamp'] = predictions_df['timestamp'].dt.tz_localize(None)
    observations_df['timestamp'] = observations_df['timestamp'].dt.tz_localize(None)
    
    # Merge using an inner join (using only round-hour observations)
    merged_data = pd.merge(predictions_df, observations_df, on='timestamp', how='inner')
    if merged_data.empty:
        print("No matching records after merging.")
        return None, None

    # Enhanced feature engineering
    # ----------------------------
    # 1. Timestamp to numeric features
    merged_data['input_time_numeric'] = merged_data['model_initialization_time'].apply(lambda x: x.timestamp())
    merged_data['output_time_numeric'] = merged_data['timestamp'].apply(lambda x: x.timestamp())
    
    # 2. Cyclical encoding of time features
    merged_data['hour_sin'] = np.sin(2 * np.pi * merged_data['timestamp'].dt.hour / 24)
    merged_data['hour_cos'] = np.cos(2 * np.pi * merged_data['timestamp'].dt.hour / 24)
    merged_data['day_sin'] = np.sin(2 * np.pi * merged_data['timestamp'].dt.day / 31)
    merged_data['day_cos'] = np.cos(2 * np.pi * merged_data['timestamp'].dt.day / 31)
    merged_data['month_sin'] = np.sin(2 * np.pi * merged_data['timestamp'].dt.month / 12)
    merged_data['month_cos'] = np.cos(2 * np.pi * merged_data['timestamp'].dt.month / 12)
    
    # 3. Time delta from initialization (in hours)
    merged_data['hours_from_init'] = (merged_data['timestamp'] - 
                                    merged_data['model_initialization_time']).dt.total_seconds() / 3600
    
    # 4. Normalize numerical features
    scaler = MinMaxScaler()
    # Normalize streamflow_value
    streamflow_values = merged_data['streamflow_value'].values.reshape(-1, 1)
    merged_data['streamflow_norm'] = scaler.fit_transform(streamflow_values).flatten()
    
    # Group by model_initialization_time to form sequences
    feature_list = []
    target_list = []
    groups = merged_data.groupby('model_initialization_time')
    for init_time, group in groups:
        # Sort group by valid output time
        group = group.sort_values('model_output_valid_time')

        # For each row, create enhanced feature set
        features = np.column_stack((
            group['input_time_numeric'].values.astype(np.float32),
            group['output_time_numeric'].values.astype(np.float32),
            group['streamflow_value'].values.astype(np.float32),
            group['streamflow_norm'].values.astype(np.float32),
            group['streamID'].values.astype(np.float32),
            group['hour_sin'].values.astype(np.float32),
            group['hour_cos'].values.astype(np.float32),
            group['day_sin'].values.astype(np.float32),
            group['day_cos'].values.astype(np.float32),
            group['month_sin'].values.astype(np.float32),
            group['month_cos'].values.astype(np.float32),
            group['hours_from_init'].values.astype(np.float32)
        ))
        targets = group['USGSFlowValue'].values.astype(np.float32)
        
        # Only add sequences with more than minimum required records
        if len(features) > window_size and len(features) == len(targets):
            feature_list.append(features)
            target_list.append(targets)
    
    if not feature_list:
        print("No valid sequences found.")
        return None, None
    
    # Implement sliding window approach if requested
    enhanced_features = []
    enhanced_targets = []
    
    if use_sliding_windows:
        for features, targets in zip(feature_list, target_list):
            # Create sliding windows for each sequence
            feature_windows = create_sliding_windows(features, window_size, step)
            target_windows = create_sliding_windows(targets, window_size, step)
            enhanced_features.extend(feature_windows)
            enhanced_targets.extend(target_windows)
        
        if enhanced_features:  # Only update if we got windows
            feature_list = enhanced_features
            target_list = enhanced_targets
    else:
        # If not using sliding windows, truncate all sequences to same length
        min_len = min(len(seq) for seq in feature_list)
        feature_list = [np.array(seq[:min_len], dtype=np.float32) for seq in feature_list]
        target_list = [np.array(seq[:min_len], dtype=np.float32) for seq in target_list]
    
    # Convert lists to numpy arrays
    X = np.stack(feature_list, axis=0)  # Shape: (num_sequences, sequence_len, num_features)
    y = np.stack(target_list, axis=0)   # Shape: (num_sequences, sequence_len)

    # Add a dimension to y so that it becomes (num_sequences, sequence_len, 1)
    y = np.expand_dims(y, axis=-1)
    
    # Convert NumPy arrays to PyTorch tensors
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    return X_tensor, y_tensor

def create_dataloaders(X, y, batch_size=32, train_ratio=0.8, val_ratio=0.1):
    """Create train, validation, and test DataLoaders."""
    dataset = TensorDataset(X, y)
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    folder_path = "./data/20380357"  # Adjust as needed
    X_tensor, y_tensor = prepare_data_for_training(folder_path, window_size=8, step=2)
    if X_tensor is not None and y_tensor is not None:
        print("X tensor shape:", X_tensor.shape)
        print("y tensor shape:", y_tensor.shape)
        print("X features: [input_time, output_time, streamflow, streamflow_norm, streamID, hour_sin, hour_cos, ...]")
        print("First sequence first timestep:", X_tensor[0, 0])
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(X_tensor, y_tensor, batch_size=32)
        print(f"Created {len(train_loader)} training batches, {len(val_loader)} validation batches, {len(test_loader)} test batches")
